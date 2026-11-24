"""Train one model. --check is offline; weight downloads are a separate explicit action."""
import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import traceback
from functools import partial
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent / "training/support"))
from fusion.runtime import (ROOT, ARTIFACTS, DIAGNOSTICS, MODELS, bootstrap, environment, fingerprint,
                            local_output, metrics_dict, now, read_json, result_root, sha256, source_hash, stamp, write_json)


class SmokeCheckpoint(Exception):
    """End the first part of a deliberate save/resume check."""


def add_arguments(parser):
    parser.add_argument("--data", help="Local YOLO data.yaml/.yml (images + labels), or detection NDJSON")
    parser.add_argument("--model-set", type=Path, help="Architecture collection created by the fusion Notebook")
    parser.add_argument("--config", default=str(ROOT / "configs/train.yaml"))
    parser.add_argument("--weights", default=str(ARTIFACTS / "weights/yolo26s.pt"), help="Original YOLO26s COCO weights")
    parser.add_argument("--vit-weights", default=str(ARTIFACTS / "weights/vit_b_16_imagenet1k_v1.pth"))
    for key in ("epochs", "batch", "seed", "imgsz", "patience", "workers"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--device", help="cpu or one CUDA device index")
    parser.add_argument("--check", action="store_true", help="Inspect data/config/structure without conversion or training")


def configuration(args):
    from ultralytics.cfg import get_cfg
    defaults = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8-sig"))
    supplied = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if not isinstance(supplied, dict) or set(supplied) - set(defaults):
        raise ValueError("Config must contain only supported keys from configs/train.yaml")
    params = {**defaults, **supplied}
    for key in ("epochs", "batch", "seed", "imgsz", "patience", "workers", "device"):
        if getattr(args, key, None) is not None:
            params[key] = getattr(args, key)
    if str(params["device"]) != "cpu" and not str(params["device"]).isdigit():
        raise ValueError("Only cpu or one CUDA device index is supported")
    if min(params["epochs"], params["batch"], params["imgsz"]) <= 0 or params["imgsz"] % 32:
        raise ValueError("epochs/batch must be positive; imgsz must be a positive multiple of 32")
    if min(params["seed"], params["workers"], params["patience"]) < 0:
        raise ValueError("seed/workers/patience must be nonnegative")
    if not params["save"] or not params["val"] or not params["pretrained"] or params["fraction"] != 1:
        raise ValueError("Controlled runs require save/val/pretrained=true and fraction=1")
    if params["amp"] or params["optimizer"] != "AdamW":
        raise ValueError("This experiment protocol fixes AMP=false and optimizer=AdamW")
    get_cfg(overrides=params)
    return params


def download_weights(which):
    from torchvision.models import ViT_B_16_Weights
    from torch.hub import download_url_to_file
    destinations = {
        "yolo": ("yolo26s.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt", None),
        "vit": ("vit_b_16_imagenet1k_v1.pth", ViT_B_16_Weights.IMAGENET1K_V1.url,
                Path(ViT_B_16_Weights.IMAGENET1K_V1.url).stem.rsplit("-", 1)[1]),
    }
    for key in destinations if which == "all" else [which]:
        name, url, hash_prefix = destinations[key]
        path = ARTIFACTS / "weights" / name
        if path.is_file():
            print(f"Reusing {path.name}: {sha256(path)}")
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            download_url_to_file(url, str(temporary), hash_prefix=hash_prefix)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        write_json(path.with_suffix(".source.json"), {"url": url, "sha256": sha256(path)})
        print(f"Saved {path}")


def resume_settings(args, data):
    from fusion.models import load_model
    checkpoint = Path(args.resume).resolve()
    if checkpoint.name != "last.pt" or not checkpoint.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("Resume requires this project's weights/last.pt and its run records")
    source = checkpoint.parent.parent
    for relative in ("run.json", "initial_transfer.json", "results.csv", "weights/last.pt", "weights/best.pt"):
        if not (source / relative).is_file():
            raise ValueError(f"Resume source is missing {relative}")
    parent = read_json(source / "run.json")
    if parent.get("status") not in ("failed", "interrupted", "diagnostic_passed"):
        raise ValueError("Resume requires a stopped incomplete run")
    if parent.get("kind") != "train" and not getattr(args, "diagnose_batches", None):
        raise ValueError("Resume of a diagnostic requires --diagnose-batches")
    if data["sha256"] != parent["data"]["sha256"]:
        raise ValueError("Resume data differs from the saved experiment")
    params = parent["effective_training"].copy()
    for key in ("epochs", "batch", "seed", "imgsz", "patience", "workers", "device"):
        value = getattr(args, key, None)
        if value is not None and str(value) != str(params[key]):
            raise ValueError(f"Resume must preserve {key}={params[key]}")
    if args.model is not None and args.model != parent["model"]:
        raise ValueError("Resume model differs from the saved experiment")
    if Path(args.config).resolve() != ROOT / "configs/train.yaml":
        raise ValueError("Resume uses the checkpoint configuration; omit --config")
    current_environment = environment(params["device"])
    if current_environment != parent["environment"]:
        differing = sorted(k for k in set(current_environment) | set(parent["environment"])
                           if current_environment.get(k) != parent["environment"].get(k))
        raise ValueError(f"Resume environment differs: {', '.join(differing)}; use the original environment")
    model = load_model(checkpoint)
    ckpt = model.ckpt
    completed = ckpt.get("epoch", -1) + 1
    if not 0 < completed < params["epochs"] or ckpt.get("optimizer") is None:
        raise ValueError("Resume requires an unfinished checkpoint with optimizer state")
    if completed != parent.get("completed_epochs") or model.model.variant != parent["model"]:
        raise ValueError("Resume checkpoint disagrees with its run record")
    if any(str(ckpt["train_args"].get(k)) != str(v) for k, v in params.items()):
        raise ValueError("Resume checkpoint training configuration disagrees with its run record")
    args.model = parent["model"]
    return source, parent, params, current_environment


def run(args):
    import torch
    from ultralytics import utils
    from fusion.data import inspect_data
    from fusion.models import build_model, load_model
    from fusion.engine import StudyValidator
    from fusion.records import refresh_safely
    from evaluate import evaluation_settings
    from fusion.diagnostics import DiagnosticComplete, DiagnosticTrainer
    diagnostic_batches = getattr(args, "diagnose_batches", None)
    data = inspect_data(args.data)
    source = parent = None
    if getattr(args, "resume", None):
        source, parent, params, current_environment = resume_settings(args, data)
    else:
        args.model = args.model or "baseline"
        params = configuration(args)
        current_environment = environment(params["device"])
    saved_models = getattr(args, "model_set", None)
    arch = {}
    if saved_models and not source:
        definitions = yaml.safe_load(saved_models.read_text(encoding="utf-8"))
        arch = {"architecture": definitions[args.model]}
    if args.check:
        if source:
            print(f"Resume checked; no training started. {args.model}: epoch {parent['completed_epochs'] + 1} "
                  f"through {params['epochs']}; original directory will be preserved: {source}")
            return
        model, _ = build_model(args.model, params["seed"], check=True, deterministic=params["deterministic"], **arch)
        import json
        print(json.dumps({"status": "structure_checked_not_trained", "model": args.model, "training": params,
                          "data": data, "parameters_80_classes": sum(p.numel() for p in model.model.parameters()),
                          "yolo_weights_available": Path(args.weights).is_file(),
                          "vit_weights_available": Path(args.vit_weights).is_file(),
                          "note": "No weights loaded; pretrained transfer is NOT checked here"}, ensure_ascii=False, indent=2))
        return
    root = DIAGNOSTICS if diagnostic_batches else result_root(data, "runs")
    directory = local_output(args.run_dir or root / f"{args.model}_seed{params['seed']}_{stamp()}")
    if diagnostic_batches and (not directory.is_relative_to(DIAGNOSTICS.resolve()) or directory == DIAGNOSTICS.resolve()):
        raise ValueError("Diagnostics must use a new subdirectory of artifacts/diagnostics/")
    directory.mkdir(parents=True, exist_ok=False)
    state = {"id": directory.relative_to(ARTIFACTS).as_posix(), "kind": "diagnostic" if diagnostic_batches else "train", "model": args.model, "seed": params["seed"],
             "status": "running", "started_utc": now(), "pid": os.getpid(), "data": data,
             "requested_training": params, "environment": current_environment, "code_sha256": source_hash()}
    if saved_models:
        state["model_set_sha256"] = sha256(saved_models)
    elif source and parent.get("model_set_sha256"):
        state["model_set_sha256"] = parent["model_set_sha256"]
    if source:
        state.update(completed_epochs=parent["completed_epochs"], best_epoch=parent.get("best_epoch"),
                     best_epoch_source=parent.get("best_epoch_source"), resume={
            "parent_id": parent["id"], "checkpoint_sha256": sha256(source / "weights/last.pt"),
            "parent_code_sha256": parent["code_sha256"],
            "interruption_epochs": parent.get("resume", {}).get("interruption_epochs", []) + [parent["completed_epochs"]],
            "method": "Ultralytics native resume: saved EMA/optimizer/scaler; RNG and partial epoch are not restored"})
    if diagnostic_batches:
        state["diagnose_batches"] = diagnostic_batches
    write_json(directory / "run.json", state)
    log = (directory / "console.log").open("w", encoding="utf-8")
    handler = logging.StreamHandler(log)
    utils.LOGGER.addHandler(handler)
    model = None
    try:
        device = str(params["device"])
        if device != "cpu" and (not torch.cuda.is_available() or int(device) >= torch.cuda.device_count()):
            raise ValueError(f"Requested CUDA device is unavailable: {device}")
        if source:
            (directory / "weights").mkdir()
            for relative in ("results.csv", "initial_transfer.json", "weights/last.pt", "weights/best.pt"):
                shutil.copy2(source / relative, directory / relative)
            model = load_model(directory / "weights/last.pt")
            transfer = read_json(directory / "initial_transfer.json")
        else:
            model, transfer = build_model(args.model, params["seed"], args.weights, args.vit_weights,
                                          deterministic=params["deterministic"], **arch)
        write_json(directory / "initial_transfer.json", transfer)
        state["yolo_sha256"] = transfer["yolo"]["sha256"]
        state["vit_sha256"] = transfer.get("vit", {}).get("sha256")
        state["model_config"] = model.model.yaml
        state["initial_weights"] = parent["initial_weights"] if source else str(Path(args.weights).resolve())
        smoke_parameters = {}

        def remember_branches(trainer):
            from fusion.mamba import ResidualSS2D
            from fusion.models import P5ResidualViT
            for name, module in trainer.model.named_modules():
                if isinstance(module, (ResidualSS2D, P5ResidualViT)):
                    mixer = module.select if isinstance(module, ResidualSS2D) else module.encoder.self_attention.in_proj_weight
                    for key, parameter in (("alpha", module.alpha), ("mixer", mixer)):
                        smoke_parameters[name + "." + key] = (parameter, parameter.detach().clone())

        def record_progress(trainer):
            effective = {k: getattr(trainer.args, k) for k in params}
            effective["amp"] = bool(trainer.amp)
            state["effective_training"] = effective
            state["validation"] = {"batch": trainer.test_loader.batch_size, "precision": "fp32",
                                   "images_seen": trainer.validator.seen}
            state["determinism"] = {"enabled": torch.are_deterministic_algorithms_enabled(),
                                    "warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}
            state["parameters"] = sum(p.numel() for p in trainer.model.parameters())
            state["protocol_key"] = fingerprint({"data": data["sha256"], "code": state["code_sha256"],
                "environment": state["environment"], "determinism": state["determinism"], "yolo": state["yolo_sha256"],
                "training": {k: v for k, v in effective.items() if k != "seed"},
                "model_set": state.get("model_set_sha256"),
                "resume_epochs": state.get("resume", {}).get("interruption_epochs", []),
                "epoch_validation": {k: v for k, v in state["validation"].items() if k != "images_seen"},
                "evaluation": evaluation_settings(params["imgsz"], params["batch"], device)})
            write_json(directory / "effective_args.json", vars(trainer.args))
            write_json(directory / "run.json", state)

        def record_checkpoint(trainer):
            state["completed_epochs"] = trainer.epoch + 1
            if trainer.best_fitness == trainer.fitness:
                state["best_epoch"] = trainer.epoch + 1
                state["best_epoch_source"] = "on_model_save: best.pt written by mAP50-95 fitness"
            record_progress(trainer)
            if getattr(args, "smoke_checkpoint", False) and trainer.epoch == 0:
                changes = {k: bool(torch.isfinite(p).all() and not torch.equal(p, initial))
                           for k, (p, initial) in smoke_parameters.items()}
                state["smoke_branch_updates"] = changes
                if not all(changes.values()):
                    raise RuntimeError(f"Smoke training did not update all branch parameters: {changes}")
                raise SmokeCheckpoint()

        model.add_callback("on_train_start", record_progress)
        if getattr(args, "smoke_checkpoint", False):
            model.add_callback("on_train_start", remember_branches)
        model.add_callback("on_model_save", record_checkpoint)
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats(int(device))
        # Dataset validation and cache preparation are shared with independent evaluation.
        trainer_class = partial(DiagnosticTrainer, batch_limit=diagnostic_batches)
        model.train(data=str(Path(args.data).resolve()), trainer=trainer_class,
                    project=str(directory.parent), name=directory.name, save_dir=str(directory),
                    exist_ok=True, resume=bool(source), **params)
        trainer = model.trainer
        if diagnostic_batches:
            raise RuntimeError("Training ended before the requested diagnostic batch limit")
        state["completed_epochs"] = trainer.epoch + 1
        state["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(int(device)) if device != "cpu" else None
        best_checkpoint = directory / "weights/best.pt"
        if not best_checkpoint.is_file():
            raise RuntimeError("Training returned without best.pt")
        settings = evaluation_settings(params["imgsz"], params["batch"], device)
        model = load_model(best_checkpoint)
        if dict(model.names) != data["names"]:
            raise ValueError("Best checkpoint class names differ from the dataset")
        model.eval()
        metrics = model.val(data=str(trainer.args.data), validator=StudyValidator,
                            project=str(directory), name="best_validation", exist_ok=True, **settings)
        write_json(directory / "metrics.json", {**metrics_dict(metrics), "split": "val", "settings": settings,
                   "data_sha256": data["sha256"], "checkpoint_sha256": sha256(best_checkpoint),
                   "checkpoint": "weights/best.pt"})
        last_checkpoint = directory / "weights/last.pt"
        if last_checkpoint.is_file():
            state["last_sha256"] = sha256(last_checkpoint)
        state["status"] = "completed"
    except SmokeCheckpoint:
        state["status"] = "interrupted"
        state["stop_reason"] = "planned first-epoch smoke checkpoint"
    except DiagnosticComplete:
        if not diagnostic_batches:
            raise
        state["status"] = "diagnostic_passed"
        print(f"Diagnostic passed: {args.model}; {model.trainer.batches_completed} batches, "
              f"{model.trainer.optimizer_steps} optimizer updates. Evidence: {directory}")
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        traceback.print_exc(file=log)
        if isinstance(getattr(model, "trainer", None), DiagnosticTrainer):
            try:
                model.trainer.save_failure(directory)
            except Exception as evidence_error:
                state["evidence_error_type"] = type(evidence_error).__name__
                traceback.print_exc(file=log)
        raise
    finally:
        if isinstance(getattr(model, "trainer", None), DiagnosticTrainer):
            trainer = model.trainer
            if diagnostic_batches:
                state.update(batches_completed=trainer.batches_completed, optimizer_steps=trainer.optimizer_steps,
                             diagnostic_phase=trainer.phase)
            trainer.close_diagnostic()
        state["finished_utc"] = now()
        write_json(directory / "run.json", state)
        utils.LOGGER.removeHandler(handler)
        log.close()
        if not diagnostic_batches:
            refresh_safely(directory)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and not argv[0].startswith("-"):
        from fusion.study import main as run_study
        return run_study(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--model", choices=MODELS, help="Default: baseline, or the saved model when resuming")
    parser.add_argument("--resume", help="Resume this project's last.pt in a new output directory; preserve original settings")
    parser.add_argument("--run-dir", help="New directory under training/artifacts/")
    parser.add_argument("--download-weights", choices=("yolo", "vit", "all"))
    parser.add_argument("--diagnose-batches", type=int, help="Single-model real diagnostic; stop after N batches")
    parser.add_argument("--smoke-checkpoint", action="store_true", help="Stop after epoch 1 of a 2-epoch save/resume check")
    args = parser.parse_args(argv)
    if args.smoke_checkpoint and (args.epochs != 2 or args.resume or args.check or args.diagnose_batches):
        parser.error("--smoke-checkpoint requires a fresh --epochs 2 run")
    if args.download_weights and (args.data or args.check or args.resume):
        parser.error("Downloading weights is a separate action; omit --data/--check")
    if not args.download_weights and not args.data:
        parser.error("--data is required")
    if args.diagnose_batches is not None:
        if args.diagnose_batches <= 0 or args.check or args.download_weights:
            parser.error("--diagnose-batches requires N > 0 and cannot be combined with --check/--download-weights")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    bootstrap()
    if args.download_weights:
        download_weights(args.download_weights)
    else:
        run(args)


if __name__ == "__main__":
    main()
