"""Train one model. --check is offline; weight downloads are a separate explicit action."""
import argparse
import logging
import os
from pathlib import Path
import traceback
import yaml
from fusion.runtime import (ROOT, ARTIFACTS, RUNS, MODELS, bootstrap, environment, fingerprint,
                            local_output, metrics_dict, now, sha256, source_hash, stamp, write_json)


def add_arguments(parser):
    parser.add_argument("--data", help="Local YOLO data.yaml/.yml (images + labels), or detection NDJSON")
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


def run(args):
    import torch
    from ultralytics import utils
    from fusion.data import inspect_data
    from fusion.models import build_model, load_model
    from fusion.engine import FusionTrainer, CompleteValidator
    from fusion.records import refresh_safely
    from evaluate import evaluation_settings
    params = configuration(args)
    data = inspect_data(args.data)
    if args.check:
        model, _ = build_model(args.model, params["seed"], check=True)
        import json
        print(json.dumps({"status": "structure_checked_not_trained", "model": args.model, "training": params,
                          "data": data, "parameters_80_classes": sum(p.numel() for p in model.model.parameters()),
                          "yolo_weights_available": Path(args.weights).is_file(),
                          "vit_weights_available": Path(args.vit_weights).is_file(),
                          "note": "No weights loaded; pretrained transfer is NOT checked here"}, ensure_ascii=False, indent=2))
        return
    directory = local_output(args.run_dir or RUNS / f"{args.model}_seed{params['seed']}_{stamp()}")
    directory.mkdir(parents=True, exist_ok=False)
    state = {"id": directory.relative_to(ARTIFACTS).as_posix(), "kind": "train", "model": args.model, "seed": params["seed"],
             "status": "running", "started_utc": now(), "pid": os.getpid(), "data": data,
             "requested_training": params, "environment": environment(params["device"]), "code_sha256": source_hash()}
    write_json(directory / "run.json", state)
    log = (directory / "console.log").open("w", encoding="utf-8")
    handler = logging.StreamHandler(log)
    utils.LOGGER.addHandler(handler)
    try:
        device = str(params["device"])
        if device != "cpu" and (not torch.cuda.is_available() or int(device) >= torch.cuda.device_count()):
            raise ValueError(f"Requested CUDA device is unavailable: {device}")
        model, transfer = build_model(args.model, params["seed"], args.weights, args.vit_weights)
        write_json(directory / "initial_transfer.json", transfer)
        state["yolo_sha256"] = transfer["yolo"]["sha256"]
        state["vit_sha256"] = transfer.get("vit", {}).get("sha256")
        state["model_config"] = model.model.yaml

        def record_progress(trainer):
            effective = {k: getattr(trainer.args, k) for k in params}
            effective["amp"] = bool(trainer.amp)
            state["effective_training"] = effective
            state["determinism"] = {"enabled": torch.are_deterministic_algorithms_enabled(),
                                    "warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}
            state["parameters"] = sum(p.numel() for p in trainer.model.parameters())
            state["protocol_key"] = fingerprint({"data": data["sha256"], "code": state["code_sha256"],
                "environment": state["environment"], "determinism": state["determinism"], "yolo": state["yolo_sha256"],
                "training": {k: v for k, v in effective.items() if k != "seed"},
                "evaluation": evaluation_settings(params["imgsz"], params["batch"], device)})
            write_json(directory / "effective_args.json", vars(trainer.args))
            write_json(directory / "run.json", state)

        def record_checkpoint(trainer):
            state["completed_epochs"] = trainer.epoch + 1
            if trainer.best_fitness == trainer.fitness:
                state["best_epoch"] = trainer.epoch + 1
                state["best_epoch_source"] = "on_model_save: best.pt written by mAP50-95 fitness"
            record_progress(trainer)

        model.add_callback("on_train_start", record_progress)
        model.add_callback("on_model_save", record_checkpoint)
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats(int(device))
        # Dataset validation and cache preparation are shared with independent evaluation.
        model.train(data=str(Path(args.data).resolve()), trainer=FusionTrainer,
                    project=str(directory.parent), name=directory.name, exist_ok=True, **params)
        trainer = model.trainer
        record_progress(trainer)
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
        metrics = model.val(data=str(trainer.args.data), validator=CompleteValidator,
                            project=str(directory), name="best_validation", exist_ok=True, **settings)
        write_json(directory / "metrics.json", {**metrics_dict(metrics), "split": "val", "settings": settings,
                   "data_sha256": data["sha256"], "checkpoint_sha256": sha256(best_checkpoint),
                   "checkpoint": "weights/best.pt"})
        state["status"] = "completed"
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error_type"] = type(exc).__name__
        traceback.print_exc(file=log)
        raise
    finally:
        state["finished_utc"] = now()
        write_json(directory / "run.json", state)
        utils.LOGGER.removeHandler(handler)
        log.close()
        refresh_safely()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument("--run-dir", help="New directory under artifacts/")
    parser.add_argument("--download-weights", choices=("yolo", "vit", "all"))
    args = parser.parse_args(argv)
    if args.download_weights and (args.data or args.check):
        parser.error("Downloading weights is a separate action; omit --data/--check")
    if not args.download_weights and not args.data:
        parser.error("--data is required")
    bootstrap()
    if args.download_weights:
        download_weights(args.download_weights)
    else:
        run(args)


if __name__ == "__main__":
    main()
