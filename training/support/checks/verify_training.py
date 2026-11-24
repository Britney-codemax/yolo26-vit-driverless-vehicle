"""Explicit CPU/GPU acceptance on local data: cache parity, native baseline, real boundary updates."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.runtime import ARTIFACTS, DIAGNOSTICS, LEGACY_MODELS as MODELS, bootstrap, environment, sha256, stamp, write_json


def verify_data(path):
    from fusion.data import read_data, convert_data
    rows, info = read_data(path)
    if Path(path).suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("This acceptance check requires a local YOLO YAML dataset")
    prepared = asyncio.run(convert_data(path))
    for row in rows[1:]:
        name = hashlib.sha256(row["file"].encode()).hexdigest()
        image = prepared.parent / "images" / row["split"] / (name + Path(row["file"]).suffix.lower())
        label = prepared.parent / "labels" / row["split"] / (name + ".txt")
        assert sha256(image) == row["image_sha256"], f"Image changed: {row['file']}"
        boxes = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        assert boxes == row["annotations"]["boxes"], f"Labels changed: {row['file']}"
    return prepared, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", required=True, help="cpu or one CUDA index; GPU runs must be explicit")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)
    if args.imgsz < 32 or args.imgsz % 32 or (args.device != "cpu" and not args.device.isdigit()):
        parser.error("imgsz must be a positive multiple of 32; device must be cpu or one CUDA index")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    bootstrap()
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    from fusion.models import build_model, make_detector, load_model
    torch.set_num_threads(2)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    directory = DIAGNOSTICS / ("boundary_acceptance_" + stamp())
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "environment": environment(args.device), "imgsz": args.imgsz,
              "scope": "real local images, no augmentation; isolated AdamW boundary steps, not long training",
              "models": {m: {"status": "not_run"} for m in MODELS}}
    cpu_batch = None
    variant = None
    try:
        prepared, info = verify_data(args.data)
        report["data"] = info
        report["cache_parity"] = "all image bytes and numeric labels identical"
        config = yaml.safe_load((ARTIFACTS.parent / "configs/train.yaml").read_text())
        cfg = get_cfg(overrides={**config, "imgsz": args.imgsz, "device": args.device})
        dataset_config = {"names": info["names"], "nc": len(info["names"])}
        dataset = build_yolo_dataset(cfg, str(prepared.parent / "images/train"), 4, dataset_config, mode="val")
        negative = [i for i, item in enumerate(dataset.labels) if len(item["cls"]) == 0]
        positive = [i for i, item in enumerate(dataset.labels) if len(item["cls"]) > 0]
        multiple = [i for i, item in enumerate(dataset.labels) if len(item["cls"]) > 1]
        if len(negative) < 4 or len(positive) < 2 or not multiple:
            raise ValueError("Boundary acceptance needs four negatives, two positives and a multi-object image")
        cases = {"all_negative": negative[:4], "mixed": negative[:2] + positive[:2],
                 "multi_object": [multiple[0]] + positive[:2] + negative[:1], "tail_batch": [positive[0]]}
        weights = ARTIFACTS / "weights/yolo26s.pt"
        vit_weights = ARTIFACTS / "weights/vit_b_16_imagenet1k_v1.pth"
        report["yolo_sha256"] = sha256(weights)
        report["vit_sha256"] = sha256(vit_weights)
        for variant in MODELS:
            report["models"][variant] = {"status": "running", "cases": {}}
            write_json(directory / "acceptance.json", report)
            wrapper, _ = build_model(variant, 0, weights, vit_weights, deterministic=cfg.deterministic)
            model = make_detector(wrapper.model.yaml, nc=len(info["names"]), weights=wrapper.model).to(device)
            model.args, model.names = cfg, info["names"]
            if variant == "baseline":
                init_seeds(0, deterministic=cfg.deterministic)
                native = DetectionModel("yolo26s.yaml", nc=len(info["names"]), verbose=False)
                native.load(YOLO(str(weights)).model, verbose=False)
                native = native.to(device)
                native.args, native.names = cfg, info["names"]
                torch.testing.assert_close(model.state_dict(), native.state_dict(), rtol=0, atol=0)
                cpu_batch = dataset.collate_fn([dataset[i] for i in cases["mixed"]])
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.items()}
                batch["img"] = batch["img"].float() / 255
                model.eval(); native.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(batch["img"]), native(batch["img"]), rtol=1e-5, atol=1e-6)
                model.train(); native.train()
                native_loss, native_items = native.loss(deepcopy(batch))
                native_loss.sum().backward()
                native_grads = {k: p.grad.detach().cpu().clone() for k, p in native.named_parameters() if p.grad is not None}
                loss, items = model.loss(deepcopy(batch))
                torch.testing.assert_close(loss, native_loss, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(items, native_items, rtol=1e-5, atol=1e-6)
                loss.sum().backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad.cpu(), native_grads[name], rtol=1e-5, atol=1e-6)
                report["native_baseline"] = "matching initialization, forward, loss and gradients"
                del native, native_grads, native_loss, native_items, loss, items, batch
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr0, weight_decay=cfg.weight_decay)
            encoder_before = model.model[10].encoder.self_attention.in_proj_weight.detach().clone() if variant != "baseline" else None
            for case, indices in cases.items():
                cpu_batch = dataset.collate_fn([dataset[i] for i in indices])
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.items()}
                batch["img"] = batch["img"].float() / 255
                optimizer.zero_grad(set_to_none=True)
                loss, items = model.loss(batch)
                assert torch.isfinite(loss).all() and torch.isfinite(items).all(), f"{variant}/{case}: loss"
                loss.sum().backward()
                for parameter in model.parameters():
                    assert parameter.grad is None or torch.isfinite(parameter.grad).all(), f"{variant}/{case}: gradient"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                parameter = list(model.parameters())[-1]
                before = parameter.detach().clone()
                optimizer.step()
                assert not torch.equal(before, parameter), f"{variant}/{case}: no weight update"
                report["models"][variant]["cases"][case] = {"batch_size": len(indices), "optimizer_updates": 1,
                                                            "loss": float(loss.detach().sum().cpu())}
            if encoder_before is not None:
                assert not torch.equal(encoder_before, model.model[10].encoder.self_attention.in_proj_weight)
                report["models"][variant]["encoder_updated"] = True
            wrapper.model = model.cpu()
            checkpoint = directory / f"{variant}.pt"
            wrapper.save(checkpoint)
            reloaded = load_model(checkpoint).model
            for key, value in model.state_dict().items():
                expected = value.half().float() if value.is_floating_point() else value
                torch.testing.assert_close(reloaded.state_dict()[key], expected, rtol=0, atol=0)
            report["models"][variant].update(status="passed", checkpoint_reload="all state tensors preserved at saved precision")
            write_json(directory / "acceptance.json", report)
            del wrapper, model, reloaded, optimizer, batch, loss, items, encoder_before, before, parameter
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        if variant is not None:
            report["models"][variant]["status"] = "failed"
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if cpu_batch is not None:
            torch.save(cpu_batch, directory / "failure_batch.pt")
        raise
    finally:
        write_json(directory / "acceptance.json", report)
        print(f"Acceptance evidence: {directory}")


if __name__ == "__main__":
    main()
