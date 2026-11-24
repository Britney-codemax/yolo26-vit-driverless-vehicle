"""Evaluate one local checkpoint on an explicitly requested YOLO/NDJSON split."""
import argparse
import asyncio
from pathlib import Path
from fusion.runtime import (ARTIFACTS, bootstrap, environment, fingerprint, local_output,
                            metrics_dict, now, sha256, source_hash, stamp, write_json, result_root)


def evaluation_settings(imgsz=640, batch=4, device="0", split="val"):
    return dict(imgsz=imgsz, batch=batch, device=device, split=split, conf=0.001, iou=0.7,
                max_det=300, half=False, rect=True, workers=0, plots=False, save_json=False,
                save_txt=False, augment=False, verbose=False)


def evaluate_checkpoint(checkpoint, data_yaml, destination, settings, expected_names):
    from fusion.models import load_model
    from fusion.engine import StudyValidator
    model = load_model(checkpoint)
    if dict(model.names) != expected_names:
        raise ValueError("Checkpoint class IDs/names differ from the dataset")
    model.eval()
    metrics = model.val(data=str(data_yaml), validator=StudyValidator,
                        project=str(destination.parent), name=destination.name, exist_ok=True, **settings)
    return {**metrics_dict(metrics), "split": settings["split"], "settings": settings,
            "checkpoint_sha256": sha256(checkpoint)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="Local YOLO data.yaml/.yml, or detection NDJSON")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)
    if args.batch <= 0 or args.imgsz <= 0 or args.imgsz % 32:
        parser.error("batch must be positive; imgsz must be a positive multiple of 32")
    checkpoint = Path(args.weights).resolve()
    if not checkpoint.is_file():
        parser.error("--weights must be an existing local checkpoint")
    bootstrap()
    from fusion.data import inspect_data, convert_data
    from fusion.models import load_model
    from fusion.records import refresh_safely
    data = inspect_data(args.data, required=(args.split,))
    settings = evaluation_settings(args.imgsz, args.batch, args.device, args.split)
    directory = local_output(result_root(data, "evaluations") / stamp())
    directory.mkdir(parents=True, exist_ok=False)
    probe = load_model(checkpoint)
    variant = getattr(probe.model, "variant", "external")
    del probe
    state = {"id": directory.relative_to(ARTIFACTS).as_posix(), "kind": "evaluation", "model": variant, "seed": None,
             "data": data, "settings": settings, "environment": environment(args.device), "code_sha256": source_hash(),
             "checkpoint_sha256": sha256(checkpoint), "status": "running", "started_utc": now()}
    if checkpoint.is_relative_to(ARTIFACTS.parent):
        state["weight_path"] = checkpoint.relative_to(ARTIFACTS.parent).as_posix()
    state["protocol_key"] = fingerprint({k: state[k] for k in ("data", "settings", "environment", "code_sha256")})
    write_json(directory / "run.json", state)
    try:
        data_yaml = asyncio.run(convert_data(args.data, required=(args.split,)))
        result = evaluate_checkpoint(checkpoint, data_yaml, directory / "validation", settings, data["names"])
        write_json(directory / "metrics.json", {**result, "data_sha256": data["sha256"]})
        state["status"] = "completed"
        print(result["overall"])
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error_type"] = type(exc).__name__
        raise
    finally:
        state["finished_utc"] = now()
        write_json(directory / "run.json", state)
        refresh_safely(directory)


if __name__ == "__main__":
    main()
