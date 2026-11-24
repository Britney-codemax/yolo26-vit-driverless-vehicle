"""Bounded epoch-validation replay from a local checkpoint; never resumes training."""
import argparse
import asyncio
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.runtime import DIAGNOSTICS, bootstrap, environment, now, sha256, stamp, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch", type=int, choices=(4, 8), required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--native", action="store_true", help="Use an unmodified Ultralytics baseline with identical weights")
    args = parser.parse_args(argv)
    if args.repeats <= 0 or (args.device != "cpu" and not args.device.isdigit()):
        parser.error("repeats must be positive; device must be cpu or a CUDA index")
    if not Path(args.weights).is_file():
        parser.error("weights must be an existing local checkpoint")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    bootstrap()
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    from fusion.data import inspect_data, convert_data
    from fusion.engine import CompleteValidator
    from fusion.models import load_model

    torch.set_num_threads(2)
    init_seeds(0, deterministic=True)
    directory = DIAGNOSTICS / ("validation_replay_" + stamp())
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_utc": now(), "environment": environment(args.device),
              "checkpoint_sha256": sha256(args.weights), "batch": args.batch, "native": args.native,
              "requested_repeats": args.repeats, "rounds": [],
              "scope": "previous saved EMA; epoch validation including loss, no optimizer, not exact crash replay"}
    write_json(directory / "acceptance.json", report)
    try:
        info = inspect_data(args.data)
        data = check_det_dataset(str(asyncio.run(convert_data(args.data))))
        wrapper = load_model(args.weights)
        model = wrapper.model.float().cpu()
        if args.native:
            if getattr(model, "variant", None) != "baseline":
                raise ValueError("Native comparison requires a baseline checkpoint")
            native = DetectionModel("yolo26s.yaml", nc=model.nc, verbose=False)
            native.load_state_dict(model.state_dict(), strict=True)
            native.names = model.names
            torch.testing.assert_close(native.state_dict(), model.state_dict(), rtol=0, atol=0)
            model = native
        cfg = get_cfg(overrides={**wrapper.ckpt["train_args"], "batch": args.batch, "device": args.device,
                                 "amp": False, "half": False, "compile": False, "plots": False, "workers": 0,
                                 "rect": True})
        model.args = cfg
        device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
        model.to(device)
        validator = CompleteValidator(save_dir=directory, args=cfg)
        validator.data = data
        validator.stride = max(int(model.stride.max()), 32)
        validator.dataloader = validator.get_dataloader(data["val"], args.batch)
        assert validator.dataloader.dataset.rect, "Epoch validation must use rectangular batches"
        trainer = SimpleNamespace(device=device, data=data, amp=False, ema=SimpleNamespace(ema=model),
            model=model, args=cfg, loss_items=torch.zeros(3, device=device),
            stopper=SimpleNamespace(possible_stop=False), epoch=wrapper.ckpt["epoch"], epochs=100, world_size=1,
            label_loss_items=lambda loss, prefix: dict(zip((f"{prefix}/{k}_loss" for k in ("box", "cls", "dfl")), loss)))
        expected = info["counts"]["val"]["images"]
        for repeat in range(args.repeats):
            metrics = validator(trainer)
            assert validator.seen == expected, f"Incomplete validation: {validator.seen}/{expected}"
            assert all(math.isfinite(v) for v in metrics.values()), "Non-finite validation metrics"
            report["rounds"].append({"repeat": repeat + 1, "images": validator.seen, "metrics": metrics})
            write_json(directory / "acceptance.json", report)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_utc"] = now()
        write_json(directory / "acceptance.json", report)
        print(f"Validation evidence: {directory}")


if __name__ == "__main__":
    main()
