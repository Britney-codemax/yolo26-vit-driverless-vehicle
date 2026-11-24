"""Serial controlled experiments; invoke explicitly to start real training."""
import argparse
from pathlib import Path
import subprocess
import sys
from fusion.runtime import PROJECT, ROOT, ARTIFACTS, LEGACY_MODELS, MODELS, bootstrap, local_output, read_json, result_root, stamp, write_json
sys.path.insert(0, str(PROJECT))
from train import add_arguments, configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(LEGACY_MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args(argv)
    if not args.data or args.seed is not None:
        parser.error("Provide --data; use --seeds instead of --seed")
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Models and nonnegative seeds must be unique")
    bootstrap()
    configuration(args)
    from fusion.data import inspect_data
    from fusion.records import refresh_safely
    data = inspect_data(args.data)
    if not args.check:
        if not Path(args.weights).is_file():
            parser.error("YOLO weights missing; run train.py --download-weights yolo first")
        if "p5_vit_pretrained" in args.models and not Path(args.vit_weights).is_file():
            parser.error("ViT weights missing; run train.py --download-weights vit before starting the batch")
    batch = local_output(result_root(data, "runs") / f"batch_{stamp()}")
    entries = [{"id": (batch / f"{model}_seed{seed}").relative_to(ARTIFACTS).as_posix(), "model": model, "seed": seed,
                "path": (batch / f"{model}_seed{seed}").relative_to(ROOT).as_posix(), "status": "not_run"}
               for model in args.models for seed in args.seeds]
    if not args.check:
        batch.mkdir(parents=True, exist_ok=False)
        write_json(batch / "plan.json", {"entries": entries})
        refresh_safely(batch)
    failed = False
    for entry in entries:
        command = [sys.executable, str(PROJECT / "train.py"), "--data", str(Path(args.data).resolve()),
                   "--model", entry["model"], "--seed", str(entry["seed"]), "--run-dir", entry["path"],
                   "--config", str(Path(args.config).resolve()), "--weights", str(Path(args.weights).resolve()),
                   "--vit-weights", str(Path(args.vit_weights).resolve())]
        for key in ("epochs", "batch", "imgsz", "patience", "workers", "device"):
            if getattr(args, key) is not None:
                command += [f"--{key}", str(getattr(args, key))]
        if args.check:
            command.append("--check")
        print(f"{entry['model']} seed={entry['seed']}", flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT)
            entry["exit_code"] = result.returncode
            entry["status"] = "process_succeeded" if result.returncode == 0 else "failed"
            failed |= result.returncode != 0
        except KeyboardInterrupt:
            entry["status"] = "interrupted"
            raise
        finally:
            if not args.check:
                write_json(batch / "plan.json", {"entries": entries})
                refresh_safely(batch)
        if failed:
            run_file = ROOT / entry["path"] / "run.json"
            epochs = read_json(run_file).get("completed_epochs", 0) if run_file.is_file() else 0
            print(f"Stopped: {entry['model']} failed after {epochs} completed epochs. "
                  f"Log: {ROOT / entry['path'] / 'console.log'}. Remaining runs were not started.", flush=True)
            break
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
