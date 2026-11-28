"""Run one round-four experiment or a sequential queue on one physical GPU."""
import argparse
import asyncio
import csv
from functools import partial
import gc
import os
from pathlib import Path

if __name__ == "__main__":
    # Set the mask before importing any package that may already query the CUDA runtime.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", metavar="F01…F09")
    parser.add_argument("--device", default="0", help="physical CUDA index on this host")
    parser.add_argument("--diagnose", action="store_true", help="one real optimizer update per group")
    args = parser.parse_args()
    if not args.device.isdecimal():
        parser.error("--device 需要单张显卡编号")
    config_dir = Path(__file__).resolve().parents[1] / "training/artifacts/config" / f"gpu{args.device}"
    os.environ.update(CUDA_VISIBLE_DEVICES=args.device, CUDA_DEVICE_ORDER="PCI_BUS_ID", YOLO_CONFIG_DIR=str(config_dir),
                      YOLO_DISABLE_CUDNN="1", CUDA_LAUNCH_BLOCKING="1", YOLO_OFFLINE="true")

import torch
import yaml
from filelock import FileLock, Timeout

from models.phase4_layers import PROJECT, ROUTES, definitions, register, legacy
from models.phase2 import DATA, PROTOCOL, SETTINGS, next_attempt, recover_source, training_protocol
from fusion import engine
from fusion.data import convert_data, inspect_data
from fusion.runtime import ARTIFACTS, DIAGNOSTICS, ROOT, STUDIES, fingerprint, now, read_json, sha256, stamp, write_json
from evaluate import evaluate_checkpoint, evaluation_settings
import train

CONFIG = PROJECT / "models/fusion_phase4.yaml"
DATAFILE = ROOT / "configs/study-data.yaml"
WEIGHTS = ARTIFACTS / "weights/yolo26n.pt"
COCO = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
BASE_HASH = "78de8d2e9d5a572c0d3d33af751ce650c59215c9d5f66f4dd0f146cf11ba8d09"
OUTPUT = STUDIES / DATA / "phase4_n"
BASE_RUN = OUTPUT.parent / "phase3_n/baseline_resume01"


def verify_epochs(key, directory, row):
    with (directory / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        epochs = [int(record["epoch"]) for record in csv.DictReader(stream, skipinitialspace=True)]
    if (row["model"] != key or row["seed"] != 0 or
            row["completed_epochs"] != 100 or epochs != list(range(1, 101)) or
            row["yolo_sha256"] != COCO or row["data"]["sha256"] != DATA or
            training_protocol(row["effective_training"]) != PROTOCOL):
        raise ValueError(f"训练协议或已保存轮次不符：{directory}")
    if key in ROUTES and row["model_config"]["fusion_spec"] != definitions()[key]["fusion_spec"]:
        raise ValueError(f"融合结构不一致：{directory}")


def verified_result(key, directory):
    row = read_json(directory / "run.json")
    verify_epochs(key, directory, row)
    metrics = read_json(directory / "metrics.json")
    best = sha256(directory / "weights/best.pt")
    if (row["status"] != "completed" or metrics["data_sha256"] != DATA or metrics["split"] != "val" or
            metrics["checkpoint"] != "weights/best.pt" or metrics["checkpoint_sha256"] != best or
            metrics["settings"] != evaluation_settings(640, 4, row["effective_training"]["device"]) or
            (key == "baseline" and best != BASE_HASH)):
        raise ValueError(f"训练记录与 best.pt 复评不符：{directory}")
    return metrics


def latest_result(key):
    if key == "baseline":
        if not (BASE_RUN / "run.json").exists():
            return "未同步", None, None
        return "已完成", BASE_RUN, verified_result(key, BASE_RUN)
    records = sorted((OUTPUT / key).glob("attempt_*/run.json"),
                     key=lambda p: int(p.parent.name.split("_")[-1]))
    completed = [p.parent for p in records if read_json(p)["status"] == "completed"]
    if len(completed) > 1:
        raise ValueError(f"{key} 有多个已完成目录，请核对原始运行：{completed}")
    if completed:
        return "已完成", completed[0], verified_result(key, completed[0])
    if records:
        return read_json(records[-1])["status"], records[-1].parent, None
    return "待训练", None, None


def run_one(key, info, diagnostic, code_hash):
    slot = OUTPUT / key
    slot.mkdir(parents=True, exist_ok=True)
    with FileLock(slot / "training.lock", timeout=0):
        source = None
        if not diagnostic:
            if latest_result(key)[2]:
                print(f"{key} 已完成，跳过", flush=True)
                return
            for path in slot.glob("attempt_*/run.json"):
                if read_json(path)["code_sha256"] != code_hash:
                    raise ValueError(f"续训实现已变化：{path}")
            source = recover_source(slot, info, "0")
        if source and source.name == "best.pt":
            directory = source.parent.parent
            row = read_json(directory / "run.json")
            verify_epochs(key, directory, row)
            if legacy.load_model(source).model.variant != key:
                raise ValueError(f"复评检查点不匹配：{source}")
            settings = evaluation_settings(640, 4, "0")
            metrics = evaluate_checkpoint(source, asyncio.run(engine.convert_data(DATAFILE)),
                                          directory / "best_validation", settings, info["names"])
            write_json(directory / "metrics.json", dict(metrics, data_sha256=DATA, checkpoint="weights/best.pt"))
            row.update(status="completed", last_sha256=sha256(source.with_name("last.pt")),
                       finished_utc=now(), recovered_validation=True)
            write_json(directory / "run.json", row)
        else:
            directory = diagnostic / key if diagnostic else next_attempt(slot)
            options = ["--resume", str(source)] if source else [
                "--model", key, "--model-set", str(CONFIG), "--weights", str(WEIGHTS)]
            if not source:
                for name, value in SETTINGS.items():
                    options.extend(["--" + name, str(value)])
            if diagnostic:
                options.extend(["--diagnose-batches", "1"])
            print(f"{key} → {directory}", flush=True)
            train.main(["--data", str(DATAFILE), "--run-dir", str(directory), "--device", "0", *options])
        if diagnostic:
            row = read_json(directory / "run.json")
            assert (row["status"], row["batches_completed"], row["optimizer_steps"]) == ("diagnostic_passed", 1, 1)
        else:
            verified_result(key, directory)


def main(args):
    if any(key not in ROUTES for key in args.models) or len(args.models) != len(set(args.models)):
        raise ValueError("请指定 F01 到 F09 中不重复的实验编号")
    # Bind the requested physical GPU before Ultralytics selects logical cuda:0.
    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA 设备 {args.device} 不可用")
    torch.cuda.init()
    print(f"物理卡 {args.device} → cuda:0，{torch.cuda.get_device_name(0)}", flush=True)
    train.bootstrap()  # Resume compares the active cuDNN settings to the saved run.
    register()
    info = inspect_data(DATAFILE)
    cfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8-sig"))
    if (info["sha256"] != DATA or sha256(WEIGHTS) != COCO or
            training_protocol(dict(cfg, **SETTINGS)) != PROTOCOL or
            yaml.safe_load(CONFIG.read_text(encoding="utf-8")) != definitions()):
        raise ValueError("数据、权重、结构或训练协议与实验设计不符")
    paths = [PROJECT / "train.py", Path(__file__), Path(__file__).with_name("phase4_layers.py"),
             ROOT / "support/evaluate.py", *(ROOT / "support/fusion").glob("*.py")]
    code_hash = fingerprint({p.relative_to(PROJECT).as_posix(): p.read_text(encoding="utf-8-sig") for p in paths})
    train.source_hash = lambda: code_hash
    engine.convert_data = partial(convert_data, cache_root=ARTIFACTS / "datasets" / f"phase4_gpu{args.device}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    diagnostic = DIAGNOSTICS / f"phase4_{stamp()}" if args.diagnose else None
    try:
        with FileLock(OUTPUT / f"gpu{args.device}.lock", timeout=0):
            for key in args.models:
                run_one(key, info, diagnostic, code_hash)
                gc.collect()
    except Timeout:
        raise SystemExit("此显卡或实验组正在运行，请查看原训练终端。") from None


if __name__ == "__main__":
    main(args)
