"""Notebook export and A/B/C queues; training and resume delegate to the original train.py."""
import argparse
import asyncio
import csv
from hashlib import sha256 as digest
import os
from pickle import UnpicklingError
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import yaml
from filelock import FileLock, Timeout

from models.phase2_layers import GROUPS, SPECS, PROJECT, fusion_spec, register
from fusion.data import inspect_data
from fusion.runtime import ARTIFACTS, MODEL_SET, ROOT, STUDIES, bootstrap, fingerprint, now, read_json, sha256, write_json

DATA = "760e212a7ba42aaf20b2f912aaa688c72c01bbc6c4eacd7af771866161605e8b"
COCO = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
BASE = "d0666955e41a8bf6a5d7a979e916ec18d56f9a651716247229c84a28f9132d7e"
PROTOCOL = "b32e364c83fcceddf16fdd62ec979058f8b620ac4ed9404e878e41fbe611eb33"
CONFIG = MODEL_SET.with_name("fusion_phase2.yaml")
OUTPUT = STUDIES / DATA / "phase2"
WEIGHTS = ARTIFACTS / "weights/yolo26s.pt"
SETTINGS = dict(seed=0, epochs=100, imgsz=640, batch=4, patience=0)


def baseline_config():
    cfg = yaml.safe_load(MODEL_SET.read_text(encoding="utf-8"))["baseline"]
    if fingerprint(cfg) != BASE:
        raise ValueError("第一轮 YOLO26s 基线结构已变化")
    return cfg


def definitions():
    base = baseline_config()
    return {key: dict(base, fusion_variant=key, fusion_spec=fusion_spec(key)) for key in SPECS}


def export(cfgs):
    canonical = definitions()
    if cfgs != canonical:
        raise ValueError("模型与 12 格设计不一致")
    document = dict(canonical, experiment={"groups": GROUPS, "data_sha256": DATA, "yolo_sha256": COCO,
                                           "protocol": PROTOCOL})
    with NamedTemporaryFile(mode="w", dir=CONFIG.parent, suffix=".tmp", encoding="utf-8",
                            newline="\n", delete=False) as f:
        path = Path(f.name)
        try:
            yaml.safe_dump(document, f, sort_keys=False, allow_unicode=True)
        except BaseException:
            f.close()
            path.unlink(missing_ok=True)
            raise
    try:
        path.replace(CONFIG)
    finally:
        path.unlink(missing_ok=True)
    return CONFIG


def training_protocol(settings):
    return fingerprint({k: v for k, v in settings.items() if k != "device"})


def identity(data):
    if not CONFIG.is_file():
        raise FileNotFoundError("请先运行第二轮 Notebook，导出 models/fusion_phase2.yaml")
    saved = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    metadata = saved.pop("experiment")
    if saved != definitions() or metadata != dict(groups=GROUPS, data_sha256=DATA, yolo_sha256=COCO,
                                                 protocol=PROTOCOL):
        raise ValueError("配置或分工与第二轮设计不符，请重新运行 Notebook")
    info = inspect_data(data)
    if info["sha256"] != DATA or sha256(WEIGHTS) != COCO:
        raise ValueError("数据指纹或 COCO 初始权重与第一轮不同")
    params = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8-sig"))
    if training_protocol(dict(params, **SETTINGS)) != PROTOCOL:
        raise ValueError("训练超参数与第一轮 100 轮协议不同")
    # Hash source as text so a Windows checkout's line endings do not separate classmates.
    files = [PROJECT / "train.py", ROOT / "support/evaluate.py", Path(__file__),
             Path(__file__).with_name("phase2_layers.py"), *(ROOT / "support/fusion").glob("*.py")]
    code = {p.relative_to(PROJECT).as_posix(): digest(p.read_text(encoding="utf-8-sig").encode()).hexdigest()
            for p in files}
    return {"data": DATA, "weights": COCO, "training": PROTOCOL, "models": fingerprint(saved),
            "implementation": fingerprint(code)}


def check_record(row, model, identity_hash, model_set_hash):
    expected = (model, 0, DATA, identity_hash, model_set_hash, PROTOCOL)
    actual = (row.get("model"), row.get("seed"), row["data"]["sha256"], row.get("code_sha256"),
              row.get("model_set_sha256"), training_protocol(row["requested_training"]))
    if actual != expected or row.get("yolo_sha256", COCO) != COCO:
        raise ValueError(f"{model} 的运行记录与当前实验不符")


def completed(slot, model, identity_hash, model_set_hash):
    found = []
    for path in slot.glob("attempt_*/run.json"):
        row = read_json(path)
        check_record(row, model, identity_hash, model_set_hash)
        if row["status"] != "completed":
            continue
        metrics = read_json(path.with_name("metrics.json"))
        if (row.get("completed_epochs") != 100 or metrics.get("split") != "val"
                or metrics.get("data_sha256") != DATA or row.get("yolo_sha256") != COCO
                or training_protocol(row["effective_training"]) != PROTOCOL
                or sha256(path.parent / "weights/best.pt") != metrics.get("checkpoint_sha256")):
            raise ValueError(f"完成记录不完整：{path.parent}")
        found.append((path.parent, row, metrics))
    if len(found) > 1:
        raise ValueError(f"同一模型有多个完整结果：{slot}")
    return found[0] if found else None


def next_attempt(slot):
    numbers = [int(p.name.split("_")[-1]) for p in slot.glob("attempt_*") if p.is_dir()]
    return slot / f"attempt_{max(numbers, default=0) + 1:02d}"


def recover_source(slot, data, device):
    """Use the most advanced valid checkpoint; leave partial/damaged attempts on disk."""
    import train
    from types import SimpleNamespace
    stopped = []
    for p in slot.glob("attempt_*/run.json"):
        row = read_json(p)
        if row["status"] == "running":
            # The group lock lives in the actual training process and is released by the OS.
            row.update(status="interrupted", stop_reason="group_process_exited")
            write_json(p, row)
        epoch = row.get("completed_epochs") or 0
        if epoch or (p.parent / "weights/last.pt").exists():
            stopped.append((epoch, p))
    errors = []
    for epoch, p in sorted(stopped, reverse=True):
        ck = p.parent / "weights/last.pt"
        args = SimpleNamespace(resume=ck, model=slot.name, config=ROOT / "configs/train.yaml", device=device)
        if epoch == SETTINGS["epochs"]:
            return ck.with_name("best.pt")
        try:
            train.resume_settings(args, data)
            return ck
        except (ValueError, RuntimeError, UnpicklingError, EOFError, OSError) as exc:
            errors.append(f"{p.parent.name}: {exc}")
    if errors:
        raise RuntimeError("没有可验证的续训检查点：" + " | ".join(errors))
    return None


def finish_validation(best, data, info, device):
    """Recover a job interrupted after its final epoch, before the best.pt report was saved."""
    from evaluate import evaluate_checkpoint, evaluation_settings
    from fusion.data import convert_data
    from fusion.models import load_model
    directory = best.parent.parent
    record = directory / "run.json"
    row = read_json(record)
    with (directory / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        epochs = [int(item["epoch"]) for item in csv.DictReader(stream, skipinitialspace=True)]
    if (epochs != list(range(1, SETTINGS["epochs"] + 1)) or row.get("yolo_sha256") != COCO
            or training_protocol(row["effective_training"]) != PROTOCOL
            or load_model(best).model.variant != row["model"]):
        raise ValueError(f"最后一轮记录与检查点不一致：{directory}")
    last_hash = sha256(best.with_name("last.pt"))
    settings = evaluation_settings(640, 4, device)
    result = evaluate_checkpoint(best, asyncio.run(convert_data(data)), directory / "best_validation",
                                 settings, info["names"])
    write_json(directory / "metrics.json", dict(result, data_sha256=DATA, checkpoint="weights/best.pt"))
    row.update(status="completed", finished_utc=now(), last_sha256=last_hash, recovered_validation=True)
    write_json(record, row)


def run_group(group, data, device, study):
    import train
    register()
    identity_hash = fingerprint(study)
    root = OUTPUT / group
    root.mkdir(parents=True, exist_ok=True)
    meta = dict(study=study, group=group, model_set_sha256=sha256(CONFIG))
    with FileLock(root / "queue.lock", timeout=0):
        manifest = root / "assignment.json"
        if manifest.exists() and read_json(manifest) != meta:
            raise ValueError(f"此目录属于不同的训练版本：{root}")
        write_json(manifest, meta)
        info = inspect_data(data)
        for model in GROUPS[group]:
            slot = root / model
            if completed(slot, model, identity_hash, meta["model_set_sha256"]):
                print(f"{model} 已完成，跳过", flush=True)
                continue
            ck = recover_source(slot, info, device)
            if ck and ck.name == "best.pt":
                print(f"{model} 已保存 100 轮，仅恢复最终验证", flush=True)
                finish_validation(ck, data, info, device)
            else:
                if ck:
                    options = ["--resume", str(ck)]
                else:
                    options = ["--model-set", str(CONFIG), "--model", model]
                    for key, value in SETTINGS.items():
                        options.extend(["--" + key, str(value)])
                dest = next_attempt(slot)
                print(f"{model} {'续训' if ck else '开始'} → {dest}", flush=True)
                original_hash = train.source_hash
                try:
                    train.source_hash = lambda: identity_hash
                    train.main(["--data", str(data), "--device", device, "--run-dir", str(dest), *options])
                finally:
                    train.source_hash = original_hash
            if not completed(slot, model, identity_hash, meta["model_set_sha256"]):
                raise RuntimeError(f"{model} 缺少 100 轮完整验证结果")


def main(argv=None):
    parser = argparse.ArgumentParser(description="第二轮实验：A/B/C 训练，review 在收齐后统一复评")
    parser.add_argument("action", choices=(*GROUPS, "review", "check"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    os.environ.setdefault("YOLO_DISABLE_CUDNN", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    bootstrap()
    data = args.data.resolve()
    study = identity(data)
    if args.action == "check":
        print("校验通过", study, GROUPS)
        return
    if args.device != "cpu" and (not args.device.isdigit() or not torch.cuda.is_available()
                                  or int(args.device) >= torch.cuda.device_count()):
        parser.error(f"设备不可用：{args.device}")
    if args.action == "review":
        from models.phase2_report import review
        try:
            review(data, args.device, study)
        except (FileNotFoundError, ValueError, PermissionError) as exc:
            parser.exit(2, f"复评未完成：{exc}\n")
        return
    try:
        run_group(args.action, data, args.device, study)
    except Timeout:
        parser.exit(2, "此组正在运行，请查看原终端。\n")


if __name__ == "__main__":
    main()
