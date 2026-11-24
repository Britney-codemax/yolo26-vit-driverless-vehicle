"""Run the five-model UGV/USV comparison, from data inspection to paired repeats."""
import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys

from fusion.runtime import (ARTIFACTS, MODEL_SET, PROJECT, SUPPORT, STUDIES, STUDY_MODELS, bootstrap, environment,
                            fingerprint, read_json, sha256, source_hash, write_json)
from fusion.data import inspect_study
from fusion.workbook import eligible, export_workbook


def train_cli(data, *options):
    definitions = [] if "--resume" in options else ["--model-set", str(MODEL_SET)]
    subprocess.run([sys.executable, str(PROJECT / "train.py"), "--data", str(data),
                    *definitions, *map(str, options)],
                   cwd=PROJECT, check=True)


def completed(slot):
    runs = []
    for path in slot.glob("*/run.json"):
        record = read_json(path)
        if record.get("status") == "completed":
            record["metrics"] = read_json(path.with_name("metrics.json"))
            if eligible(record):
                record["directory"] = path.parent
                runs.append(record)
    if len(runs) > 1:
        raise ValueError(f"同一模型和种子有多次完成的结果，请核查：{slot}")
    return runs[0] if runs else None


def fit(data, campaign, model, seed, device):
    slot = campaign / f"{model}_seed{seed}"
    record = completed(slot)
    if record:
        return record
    attempt = slot / f"attempt_{1 + len(list(slot.glob('attempt_*'))):02d}"
    train_cli(data, "--model", model, "--seed", seed, "--epochs", 100, "--imgsz", 640,
              "--patience", 0, "--batch", 4, "--device", device, "--run-dir", attempt)
    record = completed(slot)
    if not record:
        raise RuntimeError(f"100 轮验证记录不完整：{slot}")
    return record


def rank(records):
    # Within 0.1 percentage point prefer the smaller, then faster network.
    best = max(score for score, _, _, _ in records)
    close = [r for r in records if best - r[0] <= .001 + 1e-12]
    return min(close, key=lambda r: (r[1], r[2], -r[0], r[3]))[3]


def choose(groups):
    protocol = {r["protocol_key"] for rows in groups.values() for r in rows}
    if len(protocol) != 1:
        raise RuntimeError("候选训练协议不同（含中断轮次），请用同协议完整运行后比较。")
    return rank([(statistics.mean(r["metrics"]["overall"]["map50_95"] for r in rows),
                  rows[0]["parameters"], statistics.mean(r["metrics"]["inference_ms_per_image"] for r in rows), model)
                 for model, rows in groups.items()])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("inspect","fit","screen","repeat","test","all","report","resume"))
    p.add_argument("--data", help="本地新版 YOLO YAML（当前复核版使用 training/configs/study-data.yaml）")
    p.add_argument("--device", default="0")
    p.add_argument("--model", choices=STUDY_MODELS, default="p5_mamba")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args(argv)
    if args.stage == "report":
        print(export_workbook())
        return
    if not args.data:
        p.error("请指定新版 --data；公开链接可以留空。")
    data = Path(args.data).resolve()
    info = inspect_study(data, STUDIES)
    folder = STUDIES / info["sha256"]
    write_json(folder / "dataset.json", info)
    if args.stage == "inspect":
        print(info)
        print(export_workbook())
        return
    os.environ.setdefault("YOLO_DISABLE_CUDNN", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    if not MODEL_SET.is_file():
        p.error("请先运行 模型融合.ipynb，生成 models/fusion.yaml。")
    bootstrap()
    weights = ARTIFACTS / "weights/yolo26s.pt"
    key = fingerprint({"code": source_hash(), "environment": environment(args.device),
                       "weights": sha256(weights), "model_set": sha256(MODEL_SET)})
    campaign = folder / "runs" / key
    if args.stage == "resume":
        if not args.checkpoint:
            p.error("请指定 --checkpoint 指向本轮未完成运行的 weights/last.pt")
        ck = args.checkpoint.resolve()
        if not ck.is_relative_to(folder / "runs"):
            p.error("检查点不属于当前数据的正式实验。")
        slot = ck.parents[2]
        if completed(slot):
            p.error("此模型和种子已有完整结果。")
        attempt = slot / f"attempt_{1 + len(list(slot.glob('attempt_*'))):02d}"
        train_cli(data, "--resume", ck, "--run-dir", attempt)
        return
    if args.stage == "fit":
        fit(data, campaign, args.model, args.seed, args.device)
        return
    first = {}
    for model in STUDY_MODELS:
        if args.stage in ("screen","all"):
            record = fit(data, campaign, model, 0, args.device)
        else:
            record = completed(campaign / f"{model}_seed0")
            if not record:
                p.error(f"请先运行 screen：缺少 {model} 的完整 seed 0 结果。")
        first[model] = record
    best = choose({m: [first[m]] for m in STUDY_MODELS[2:]})
    selected = list(dict.fromkeys(["baseline","p5_vit_small",best] + (["p4_mamba"] if best == "p4_mamba_p5_vit" else [])))
    print(f"首轮候选 {best}；复验 {selected}", flush=True)
    if args.stage == "screen":
        return
    groups = {m: [first[m]] for m in selected}
    for m in selected:
        for seed in (1,2):
            if args.stage == "test":
                record = completed(campaign / f"{m}_seed{seed}")
                if not record:
                    p.error("独立测试前请先完成 repeat。")
            else:
                record = fit(data, campaign, m, seed, args.device)
            groups[m].append(record)
    winner = choose(groups)
    selection = {"screen_winner": best, "repeated": selected, "final_model": winner, "seeds": [0,1,2],
                 "selection_split": "val", "test_images": info["counts"]["test"]["images"]}
    write_json(campaign / "selection.json", selection)
    print(selection, flush=True)
    if args.stage in ("test","all") and selection["test_images"]:
        ck = groups[winner][0]["directory"] / "weights/best.pt"
        digest = sha256(ck)
        prior = [read_json(f) for f in (folder / "evaluations").glob("*/metrics.json")]
        if not any(r["split"] == "test" and r["checkpoint_sha256"] == digest for r in prior):
            subprocess.run([sys.executable, str(SUPPORT / "evaluate.py"), "--data", str(data), "--split", "test",
                            "--weights", str(ck), "--device", args.device], cwd=PROJECT, check=True)
    print(export_workbook())


if __name__ == "__main__":
    main()
