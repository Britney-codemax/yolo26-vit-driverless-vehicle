"""Supervise the current 100-epoch study without changing its training fingerprint."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from filelock import FileLock, Timeout

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from fusion.data import inspect_data
from fusion.runtime import (ARTIFACTS, MODEL_SET, PROJECT, STUDIES, bootstrap, environment,
                            fingerprint, now, read_json, sha256, source_hash, write_json)
from fusion.study import completed


def campaign_for(data, device):
    key = fingerprint({"code": source_hash(), "environment": environment(device),
                       "weights": sha256(ARTIFACTS / "weights/yolo26s.pt"), "model_set": sha256(MODEL_SET)})
    return STUDIES / inspect_data(data)["sha256"] / "runs" / key


def active_job(data, campaign, device):
    """Prefer the queue over an orphan worker. Match paths as well as the PID."""
    worker = None
    for proc in psutil.process_iter(["name", "cmdline"]):
        args = proc.info["cmdline"] or []
        if len(args) < 4 or Path(args[1]).name.lower() != "train.py" or "--data" not in args:
            continue
        try:
            cwd = Path(proc.cwd())
            if (cwd / args[1]).resolve() != PROJECT / "train.py":
                continue
            if (cwd / args[args.index("--data") + 1]).resolve() != data:
                continue
            queue_device = args[args.index("--device") + 1] if "--device" in args else "0"
            if args[2] == "all" and queue_device == device:
                return proc, True
            if "--run-dir" in args and (cwd / args[args.index("--run-dir") + 1]).resolve().is_relative_to(campaign):
                worker = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    return (worker, False) if worker else (None, False)


def pending(campaign):
    """Return a checkpoint only for an unfinished slot in this campaign."""
    for slot in sorted(campaign.glob("*_seed*")):
        paths = sorted(slot.glob("attempt_*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not paths or completed(slot):
            continue
        records = [(p, read_json(p)) for p in paths]
        if any((r.get("completed_epochs") or 0) >= 100 for _, r in records):
            raise RuntimeError(f"已到第 100 轮，需检查权重或最终验证记录：{slot}")
        candidates = [(r.get("completed_epochs", 0), p, r) for p, r in records
                      if r.get("completed_epochs") and (p.parent / "weights/last.pt").is_file()]
        if candidates:
            epoch, path, record = max(candidates, key=lambda item: item[0])
            if record["status"] == "running":
                # The queue and its workers have exited without recording the interruption.
                record.update(status="interrupted", stop_reason="worker_exited", finished_utc=now())
                write_json(path, record)
            return slot, epoch, path.parent / "weights/last.pt"
        if any(r.get("completed_epochs") for _, r in records):
            raise RuntimeError(f"找不到已保存轮次的 last.pt，请检查：{slot}")
        return slot, 0, None
    return None, 0, None


def user_stopped(code, campaign):
    if code in (130, -2, -1073741510, 3221225786):
        return True
    paths = list(campaign.glob("*_seed*/attempt_*/run.json"))
    if not paths:
        return False
    latest = read_json(max(paths, key=lambda p: p.stat().st_mtime))
    return latest.get("error_type") == "KeyboardInterrupt" or latest.get("stop_reason") == "user_requested"


def supervise(data, campaign, device):
    last_position, retries = None, 0
    while True:
        proc, is_queue = active_job(data, campaign, device)
        if proc:
            print(f"{now()} 正在守候已有进程 {proc.pid}", flush=True)
            code = proc.wait()
            if user_stopped(code, campaign) or (is_queue and code == 0):
                return
            if active_job(data, campaign, device)[0]:
                continue
            if is_queue and not pending(campaign)[0] and code:
                raise RuntimeError("训练队列在模型训练之外退出，请查看原终端或实验记录。")
        if (campaign / "selection.json").is_file():
            return
        if campaign_for(data, device) != campaign:
            raise RuntimeError("数据、代码、环境或权重已变化，请核对原实验后续训。")
        slot, epoch, checkpoint = pending(campaign)
        if slot:
            position = (slot, epoch)
            retries = retries + 1 if position == last_position else 1
            last_position = position
            if retries > 3:
                raise RuntimeError(f"{slot.name} 在第 {epoch} 轮后连续三次恢复均无进展，已停止。")
            print(f"{now()} {slot.name} 从第 {epoch + 1}/100 轮继续，第 {retries} 次恢复", flush=True)
        command = ["resume", "--checkpoint", str(checkpoint)] if checkpoint else ["all"]
        if slot and not checkpoint:
            print("尚无首轮检查点，本模型从第 1 轮开始。", flush=True)
        result = subprocess.run([sys.executable, str(PROJECT / "train.py"), *command,
                                 "--data", str(data), "--device", device], cwd=PROJECT)
        if user_stopped(result.returncode, campaign):
            return
        if not checkpoint and result.returncode == 0:
            return
        if result.returncode:
            if not pending(campaign)[0]:
                raise RuntimeError(f"训练队列退出，退出码 {result.returncode}，请检查上方日志。")
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    data = args.data.resolve()
    os.environ.setdefault("YOLO_DISABLE_CUDNN", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    bootstrap()
    campaign = campaign_for(data, args.device)
    if not campaign.is_dir():
        parser.error("当前数据、代码、权重和环境没有对应的正式训练。请核对原实验设置。")
    try:
        with FileLock(campaign / "auto_resume.lock", timeout=0):
            print(f"{now()} 自动续训已启用：{campaign}", flush=True)
            supervise(data, campaign, args.device)
    except Timeout:
        print("本实验已有自动续训进程。", flush=True)


if __name__ == "__main__":
    main()
