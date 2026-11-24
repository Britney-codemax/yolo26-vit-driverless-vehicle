"""Export public experiment evidence from local manifests; no data or checkpoint downloads."""
import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from tempfile import NamedTemporaryFile, TemporaryDirectory

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .records import collect
from .runtime import MODEL_SET, ROOT, STUDIES, STUDY_MODELS, fingerprint, read_json, sha256


def eligible(row):
    m = row.get("metrics", {})
    score = m.get("overall", {}).get("map50_95")
    return (row.get("kind") == "train" and row.get("status") == "completed"
            and row.get("completed_epochs") == 100 and row.get("protocol_key")
            and m.get("split") == "val" and m.get("data_sha256") == row["data"]["sha256"]
            and score is not None and math.isfinite(score))


def collect_weights(row):
    run = ROOT / row["path"]
    target = MODEL_SET.parent / row["model"] / f"seed{row['seed']}_{fingerprint(row['id'])[:12]}"
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in (("best.pt", row["metrics"]["checkpoint_sha256"]), ("last.pt", row["last_sha256"])):
        source, output = run / "weights" / name, target / name
        if output.exists() and (output.samefile(source) or sha256(output) == digest):
            continue
        with TemporaryDirectory(dir=target) as tmp:
            ready = Path(tmp) / name
            try:
                ready.hardlink_to(source)
            except OSError:
                shutil.copy2(source, ready)
            if sha256(ready) != digest:
                raise ValueError(f"Checkpoint differs from its run record: {source}")
            ready.replace(output)
    return Path(os.path.relpath(target / "best.pt", ROOT)).as_posix()


def render_book(rows, path, inspections=()):
    book = Workbook()
    old = book.active
    old.title = "旧版历史参考"
    with (ROOT / "docs/历史验证集.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        old.append(next(reader))
        for model, *rest in reader:
            old.append([model, *(float(x) for x in rest[:-1]), rest[-1]])

    detail = book.create_sheet("新版逐次实验")
    detail.append(["数据指纹", "协议指纹", "模型", "seed", "集合", "状态", "轮数", "最佳轮",
                   "mAP50", "mAP50–95", "ΔmAP50–95 (百分点)", "召回率",
                   "UGV AP50", "UGV AP50–95", "UGV 召回率", "USV AP50", "USV AP50–95", "USV 召回率",
                   "负样本数", "有误报图片数 @0.25", "误报框 @0.25", "参数量", "峰值显存 MiB",
                   "验证前向 ms/图", "best.pt 相对路径", "最佳权重 SHA256", "末轮权重 SHA256", "实验目录", "新版数据集链接"])
    groups = defaultdict(lambda: defaultdict(list))
    present = set()
    statuses = {"completed": "完成", "running": "运行中", "failed": "失败", "interrupted": "中断",
                "not_run": "待训练"}
    for row in rows:
        if eligible(row):
            group = (row["data"]["sha256"], row["protocol_key"], row["model"])
            groups[group][row["seed"]].append(row)
    def score(row, key="map50_95"):
        return row["metrics"]["overall"][key]
    for row in rows:
        is_eval = row.get("kind") == "evaluation"
        m = row.get("metrics", {}) if row.get("status") == "completed" else {}
        overall = m.get("overall", {})
        cls = {k.strip().upper(): v for k, v in m.get("per_class", {}).items()}
        data_id, protocol = row.get("data", {}).get("sha256"), row.get("protocol_key")
        model, seed = row.get("model"), row.get("seed")
        if not is_eval:
            present.add((data_id, model, seed))
        delta = None
        if eligible(row):
            own = groups[(data_id, protocol, model)][seed]
            base = groups.get((data_id, protocol, "baseline"), {}).get(seed, [])
            if len(own) == len(base) == 1:
                delta = (score(row) - score(base[0])) * 100
        neg = m.get("negative_images") or {}
        loc = Path(row["path"])
        checkpoint = m.get("checkpoint")
        if checkpoint:
            checkpoint = row.get("exported_best") or (loc / checkpoint).as_posix()
        elif is_eval:
            checkpoint = row.get("weight_path")
        detail.append([data_id, protocol, model, seed, m.get("split", row.get("settings", {}).get("split", "val")),
                       statuses.get(row["status"], row["status"]), row.get("completed_epochs"), row.get("best_epoch"),
                       overall.get("map50"), overall.get("map50_95"), delta, overall.get("recall"),
                       *(cls.get(name, {}).get(k) for name in ("UGV", "USV") for k in ("map50","map50_95","recall")),
                       neg.get("images"), neg.get("flagged_images") if neg.get("images") else ("不适用" if neg else None),
                       neg.get("boxes") if neg.get("images") else ("不适用" if neg else None), row.get("parameters"),
                       row["peak_cuda_memory_bytes"] / 2**20 if row.get("peak_cuda_memory_bytes") else None,
                       m.get("inference_ms_per_image"), checkpoint, m.get("checkpoint_sha256"), row.get("last_sha256"),
                       loc.as_posix(), None])
    data_ids = {info["sha256"] for info in inspections} | {r["data"]["sha256"] for r in rows if r.get("data")}
    for data_id in sorted(data_ids) or [None]:
        for model in STUDY_MODELS:
            if (data_id, model, 0) not in present:
                detail.append([data_id, None, model, 0, "val", "待训练" if data_id else "待新版数据"])

    summary = book.create_sheet("新版多种子汇总")
    summary.append(["数据指纹", "协议指纹", "模型", "有效种子", "n", "mAP50 均值", "mAP50 标准差",
                    "mAP50–95 均值", "mAP50–95 标准差", "与基线配对种子", "配对 ΔmAP50–95 (百分点)",
                    "排除的重复种子"])
    for (data_id, protocol, model), seeds in sorted(groups.items()):
        valid = {s: items[0] for s, items in seeds.items() if len(items) == 1}
        base = groups.get((data_id, protocol, "baseline"), {})
        paired = sorted(s for s in valid if len(base.get(s, [])) == 1)
        values = []
        for k in ("map50","map50_95"):
            xs = [score(r,k) for r in valid.values()]
            values.extend([mean(xs) if xs else None, stdev(xs) if len(xs)>1 else None])
        delta = mean((score(valid[s])-score(base[s][0]))*100 for s in paired) if paired else None
        summary.append([data_id,protocol,model,str(sorted(valid)),len(valid),*values,str(paired),delta,
                        str(sorted(s for s in seeds if s not in valid))])
    for ws, percent in ((old, range(2,8)), (detail,(9,10,12,13,14,15,16,17,18)), (summary,range(6,10))):
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = PatternFill("solid", fgColor="264761")
            cell.alignment = Alignment(vertical="center")
            width = max(16, min(40, len(str(cell.value))*2+5))
            ws.column_dimensions[get_column_letter(cell.column)].width = width
        ws.row_dimensions[1].height = 27
        for fields in ws.iter_rows(min_row=2):
            for cell in fields:
                if cell.column in percent:
                    cell.number_format = "0.00%"
                elif isinstance(cell.value,float):
                    cell.number_format = "0.00"
    old.column_dimensions["A"].width = 50
    detail.column_dimensions["C"].width = 23
    summary.column_dimensions["C"].width = 23
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, suffix=".xlsx", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        book.save(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def export_workbook():
    rows, inspections = [], []
    for folder in sorted(STUDIES.glob("*")):
        if not (folder / "dataset.json").is_file():
            continue
        inspections.append(read_json(folder / "dataset.json"))
        for subdir in ("runs","evaluations"):
            rows.extend(collect(folder / subdir))
    for row in rows:
        if eligible(row) and row.get("last_sha256"):
            row["exported_best"] = collect_weights(row)
    return render_book(rows, ROOT / "实验结果汇总.xlsx", inspections)
