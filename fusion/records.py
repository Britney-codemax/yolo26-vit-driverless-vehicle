"""Read-only experiment collection and repeatable, compact Markdown summaries."""
from collections import Counter, defaultdict
import csv
import io
import statistics
from pathlib import Path
from .runtime import ROOT, ARTIFACTS, RUNS, EVALUATIONS, METRICS, read_json, write_json

START, END = "<!-- AUTO_RESULTS_START -->", "<!-- AUTO_RESULTS_END -->"


def collect(root):
    root = Path(root)
    rows, known = [], set()
    for path in sorted(root.rglob("run.json")):
        row = read_json(path)
        row["path"] = path.parent.relative_to(ROOT).as_posix()
        row["id"] = row.get("id", path.parent.name)
        metrics = path.parent / "metrics.json"
        if metrics.is_file():
            row["metrics"] = read_json(metrics)
        rows.append(row)
        known.add(path.parent.resolve())
    for path in sorted(root.rglob("plan.json")):
        for entry in read_json(path)["entries"]:
            directory = (ROOT / entry["path"]).resolve()
            if directory not in known:
                rows.append({**entry, "kind": "train", "status": entry.get("status", "not_run")})
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("kind") == "train" and row.get("status") == "completed" and row.get("protocol_key"):
            if row.get("metrics", {}).get("overall", {}).get("map50") is not None:
                groups[(row["protocol_key"], row["model"], row.get("vit_sha256"))].append(row)
    result = []
    for (protocol, model, vit_hash), items in sorted(groups.items(), key=lambda x: str(x[0])):
        seeds = Counter(row["seed"] for row in items)
        unique = [row for row in items if seeds[row["seed"]] == 1]
        metrics = {}
        for metric in METRICS:
            values = [row["metrics"]["overall"][metric] for row in unique if row["metrics"]["overall"].get(metric) is not None]
            metrics[metric] = {"mean": statistics.mean(values) if values else None,
                               "std": statistics.stdev(values) if len(values) > 1 else None}
        result.append({"protocol_key": protocol, "model": model, "vit_sha256": vit_hash,
                       "n": len(unique), "seeds": sorted(row["seed"] for row in unique),
                       "excluded_duplicate_seeds": sorted(s for s, count in seeds.items() if count > 1),
                       "metrics": metrics})
    baselines = {g["protocol_key"]: g for g in result if g["model"] == "baseline"}
    for group in result:
        baseline = baselines.get(group["protocol_key"])
        a = group["metrics"]["map50"]["mean"]
        b = baseline["metrics"]["map50"]["mean"] if baseline else None
        group["delta_map50"] = a - b if a is not None and b is not None and group["seeds"] == baseline["seeds"] else None
    return result


def cell(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows])


def render(rows, groups):
    if not rows:
        return "## 自动实验结果\n\n暂无新版数据实验结果。旧运行目录不参与当前汇总。\n"
    statuses = {"not_run": "未运行", "completed": "完成", "failed": "失败", "interrupted": "中断",
                "running": "运行中或异常中断", "process_succeeded": "进程结束，缺少结果记录"}
    values = []
    class_values = []
    for row in rows:
        metrics = row.get("metrics", {}).get("overall", {})
        values.append([row["id"], row.get("kind", "train"), row.get("model"), row.get("seed"),
                       row.get("data", {}).get("sha256", "")[:12], row.get("protocol_key", "")[:12],
                       row.get("best_epoch"), row.get("completed_epochs"),
                       *(metrics.get(k) for k in METRICS), row.get("parameters"),
                       statuses.get(row.get("status"), row.get("status")), row["path"]])
        for name, metric in row.get("metrics", {}).get("per_class", {}).items():
            class_values.append([row["id"], name, metric.get("instances"), *(metric.get(k) for k in METRICS)])
    text = ["## 自动实验结果", "", "best.pt 按 mAP50–95 fitness 选择；P/R 为最大平均 F1 工作点。指标缺失不补 0。",
            "数据、代码、实际配置或环境不同时分组；独立评估不参与训练种子统计。", "",
            table(["实验", "类型", "模型", "seed", "数据哈希", "协议", "最佳轮", "已跑轮", "P", "R", "mAP50", "mAP50–95", "参数量", "状态", "本地相对路径"], values), "",
            "## 每类指标", "", table(["实验", "类别", "实例数", "P", "R", "AP50", "AP50–95"], class_values), "",
            "## 同条件复验汇总", "", "标准差使用 ddof=1；单种子标准差为 N/A。同模型同条件重复 seed 全部排除，不挑最高值。",
            "ΔmAP50 仅在 baseline 与候选具有相同有效 seed 集合时给出。", "",
            table(["协议", "模型", "seeds", "n", "mAP50 均值", "标准差", "mAP50–95 均值", "标准差", "ΔmAP50", "排除的重复 seed"],
                  [[g["protocol_key"][:12], g["model"], g["seeds"], g["n"],
                    g["metrics"]["map50"]["mean"], g["metrics"]["map50"]["std"],
                    g["metrics"]["map50_95"]["mean"], g["metrics"]["map50_95"]["std"],
                    g["delta_map50"], g["excluded_duplicate_seeds"]] for g in groups]), ""]
    return "\n".join(text)


def refresh(root=None, destination=None, include_evaluations=True):
    rows = collect(root or RUNS)
    if include_evaluations:
        rows.extend(collect(EVALUATIONS))
    groups = aggregate(rows)
    destination = Path(destination or ROOT / "docs/EXPERIMENTS.md")
    content = destination.read_text(encoding="utf-8")
    if content.count(START) != 1 or content.count(END) != 1 or content.index(START) >= content.index(END):
        raise ValueError("Markdown requires exactly one ordered AUTO_RESULTS marker pair")
    updated = content[:content.index(START) + len(START)] + "\n\n" + render(rows, groups) + "\n" + content[content.index(END):]
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(destination)
    report = ARTIFACTS / "reports"
    write_json(report / "summary.json", {"runs": rows, "aggregate": groups})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["id", "model", "seed", "status", "protocol_key", *METRICS, "path"])
    writer.writeheader()
    for row in rows:
        writer.writerow({**{k: row.get(k) for k in ("id", "model", "seed", "status", "protocol_key", "path")},
                         **{k: row.get("metrics", {}).get("overall", {}).get(k) for k in METRICS}})
    (report / "runs.csv").write_text(output.getvalue(), encoding="utf-8-sig")
    return rows, groups


def refresh_safely():
    try:
        refresh()
    except Exception as exc:
        print(f"Summary refresh failed ({type(exc).__name__}); raw run files remain available. Run compare.py after fixing it.")
