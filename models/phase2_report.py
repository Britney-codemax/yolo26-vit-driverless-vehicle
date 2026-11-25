"""Re-evaluate the 14 candidates together and append one sheet to the original workbook."""
import asyncio
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from models.phase2 import COCO, DATA, GROUPS, OUTPUT, PROTOCOL, ROOT, SPECS, STUDIES, completed, training_protocol
from fusion.data import convert_data
from fusion.runtime import environment, fingerprint, read_json, sha256, write_json
from evaluate import evaluate_checkpoint, evaluation_settings

# Baseline supplies the USV floor; only R4 and R5 participate in the ranking.
REFERENCES = {
    "baseline": ("baseline", "a3c5e3b11a0a1712ec77660094572f2c3f2673b24e0ff152594c0187f7aa74dc"),
    "R4": ("p4_mamba", "7ecc4160766134d5d20a46d0908a23536f93213df39539254ad0e63c328692ad"),
    "R5": ("p4_mamba_p5_vit", "a3062f107b960e30281f04eee64493e2abb0cf2cae17fc40dbae77a4c601a35f"),
}
HEADERS = ["模型", "UGV mAP50", "UGV mAP50–95", "USV mAP50", "USV mAP50–95",
           "Overall mAP50", "Overall mAP50–95"]


def references():
    sources = {}
    for key, (name, expected) in REFERENCES.items():
        for path in (STUDIES / DATA / "runs").glob(f"*/{name}_seed0/*/metrics.json"):
            metrics = read_json(path)
            if metrics["checkpoint_sha256"] != expected:
                continue
            row = read_json(path.with_name("run.json"))
            checkpoint = path.parent / "weights/best.pt"
            if (row["status"] != "completed" or row["completed_epochs"] != 100 or row["seed"] != 0
                    or row["model"] != name or row["yolo_sha256"] != COCO or metrics["split"] != "val"
                    or metrics["data_sha256"] != DATA or training_protocol(row["effective_training"]) != PROTOCOL
                    or sha256(checkpoint) != expected):
                raise ValueError(f"第一轮对照记录不符：{path}")
            sources[key] = (checkpoint, row)
            break
        if key not in sources:
            raise FileNotFoundError(f"请在有原始 {name} 训练记录与 best.pt 的汇总机上复评")
    return sources


def select(metrics):
    floors = {"UGV": metrics["R4"]["per_class"]["UGV"]["map50"],
              "USV": metrics["baseline"]["per_class"]["USV"]["map50"]}
    if not all(isinstance(v, (float, int)) and 0 <= v <= 1 for v in floors.values()):
        raise ValueError("对照组的 AP50 门槛不完整")
    keys = ["R4", "R5", *SPECS]
    missing = {}
    for key in keys:
        m = metrics[key]
        values = [m["per_class"][c][n] for c in floors for n in ("map50", "map50_95")]
        values += [m["overall"]["map50"], m["overall"]["map50_95"]]
        latency = m["inference_ms_per_image"]
        if (not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in values)
                or not isinstance(latency, (int, float)) or not isfinite(latency) or latency < 0):
            raise ValueError(f"{key}: 验证指标不完整")
        missing[key] = [c for c, floor in floors.items() if m["per_class"][c]["map50"] < floor]
    qualified = [key for key in keys if not missing[key]]
    def rank(key):
        m = metrics[key]
        return (-m["overall"]["map50"], -m["per_class"]["USV"]["map50_95"], m["inference_ms_per_image"], key)
    winner = min(qualified or keys, key=rank)
    return dict(winner=winner, floors=floors, unmet=missing[winner], qualified=qualified,
                map50_090=metrics[winner]["overall"]["map50"] >= .9)


def write_sheet(path, metrics, decision):
    book = load_workbook(path)
    title = "第二轮14组对比"
    if title in book.sheetnames:
        del book[title]
    sheet = book.create_sheet(title)
    sheet.append(HEADERS)
    for key in ["R4", "R5", *SPECS]:
        m = metrics[key]
        if key in REFERENCES:
            label = key + " · " + REFERENCES[key][0]
        else:
            layer, route, vit = SPECS[key]
            routing = {"both": "双路径", "skip": "仅跳连", "main": "仅主路"}[route]
            label = f"{key} · Mamba L{layer} {routing}" + (f" + ViT L{vit}" if vit else "")
        sheet.append([label, *[group[n] for group in (m["per_class"]["UGV"], m["per_class"]["USV"], m["overall"])
                                for n in ("map50", "map50_95")]])
        for cell in sheet[sheet.max_row][1:]:
            cell.number_format = "0.00%"
    for c in sheet[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="244B61")
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = "A1:G15"
    for i in range(1, 8):
        sheet.column_dimensions[get_column_letter(i)].width = 47 if i == 1 else 21
    winner = decision["winner"]
    outcome = "通过两类门槛" if not decision["unmet"] else "未达到 " + "、".join(decision["unmet"]) + " 门槛"
    sheet.merge_cells("A17:G17")
    sheet["A17"] = f"推荐 {winner}，{outcome}。单种子 Overall mAP50 = {metrics[winner]['overall']['map50']:.4%}。"
    with NamedTemporaryFile(dir=path.parent, suffix=".xlsx", delete=False) as f:
        tmp = Path(f.name)
    try:
        book.save(tmp)
        book.close()
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def review(data, device, study):
    sources = references()
    missing = []
    for group, names in GROUPS.items():
        manifest = OUTPUT / group / "assignment.json"
        if not manifest.is_file():
            missing.extend(names)
            continue
        meta = read_json(manifest)
        if meta["study"] != study or meta["group"] != group:
            raise ValueError(f"{group} 的实现、数据或配置与汇总机不同")
        for name in names:
            result = completed(manifest.parent / name, name, fingerprint(study), meta["model_set_sha256"])
            if result:
                directory, row, _ = result
                sources[name] = (directory / "weights/best.pt", row)
            else:
                missing.append(name)
    if missing:
        raise ValueError("请先收齐这几组的完整结果：" + "、".join(missing))
    settings = evaluation_settings(640, 4, device)
    machine = environment(device)
    out = OUTPUT / "review"
    prepared = asyncio.run(convert_data(data))
    results, provenance = {}, {}
    for key, (ck, run) in sources.items():
        print(f"同机复评 {key}，{ck.name}", flush=True)
        metrics = evaluate_checkpoint(ck, prepared, out / key, settings, {0: "UGV", 1: "USV"})
        results[key] = metrics
        provenance[key] = {"training_gpu": run["environment"]["gpu"], "best_epoch": run["best_epoch"],
                           "resume_epochs": run.get("resume", {}).get("interruption_epochs", []),
                           "training_protocol": run["protocol_key"]}
    decision = select(results)
    write_json(out / "comparison.json", dict(data_sha256=DATA, seed=0, environment=machine, settings=settings,
                                             results=results, training=provenance, selection=decision))
    path = ROOT / "实验结果汇总.xlsx"
    write_sheet(path, results, decision)
    print(f"推荐 {decision['winner']}，详见 {path} 的第二轮14组对比")
