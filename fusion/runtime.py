"""Local runtime paths and experiment evidence; no model downloads."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RUNS = ARTIFACTS / "runs" / "roboflow_v3"
EVALUATIONS = ARTIFACTS / "evaluations" / "roboflow_v3"
MODELS = ("baseline", "p5_vit_small", "p5_vit_random", "p5_vit_pretrained")
METRICS = ("precision", "recall", "map50", "map50_95")


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_hash():
    paths = sorted([*ROOT.glob("*.py"), *(ROOT / "fusion").glob("*.py"), *(ROOT / "configs").glob("*.yaml")])
    return fingerprint({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})


def local_output(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ARTIFACTS.resolve()) or path == ARTIFACTS.resolve():
        raise ValueError("Output must be a subdirectory of this project's artifacts/")
    return path


def bootstrap():
    for key, relative in {"YOLO_CONFIG_DIR": "config", "MPLCONFIGDIR": "cache/matplotlib",
                          "TORCH_HOME": "cache/torch", "TEMP": "cache/tmp", "TMP": "cache/tmp"}.items():
        path = ARTIFACTS / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", MPLBACKEND="Agg",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for name, version in {"ultralytics": "8.4.14", "torch": "2.13.0", "torchvision": "0.28.0"}.items():
        actual = importlib.metadata.version(name).split("+")[0]
        if actual != version:
            raise RuntimeError(f"Expected {name}=={version}, found {actual}; use requirements.txt")
    from ultralytics import utils
    settings = {k: False for k, v in utils.SETTINGS.items() if isinstance(v, bool)}
    for key, relative in {"datasets_dir": "datasets", "weights_dir": "weights", "runs_dir": "runs"}.items():
        path = ARTIFACTS / relative
        path.mkdir(exist_ok=True)
        settings[key] = str(path)
    utils.SETTINGS.update(settings)
    utils.DATASETS_DIR, utils.WEIGHTS_DIR, utils.RUNS_DIR = (Path(settings[k]) for k in ("datasets_dir", "weights_dir", "runs_dir"))


def environment(device=None):
    import torch
    index = int(device) if device is not None and str(device).isdigit() else 0
    gpu = torch.cuda.get_device_name(index) if str(device) != "cpu" and torch.cuda.is_available() and index < torch.cuda.device_count() else None
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "aiohttp")},
            "cuda": torch.version.cuda, "gpu": gpu}


def metrics_dict(metrics):
    index = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    counts = metrics.nt_per_class
    names = dict(enumerate(metrics.names)) if isinstance(metrics.names, list) else metrics.names
    per_class = {}
    for cls, name in names.items():
        instances = int(counts[cls]) if len(counts) > cls else 0
        values = metrics.box.class_result(index[cls]) if cls in index and instances else [None] * 4
        per_class[name] = {**dict(zip(METRICS, (float(v) if v is not None else None for v in values))), "instances": instances}
    overall = dict(zip(METRICS, map(float, metrics.box.mean_results()))) if sum(counts) else dict.fromkeys(METRICS)
    return {"overall": overall, "per_class": per_class,
            "precision_recall_definition": "Ultralytics maximum mean-F1 operating point; not conf=0.25"}
