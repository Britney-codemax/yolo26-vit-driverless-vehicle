"""Local runtime paths and experiment evidence; no model downloads."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta

SUPPORT = Path(__file__).resolve().parents[1]
ROOT = SUPPORT.parent  # Data, configuration and run records are relative to training/.
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
RUNS = ARTIFACTS / "runs" / "roboflow_v3"
EVALUATIONS = ARTIFACTS / "evaluations" / "roboflow_v3"
DIAGNOSTICS = ARTIFACTS / "diagnostics"
LEGACY_MODELS = ("baseline", "p5_vit_small", "p5_vit_random", "p5_vit_pretrained")
STUDY_MODELS = ("baseline", "p5_vit_small", "p5_mamba", "p4_mamba", "p4_mamba_p5_vit")
MODELS = LEGACY_MODELS + STUDY_MODELS[2:]
STUDIES = ARTIFACTS / "studies"
MODEL_SET = PROJECT / "models" / "fusion.yaml"
LEGACY_CONTENT = "1093f2c192255a66b8bc2aa3d5aca676b22cc08ae3206b06eb89e3ccc5631b83"


def result_root(data, kind):
    if data.get("content_sha256") and data["content_sha256"] != LEGACY_CONTENT:
        return STUDIES / data["sha256"] / kind
    return RUNS if kind == "runs" else EVALUATIONS
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
    paths = sorted([PROJECT / "train.py", *SUPPORT.glob("*.py"), *(SUPPORT / "fusion").glob("*.py"), *(ROOT / "configs").glob("*.yaml"),
                    *(ROOT / "configs").glob("*.csv")])
    return fingerprint({p.relative_to(PROJECT).as_posix(): sha256(p) for p in paths})


def local_output(path):
    path = Path(path)
    base = PROJECT if path.parts[:1] == (ROOT.name,) else ROOT
    path = (base / path).resolve()
    if not path.is_relative_to(ARTIFACTS.resolve()) or path == ARTIFACTS.resolve():
        raise ValueError("Output must be a subdirectory of this project's training/artifacts/")
    return path


def bootstrap():
    for key, relative in {"YOLO_CONFIG_DIR": "config", "MPLCONFIGDIR": "cache/matplotlib",
                          "TORCH_HOME": "cache/torch", "TEMP": "cache/tmp", "TMP": "cache/tmp"}.items():
        path = ARTIFACTS / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", MPLBACKEND="Agg",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    check_package_versions()
    check_isolated_environment()
    if os.environ.get("YOLO_DISABLE_CUDNN") == "1":
        import torch
        torch.backends.cudnn.enabled = False
    from ultralytics import utils
    settings = {k: False for k, v in utils.SETTINGS.items() if isinstance(v, bool)}
    for key, relative in {"datasets_dir": "datasets", "weights_dir": "weights", "runs_dir": "runs"}.items():
        path = ARTIFACTS / relative
        path.mkdir(exist_ok=True)
        settings[key] = str(path)
    utils.SETTINGS.update(settings)
    utils.DATASETS_DIR, utils.WEIGHTS_DIR, utils.RUNS_DIR = (Path(settings[k]) for k in ("datasets_dir", "weights_dir", "runs_dir"))


def check_package_versions():
    actual = tuple(importlib.metadata.version(name).split("+")[0]
                   for name in ("ultralytics", "torch", "torchvision"))
    if actual not in (("8.4.14", "2.13.0", "0.28.0"), ("8.4.14", "2.12.1", "0.27.1")):
        raise RuntimeError(f"Unsupported ultralytics/torch/torchvision combination: {actual}; use a pinned matching pair")


def check_isolated_environment():
    import site
    prefix = Path(sys.prefix).resolve()
    config = prefix / "pyvenv.cfg"
    if (sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE or not config.is_file()
            or "include-system-site-packages = false" not in config.read_text(encoding="utf-8").lower()):
        raise RuntimeError("Use .venv-cuda with system/user site packages disabled; see README installation steps")
    for name in ("torch", "torchvision", "ultralytics"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError(f"{name} was loaded outside this environment: {module.__file__}; use .venv-cuda")


def environment(device=None):
    import torch
    import site
    import torchvision
    import ultralytics
    index = int(device) if device is not None and str(device).isdigit() else 0
    gpu = torch.cuda.get_device_name(index) if str(device) != "cpu" and torch.cuda.is_available() and index < torch.cuda.device_count() else None
    driver = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5, check=True)
        versions = result.stdout.strip().splitlines()
        driver = versions[index] if index < len(versions) else None
    except (OSError, subprocess.SubprocessError):
        pass  # Driver information is optional on CPU-only hosts.
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "aiohttp")},
            "cuda": torch.version.cuda, "gpu": gpu, "driver": driver, "cudnn": torch.backends.cudnn.version(),
            "python_executable": sys.executable, "torch_path": torch.__file__,
            "python_prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "user_site_enabled": site.ENABLE_USER_SITE,
            "torchvision_path": torchvision.__file__, "ultralytics_path": ultralytics.__file__,
            "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "cudnn_v8_api_disabled": os.environ.get("TORCH_CUDNN_V8_API_DISABLED"),
            "cudnn_enabled": torch.backends.cudnn.enabled}


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
            "negative_images": getattr(metrics, "negative_images", None),
            "inference_ms_per_image": getattr(metrics, "speed", {}).get("inference"),
            "precision_recall_definition": "Ultralytics maximum mean-F1 operating point; not conf=0.25"}
