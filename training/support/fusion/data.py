"""Local YOLO/NDJSON detection validation and short-name caching."""
import asyncio
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
from urllib.parse import urlsplit, urlunsplit
import aiohttp
from PIL import Image
import yaml
from .runtime import ARTIFACTS, LEGACY_CONTENT, ROOT, fingerprint, sha256


def read_yolo(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not isinstance(config.get("names"), (list, dict)):
        raise ValueError("YOLO YAML requires a names list or class-ID mapping")
    names = config["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names
    if "nc" in config and config["nc"] != len(names):
        raise ValueError("nc must match the number of class names")
    root = (path.parent / config.get("path", ".")).resolve()
    source = config.get("roboflow", {})
    if not isinstance(source, dict):
        raise ValueError("roboflow metadata must be a mapping")
    source = {k: source[k] for k in ("workspace", "project", "version", "license", "url") if k in source}
    rows = [{"type": "dataset", "task": "detect", "class_names": names,
             "local_root": str(root), "source": source}]
    identities = {}
    exclusions = {}
    if config.get("train_exclusions"):
        manifest = (path.parent / config["train_exclusions"]).resolve()
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for pair in csv.DictReader(stream):
                ref = (root / pair["validation_image"]).resolve()
                if not ref.is_relative_to(root) or sha256(ref) != pair["validation_sha256"]:
                    raise ValueError("Validation image differs from the duplicate audit")
                exclusions[pair["train_image"]] = pair["train_sha256"]
    excluded = set()
    extensions = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
    for split in ("train", "val", "test"):
        value = config.get(split)
        if not value:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{split} must be one local image directory")
        value = value.replace("\\", "/")
        images = (root / value).resolve()
        # Roboflow ZIPs can use ../train/images even though train is inside the package.
        if not images.is_dir() and value.startswith("../"):
            images = (root / value[3:]).resolve()
        if not images.is_relative_to(root):
            raise ValueError(f"{split} image directory is outside the dataset root")
        if not images.is_dir():
            continue  # The common validator rejects missing required splits below.
        parts = list(images.relative_to(root).parts)
        positions = [i for i, part in enumerate(parts) if part in ("images", "imgs")]
        if not positions:
            raise ValueError(f"{split} requires an images/ or imgs/ directory with sibling labels/")
        parts[positions[-1]] = "labels"
        labels = root.joinpath(*parts)
        expected_labels = set()
        for image in sorted(images.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in extensions:
                continue
            label = labels / image.relative_to(images).with_suffix(".txt")
            if not image.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
                raise ValueError("Image or label resolves outside the dataset root")
            if not label.is_file():
                raise FileNotFoundError(f"Missing label file: {label.relative_to(root)}; negatives require an empty .txt")
            expected_labels.add(label)
            relative = image.relative_to(root).as_posix()
            if split == "train" and relative in exclusions:
                if sha256(image) != exclusions[relative]:
                    raise ValueError(f"Training image differs from the duplicate audit: {relative}")
                excluded.add(relative)
                continue
            verify_image(image)
            digest = sha256(image)
            if digest in identities and identities[digest] != split:
                raise ValueError("Identical image content occurs across splits")
            identities[digest] = split
            label_bytes = label.read_bytes()
            boxes = []
            for number, line in enumerate(label_bytes.decode("utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    box = list(map(float, line.split()))
                    if len(box) != 5:
                        raise ValueError("Expected five columns")
                except ValueError as exc:
                    raise ValueError(f"Invalid label {label.relative_to(root)}:{number}") from exc
                boxes.append(box)
            rows.append({"type": "image", "file": image.relative_to(root).as_posix(), "split": split,
                         "annotations": {"boxes": boxes}, "image_sha256": digest,
                         "label_sha256": hashlib.sha256(label_bytes).hexdigest()})
        if set(labels.rglob("*.txt")) - expected_labels:
            raise ValueError(f"Orphan label files in {split}")
    if excluded != set(exclusions):
        raise ValueError("Missing training images from the duplicate audit")
    if excluded:
        rows[0]["verified_train_exclusions"] = sorted(excluded)
    return rows


def read_data(path, required=("train", "val")):
    path = Path(path).resolve()
    is_yolo = path.suffix.lower() in (".yaml", ".yml")
    if is_yolo:
        rows = read_yolo(path)
    elif path.suffix.lower() == ".ndjson":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        raise ValueError("Provide a local YOLO .yaml/.yml or detection .ndjson file")
    if not rows or rows[0].get("type") != "dataset" or rows[0].get("task") != "detect":
        raise ValueError("First NDJSON record must describe a detection dataset")
    names = {int(k): str(v) for k, v in rows[0].get("class_names", {}).items()}
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs must be contiguous starting at 0")
    counts = {s: {"images": 0, "boxes": 0, "boxless_images": 0} for s in ("train", "val", "test")}
    seen, identities, groups = set(), {}, {}
    grouped = 0
    for row in rows[1:]:
        split, name = row.get("split"), row.get("file", "")
        if row.get("type") != "image" or split not in counts:
            raise ValueError("Images require type=image and split=train/val/test")
        relative = PurePosixPath(name.replace("\\", "/"))
        if not name or relative.is_absolute() or ".." in relative.parts or ":" in name:
            raise ValueError("Unsafe image filename")
        key = (split, str(relative.with_suffix("")).casefold())
        if key in seen:
            raise ValueError(f"Duplicate image/label stem in {split}")
        seen.add(key)
        url = urlsplit(row.get("url", ""))
        if row.get("url") and (url.scheme not in ("http", "https") or not url.netloc):
            raise ValueError("Image URLs must be http(s)")
        identity = urlunsplit((url.scheme, url.netloc, url.path, "", "")) if row.get("url") else str(relative).casefold()
        if identity in identities and identities[identity] != split:
            raise ValueError("An image occurs across splits")
        identities[identity] = split
        annotations = row.get("annotations", {})
        if not isinstance(annotations, dict) or set(annotations) - {"boxes"}:
            raise ValueError("Only detection boxes are supported")
        boxes = annotations.get("boxes", [])
        if not isinstance(boxes, list):
            raise ValueError("boxes must be a list")
        for box in boxes:
            if not isinstance(box, list) or len(box) != 5 or not all(type(v) in (int, float) and math.isfinite(v) for v in box):
                raise ValueError("Boxes must contain five finite numbers")
            cls, x, y, w, h = box
            if cls != int(cls) or int(cls) not in names or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("Expected [class, normalized cx, cy, width, height]")
            # Exported normalized centers/sizes are rounded; allow tiny edge roundoff without rewriting labels.
            if min(x - w / 2, y - h / 2) < -1e-4 or max(x + w / 2, y + h / 2) > 1 + 1e-4:
                raise ValueError("Box extends outside the image")
        counts[split]["images"] += 1
        counts[split]["boxes"] += len(boxes)
        counts[split]["boxless_images"] += not boxes
        if row.get("group_id") is not None:
            grouped += 1
            group = str(row["group_id"])
            if group in groups and groups[group] != split:
                raise ValueError("group_id crosses splits")
            groups[group] = split
    for split in required:
        if not counts[split]["images"]:
            raise ValueError(f"Missing required split '{split}'; no fallback")
    warnings = []
    if grouped != len(rows) - 1:
        warnings.append("Source-group isolation unverified; visual duplicates were not checked")
    if not counts["test"]["images"]:
        warnings.append("No independent test split")
    digest = fingerprint({"names": names, "source": rows[0]["source"], "images": rows[1:]}) if is_yolo else sha256(path)
    info = {"sha256": digest, "names": names, "counts": counts,
                  "grouping": "verified_ids" if grouped == len(rows) - 1 else "unverified",
                  "warnings": warnings}
    if is_yolo:
        info["source"] = rows[0]["source"]
        info["content_sha256"] = fingerprint({"names": names, "files": sorted(
            (r["split"], r["image_sha256"], r["label_sha256"]) for r in rows[1:])})
        if (path == (ROOT / "dataset/data.yaml").resolve()
                and info["content_sha256"] == "ff73b4fe89effd9779f71bea0881c682b33f278b4f89736101deffb242cf8a78"):
            return read_data(ROOT / "configs/study-data.yaml", required)
        if info["content_sha256"] != LEGACY_CONTENT:
            info["sha256"] = fingerprint({"names": names, "images": rows[1:]})
        if rows[0].get("verified_train_exclusions"):
            info["verified_train_exclusions"] = rows[0]["verified_train_exclusions"]
            info["warnings"] = [w for w in warnings if not w.startswith("Source-group")]
            info["warnings"].append("Known train/val duplicate pairs excluded from training; source groups not verified")
    return rows, info


def inspect_study(path, manifest_root=None):
    if not path or not Path(path).is_file() or Path(path).suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("请先设置新版数据集的本地 DATA_YAML（YOLO .yaml/.yml）。")
    rows, info = read_data(path)
    if info["content_sha256"] == LEGACY_CONTENT:
        raise ValueError("这是历史 v3 数据，请提供新版 DATA_YAML。")
    names = {k: v.strip().upper() for k, v in info["names"].items()}
    if len(names) != 2 or set(names.values()) != {"UGV", "USV"}:
        raise ValueError(f"期望 UGV 和 USV 两个类别，实际为 {info['names']}。")
    sizes = {split: {name: dict(small=0, medium=0, large=0) for name in names.values()}
             for split in info["counts"]}
    root = Path(rows[0]["local_root"])
    for row in rows[1:]:
        with Image.open(root / row["file"]) as image:
            w, h = image.size
        scale = 640 / max(w, h)
        for cls, _, _, bw, bh in row["annotations"]["boxes"]:
            area = bw * w * bh * h * scale ** 2
            bucket = "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"
            sizes[row["split"]][names[int(cls)]][bucket] += 1
    info["box_sizes_long_edge_640"] = sizes
    info.pop("source", None)
    info["public_dataset_url"] = ""
    if manifest_root:
        folder = Path(manifest_root) / info["sha256"]
        folder.mkdir(parents=True, exist_ok=True)
        for split, counts in info["counts"].items():
            if counts["images"]:
                paths = [row["file"] for row in rows[1:] if row["split"] == split]
                (folder / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    return info


def inspect_data(path, required=("train", "val")):
    return read_data(path, required)[1]


def verify_image(path):
    with Image.open(path) as image:
        image.verify()


async def convert_data(path, required=("train", "val"), cache_root=None):
    path = Path(path).resolve()
    rows, info = read_data(path, required)
    is_yolo = path.suffix.lower() in (".yaml", ".yml")
    local_root = Path(rows[0]["local_root"]) if is_yolo else path.parent
    directory = Path(cache_root or ARTIFACTS / "datasets") / info["sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(16)

    async def process(session, row):
        async with semaphore:
            split, original = row["split"], row["file"]
            suffix = PurePosixPath(original.replace("\\", "/")).suffix.lower()
            short = hashlib.sha256(original.encode("utf-8")).hexdigest()
            image_path = directory / "images" / split / (short + suffix)
            label_path = directory / "labels" / split / (short + ".txt")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = image_path.with_suffix(image_path.suffix + ".part")
            try:
                if not image_path.exists():
                    local = local_root / original
                    if local.is_file():
                        shutil.copyfile(local, temporary)
                    elif row.get("url"):
                        for attempt in range(3):
                            try:
                                async with session.get(row["url"], timeout=aiohttp.ClientTimeout(total=30)) as response:
                                    response.raise_for_status()
                                    temporary.write_bytes(await response.read())
                                verify_image(temporary)
                                break
                            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                                if attempt == 2:
                                    raise
                                await asyncio.sleep(2 ** attempt)
                    else:
                        raise FileNotFoundError("Neither local image nor URL is available")
                    verify_image(temporary)
                    temporary.replace(image_path)
                verify_image(image_path)
                if is_yolo and sha256(image_path) != row["image_sha256"]:
                    raise ValueError("Image content changed after inspection or cache was modified")
                boxes = row.get("annotations", {}).get("boxes", [])
                label_path.write_text("".join(" ".join(map(str, box)) + "\n" for box in boxes), encoding="utf-8")
                return True
            except Exception as exc:
                # Signed URLs can contain credentials: report the hashed record, not the exception URL.
                print(f"Image {split}/{short}: {type(exc).__name__}")
                return False
            finally:
                temporary.unlink(missing_ok=True)

    async with aiohttp.ClientSession() as session:
        completed = await asyncio.gather(*(process(session, row) for row in rows[1:]))
    if not all(completed):
        raise RuntimeError(f"Incomplete dataset: {sum(completed)}/{len(completed)} images; refusing partial evaluation/training")
    config = {"path": str(directory.resolve()), "names": info["names"], "train": None, "val": None,
              "expected_images": {s: c["images"] for s, c in info["counts"].items()}}
    for split, count in info["counts"].items():
        if count["images"]:
            config[split] = f"images/{split}"
    yaml_path = directory / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path
