"""Preview both CPU detectors on two examples, or benchmark the validation set."""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODELS = ("p4_mamba", "p4_mamba_p5_vit")
CLASSES = ("UGV", "USV")
SAMPLES = (
    "G2_bing_020_jpg.rf.29c840849ac8d44560fc7cf64f27f2b5.jpg",
    "XOCEAN_XO-450_7fd59484_jpg.rf.53ea2ac582c08c15a2fac4431453e464.jpg",
)


def load_session(name):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(ROOT / f"{name}.onnx"), options,
                                providers=["CPUExecutionProvider"])


def detect(session, path):
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f"Could not read {path}")
    height, width = image.shape[:2]
    gain = 640 / max(height, width)
    w, h = round(width * gain), round(height * gain)
    small = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    left, top = (640 - w) // 2, (640 - h) // 2
    padded = cv2.copyMakeBorder(small, top, 640 - h - top, left, 640 - w - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255
    start = perf_counter()
    raw = session.run(["output0"], {"images": tensor})[0][0]
    model_ms = (perf_counter() - start) * 1000
    # YOLO26 end-to-end output is already xyxy, confidence, class.
    boxes = raw[raw[:, 4] > 0.25].copy()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / gain).clip(0, width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / gain).clip(0, height)
    return image, boxes, model_ms


def preview(session, directory):
    directory.mkdir(parents=True, exist_ok=True)
    detections = {}
    for filename in SAMPLES:
        image, boxes, _ = detect(session, ROOT / "images" / filename)
        detections[filename] = boxes.tolist()
        for x1, y1, x2, y2, score, cls in boxes:
            x, y = round(x1), round(y1)
            cv2.rectangle(image, (x, y), (round(x2), round(y2)), (0, 220, 140), 2)
            cv2.putText(image, f"{CLASSES[int(cls)]} {score:.3f}", (x, max(18, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 110, 20), 2)
        if not cv2.imwrite(str(directory / filename), image):
            raise OSError(f"Could not save {filename}")
    (directory / "detections.json").write_text(json.dumps(detections, indent=2), encoding="utf-8")


def pi_status():
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    temp = float(thermal.read_text()) / 1000 if thermal.is_file() else None
    flags = None
    if shutil.which("vcgencmd"):
        result = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True)
        if result.returncode == 0:
            flags = result.stdout.strip().removeprefix("throttled=")
    return temp, flags


def benchmark(session, paths, model):
    for path in paths[:10]:
        detect(session, path)
    temp_before, flags_before = pi_status()
    model_times, total_times = [], []
    start = perf_counter()
    for path in paths:
        begin = perf_counter()
        _, _, model_ms = detect(session, path)
        total_times.append((perf_counter() - begin) * 1000)
        model_times.append(model_ms)
    elapsed = perf_counter() - start
    temp_after, flags_after = pi_status()
    return dict(model=model, images=len(paths), threads=4, warmup=10,
                model_p50_ms=round(np.median(model_times), 2),
                model_p95_ms=round(np.percentile(model_times, 95), 2),
                total_p50_ms=round(np.median(total_times), 2),
                total_p95_ms=round(np.percentile(total_times, 95), 2),
                images_per_sec=round(len(paths) / elapsed, 3),
                temp_before_c=temp_before, temp_after_c=temp_after,
                throttled_before=flags_before, throttled_after=flags_after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "bench"))
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    paths = sorted((ROOT / "images").glob("*.jpg"))
    if args.mode == "bench" and len(paths) != 93:
        parser.error(f"Expected all 93 validation images, found {len(paths)}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in MODELS:
        session = load_session(name)
        if args.mode == "preview":
            preview(session, args.out / name)
        else:
            row = benchmark(session, paths, name)
            rows.append(row)
            print(row, flush=True)
        del session
    if rows:
        with (args.out / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
