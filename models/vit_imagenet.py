"""Matched ImageNet initialization for the first-round P4 Mamba + P5 ViT model."""
import argparse
import os
from pathlib import Path

import torch
import yaml
from filelock import FileLock

import train
from fusion import models
from fusion.data import inspect_data
from fusion.runtime import MODEL_SET, ROOT, STUDIES, bootstrap, fingerprint, read_json, sha256
from models.phase2 import (COCO, DATA, PROTOCOL, SETTINGS, WEIGHTS, completed, finish_validation,
                           next_attempt, recover_source, training_protocol)

MODEL = "p4_mamba_p5_vit"
ARCHITECTURE = "910ff25326f9d094d4d68e15ad509618050b1b2cd5343f7798ce84658172a578"
SLOT = STUDIES / DATA / "vit_imagenet" / MODEL
DONOR = WEIGHTS.with_name("pit_b_224_in1k.bin")
REVISION = "f46cba2050d0bb3f6faa0e5fcf5e8edda35f1436"
URL = f"https://huggingface.co/timm/pit_b_224.in1k/resolve/{REVISION}/pytorch_model.bin"
DIGEST = "296a53a0c700167e5300c08763a8f6147485c9796cb68cdbaa843d2809c9c9b5"
PREFIX = "transformers.0.blocks.0."
CONTROL_CHECKPOINT = "a3062f107b960e30281f04eee64493e2abb0cf2cae17fc40dbae77a4c601a35f"


def load_encoder(encoder):
    if not DONOR.is_file():
        raise FileNotFoundError("请先运行 python -m models.vit_imagenet prepare")
    if sha256(DONOR) != DIGEST:
        raise ValueError(f"PiT-B ImageNet 权重校验失败：{DONOR}")
    source = torch.load(DONOR, weights_only=True, map_location="cpu", mmap=True)
    modules = {"norm1": "ln_1", "attn.proj": "self_attention.out_proj", "norm2": "ln_2",
               "mlp.fc1": "mlp.0", "mlp.fc2": "mlp.3"}
    mapping = {}
    for key in source:
        if key.startswith(PREFIX):
            layer, kind = key[len(PREFIX):].rsplit(".", 1)
            name = "self_attention.in_proj_" + kind if layer == "attn.qkv" else f"{modules[layer]}.{kind}"
            mapping[name] = key
    if encoder.self_attention.num_heads != 4 or len(mapping) != 12:
        raise ValueError("需要四头注意力和完整的 12 个编码块张量")
    encoder.load_state_dict({name: source[key] for name, key in mapping.items()}, strict=True)
    return dict(source=URL, sha256=DIGEST, block=PREFIX.rstrip("."), loaded=mapping)


def prepare(architecture):
    if not DONOR.is_file():
        torch.hub.download_url_to_file(URL, str(DONOR), hash_prefix=DIGEST)
    wrapper, _ = models.build_model(MODEL, seed=0, yolo_weights=WEIGHTS, architecture=architecture)
    encoder = wrapper.model.model[10].encoder
    before = {name: value.clone() for name, value in encoder.state_dict().items()}
    report = load_encoder(encoder)
    if all(torch.equal(before[k], v) for k, v in encoder.state_dict().items()):
        raise RuntimeError("ViT 初始化未产生变化")
    # Training reconstructs the 80-class source into a two-class detector before its first update.
    rebuilt = models.make_detector(wrapper.model.yaml, nc=2, weights=wrapper.model)
    for key, value in rebuilt.model[10].encoder.state_dict().items():
        torch.testing.assert_close(value, encoder.state_dict()[key], rtol=0, atol=0)
    print(f"准备完成：PiT-B 的 {len(report['loaded'])} 个张量已逐项通过两类别模型重建校验。")


def show_result(directory, metrics):
    print(f"正式验证结果：{directory / 'metrics.json'}")
    controls = (STUDIES / DATA / "runs").glob(f"*/{MODEL}_seed0/attempt_*/metrics.json")
    reference = next((m for m in map(read_json, controls)
                      if m.get("checkpoint_sha256") == CONTROL_CHECKPOINT and m.get("data_sha256") == DATA), None)
    print("指标                 ImageNet 初始化    原第五组     差值（百分点）")
    for label in ("UGV", "USV", "Overall"):
        for key in ("map50", "map50_95"):
            group = metrics["overall"] if label == "Overall" else metrics["per_class"][label]
            value = group[key]
            line = f"{label + ' ' + key:<21} {value:7.2%}"
            if reference:
                control = reference["overall"] if label == "Overall" else reference["per_class"][label]
                line += f"        {control[key]:7.2%}       {(value - control[key]) * 100:+.2f}"
            print(line)
    if not reference:
        print("本机缺少已核验的第五组原始 metrics.json，未计算差值。")


def fit(data, info, device):
    implementation = fingerprint([train.source_hash(), sha256(Path(__file__)),
                                  sha256(Path(__file__).with_name("phase2.py")), DIGEST])
    model_set = sha256(MODEL_SET)
    SLOT.mkdir(parents=True, exist_ok=True)
    with FileLock(SLOT / "queue.lock", timeout=0):
        done = completed(SLOT, MODEL, implementation, model_set)
        if not done:
            checkpoint = recover_source(SLOT, info, device)
            if checkpoint and checkpoint.name == "best.pt":
                finish_validation(checkpoint, data, info, device)
            else:
                options = ["--model", MODEL, "--model-set", str(MODEL_SET)]
                for name, value in SETTINGS.items():
                    options.extend(["--" + name, str(value)])
                if checkpoint:
                    options = ["--resume", str(checkpoint)]
                original_build, original_hash = models.build_model, train.source_hash

                def build(*args, **kwargs):
                    model, transfer = original_build(*args, **kwargs)
                    transfer["vit"] = load_encoder(model.model.model[10].encoder)
                    return model, transfer

                # Scope the initialization to this CLI. The original trainer owns fitting and validation.
                try:
                    models.build_model = build
                    train.source_hash = lambda: implementation
                    train.main(["--data", str(data), "--device", device, "--run-dir", str(next_attempt(SLOT)), *options])
                finally:
                    models.build_model, train.source_hash = original_build, original_hash
            done = completed(SLOT, MODEL, implementation, model_set)
        if not done or done[1].get("vit_sha256") != DIGEST:
            raise RuntimeError("缺少本次 ImageNet 初始化对应的完整验证记录")
        if done[1].get("resume"):
            print("本次有断点续训，原第五组连续完成。比较时请一并记录中断轮次：",
                  done[1]["resume"]["interruption_epochs"])
        show_result(done[0], done[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--data", type=Path, default=ROOT / "configs/study-data.yaml")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    os.environ["YOLO_DISABLE_CUDNN"] = os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    bootstrap()
    data = args.data.resolve()
    info = inspect_data(data)
    config = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8-sig"))
    architecture = yaml.safe_load(MODEL_SET.read_text(encoding="utf-8"))[MODEL]
    if fingerprint(architecture) != ARCHITECTURE:
        raise ValueError("模型定义与原第五组结构不一致")
    if info["sha256"] != DATA or sha256(WEIGHTS) != COCO or training_protocol(dict(config, **SETTINGS)) != PROTOCOL:
        raise ValueError("数据、COCO 权重或训练协议与原第五组不一致")
    if args.action == "prepare":
        torch.set_num_threads(2)
        prepare(architecture)
    else:
        fit(data, info, args.device)


if __name__ == "__main__":
    main()
