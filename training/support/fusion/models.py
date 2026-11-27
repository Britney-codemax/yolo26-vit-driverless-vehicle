"""Native YOLO26n/s with serializable residual branches at P4 and P5."""
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import yaml
from torchvision.models.vision_transformer import EncoderBlock
from ultralytics import YOLO
from ultralytics.nn.modules import C2PSA
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.torch_utils import init_seeds
from .runtime import ROOT, MODELS, sha256
from .mamba import ResidualSS2D

SPECS = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))


class P5ResidualViT(nn.Module):
    def __init__(self, base, dim, heads):
        super().__init__()
        self.base = base
        channels = base.cv2.conv.out_channels
        self.input_projection = nn.Conv2d(channels, dim, 1)
        self.encoder = EncoderBlock(heads, dim, dim * 4, dropout=0.0, attention_dropout=0.0)
        self.output_projection = nn.Conv2d(dim, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.dim = dim
        # Ultralytics routes layers by these attributes, including the P5 skip connection.
        self.i, self.f, self.type = base.i, base.f, "P5ResidualViT"
        self.np = sum(p.numel() for p in self.parameters())

    def position(self, h, w, device, dtype):
        freq = 10000.0 ** (-torch.arange(self.dim // 4, device=device).float() / (self.dim // 4))
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        x, y = x.flatten()[:, None] * freq, y.flatten()[:, None] * freq
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1).to(dtype).unsqueeze(0)

    def forward(self, x):
        x = self.base(x)
        features = self.input_projection(x)
        n, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + self.position(h, w, tokens.device, tokens.dtype))
        branch = self.output_projection(tokens.transpose(1, 2).reshape(n, c, h, w))
        return x + self.alpha * branch


class FusionDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26s.yaml", ch=3, nc=None, verbose=False):
        cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        variant = cfg.get("fusion_variant", "baseline")
        if variant not in MODELS:
            raise ValueError(f"Unknown fusion variant: {variant}")
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        expected_hidden = {"n": 128, "s": 256}.get(cfg.get("scale", "s"))
        if not isinstance(self.model[10], C2PSA) or self.model[10].c != expected_hidden:
            raise ValueError("Layer 10 does not match the requested YOLO26n/s architecture")
        if not self.end2end or self.model[-1].reg_max != 1:
            raise ValueError("Expected YOLO26 end-to-end detection")
        self.variant = variant
        spec = cfg.get("fusion_spec", SPECS[variant])
        channels = {i: self.model[i].cv2.conv.out_channels for i in (10, 13)}
        if spec.get("dim"):
            self.model[10] = P5ResidualViT(self.model[10], spec["dim"], spec["heads"])
        if "scan_layer" in spec:
            index = spec["scan_layer"]
            self.model[index] = ResidualSS2D(self.model[index], channels[index],
                                             spec["width"], spec["state_size"])
        self.yaml["fusion_variant"] = variant
        self.transfer = None

    def load(self, weights, verbose=True):
        source = weights["model"] if isinstance(weights, dict) else weights
        state = source.float().state_dict()
        target = self.state_dict()
        wrappers = (P5ResidualViT, ResidualSS2D)
        wrapped_source = tuple(f"model.{i}." for i in (10, 13) if isinstance(source.model[i], wrappers))
        if wrapped_source and getattr(source, "variant", None) != self.variant:
            raise ValueError("Checkpoint variant differs from the requested fusion model")
        loaded, unmatched, used, mapping = {}, {}, set(), {}
        for key, value in target.items():
            source_key = key
            if not wrapped_source and key.startswith(("model.10.base.", "model.13.base.")):
                source_key = key.replace(".base.", ".", 1)
            if source_key in state and value.shape == state[source_key].shape:
                loaded[key] = state[source_key]
                used.add(source_key)
                mapping[key] = source_key
            else:
                unmatched[key] = {"target": list(value.shape),
                                  "source": list(state[source_key].shape) if source_key in state else None}
        self.load_state_dict(loaded, strict=False)
        self.transfer = {"loaded": mapping, "unmatched": unmatched, "unused_source": sorted(set(state) - used),
                         "matched_tensors": len(loaded), "target_tensors": len(target)}
        if wrapped_source:
            for key in target:
                if key.startswith(wrapped_source) and ".base." not in key:
                    if key not in loaded or not torch.equal(target[key], state[key]):
                        raise RuntimeError(f"Fusion parameters lost during rebuild: {key}")
        return self


def load_vit_block(detector, path):
    from torchvision.models import ViT_B_16_Weights
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("ViT weights missing; run train.py --download-weights vit explicitly")
    digest = sha256(path)
    expected_prefix = Path(ViT_B_16_Weights.IMAGENET1K_V1.url).stem.rsplit("-", 1)[1]
    if not digest.startswith(expected_prefix):
        raise ValueError("ViT checkpoint hash does not match IMAGENET1K_V1; use the explicit download command")
    state = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "encoder.layers.encoder_layer_0."
    block_state = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    detector.model[10].encoder.load_state_dict(block_state, strict=True)
    return {"source": "torchvision ViT_B_16_Weights.IMAGENET1K_V1", "sha256": digest,
            "block": "encoder.layers.encoder_layer_0", "loaded": sorted(prefix + k for k in block_state),
            "excluded": "other encoder blocks, patch embedding, position embedding, class token, classifier"}


def make_detector(cfg, nc=None, weights=None):
    init_seeds(int(cfg.get("fusion_seed", 0)), deterministic=cfg.get("fusion_deterministic", True))
    model = FusionDetectionModel(cfg, nc=nc)
    if weights is not None:
        model.load(weights, verbose=False)
    return model


def build_model(variant, seed=0, yolo_weights=None, vit_weights=None, check=False, deterministic=True,
                architecture=None):
    if variant not in MODELS:
        raise ValueError(variant)
    cfg = deepcopy(architecture) if architecture is not None else yaml_model_load("yolo26s.yaml")
    if architecture is not None and cfg.get("fusion_variant") != variant:
        raise ValueError("Notebook architecture and requested model differ")
    cfg.update(fusion_variant=variant, fusion_seed=seed, fusion_deterministic=deterministic)
    # Constructing from YAML never downloads weights.
    scale = cfg.get("scale", "s")
    wrapper = YOLO(f"yolo26{scale}.yaml", task="detect", verbose=False)
    wrapper.model = make_detector(cfg)
    reports = {}
    if not check:
        path = Path(yolo_weights).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YOLO26{scale} weights missing: {path}")
        source = YOLO(str(path), task="detect", verbose=False)
        if (source.model.yaml.get("scale") != scale or not isinstance(source.model.model[10], C2PSA)
                or source.model.model[-1].nc != 80):
            raise ValueError(f"Initialization requires the original 80-class YOLO26{scale} COCO checkpoint")
        if not source.model.end2end or source.model.model[-1].reg_max != 1:
            raise ValueError("Initialization checkpoint is not YOLO26")
        wrapper.model.load(source.model)
        reports["yolo"] = {"sha256": sha256(path), **wrapper.model.transfer}
        if SPECS[variant]["pretrained"]:
            reports["vit"] = load_vit_block(wrapper.model, vit_weights)
        # Model.train passes the initialized model to Trainer.get_model only with a checkpoint.
        wrapper.ckpt = {"model": wrapper.model}
    return wrapper, reports


def load_model(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return YOLO(str(path), task="detect", verbose=False)
