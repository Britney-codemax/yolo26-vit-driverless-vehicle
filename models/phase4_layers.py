"""YOLO26n edge adapters. The native 24 layers and checkpoint names stay at their original indices."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import EncoderBlock
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils.torch_utils import init_seeds

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training/support"))
from fusion import models as legacy
from fusion.mamba import ResidualSS2D

# Each tuple is (producer, consumer); None means that branch is absent.
M_END, M_SKIP = (22, 23), (10, 21)
V_HEAD, V_SKIP, V_OLD = (19, 23), (13, 18), (10, 10)
ROUTES = {
    "F01": (M_END, None), "F02": (None, V_HEAD), "F03": (M_END, V_HEAD),
    "F04": (M_SKIP, None), "F05": (None, V_SKIP), "F06": (M_SKIP, V_SKIP),
    "F07": (M_END, V_SKIP), "F08": (M_SKIP, V_HEAD), "F09": (None, V_OLD),
}
NATIVE = legacy.make_detector


def spec(name):
    mamba, vit = ROUTES[name]
    out = {"pretrained": False, "mamba_edge": list(mamba) if mamba else None,
           "vit_edge": list(vit) if vit else None}
    if vit == V_OLD:
        out.update(dim=128, heads=4)  # Same P5ResidualViT as round three, without the P4 scan.
    return out


def definitions():
    base = yaml_model_load("yolo26n.yaml")
    return {k: dict(base, fusion_variant=k, fusion_spec=spec(k)) for k in ROUTES}


class PooledViT(nn.Module):
    dim = 64
    position = legacy.P5ResidualViT.position

    def __init__(self, channels):
        super().__init__()
        self.input_projection = nn.Conv2d(channels, self.dim, 1)
        self.encoder = EncoderBlock(4, self.dim, 128, dropout=0., attention_dropout=0.)
        self.output_projection = nn.Conv2d(self.dim, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        # ceil_mode also supports the small/rectangular tensors used by validation and profiling.
        low = self.input_projection(F.avg_pool2d(x, kernel_size=4, stride=4, ceil_mode=True))
        b, c, h, w = low.shape
        tokens = low.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + self.position(h, w, tokens.device, tokens.dtype))
        delta = self.output_projection(tokens.transpose(1, 2).reshape(b, c, h, w))
        return x + self.alpha * F.interpolate(delta, size=x.shape[-2:], mode="bilinear", align_corners=False)


def apply_input_branches(consumer, args):
    # Copy the edge list, never alter a producer's saved feature or its other consumers.
    features = list(args[0])
    for pos, branch in consumer.input_branches.items():
        features[int(pos)] = branch(features[int(pos)])
    return (features,)


class Phase4Detector(legacy.FusionDetectionModel):
    def __init__(self, cfg, nc=None):
        super().__init__(cfg, nc=nc)
        if cfg["scale"] != "n" or cfg["fusion_spec"] != spec(self.variant):
            raise ValueError(f"{self.variant}: expected the round-four YOLO26n definition")
        mamba, vit = ROUTES[self.variant]
        for edge in (vit, mamba):
            if edge is None or edge == V_OLD:
                continue
            src, dst = edge
            consumer = self.model[dst]
            pos = consumer.f.index(src)
            channels = self.model[src].cv2.conv.out_channels
            # All matched single/dual comparisons start with the same branch tensors.
            with torch.random.fork_rng(devices=[]):
                if edge == mamba:
                    identity = nn.Identity()
                    identity.i, identity.f = src, -1
                    branch = ResidualSS2D(identity, channels, 32, 8)
                else:
                    branch = PooledViT(channels)
            if not hasattr(consumer, "input_branches"):
                consumer.input_branches = nn.ModuleDict()
                consumer.register_forward_pre_hook(apply_input_branches)
            consumer.input_branches[str(pos)] = branch

    def load(self, weights, verbose=True):
        source = weights["model"] if isinstance(weights, dict) else weights
        if isinstance(source, Phase4Detector) and source.variant != self.variant:
            raise ValueError(f"Cannot rebuild {self.variant} from {source.variant}")
        super().load(weights, verbose=verbose)
        head = self.model[-1].i
        for key in self.transfer["unmatched"]:
            if key.startswith((f"model.{head}.cv3.", f"model.{head}.one2one_cv3.")):
                continue
            if not isinstance(source, Phase4Detector) and (
                    ".input_branches." in key or (self.variant == "F09" and key.startswith("model.10.") and ".base." not in key)):
                continue
            raise ValueError(f"Unexpected unmatched tensor: {key}")
        return self


def make_detector(cfg, nc=None, weights=None):
    if cfg.get("fusion_variant") not in ROUTES:
        return NATIVE(cfg, nc=nc, weights=weights)
    init_seeds(int(cfg.get("fusion_seed", 0)), deterministic=cfg.get("fusion_deterministic", True))
    model = Phase4Detector(cfg, nc=nc)
    return model.load(weights, verbose=False) if weights is not None else model


def register():
    """Register only inside the phase-four process, including the trainer's two-class rebuild."""
    import train
    from fusion import engine
    train.MODELS = legacy.MODELS = tuple(dict.fromkeys((*legacy.MODELS, *ROUTES)))
    legacy.SPECS.update({k: spec(k) for k in ROUTES})
    legacy.make_detector = engine.make_detector = make_detector
