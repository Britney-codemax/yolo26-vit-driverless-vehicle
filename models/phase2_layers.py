"""Round-two routing on the native YOLO26s graph. Layer numbers stay unchanged."""
from copy import deepcopy
from pathlib import Path
import sys

from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training/support"))
from fusion import models as legacy
from fusion.mamba import ResidualSS2D
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds

# (Mamba layer, recipient of fused features, ViT layer). None means no ViT.
SPECS = {
    "N01": (6, "both", None), "N02": (6, "both", 10), "N03": (6, "both", 22),
    "N04": (6, "skip", None), "N05": (6, "skip", 10), "N06": (6, "skip", 22),
    "N07": (13, "both", 22),
    "N08": (13, "skip", None), "N09": (13, "skip", 10), "N10": (13, "skip", 22),
    "N11": (13, "main", None), "N12": (13, "main", 10),
}
EDGES = {6: (7, 12), 13: (14, 18)}
GROUPS = {"A": ["N07", "N10", "N01", "N05"], "B": ["N08", "N11", "N02", "N06"],
          "C": ["N09", "N12", "N03", "N04"]}
NATIVE_FACTORY = legacy.make_detector


class EdgeScan(nn.Module):
    """Transform one consumer's input; the shared producer still returns its raw output."""
    def __init__(self, consumer, skip):
        super().__init__()
        self.base = consumer
        self.skip_input = skip
        identity = nn.Identity()
        identity.i, identity.f = consumer.i, -1
        self.scan = ResidualSS2D(identity, 256, 64, 8)
        self.i, self.f, self.type = consumer.i, consumer.f, type(self).__name__
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if self.skip_input:
            return self.base([x[0], self.scan(x[1])])
        return self.base(self.scan(x))


class Phase2Detector(DetectionModel):
    def __init__(self, cfg, nc=None):
        super().__init__(deepcopy(cfg), nc=nc, verbose=False)
        self.variant = cfg["fusion_variant"]
        layer, route, vit = SPECS[self.variant]
        spec = cfg["fusion_spec"]
        if spec != fusion_spec(self.variant):
            raise ValueError(f"{self.variant}: architecture does not match the exported design")
        main, skip = EDGES[layer]
        if self.model[main].f != -1 or self.model[skip].f != [-1, layer]:
            raise ValueError("Unexpected YOLO graph: check the two consumers before attaching SS2D")
        # Preserve the first round's seeded branch initialization order (ViT, then SS2D).
        if vit is not None:
            self.model[vit] = legacy.P5ResidualViT(self.model[vit], 256, 4)
        if route == "both":
            scan_index = layer
            self.model[layer] = ResidualSS2D(self.model[layer], 256, 64, 8)
        else:
            scan_index = skip if route == "skip" else main
            self.model[scan_index] = EdgeScan(self.model[scan_index], route == "skip")
        self.wrapped_indices = (scan_index, vit)

    def load(self, weights, verbose=True):
        donor = weights["model"] if isinstance(weights, dict) else weights
        restoring = isinstance(donor, Phase2Detector)
        if restoring and donor.variant != self.variant:
            raise ValueError(f"Cannot rebuild {self.variant} from {donor.variant}")
        state = donor.float().state_dict()
        target = self.state_dict()
        mapped, tensors, unmatched = {}, {}, {}
        for key, tensor in target.items():
            prefix, index, rest = key.split(".", 2)
            wrapped = int(index) in self.wrapped_indices
            native = not wrapped or rest.startswith("base.")
            source = key
            if wrapped and native and not restoring:
                source = f"{prefix}.{index}.{rest.removeprefix('base.')}"
            if (restoring or native) and source in state and state[source].shape == tensor.shape:
                tensors[key], mapped[key] = state[source], source
            else:
                unmatched[key] = {"target": list(tensor.shape),
                                  "source": list(state[source].shape) if source in state else None}
                # Only the class head is allowed to differ on native 80 -> dataset-nc rebuilds.
                if (restoring or native) and int(index) != len(self.model) - 1:
                    raise ValueError(f"Unexpected missing or incompatible weight: {key}")
        self.load_state_dict(tensors, strict=False)
        self.transfer = {"loaded": mapped, "unmatched": unmatched,
                         "unused_source": sorted(set(state) - set(mapped.values())),
                         "matched_tensors": len(mapped), "target_tensors": len(target)}
        return self


def fusion_spec(model):
    layer, route, vit = SPECS[model]
    return dict(scan_layer=layer, route=route, width=64, state_size=8,
                vit_layer=vit, dim=256 if vit else 0, heads=4 if vit else 0, pretrained=False)


def make_detector(cfg, nc=None, weights=None):
    if cfg.get("fusion_variant") not in SPECS:
        return NATIVE_FACTORY(cfg, nc=nc, weights=weights)
    init_seeds(int(cfg.get("fusion_seed", 0)), deterministic=cfg.get("fusion_deterministic", True))
    model = Phase2Detector(cfg, nc)
    if weights is not None:
        model.load(weights)
    return model


def register():
    """The CLI parser, initial build and Trainer rebuild must agree inside this process."""
    import train
    from fusion import engine
    choices = tuple(dict.fromkeys((*legacy.MODELS, *SPECS)))
    train.MODELS = legacy.MODELS = choices
    legacy.SPECS.update({name: fusion_spec(name) for name in SPECS})
    legacy.make_detector = engine.make_detector = make_detector
