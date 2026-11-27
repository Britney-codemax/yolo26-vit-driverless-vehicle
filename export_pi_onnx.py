"""Export the two first-round Pi 5 detectors from their archived best checkpoints."""
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training" / "support"))
from fusion.models import load_model

CHECKPOINTS = {
    "p4_mamba": "seed0_d89b9792b891",
    "p4_mamba_p5_vit": "seed0_0732d8ab7490",
}


def main():
    output = ROOT / "pi-deploy"
    output.mkdir(exist_ok=True)
    for name, run in CHECKPOINTS.items():
        source = ROOT / "models" / name / run / "best.pt"
        # Ultralytics writes beside its input. Stage the copy on the output volume.
        with TemporaryDirectory(dir=output) as stage:
            checkpoint = Path(stage) / f"{name}.pt"
            shutil.copyfile(source, checkpoint)
            detector = load_model(checkpoint)
            previous = torch.backends.mha.get_fastpath_enabled()
            try:
                if name == "p4_mamba_p5_vit":
                    torch.backends.mha.set_fastpath_enabled(False)
                exported = Path(detector.export(
                    format="onnx", device="cpu", imgsz=640, batch=1, opset=17,
                    half=False, dynamic=False, simplify=False, nms=False,
                ))
            finally:
                torch.backends.mha.set_fastpath_enabled(previous)
            graph = onnx.load(exported)
            onnx.checker.check_model(graph)
            input_type = graph.graph.input[0].type.tensor_type
            assert input_type.elem_type == onnx.TensorProto.FLOAT
            assert [d.dim_value for d in input_type.shape.dim] == [1, 3, 640, 640]
            assert next(s.version for s in graph.opset_import if not s.domain) == 17
            assert all(not t.external_data for t in graph.graph.initializer)
            destination = exported.replace(output / f"{name}.onnx")
            print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
