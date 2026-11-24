"""Validation failures must retain the actual CPU input and original exception."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fusion.runtime import ARTIFACTS, ROOT, bootstrap, check_isolated_environment, check_package_versions, read_json, stamp

bootstrap()
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.ops import Profile
from fusion.engine import CompleteValidator, FusionTrainer
from fusion.diagnostics import DiagnosticTrainer
from fusion.models import FusionDetectionModel
from test_yolo_data import yolo_fixture
import train


class ValidationEvidence(unittest.TestCase):
    def test_runtime_accepts_only_matching_version_pairs(self):
        for pair in (("2.13.0+cu126", "0.28.0+cu126"), ("2.12.1+cu126", "0.27.1+cu126")):
            versions = dict(zip(("ultralytics", "torch", "torchvision"), ("8.4.14", *pair)))
            with patch("fusion.runtime.importlib.metadata.version", side_effect=versions.__getitem__):
                check_package_versions()
        versions["torchvision"] = "0.28.0+cu126"
        with patch("fusion.runtime.importlib.metadata.version", side_effect=versions.__getitem__):
            with self.assertRaisesRegex(RuntimeError, "matching pair"):
                check_package_versions()

    def test_import_isolation_rejects_user_site_and_external_torch(self):
        check_isolated_environment()
        with patch("site.ENABLE_USER_SITE", True):
            with self.assertRaisesRegex(RuntimeError, "site packages disabled"):
                check_isolated_environment()
        with patch("torch.__file__", str(ROOT / "outside_environment/torch/__init__.py")):
            with self.assertRaisesRegex(RuntimeError, "outside this environment"):
                check_isolated_environment()

    def test_validation_loader_uses_training_batch(self):
        trainer = FusionTrainer.__new__(FusionTrainer)
        trainer.args = SimpleNamespace(batch=4)
        trainer.data = {}
        with patch.object(DetectionTrainer, "get_dataloader", return_value=object()) as build:
            trainer.get_dataloader("images", batch_size=8, mode="val")
            self.assertEqual(build.call_args.args[1], 4)
            trainer.get_dataloader("images", batch_size=4, mode="train")
            self.assertEqual(build.call_args.args[1], 4)

    def test_formal_validation_failure_preserves_first_error_and_cpu_batch(self):
        torch.set_num_threads(2)
        root = ARTIFACTS / "checks" / ("validation_failure_" + stamp())
        data = yolo_fixture(root / "data", splits=("train", "val"))
        weights = ARTIFACTS / "weights/yolo26s.pt"
        if not weights.is_file():
            self.skipTest("Local official weights required")
        args = SimpleNamespace(data=str(data), config=str(ROOT / "configs/train.yaml"), model="baseline",
            weights=str(weights), vit_weights="unused", check=False, run_dir=str(root / "run"),
            epochs=1, batch=2, seed=0, imgsz=64, patience=0, workers=0, device="cpu")
        original_predict, original_exit = FusionDetectionModel.predict, Profile.__exit__

        def predict(model, *a, **kw):
            if not model.training:
                raise RuntimeError("injected validation forward failure")
            return original_predict(model, *a, **kw)

        def exit_profile(profile, kind, error, tb):
            if error is not None:
                raise RuntimeError("secondary synchronize failure")
            return original_exit(profile, kind, error, tb)

        with patch.object(FusionDetectionModel, "predict", predict), patch.object(Profile, "__exit__", exit_profile), \
             patch.object(DiagnosticTrainer, "check_gradients", side_effect=AssertionError("Diagnostic checks in formal run")), \
             patch.object(DiagnosticTrainer, "check_loss", side_effect=AssertionError("Diagnostic checks in formal run")), \
             patch("fusion.records.refresh_safely"):
            with self.assertRaisesRegex(RuntimeError, "injected validation forward failure"):
                train.run(args)
        run = Path(args.run_dir)
        evidence = read_json(run / "failure_batch.json")
        self.assertEqual(evidence["phase"], "validation_forward")
        self.assertEqual(evidence["epoch"], 1)
        self.assertEqual(evidence["batch_in_epoch"], 1)
        self.assertEqual(evidence["input_dtype"], "torch.float32")
        self.assertEqual(evidence["input_shape"][:2], [2, 3])
        self.assertIn("injected validation forward failure", evidence["first_error"])
        saved = torch.load(run / "failure_batch.pt", map_location="cpu", weights_only=False)
        self.assertEqual(evidence["input_shape"], list(saved["batch"]["img"].shape))
        self.assertEqual(saved["batch"]["img"].dtype, torch.uint8)
        self.assertTrue(all("val" in p for p in saved["batch"]["im_file"]))
        self.assertTrue(all(v.device.type == "cpu" for v in saved["batch"].values() if isinstance(v, torch.Tensor)))
        self.assertIn("secondary synchronize failure", (run / "failure_traceback.txt").read_text())
        self.assertEqual(read_json(run / "run.json")["status"], "failed")

    def test_failure_writer_never_reads_cuda(self):
        validator = CompleteValidator.__new__(CompleteValidator)
        validator.save_dir = ARTIFACTS / "checks" / ("cpu_evidence_" + stamp())
        validator.last_batch = {"epoch": 3, "batch_in_epoch": 2,
                                "batch": {"img": torch.zeros(1, 3, 32, 32), "im_file": ["val.png"]}}
        validator.phase = "validation_forward"
        validator.input_details = {"input_shape": [1, 3, 32, 32], "input_dtype": "torch.float32"}
        with patch("torch.cuda.synchronize", side_effect=AssertionError("CUDA touched")), \
             patch("torch.cuda.memory_allocated", side_effect=AssertionError("CUDA touched")), \
             patch("torch.Tensor.cpu", side_effect=AssertionError("GPU transfer attempted")):
            validator.save_failure(RuntimeError("first"), "trace")
        self.assertEqual(read_json(validator.save_dir / "failure_batch.json")["first_error"], "first")

    def test_formal_training_failure_saves_augmented_cpu_input(self):
        torch.set_num_threads(2)
        root = ARTIFACTS / "checks" / ("training_failure_" + stamp())
        data = yolo_fixture(root / "data", splits=("train", "val"))
        args = SimpleNamespace(data=str(data), config=str(ROOT / "configs/train.yaml"), model="baseline",
            weights=str(ARTIFACTS / "weights/yolo26s.pt"), vit_weights="unused", check=False,
            run_dir=str(root / "run"), epochs=1, batch=2, seed=0, imgsz=64, patience=0, workers=0, device="cpu")
        original = FusionDetectionModel.predict

        def predict(model, *a, **kw):
            if model.training and hasattr(model, "criterion"):
                raise RuntimeError("injected formal training failure")
            return original(model, *a, **kw)

        with patch.object(FusionDetectionModel, "predict", predict), patch("fusion.records.refresh_safely"):
            with self.assertRaisesRegex(RuntimeError, "injected formal training failure"):
                train.run(args)
        run = Path(args.run_dir)
        evidence = read_json(run / "failure_batch.json")
        self.assertEqual((evidence["epoch"], evidence["batch_in_epoch"]), (1, 1))
        self.assertEqual(evidence["phase"], "training_step")
        saved = torch.load(run / "failure_batch.pt", map_location="cpu", weights_only=False)
        self.assertEqual(saved["batch"]["img"].dtype, torch.uint8)
        self.assertTrue(all("train" in p for p in saved["batch"]["im_file"]))
        self.assertEqual(read_json(run / "run.json")["status"], "failed")

    def test_diagnostic_does_not_replace_saved_validation_input(self):
        trainer = DiagnosticTrainer.__new__(DiagnosticTrainer)
        trainer.validator = SimpleNamespace(failure_saved=True)
        trainer.last_batch = {"batch": {"img": torch.zeros(1, 3, 32, 32), "im_file": ["train.png"]}}
        with patch("torch.save") as save:
            trainer.save_failure(ROOT / "must_not_write")
            save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
