"""CPU checks for bounded training; real GPU work has a separate explicit entry point."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fusion.runtime import ARTIFACTS, ROOT, DIAGNOSTICS, MODELS, bootstrap, environment, fingerprint, read_json, stamp, write_json

bootstrap()
import torch
torch.set_num_threads(2)
from fusion.diagnostics import DiagnosticTrainer
from fusion.engine import FusionTrainer
from fusion.models import build_model
from fusion.records import aggregate, collect
from test_yolo_data import yolo_fixture
import train


class TrainingDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = DIAGNOSTICS / ("cpu_acceptance_" + stamp())
        cls.root.mkdir(parents=True)
        cls.data = yolo_fixture(cls.root / "data", splits=("train", "val"))
        cls.weights = ARTIFACTS / "weights/yolo26s.pt"
        cls.vit = ARTIFACTS / "weights/vit_b_16_imagenet1k_v1.pth"

    def test_invalid_arguments_and_environment_before_bootstrap(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(SystemExit), patch("train.bootstrap") as boot:
                train.main(["--data", "unused.yaml", "--diagnose-batches", value])
            boot.assert_not_called()
        with patch.dict(os.environ, {}, clear=False), patch("train.run") as run, \
             patch("train.bootstrap", side_effect=lambda: self.assertEqual(os.environ["CUDA_LAUNCH_BLOCKING"], "1")):
            train.main(["--data", "unused.yaml", "--diagnose-batches", "2"])
            self.assertEqual(run.call_args.args[0].diagnose_batches, 2)

    def test_cudnn_backend_is_recorded_in_protocol(self):
        records = []
        for value in ("0", "1"):
            with patch.dict(os.environ, {"TORCH_CUDNN_V8_API_DISABLED": value}):
                record = environment("cpu")
            self.assertEqual(record["cudnn_v8_api_disabled"], value)
            records.append(record)
        self.assertNotEqual(fingerprint(records[0]), fingerprint(records[1]))

    def test_no_cudnn_bootstrap_and_record(self):
        original = torch.backends.cudnn.enabled
        try:
            for value, expected in (("0", True), ("1", False)):
                torch.backends.cudnn.enabled = True
                with patch.dict(os.environ, {"YOLO_DISABLE_CUDNN": value}):
                    bootstrap()
                    self.assertEqual(torch.backends.cudnn.enabled, expected)
                    self.assertEqual(environment("cpu")["cudnn_enabled"], expected)
        finally:
            torch.backends.cudnn.enabled = original

    def test_deterministic_config_survives_initial_build_and_rebuild(self):
        for value in (False, True):
            wrapper, _ = build_model("p5_vit_small", seed=7, check=True, deterministic=value)
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), value)
            trainer = FusionTrainer.__new__(FusionTrainer)
            trainer.args = SimpleNamespace(deterministic=value)
            trainer.data = {"nc": 2}
            trainer.save_dir = self.root / f"rebuild_{value}"
            cfg = deepcopy(wrapper.model.yaml)
            cfg["fusion_deterministic"] = not value
            model = trainer.get_model(cfg, wrapper.model)
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), value)
            self.assertEqual(torch.backends.cudnn.deterministic, value)
            self.assertEqual(model.yaml["fusion_deterministic"], value)

    def test_nonfinite_loss_and_gradient_stop_before_update(self):
        trainer = DiagnosticTrainer.__new__(DiagnosticTrainer)
        trainer.model = torch.nn.Linear(2, 1)
        trainer.batch_limit = 1
        trainer.optimizer_steps = 0
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            trainer.check_loss(trainer.model, ({},), (torch.tensor(float("nan")), torch.zeros(3)))
        for parameter in trainer.model.parameters():
            parameter.grad = torch.full_like(parameter, float("inf"))
        with patch.object(FusionTrainer, "optimizer_step") as update:
            with self.assertRaisesRegex(FloatingPointError, "gradient"):
                trainer.optimizer_step()
            update.assert_not_called()
        self.assertEqual(trainer.optimizer_steps, 0)

    def test_cpu_failure_snapshot_is_independent_and_never_reads_cuda(self):
        trainer = DiagnosticTrainer.__new__(DiagnosticTrainer)
        trainer.epoch = 2
        trainer.batch_in_epoch = 0
        trainer.batches_completed = 7
        trainer.optimizer_steps = 2
        batch = {"img": torch.zeros(1, 3, 32, 32, dtype=torch.uint8), "cls": torch.empty(0, 1),
                 "bboxes": torch.empty(0, 4), "batch_idx": torch.empty(0), "im_file": ["negative.png"]}
        with patch.object(FusionTrainer, "preprocess_batch", side_effect=RuntimeError("simulated CUDA failure")):
            with self.assertRaises(RuntimeError):
                trainer.preprocess_batch(batch)
        batch["img"].fill_(255)
        directory = self.root / "failure_snapshot"
        directory.mkdir()
        with patch("torch.cuda.synchronize", side_effect=AssertionError("CUDA touched")), \
             patch("torch.Tensor.cpu", side_effect=AssertionError("GPU transfer attempted")):
            trainer.save_failure(directory)
        saved = torch.load(directory / "failure_batch.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["epoch"], 3)
        self.assertEqual(saved["batch_number"], 8)
        self.assertEqual(saved["batch"]["img"].sum().item(), 0)
        self.assertEqual(saved["batch"]["im_file"], ["negative.png"])

    def test_real_cpu_updates_exact_limit_and_report_isolation(self):
        if not self.weights.is_file() or not self.vit.is_file():
            self.skipTest("Official local weights required for real CPU integration")
        configs = {value: self.root / f"config_{value}.yaml" for value in (False, True)}
        for value, path in configs.items():
            path.write_text(f"deterministic: {str(value).lower()}\n", encoding="utf-8")
        for index, variant in enumerate(MODELS):
            value = bool(index % 2)
            directory = self.root / variant
            args = SimpleNamespace(data=str(self.data), config=str(configs[value]), model=variant,
                weights=str(self.weights), vit_weights=str(self.vit), check=False, run_dir=str(directory),
                epochs=3, batch=2, seed=0, imgsz=64, patience=0, workers=0, device="cpu", diagnose_batches=1)
            with self.subTest(model=variant), patch("fusion.records.refresh_safely") as refresh, \
                 patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")), \
                 patch("aiohttp.ClientSession._request", side_effect=AssertionError("Network forbidden")):
                train.run(args)
                refresh.assert_not_called()
            record = read_json(directory / "run.json")
            self.assertEqual(record["status"], "diagnostic_passed")
            self.assertEqual(record["batches_completed"], 1)
            self.assertGreater(record["optimizer_steps"], 0)
            self.assertEqual(record["determinism"]["enabled"], value)
            self.assertFalse((directory / "weights/best.pt").exists())
            self.assertFalse((directory / "metrics.json").exists())
        self.assertEqual(aggregate(collect(self.root)), [])
        write_json(self.root / "acceptance.json", {"device": "cpu", "real_optimizer_updates": True,
                   "models": {m: read_json(self.root / m / "run.json")["status"] for m in MODELS},
                   "cuda_stability": "not_tested"})

    def test_diagnostic_crosses_epoch_and_failure_saves_batch(self):
        if not self.weights.is_file():
            self.skipTest("Official local weights required")
        args = SimpleNamespace(data=str(self.data), config=str(ROOT / "configs/train.yaml"), model="baseline",
            weights=str(self.weights), vit_weights=str(self.vit), check=False,
            run_dir=str(self.root / "cross_epoch"), epochs=3, batch=2, seed=0, imgsz=64,
            patience=0, workers=0, device="cpu", diagnose_batches=2)
        with patch("fusion.records.refresh_safely") as refresh:
            train.run(args)
            refresh.assert_not_called()
        record = read_json(Path(args.run_dir) / "run.json")
        self.assertEqual(record["status"], "diagnostic_passed")
        self.assertEqual(record["batches_completed"], 2)
        self.assertEqual(record["completed_epochs"], 1)
        args.run_dir = str(self.root / "injected_loss_failure")
        with patch.object(DiagnosticTrainer, "check_loss", side_effect=FloatingPointError("injected NaN")), \
             patch("fusion.records.refresh_safely") as refresh:
            with self.assertRaises(FloatingPointError):
                train.run(args)
            refresh.assert_not_called()
        record = read_json(Path(args.run_dir) / "run.json")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["optimizer_steps"], 0)
        self.assertTrue((Path(args.run_dir) / "failure_batch.pt").is_file())
        args.run_dir = str(self.root / "insufficient_batches")
        args.epochs = 1
        with self.assertRaisesRegex(RuntimeError, "before the requested"):
            train.run(args)
        self.assertEqual(read_json(Path(args.run_dir) / "run.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
