"""Resume the saved optimizer and epoch into a new directory, keeping the source intact."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fusion.runtime import ARTIFACTS, ROOT, bootstrap, read_json, sha256, stamp
bootstrap()
import torch
from fusion.diagnostics import DiagnosticTrainer
from fusion.models import load_model
from test_yolo_data import yolo_fixture
import train


class Resume(unittest.TestCase):
    variant = "baseline"

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = ARTIFACTS / "checks" / ("resume_test_" + stamp())
        cls.data = yolo_fixture(cls.root / "data", splits=("train", "val"))
        cls.args = SimpleNamespace(data=str(cls.data), config=str(ROOT / "configs/train.yaml"),
            model=cls.variant, weights=str(ARTIFACTS / "weights/yolo26s.pt"),
            vit_weights=str(ARTIFACTS / "weights/vit_b_16_imagenet1k_v1.pth"),
            check=False, run_dir=str(cls.root / "source"), epochs=3, batch=2, seed=0, imgsz=64,
            patience=0, workers=0, device="cpu")
        original = DiagnosticTrainer.preprocess_batch

        def interrupt(trainer, batch):
            if trainer.epoch == 1:
                raise KeyboardInterrupt("test interruption after one saved epoch")
            return original(trainer, batch)

        with patch.object(DiagnosticTrainer, "preprocess_batch", interrupt), patch("fusion.records.refresh_safely"):
            with unittest.TestCase().assertRaises(KeyboardInterrupt):
                train.run(cls.args)
        cls.source = Path(cls.args.run_dir)
        cls.checkpoint = cls.source / "weights/last.pt"
        cls.hashes = {p: sha256(p) for p in cls.source.rglob("*") if p.is_file()}

    def resume_args(self, name):
        args = deepcopy(self.args)
        args.resume = str(self.checkpoint)
        args.run_dir = str(self.root / name)
        args.model = None
        for key in ("epochs", "batch", "seed", "imgsz", "patience", "workers", "device"):
            setattr(args, key, None)
        return args

    def test_resume_restores_state_and_finishes_without_changing_source(self):
        args = self.resume_args("resumed")
        saved = load_model(self.checkpoint).ckpt
        original_start = DiagnosticTrainer.start_diagnostic
        observed = []

        def inspect(trainer, _):
            self.assertEqual(trainer.start_epoch, 1)
            self.assertEqual(trainer.epochs, 3)
            self.assertEqual(trainer.scheduler.last_epoch, 0)
            self.assertEqual(trainer.save_dir, Path(args.run_dir))
            self.assertEqual(sha256(trainer.best), self.hashes[self.source / "weights/best.pt"])
            actual = trainer.optimizer.state_dict()
            self.assertEqual(actual["param_groups"], saved["optimizer"]["param_groups"])
            for key, values in saved["optimizer"]["state"].items():
                for name, expected in values.items():
                    torch.testing.assert_close(actual["state"][key][name], expected, check_dtype=False, rtol=0, atol=0)
            self.assertEqual(trainer.ema.updates, saved["updates"])
            torch.testing.assert_close(trainer.model.state_dict(), saved["ema"].float().state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(trainer.ema.ema.state_dict(), saved["ema"].state_dict(), rtol=0, atol=0)
            observed.append(trainer.start_epoch)
            original_start(trainer, trainer)

        with patch.object(DiagnosticTrainer, "start_diagnostic", inspect), patch("fusion.records.refresh_safely"):
            train.run(args)
        self.assertEqual(observed, [1])
        result = read_json(Path(args.run_dir) / "run.json")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_epochs"], 3)
        self.assertEqual(result["resume"]["interruption_epochs"], [1])
        self.assertNotEqual(result["protocol_key"], read_json(self.source / "run.json")["protocol_key"])
        self.assertTrue((Path(args.run_dir) / "metrics.json").is_file())
        self.assertEqual(len((Path(args.run_dir) / "results.csv").read_text().splitlines()), 4)
        self.assertTrue(all(sha256(path) == digest for path, digest in self.hashes.items()))

    def test_resume_rejects_changed_protocol_before_creating_output(self):
        wrong_model = "baseline" if self.variant != "baseline" else "p5_vit_small"
        for key, value in (("epochs", 100), ("batch", 4), ("model", wrong_model)):
            args = self.resume_args("invalid_" + key)
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Resume"):
                train.run(args)
            self.assertFalse(Path(args.run_dir).exists())
        args = self.resume_args("wrong_data")
        args.data = str(yolo_fixture(self.root / "other_data", splits=("train", "val")))
        label = Path(args.data).parent / "train/labels/image 0.txt"
        label.write_text("0 .2 .5 .1 .2\n")
        with self.assertRaisesRegex(ValueError, "Resume.*data"):
            train.run(args)
        self.assertFalse(Path(args.run_dir).exists())
        args = self.resume_args("wrong_environment")
        with patch("train.environment", return_value={"different": True}):
            with self.assertRaisesRegex(ValueError, "Resume.*environment"):
                train.run(args)
        self.assertFalse(Path(args.run_dir).exists())


class FusionResume(Resume):
    variant = "p5_vit_small"


class RandomResume(Resume):
    variant = "p5_vit_random"


class PretrainedResume(Resume):
    variant = "p5_vit_pretrained"


if __name__ == "__main__":
    unittest.main()
