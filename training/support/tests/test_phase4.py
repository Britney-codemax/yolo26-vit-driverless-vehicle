"""Exercise native checkpoint recovery and completed-attempt discovery for edge adapters."""
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import yaml

from models import phase4 as p
from fusion.runtime import read_json, sha256, write_json
from fusion.diagnostics import DiagnosticTrainer
from test_yolo_data import yolo_fixture


class RoundFour(unittest.TestCase):
    def test_resume_serialized_edges(self):
        for module, name in ((p.train, "MODELS"), (p.legacy, "MODELS"),
                             (p.legacy, "make_detector"), (p.engine, "make_detector")):
            self.addCleanup(setattr, module, name, getattr(module, name))
        self.enterContext(patch.dict(p.legacy.SPECS))
        p.register()
        p.train.bootstrap()
        torch.set_num_threads(2)
        with TemporaryDirectory(dir=p.ARTIFACTS) as temporary:
            root = Path(temporary)
            data = yolo_fixture(root / "data", splits=("train", "val"))
            slot = root / "F03"
            first, second = slot / "attempt_01", slot / "attempt_02"
            preprocess = DiagnosticTrainer.preprocess_batch

            def interrupt(trainer, batch):
                if trainer.epoch == 1:
                    raise KeyboardInterrupt()
                return preprocess(trainer, batch)

            with patch("fusion.records.refresh_safely"), patch.object(p.engine, "convert_data",
                    partial(p.convert_data, cache_root=root / "cache")):
                with patch.object(DiagnosticTrainer, "preprocess_batch", interrupt), self.assertRaises(KeyboardInterrupt):
                    p.train.main(["--model", "F03", "--data", str(data), "--model-set", str(p.CONFIG),
                                  "--weights", str(p.WEIGHTS), "--epochs", "3", "--batch", "2", "--imgsz", "64",
                                  "--device", "cpu", "--run-dir", str(first)])
                checkpoint = p.recover_source(slot, p.inspect_data(data), "cpu")
                self.assertEqual(checkpoint, first / "weights/last.pt")
                original = sha256(checkpoint)
                p.train.main(["--data", str(data), "--resume", str(checkpoint), "--device", "cpu",
                              "--run-dir", str(second)])
            row = read_json(second / "run.json")
            self.assertEqual((row["completed_epochs"], row["status"]), (3, "completed"))
            self.assertEqual(row["resume"]["interruption_epochs"], [1])
            self.assertEqual(original, sha256(checkpoint))
            trained = p.legacy.load_model(second / "weights/best.pt").model
            self.assertEqual(set(trained.model[23].input_branches), {"1", "2"})
            self.assertEqual(read_json(second / "metrics.json")["checkpoint_sha256"],
                             sha256(second / "weights/best.pt"))

    def test_summary_finds_finished_attempt_and_rejects_changed_best(self):
        with TemporaryDirectory() as temporary, patch.object(p, "OUTPUT", Path(temporary)):
            key = "F06"
            slot = p.OUTPUT / key
            self.assertEqual(p.latest_result(key), ("待训练", None, None))
            done = slot / "attempt_02"
            for suffix in ("01", "03"):
                write_json(slot / f"attempt_{suffix}/run.json", {"status": "interrupted"})
            cfg = yaml.safe_load((p.ROOT / "configs/train.yaml").read_text(encoding="utf-8-sig"))
            write_json(done / "run.json", dict(model=key, seed=0, completed_epochs=100, status="completed",
                       data={"sha256": p.DATA}, yolo_sha256=p.COCO, effective_training=dict(cfg, **p.SETTINGS),
                       model_config=p.definitions()[key]))
            (done / "results.csv").write_text("epoch\n" + "".join(f"{i}\n" for i in range(1, 101)))
            checkpoint = done / "weights/best.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"test best")
            write_json(done / "metrics.json", dict(data_sha256=p.DATA, split="val", checkpoint="weights/best.pt",
                       checkpoint_sha256=sha256(checkpoint), settings=p.evaluation_settings(640, 4, "0")))
            self.assertEqual(p.latest_result(key)[:2], ("已完成", done))
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                p.latest_result(key)


if __name__ == "__main__":
    unittest.main()
