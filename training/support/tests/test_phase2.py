"""Exercise queue and ranking boundaries with temporary records, without training."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openpyxl import Workbook, load_workbook
import yaml

from models import phase2 as p, phase2_report as report
from fusion.runtime import fingerprint, read_json, sha256, write_json
import train


class Phase2(unittest.TestCase):
    def test_queues_resume_skip_and_review(self):
        study = {"test": True}
        params = yaml.safe_load((p.ROOT / "configs/train.yaml").read_text(encoding="utf-8-sig"))
        params.update(p.SETTINGS)
        info = {"sha256": p.DATA, "names": {0: "UGV", 1: "USV"}}
        calls = []
        def simulate(argv):
            dest = Path(argv[argv.index("--run-dir") + 1])
            key = dest.parent.name
            calls.append((key, "--resume" in argv))
            if "--resume" in argv:
                self.assertEqual(Path(argv[argv.index("--resume") + 1]), dest.parent / "attempt_01/weights/last.pt")
            else:
                self.assertEqual(argv[argv.index("--model") + 1], key)
                self.assertEqual(argv[argv.index("--epochs") + 1], "100")
            row = dict(model=key, seed=0, data=info, requested_training=params, effective_training=params,
                       yolo_sha256=p.COCO, code_sha256=train.source_hash(), model_set_sha256=sha256(p.CONFIG),
                       status="completed", completed_epochs=100, environment={"gpu": "fixture"},
                       protocol_key="test", best_epoch=67)
            ck = dest / "weights/best.pt"
            ck.parent.mkdir(parents=True)
            ck.write_bytes(key.encode())
            if len(calls) == 1:
                ck.with_name("last.pt").write_bytes(b"incomplete")
                row.update(status="running", completed_epochs=37)
                write_json(dest / "run.json", row)
                raise KeyboardInterrupt()
            write_json(dest / "run.json", row)
            write_json(dest / "metrics.json", dict(split="val", data_sha256=p.DATA, checkpoint_sha256=sha256(ck)))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "phase2"
            book = Workbook()
            book.active["A1"] = "第一轮五组"
            dest = root / "实验结果汇总.xlsx"
            book.save(dest)
            initial_hash = sha256(dest)
            with (patch.object(p, "OUTPUT", output), patch.object(report, "OUTPUT", output),
                  patch.object(report, "ROOT", root), patch.object(p, "inspect_data", return_value=info),
                  patch.object(p, "register"), patch.object(train, "main", side_effect=simulate),
                  patch.object(train, "resume_settings") as resume):
                with self.assertRaises(KeyboardInterrupt):
                    p.run_group("A", None, "0", study)
                with patch.object(report, "references", return_value={}):
                    with self.assertRaisesRegex(ValueError, "N07"):
                        report.review(None, "0", study)
                self.assertEqual(initial_hash, sha256(dest))
                for group in p.GROUPS:
                    p.run_group(group, None, "0", study)
                    p.run_group(group, None, "0", study)
                self.assertEqual(resume.call_count, 1)
                self.assertEqual(calls, [("N07", False), ("N07", True)] +
                                 [(k, False) for k in sum(p.GROUPS.values(), [])[1:]])
                stopped = read_json(output / "A/N07/attempt_01/run.json")
                self.assertEqual((stopped["status"], stopped["completed_epochs"]), ("interrupted", 37))
                with self.assertRaisesRegex(ValueError, "不同的训练版本"):
                    p.run_group("A", None, "0", {})

                old = {k: (root / k, dict(environment={"gpu": "fixture"}, best_epoch=67, protocol_key="test"))
                       for k in report.REFERENCES}
                with (patch.object(report, "references", return_value=old),
                      patch.object(report, "environment", return_value={}), patch.object(report, "convert_data"),
                      patch.object(report, "evaluate_checkpoint", return_value=self.metric(.92, .82)) as evaluate):
                    report.review(None, "0", study)
                self.assertEqual(evaluate.call_count, 15)
                for call in evaluate.call_args_list:
                    self.assertEqual(call.args[3], report.evaluation_settings(640, 4, "0"))
                book = load_workbook(dest)
                self.assertEqual(book.worksheets[0]["A1"].value, "第一轮五组")
                sheet = book["第二轮14组对比"]
                self.assertEqual([c.value for c in sheet[1]], report.HEADERS)
                self.assertEqual([sheet.cell(i, 1).value.split()[0] for i in range(2, 16)], ["R4", "R5", *p.SPECS])
                self.assertEqual(sheet["B2"].value, .92)
                book.close()
                self.assertEqual(len(read_json(output / "review/comparison.json")["selection"]["qualified"]), 14)
                (output / "A/N07/attempt_02/weights/best.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "完成记录不完整"):
                    p.completed(output / "A/N07", "N07", fingerprint(study), sha256(p.CONFIG))

    def test_last_epoch_revalidates_best(self):
        with TemporaryDirectory() as tmp:
            slot = Path(tmp) / "N01"
            run = slot / "attempt_01"
            weights = run / "weights"
            weights.mkdir(parents=True)
            (weights / "last.pt").write_bytes(b"last")
            best = weights / "best.pt"
            best.write_bytes(b"best")
            params = yaml.safe_load((p.ROOT / "configs/train.yaml").read_text(encoding="utf-8-sig"))
            params.update(p.SETTINGS)
            write_json(run / "run.json", dict(model="N01", completed_epochs=100, status="running",
                                              yolo_sha256=p.COCO, effective_training=params))
            self.assertEqual(p.recover_source(slot, {}, "0"), best)
            (run / "results.csv").write_text("epoch\n" + "\n".join(str(i) for i in range(1, 101)))
            with (patch("fusion.models.load_model", return_value=SimpleNamespace(model=SimpleNamespace(variant="N01"))),
                  patch("fusion.data.convert_data"), patch("evaluate.evaluate_checkpoint", return_value={}) as ev):
                p.finish_validation(best, None, {"names": {0: "UGV", 1: "USV"}}, "0")
                self.assertEqual(ev.call_args.args[0], best)
            self.assertEqual(read_json(run / "run.json")["status"], "completed")

    @staticmethod
    def metric(ugv, usv, localization=.6, ms=2):
        return dict(overall=dict(map50=(ugv + usv) / 2, map50_95=.6),
                    per_class={"UGV": dict(map50=ugv, map50_95=.6),
                               "USV": dict(map50=usv, map50_95=localization)}, inference_ms_per_image=ms)

    def test_raw_thresholds_ties_and_fallback(self):
        scores = {k: self.metric(.91, .84) for k in ["R4", "R5", *p.SPECS]}
        scores.update(baseline=self.metric(.90, .815801), R4=self.metric(.922601, .8148),
                      N01=self.metric(.922600, .99))
        fallback = report.select(scores)
        self.assertEqual((fallback["winner"], fallback["unmet"]), ("N01", ["UGV"]))
        scores.update(N02=self.metric(.93, .87, .61, 2), N03=self.metric(.93, .87, .62, 3),
                      N04=self.metric(.93, .87, .62, 1))
        self.assertEqual(report.select(scores)["winner"], "N04")
        scores["baseline"] = self.metric(.90, float("nan"))
        with self.assertRaises(ValueError):
            report.select(scores)
