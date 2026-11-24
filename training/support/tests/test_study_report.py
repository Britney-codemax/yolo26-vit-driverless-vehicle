import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from openpyxl import load_workbook

from fusion.workbook import collect_weights, render_book
from fusion.runtime import sha256
from fusion.study import choose


def result(model, seed, value, protocol="A", params=10):
    return {"kind":"train", "data":{"sha256":"data"}, "protocol_key":protocol, "model":model, "seed":seed,
            "status":"completed", "completed_epochs":100, "parameters":params, "path":"artifacts/example",
            "metrics":{"split":"val","data_sha256":"data","overall":{"map50":value, "map50_95":value},
                       "negative_images":{"images":0,"boxes":0,"flagged_images":0},"inference_ms_per_image":1.0}}


class Report(unittest.TestCase):
    def test_completed_checkpoints_are_exposed_in_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            training, models = Path(tmp) / "training", Path(tmp) / "models"
            run = training / "artifacts/example/weights"
            run.mkdir(parents=True)
            for name in ("best.pt", "last.pt"):
                (run / name).write_bytes(name.encode())
            row = result("p5_mamba", 0, .5)
            row.update(id="example", last_sha256=sha256(run / "last.pt"))
            row["metrics"]["checkpoint_sha256"] = sha256(run / "best.pt")
            with patch("fusion.workbook.ROOT", training), patch("fusion.workbook.MODEL_SET", models / "fusion.yaml"):
                best = (training / collect_weights(row)).resolve()
                self.assertTrue(best.is_relative_to(models))
                self.assertEqual(best.read_bytes(), b"best.pt")
                self.assertEqual(best.with_name("last.pt").read_bytes(), b"last.pt")
                self.assertEqual((training / collect_weights(row)).resolve(), best)

    def test_paired_seeds_ignore_duplicates_and_foreign_protocol(self):
        records = [result("baseline",0,.4),result("baseline",1,.5),result("p5_mamba",0,.42),
                   result("p5_mamba",1,.7),result("p5_mamba",1,.8),result("p5_mamba",2,0,"B")]
        with tempfile.TemporaryDirectory() as folder:
            path = render_book(records, Path(folder)/"report.xlsx")
            book = load_workbook(path,read_only=True,data_only=True)
            self.assertEqual(book.sheetnames,["旧版历史参考","新版逐次实验","新版多种子汇总"])
            summary=list(book.worksheets[2].values)
            row=next(r for r in summary if r[1:3]==("A","p5_mamba"))
            self.assertEqual(row[3:5],("[0]",1))
            self.assertAlmostEqual(row[10],2.)
            self.assertEqual(row[11],"[1]")
            self.assertIsNone(next(r for r in summary if r[1]=="B")[10])
            new=list(book.worksheets[1].values)
            self.assertEqual(new[1][19],"不适用")
            self.assertIsNone(new[1][-1])
            book.close()

    def test_tied_accuracy_prefers_smaller_model(self):
        a=result("p5_mamba",0,.622,params=100)
        b=result("hybrid",0,.6225,params=200)
        self.assertEqual(choose({"p5_mamba":[a],"hybrid":[b]}),"p5_mamba")
        a["protocol_key"]="other"
        with self.assertRaises(RuntimeError):
            choose({"p5_mamba":[a],"hybrid":[b]})
