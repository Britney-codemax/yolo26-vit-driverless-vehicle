"""Offline acceptance: synthetic images/tensors, no training loop or optimizer updates."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fusion.runtime import ROOT, ARTIFACTS, LEGACY_MODELS as MODELS, METRICS, bootstrap, environment, read_json, stamp, write_json

bootstrap()
import torch
torch.set_num_threads(2)
from PIL import Image
from ultralytics.cfg import DEFAULT_CFG_DICT
from torchvision.models import ViT_B_16_Weights
from fusion.models import build_model, make_detector, load_model, load_vit_block, P5ResidualViT
from fusion.data import inspect_data, convert_data
from fusion.engine import FusionTrainer
from fusion.records import START, END, aggregate, collect, refresh
from test_yolo_data import yolo_fixture


class BeforeLoop(Exception):
    pass


class StopBeforeLoop(FusionTrainer):
    def _do_train(self, *args, **kwargs):
        raise BeforeLoop("Stopped before building an optimizer or entering the training loop")


def fixture(directory, splits=("train", "val", "test"), long_name=False):
    directory.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "dataset", "task": "detect", "class_names": {"0": "UGV", "1": "USV"}}]
    for split in splits:
        for i in range(2):
            name = f"{split}_{i}_" + ("x" * 150 if long_name else "image") + ".png"
            Image.new("RGB", (64, 64), (40 + i, 60, 90)).save(directory / name)
            rows.append({"type": "image", "file": name, "split": split, "group_id": name,
                         "annotations": {"boxes": [[0, .4, .5, .3, .3], [1, .7, .5, .2, .2]] if i == 0 else []}})
    path = directory / "synthetic.ndjson"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


class Acceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ARTIFACTS / "checks" / ("acceptance_" + stamp())
        cls.root.mkdir(parents=True)
        cls.report = {"scope": "synthetic only; network blocked; no training loop or optimizer updates",
                      "environment": environment(), "passed_checks": [], "real_vit_transfer": "not_checked"}
        original_connect = socket.socket.connect
        def local_event_loop_only(sock, address):
            if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
                return original_connect(sock, address)
            raise AssertionError("External network forbidden")
        # Windows asyncio creates a loopback socket pair even when no HTTP request is made.
        cls.network = patch.object(socket.socket, "connect", new=local_event_loop_only)
        cls.network.start()
        cls.http = [patch(target, side_effect=AssertionError("HTTP forbidden")) for target in
                    ("aiohttp.ClientSession._request", "requests.sessions.Session.request", "urllib.request.urlopen")]
        for item in cls.http:
            item.start()
        cls.optimizers = [patch.object(t, "step", side_effect=AssertionError("Optimizer update forbidden"))
                          for t in (torch.optim.AdamW, torch.optim.SGD)]
        for item in cls.optimizers:
            item.start()
        cls.weights = ARTIFACTS / "weights/yolo26s.pt"
        if not cls.weights.is_file():
            wrapper, _ = build_model("baseline", check=True)
            cls.weights = cls.root / "synthetic_yolo.pt"
            wrapper.save(cls.weights)
            cls.report["real_yolo_transfer"] = "skipped: no local official weights"
        else:
            cls.report["real_yolo_transfer"] = "local checkpoint checked"
        wrapper, _ = build_model("p5_vit_random", seed=17, check=True)
        cls.fake_vit = cls.root / "synthetic_vit.pth"
        torch.save({"encoder.layers.encoder_layer_0." + k: v.clone() + .001
                    for k, v in wrapper.model.model[10].encoder.state_dict().items()}, cls.fake_vit)
        cls.prefix = Path(ViT_B_16_Weights.IMAGENET1K_V1.url).stem.rsplit("-", 1)[1]

    @classmethod
    def tearDownClass(cls):
        for item in cls.optimizers:
            item.stop()
        for item in cls.http:
            item.stop()
        cls.network.stop()
        write_json(cls.root / "acceptance.json", cls.report)
        print(f"\nAcceptance evidence: {cls.root}")

    def done(self, name):
        self.report["passed_checks"].append(name)

    def build(self, variant, seed=17):
        if variant == "p5_vit_pretrained":
            # This is explicitly a synthetic loader-path test, not official-weight verification.
            from fusion.runtime import sha256
            def digest(path):
                return self.prefix + "0" * (64 - len(self.prefix)) if Path(path) == self.fake_vit else sha256(path)
            with patch("fusion.models.sha256", side_effect=digest):
                return build_model(variant, seed, self.weights, self.fake_vit)
        return build_model(variant, seed, self.weights)

    def test_01_identity_and_branch_gradients(self):
        for variant in MODELS[1:]:
            wrapper, _ = build_model(variant, seed=17, check=True)
            layer = wrapper.model.model[10].eval()
            for h, w in ((20, 20), (12, 20)):
                x = torch.randn(1, 512, h, w)
                with torch.no_grad():
                    torch.testing.assert_close(layer(x), layer.base(x), atol=0, rtol=0)
            x = torch.randn(1, 512, 4, 6)
            layer(x).square().mean().backward()
            self.assertGreater(layer.alpha.grad.abs().item(), 0)
            self.assertEqual(layer.encoder.self_attention.in_proj_weight.grad.abs().sum().item(), 0)
            layer.zero_grad(set_to_none=True)
            with torch.no_grad():
                layer.alpha.fill_(.1)
            layer(x).square().mean().backward()
            for param in (layer.encoder.self_attention.in_proj_weight, layer.input_projection.weight, layer.output_projection.weight):
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(param.grad.abs().sum().item(), 0)
        self.done("all residual branches: square/rectangular identity, gate gradient and active-branch gradients")

    def test_02_transfer_detection_and_reload(self):
        original = load_model(self.weights).model.float().state_dict()
        for variant in MODELS:
            wrapper, report = self.build(variant)
            target = wrapper.model.state_dict()
            self.assertEqual(report["yolo"]["matched_tensors"], len(original))
            for key, source in report["yolo"]["loaded"].items():
                self.assertTrue(torch.equal(target[key], original[source]))
            write_json(self.root / f"initial_{variant}.json", report)
            model = make_detector(wrapper.model.yaml, nc=2, weights=wrapper.model)
            model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
            model.names = {0: "UGV", 1: "USV"}
            model.train()
            if variant != "baseline":
                with torch.no_grad():
                    model.model[10].alpha.fill_(.1)
            for positive in (True, False):
                model.zero_grad(set_to_none=True)
                batch = {"img": torch.rand(2, 3, 64, 96),
                         "batch_idx": torch.tensor([0., 1.]) if positive else torch.empty(0),
                         "cls": torch.tensor([[0.], [1.]]) if positive else torch.empty(0, 1),
                         "bboxes": torch.tensor([[.5, .5, .4, .4], [.5, .5, .2, .3]]) if positive else torch.empty(0, 4)}
                loss, items = model.loss(batch)
                self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(items).all())
                loss.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                if variant != "baseline":
                    self.assertGreater(model.model[10].encoder.self_attention.in_proj_weight.grad.abs().sum().item(), 0)
            model.eval()
            for h, w in ((64, 64), (64, 96)):
                with torch.no_grad():
                    output = model(torch.zeros(1, 3, h, w))[0]
                self.assertEqual(output.shape[-1], 6)
                self.assertTrue(torch.isfinite(output).all())
            model.args = dict(DEFAULT_CFG_DICT)
            wrapper.model = model
            checkpoint = self.root / (variant + ".pt")
            wrapper.save(checkpoint)
            reloaded = load_model(checkpoint).model.eval()
            self.assertEqual(reloaded.variant, variant)
            with torch.no_grad():
                expected = reloaded(torch.zeros(1, 3, 64, 96))[0]
            child_output = self.root / (variant + "_output.pt")
            code = """import socket,sys
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Network forbidden'))
from fusion.runtime import bootstrap
bootstrap()
import torch
torch.set_num_threads(2)
from fusion.models import load_model
m=load_model(sys.argv[1]).model.eval()
with torch.no_grad(): y=m(torch.zeros(1,3,64,96))[0]
torch.save(y,sys.argv[2])
"""
            subprocess.run([sys.executable, "-B", "-c", code, str(checkpoint), str(child_output)], cwd=ROOT / "support", check=True)
            torch.testing.assert_close(torch.load(child_output, weights_only=True), expected)
        self.done("four variants: mapped YOLO tensors, two-class positive/empty losses, finite gradients, fresh-process reload parity")

    def test_03_pretrained_control_and_optional_real_weights(self):
        random, _ = self.build("p5_vit_random")
        transferred, _ = self.build("p5_vit_pretrained")
        a, b = random.model.state_dict(), transferred.model.state_dict()
        self.assertEqual(a.keys(), b.keys())
        different = []
        for key in a:
            self.assertEqual(a[key].shape, b[key].shape)
            if not torch.equal(a[key], b[key]):
                different.append(key)
        self.assertTrue(different)
        self.assertTrue(all(k.startswith("model.10.encoder.") for k in different))
        with self.assertRaisesRegex(ValueError, "hash"):
            load_vit_block(transferred.model, self.fake_vit)
        self.done("random/pretrained control: identical adapter/gate/base initialization; only encoder weights differ (synthetic source)")
        real = ARTIFACTS / "weights/vit_b_16_imagenet1k_v1.pth"
        if not real.is_file():
            self.report["real_vit_transfer"] = "SKIPPED: official ViT weights not present; synthetic mapping is not real-weight verification"
            self.skipTest("Official ViT checkpoint absent; real-weight transfer unverified")
        model, report = build_model("p5_vit_pretrained", 17, self.weights, real)
        state = torch.load(real, map_location="cpu", weights_only=True)
        for key, value in model.model.model[10].encoder.state_dict().items():
            source_key = "encoder.layers.encoder_layer_0." + key
            # Official V1 weights predate torchvision's numbered MLP layer names.
            if source_key not in state:
                source_key = source_key.replace("mlp.0.", "mlp.linear_1.").replace("mlp.3.", "mlp.linear_2.")
            self.assertTrue(torch.equal(value, state[source_key]))
        write_json(self.root / "real_vit_transfer.json", report["vit"])
        self.report["real_vit_transfer"] = "passed: official hash and all encoder tensors verified"

    def test_04_ndjson_boundaries_and_cache(self):
        path = fixture(self.root / "data", long_name=True)
        before = path.read_bytes()
        info = inspect_data(path)
        self.assertEqual(info["counts"]["val"]["boxes"], 2)
        data_yaml = asyncio.run(convert_data(path, cache_root=self.root / "cache"))
        self.assertEqual(path.read_bytes(), before)
        cached = list(data_yaml.parent.glob("images/*/*"))
        self.assertEqual(len(cached), 6)
        self.assertTrue(all(len(p.stem) == 64 for p in cached))
        self.assertEqual(data_yaml, asyncio.run(convert_data(path, cache_root=self.root / "cache")))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        def reject(mutator, pattern):
            copied = deepcopy(rows)
            mutator(copied)
            invalid = path.parent / "invalid.ndjson"
            invalid.write_text("\n".join(map(json.dumps, copied)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, pattern):
                inspect_data(invalid)
        reject(lambda r: r[3].update(group_id=r[1]["group_id"]), "group_id")
        reject(lambda r: [r[i].update(url="https://example.invalid/image?sig=" + str(i)) for i in (1, 3)], "across splits")
        reject(lambda r: r[1].update(file="../escape.png"), "Unsafe")
        reject(lambda r: r[1]["annotations"].update(boxes=[[0, .9, .5, .5, .3]]), "outside")
        reject(lambda r: r[1]["annotations"].update(boxes=[[8, .5, .5, .3, .3]]), "Expected")
        cached[0].write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            asyncio.run(convert_data(path, cache_root=self.root / "cache"))
        no_test = fixture(self.root / "no_test", splits=("train", "val"))
        with self.assertRaisesRegex(ValueError, "test"):
            inspect_data(no_test, required=("test",))
        single = fixture(self.root / "test_only", splits=("test",))
        asyncio.run(convert_data(single, required=("test",), cache_root=self.root / "cache"))
        self.done("NDJSON: long names, immutable input, cache reuse, corrupt cache refusal, leakage/box/path checks, missing/test-only splits")

    def test_05_actual_train_entry_rebuild_stops_before_loop(self):
        inputs = [fixture(self.root / "trainer_ndjson"), yolo_fixture(self.root / "trainer_yolo")]
        for path in inputs:
            for variant in MODELS:
                with self.subTest(format=path.suffix, model=variant):
                    wrapper, _ = self.build(variant)
                    before = {k: v.clone() for k, v in wrapper.model.state_dict().items() if k.startswith("model.10.")}
                    with self.assertRaises(BeforeLoop):
                        wrapper.train(data=str(path), trainer=StopBeforeLoop, device="cpu", workers=0,
                                      imgsz=64, batch=2, epochs=1, amp=False, plots=False, optimizer="AdamW", lr0=.001,
                                      project=str(self.root / "trainer"), name=variant + path.suffix, exist_ok=True)
                    trainer = wrapper.trainer
                    self.assertEqual(trainer.model.model[-1].nc, 2)
                    self.assertEqual(trainer.model.variant, variant)
                    self.assertTrue(str(trainer.args.data).endswith(".yaml"))
                    for key, value in before.items():
                        self.assertTrue(torch.equal(trainer.model.state_dict()[key], value))
                    self.assertTrue((trainer.save_dir / "rebuild_transfer.json").is_file())
        self.done("actual model.train with YOLO YAML/NDJSON and all-variant Trainer rebuild; stopped before optimizer/training loop")

    def test_06_validator_on_synthetic_images(self):
        from evaluate import evaluate_checkpoint, evaluation_settings
        data = yolo_fixture(self.root / "eval_data", splits=("test",))
        prepared = asyncio.run(convert_data(data, required=("test",), cache_root=self.root / "cache"))
        wrapper, _ = self.build("baseline")
        wrapper.model = make_detector(wrapper.model.yaml, nc=2, weights=wrapper.model)
        wrapper.model.names = {0: "UGV", 1: "USV"}
        wrapper.model.args = dict(DEFAULT_CFG_DICT)
        checkpoint = self.root / "validation_synthetic.pt"
        wrapper.save(checkpoint)
        metrics = evaluate_checkpoint(checkpoint, prepared, self.root / "validation",
                                      evaluation_settings(64, 2, "cpu", "test"), {0: "UGV", 1: "USV"})
        self.assertEqual(metrics["split"], "test")
        self.assertEqual(set(metrics["per_class"]), {"UGV", "USV"})
        self.assertTrue(all(v is not None for v in metrics["overall"].values()))
        from evaluate import main as evaluate_main
        self.assertEqual(metrics["negative_images"]["images"], 1)
        with patch("fusion.records.refresh_safely"), patch("evaluate.result_root", return_value=self.root / "evaluations"):
            evaluate_main(["--weights", str(checkpoint), "--data", str(data), "--split", "test",
                           "--device", "cpu", "--batch", "2", "--imgsz", "64"])
        saved = list((self.root / "evaluations").glob("*/run.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(read_json(saved[0])["status"], "completed")
        self.assertEqual(read_json(saved[0])["data"]["counts"]["test"]["boxless_images"], 1)
        self.done("actual evaluate.py with YOLO test-only data including a negative; explicit eval/val and metric serialization")

    def test_07_records_and_duplicate_seeds(self):
        root = self.root / "records"
        rows = []
        for model in ("baseline", "p5_vit_small"):
            for seed in (0, 1, 2):
                directory = root / f"{model}_{seed}"
                row = {"id": directory.name, "kind": "train", "model": model, "seed": seed,
                       "status": "completed", "protocol_key": "same", "data": {"sha256": "abc"}}
                write_json(directory / "run.json", row)
                write_json(directory / "metrics.json", {"overall": dict.fromkeys(METRICS, .5 + seed * .1), "per_class": {}})
        write_json(root / "failed/run.json", {"id": "failed", "kind": "train", "model": "baseline", "seed": 7, "status": "failed"})
        write_json(root / "plan.json", {"entries": [{"id": "pending", "model": "baseline", "seed": 8, "status": "not_run",
                                                       "path": (root / "pending").relative_to(ROOT).as_posix()}]})
        journal = self.root / "journal.md"
        journal.write_text("MANUAL\n" + START + "\nold\n" + END + "\nTAIL", encoding="utf-8")
        rows, groups = refresh(root, journal, include_evaluations=False)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(groups[0]["metrics"]["map50"]["std"], .1)
        self.assertEqual(groups[1]["delta_map50"], 0)
        before = journal.read_bytes()
        refresh(root, journal, include_evaluations=False)
        self.assertEqual(before, journal.read_bytes())
        self.assertIn("MANUAL", journal.read_text(encoding="utf-8"))
        self.assertIn("未运行", journal.read_text(encoding="utf-8"))
        rows.append(deepcopy(rows[0]))
        duplicated = aggregate(rows)
        group = next(g for g in duplicated if g["model"] == rows[0]["model"])
        self.assertIn(rows[0]["seed"], group["excluded_duplicate_seeds"])
        self.assertEqual(group["n"], 2)
        self.assertIsNone(group["delta_map50"] if group["model"] != "baseline" else next(g for g in duplicated if g["model"] != "baseline")["delta_map50"])
        one = aggregate([r for r in rows if r.get("model") == "p5_vit_small" and r.get("seed") == 1])[0]
        self.assertIsNone(one["metrics"]["map50"]["std"])
        distinct = deepcopy(rows[1]); distinct["protocol_key"] = "different"
        self.assertEqual(len(aggregate(rows + [distinct])), 3)
        self.done("summary: failures/missing values, manual text preservation, idempotence, sample std, duplicate-seed exclusion, protocol separation")

    def test_08_training_lifecycle_without_training(self):
        import train
        data = fixture(self.root / "lifecycle_data")
        directory = self.root / "lifecycle"
        args = SimpleNamespace(data=str(data), config=str(ROOT / "configs/train.yaml"), model="baseline", weights=str(self.weights),
                               vit_weights=str(self.fake_vit), check=False, run_dir=str(directory), epochs=3, batch=2,
                               seed=0, imgsz=64, patience=2, workers=0, device="cpu")
        callbacks = {}
        flags = {"eval": False, "train": False}
        metric = SimpleNamespace(names={0: "UGV", 1: "USV"}, nt_per_class=[1, 1],
            box=SimpleNamespace(ap_class_index=[0, 1], class_result=lambda i: [.2, .3, .4, .1], mean_results=lambda: [.2, .3, .4, .1]))
        net = torch.nn.Linear(1, 1)
        net.yaml = {"fusion_variant": "baseline"}
        fake = SimpleNamespace(model=net, names={0: "UGV", 1: "USV"}, add_callback=lambda name, callback: callbacks.update({name: callback}))
        def fake_train(**kwargs):
            flags["train"] = True
            self.assertEqual(kwargs["data"], str(data.resolve()))
            self.assertEqual(kwargs["save_dir"], str(directory))
            from fusion.diagnostics import DiagnosticTrainer
            self.assertIs(kwargs["trainer"].func, DiagnosticTrainer)
            self.assertIsNone(kwargs["trainer"].keywords["batch_limit"])
            effective = train.configuration(args)
            trainer = SimpleNamespace(args=SimpleNamespace(**effective, data="synthetic.yaml"), amp=False, model=net,
                                      epoch=0, fitness=.2, best_fitness=.2,
                                      test_loader=SimpleNamespace(batch_size=2), validator=SimpleNamespace(seen=2))
            fake.trainer = trainer
            torch.use_deterministic_algorithms(True, warn_only=True)
            callbacks["on_train_start"](trainer)
            callbacks["on_model_save"](trainer)
            trainer.epoch = 2; trainer.fitness = .1
            callbacks["on_model_save"](trainer)
            flags["training_protocol"] = read_json(directory / "run.json")["protocol_key"]
            torch.use_deterministic_algorithms(False)
            (directory / "weights").mkdir()
            (directory / "weights/best.pt").write_bytes(b"synthetic checkpoint placeholder")
        fake.train = fake_train
        fake.eval = lambda: flags.update(eval=True)
        def fake_val(**kwargs):
            self.assertTrue(flags["eval"])
            self.assertEqual(kwargs["data"], "synthetic.yaml")
            return metric
        fake.val = fake_val
        with patch("fusion.models.build_model", return_value=(fake, {"yolo": {"sha256": "synthetic"}})), \
             patch("fusion.models.load_model", return_value=fake), patch("fusion.records.refresh_safely"):
            train.run(args)
        state = read_json(directory / "run.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["id"], directory.relative_to(ARTIFACTS).as_posix())
        self.assertIsNone(state["environment"]["gpu"])
        self.assertIn("warn_only", state["determinism"])
        self.assertTrue(state["determinism"]["enabled"])
        self.assertEqual(state["protocol_key"], flags["training_protocol"])
        self.assertEqual(state["best_epoch"], 1)
        self.assertEqual(state["completed_epochs"], 3)
        self.assertEqual(read_json(directory / "metrics.json")["overall"]["map50"], .4)
        args.run_dir = str(self.root / "failed_lifecycle")
        with patch("fusion.models.build_model", side_effect=RuntimeError("synthetic failure")), patch("fusion.records.refresh_safely"):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                train.run(args)
        self.assertEqual(read_json(Path(args.run_dir) / "run.json")["status"], "failed")
        self.done("mocked train.py lifecycle: NDJSON input, explicit eval/val, exact best epoch, actual config, completion/failure manifests")

    def test_09_serial_runner_manifest(self):
        import run_experiments
        data = yolo_fixture(self.root / "batch_data")
        batch = self.root / "batch_runner"
        commands = []
        def child(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=1 if len(commands) == 1 else 0)
        with patch("run_experiments.local_output", return_value=batch), \
             patch("run_experiments.subprocess.run", side_effect=child), patch("fusion.records.refresh_safely"):
            with self.assertRaises(SystemExit) as raised:
                run_experiments.main(["--models", "baseline", "p5_vit_small", "--seeds", "0", "1", "--data", str(data),
                                      "--weights", str(self.weights)])
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(len(commands), 1)
        plan = read_json(batch / "plan.json")["entries"]
        self.assertEqual(plan[0]["status"], "failed")
        self.assertTrue(all(p["status"] == "not_run" for p in plan[1:]))
        self.assertEqual(len({p["id"] for p in plan}), 4)
        self.assertTrue(all("--run-dir" in command and "--seed" in command for command in commands))
        self.assertTrue(all(command[command.index("--data") + 1] == str(data.resolve()) for command in commands))
        with patch("run_experiments.subprocess.run") as child:
            with self.assertRaises(SystemExit):
                run_experiments.main(["--models", "p5_vit_pretrained", "--data", str(data), "--weights", str(self.weights),
                                      "--vit-weights", str(self.root / "missing_vit.pth")])
            child.assert_not_called()
        self.done("serial runner: shared CLI arguments, unique IDs, first failure stops subsequent runs; child processes mocked")

    def test_10_summary_excludes_old_directories(self):
        from fusion.runtime import RUNS, EVALUATIONS
        self.assertEqual(RUNS, ARTIFACTS / "runs/roboflow_v3")
        self.assertEqual(EVALUATIONS, ARTIFACTS / "evaluations/roboflow_v3")
        area = self.root / "isolation"
        for folder in ("runs/old", "evaluations/old", "runs/roboflow_v3/new"):
            write_json(area / folder / "run.json", {"id": folder, "model": "baseline", "status": "running"})
        journal = area / "journal.md"
        journal.write_text(START + "\n" + END, encoding="utf-8")
        with patch("fusion.records.RUNS", area / "runs/roboflow_v3"), \
             patch("fusion.records.EVALUATIONS", area / "evaluations/roboflow_v3"):
            rows, _ = refresh(destination=journal)
        self.assertEqual([r["id"] for r in rows], ["runs/roboflow_v3/new"])
        self.assertTrue((area / "runs/old/run.json").exists())
        with patch("fusion.records.RUNS", area / "empty"), patch("fusion.records.EVALUATIONS", area / "empty_eval"):
            rows, _ = refresh(destination=journal)
        self.assertEqual(rows, [])
        self.assertIn("暂无新版数据实验结果", journal.read_text(encoding="utf-8"))
        self.done("only roboflow_v3 runs/evaluations enter summaries; old files preserved; no fabricated empty-run metrics")


if __name__ == "__main__":
    unittest.main()
