import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import auto_resume as recovery


class AutoResume(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.campaign = Path(temp.name)
        self.enterContext(patch.object(recovery, "campaign_for", return_value=self.campaign))
        self.slot = self.campaign / "baseline_seed0"
        self.attempt = self.slot / "attempt_01"
        self.checkpoint = self.attempt / "weights/last.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"fixture")
        self.manifest = self.attempt / "run.json"
        recovery.write_json(self.manifest, {"status": "running", "completed_epochs": 48})

    def test_orphan_checkpoint_resumes_then_queue_continues(self):
        parent = SimpleNamespace(pid=7, wait=lambda: 1)
        jobs = iter([(parent, True), (None, False), (None, False)])
        calls = []

        def dispatch(args, **kwargs):
            calls.append(args)
            if args[2] == "resume":
                self.assertEqual(recovery.read_json(self.manifest)["status"], "interrupted")
                self.assertEqual(Path(args[4]), self.checkpoint)
            return SimpleNamespace(returncode=0)

        with patch.object(recovery, "active_job", side_effect=lambda *a: next(jobs)), \
             patch.object(recovery, "completed", side_effect=lambda slot: bool(calls)), \
             patch.object(recovery.subprocess, "run", side_effect=dispatch):
            recovery.supervise(Path("data.yaml"), self.campaign, "0")
        self.assertEqual([a[2] for a in calls], ["resume", "all"])

    def test_repeated_failures_stop_at_three(self):
        with patch.object(recovery, "active_job", return_value=(None, False)), \
             patch.object(recovery.time, "sleep"), \
             patch.object(recovery.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError, "连续三次"):
                recovery.supervise(Path("data.yaml"), self.campaign, "0")
        self.assertEqual(run.call_count, 3)

    def test_ctrl_c_keeps_training_stopped(self):
        parent = SimpleNamespace(pid=7, wait=lambda: 3221225786)
        with patch.object(recovery, "active_job", return_value=(parent, True)), \
             patch.object(recovery.subprocess, "run") as run:
            recovery.supervise(Path("data.yaml"), self.campaign, "0")
        run.assert_not_called()
