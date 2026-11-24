"""Extensions of the pinned trainer; no installed-source patches or parser registration."""
import asyncio
from copy import deepcopy
from pathlib import Path
import traceback
import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER
from .data import convert_data
from .models import make_detector
from .runtime import write_json


class CompleteValidator(DetectionValidator):
    def __call__(self, trainer=None, model=None):
        self.last_batch = None
        self.input_details = {}
        self.phase = "validation_setup"
        self.evidence_epoch = trainer.epoch + 1 if trainer is not None else None
        self.failure_saved = False
        self.forward_hooks = []
        try:
            return super().__call__(trainer=trainer, model=model)
        except BaseException as exc:
            first = exc
            while first.__cause__ is not None or (first.__context__ is not None and not first.__suppress_context__):
                first = first.__cause__ or first.__context__
            try:
                self.save_failure(first, traceback.format_exc())
                self.failure_saved = True
            except Exception as evidence_error:
                LOGGER.warning(f"Unable to save validation evidence: {evidence_error}")
            # Profile.__exit__ may raise again on synchronize after the actual CUDA failure.
            raise first.with_traceback(first.__traceback__) from None
        finally:
            for hook in self.forward_hooks:
                hook.remove()
            self.last_batch = None

    def preprocess(self, batch):
        self.phase = "validation_preprocess"
        snapshot = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise ValueError("Validation evidence requires a CPU dataloader batch")
                snapshot[key] = value.detach().clone()
            else:
                snapshot[key] = deepcopy(value)
        self.last_batch = {"epoch": self.evidence_epoch, "batch_in_epoch": self.batch_i + 1, "batch": snapshot}
        self.input_details = {"input_shape": list(batch["img"].shape),
                              "input_dtype": "torch.float16" if self.args.half else "torch.float32",
                              "device": str(self.device), "cuda_memory_before_batch": None}
        if self.device.type == "cuda":
            self.input_details["cuda_memory_before_batch"] = {
                "allocated": torch.cuda.memory_allocated(self.device),
                "reserved": torch.cuda.memory_reserved(self.device),
                "peak_allocated": torch.cuda.max_memory_allocated(self.device)}
        return super().preprocess(batch)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.forward_hooks = [model.register_forward_pre_hook(self.before_forward),
                              model.register_forward_hook(self.after_forward)]

    def before_forward(self, model, inputs):
        self.phase = "validation_forward"

    def after_forward(self, model, inputs, output):
        self.phase = "validation_loss" if self.training else "validation_postprocess"

    def postprocess(self, preds):
        self.phase = "validation_postprocess"
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        self.phase = "validation_metrics"
        super().update_metrics(preds, batch)
        self.phase = "validation_finalize"
        self.last_batch = None
        self.input_details = {}

    def save_failure(self, first, trace):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        evidence = {"phase": self.phase, **self.input_details,
                    "first_error_type": type(first).__name__, "first_error": str(first),
                    "replay_limit": "CPU input only; last.pt is from the preceding saved epoch, not exact failure state."}
        if self.last_batch is not None:
            temporary = self.save_dir / "failure_batch.pt.tmp"
            torch.save(self.last_batch, temporary)
            temporary.replace(self.save_dir / "failure_batch.pt")
            evidence.update({k: v for k, v in self.last_batch.items() if k != "batch"})
            evidence.update(image_files=self.last_batch["batch"].get("im_file", []),
                            snapshot="failure_batch.pt", input_stage="CPU validation batch before GPU preprocessing")
        (self.save_dir / "failure_traceback.txt").write_text(trace, encoding="utf-8")
        write_json(self.save_dir / "failure_batch.json", evidence)

    def get_dataloader(self, dataset_path, batch_size):
        loader = super().get_dataloader(dataset_path, batch_size)
        expected = self.data.get("expected_images", {}).get(self.args.split)
        if expected is not None and len(loader.dataset) != expected:
            raise RuntimeError(f"Incomplete {self.args.split} set: {len(loader.dataset)}/{expected}")
        return loader


class StudyValidator(CompleteValidator):
    """Count predictions on empty-label images during standalone evaluation."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.negatives = {"confidence": 0.25, "images": 0, "flagged_images": 0, "boxes": 0}

    def update_metrics(self, preds, batch):
        present = set(batch["batch_idx"].flatten().long().cpu().tolist())
        for index, pred in enumerate(preds):
            if index not in present:
                count = int((pred["conf"] >= self.negatives["confidence"]).sum())
                self.negatives["images"] += 1
                self.negatives["flagged_images"] += int(count > 0)
                self.negatives["boxes"] += count
        super().update_metrics(preds, batch)

    def finalize_metrics(self):
        super().finalize_metrics()
        self.metrics.negative_images = self.negatives


class FusionTrainer(DetectionTrainer):
    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            target = overrides.get("save_dir")
            if not target or Path(target).resolve() == Path(self.args.save_dir).resolve():
                raise ValueError("Resume requires a new explicit output directory")
            # Upstream restores the old save_dir from the checkpoint, ignoring project/name overrides.
            for key in ("save_dir", "project", "name", "data"):
                setattr(self.args, key, overrides[key])

    def get_dataset(self):
        if str(self.args.data).lower().endswith((".ndjson", ".yaml", ".yml")):
            self.args.data = str(asyncio.run(convert_data(self.args.data)))
        return super().get_dataset()

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None:
            raise ValueError("Use train.py's initialized model; random whole-detector training is not supported")
        cfg = deepcopy(cfg)
        cfg["fusion_deterministic"] = self.args.deterministic
        model = make_detector(cfg, nc=self.data["nc"], weights=weights)
        write_json(self.save_dir / "rebuild_transfer.json", model.transfer)
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "val":
            batch_size = self.args.batch
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        expected = self.data.get("expected_images", {}).get(mode)
        if expected is not None and len(loader.dataset) != expected:
            raise RuntimeError(f"Incomplete {mode} set: {len(loader.dataset)}/{expected}")
        return loader

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        from copy import copy
        return CompleteValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
