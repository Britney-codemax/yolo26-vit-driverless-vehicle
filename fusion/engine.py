"""Extensions of the pinned trainer; no installed-source patches or parser registration."""
import asyncio
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from .data import convert_data
from .models import make_detector
from .runtime import write_json


class CompleteValidator(DetectionValidator):
    def get_dataloader(self, dataset_path, batch_size):
        loader = super().get_dataloader(dataset_path, batch_size)
        expected = self.data.get("expected_images", {}).get(self.args.split)
        if expected is not None and len(loader.dataset) != expected:
            raise RuntimeError(f"Incomplete {self.args.split} set: {len(loader.dataset)}/{expected}")
        return loader


class FusionTrainer(DetectionTrainer):
    def get_dataset(self):
        if str(self.args.data).lower().endswith((".ndjson", ".yaml", ".yml")):
            self.args.data = str(asyncio.run(convert_data(self.args.data)))
        return super().get_dataset()

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None:
            raise ValueError("Use train.py's initialized model; random whole-detector training is not supported")
        model = make_detector(cfg, nc=self.data["nc"], weights=weights)
        write_json(self.save_dir / "rebuild_transfer.json", model.transfer)
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        expected = self.data.get("expected_images", {}).get(mode)
        if expected is not None and len(loader.dataset) != expected:
            raise RuntimeError(f"Incomplete {mode} set: {len(loader.dataset)}/{expected}")
        return loader

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        from copy import copy
        return CompleteValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
