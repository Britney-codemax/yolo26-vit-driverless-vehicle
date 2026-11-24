"""CPU failure evidence for training, with optional bounded diagnostic checks."""
from copy import deepcopy
import torch
from .engine import FusionTrainer
from .runtime import write_json


class DiagnosticComplete(Exception):
    """Internal stop signal: the requested batches finished, not a training failure."""


class DiagnosticTrainer(FusionTrainer):
    def __init__(self, *args, batch_limit, **kwargs):
        self.batch_limit = batch_limit
        self.batches_completed = 0
        self.optimizer_steps = 0
        self.last_batch = None
        self.batch_in_epoch = 0
        self.phase = "setup"
        self.loss_hook = None
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_start", self.start_diagnostic)
        self.add_callback("on_train_epoch_start", self.start_epoch_diagnostic)
        self.add_callback("on_train_batch_end", self.finish_batch)

    def start_diagnostic(self, trainer):
        if self.batch_limit is not None:
            self.loss_hook = self.model.register_forward_hook(self.check_loss)

    def start_epoch_diagnostic(self, trainer):
        self.batch_in_epoch = 0
        self.last_batch = None

    def preprocess_batch(self, batch):
        self.phase = "preprocess"
        self.batch_in_epoch += 1
        # Keep the augmented CPU input before .to(device); never copy a failed GPU tensor back.
        snapshot = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise ValueError("Diagnostic snapshot requires a CPU dataloader batch")
                snapshot[key] = value.detach().clone()
            else:
                snapshot[key] = deepcopy(value)
        self.last_batch = {"epoch": self.epoch + 1, "batch_in_epoch": self.batch_in_epoch,
                           "batch_number": self.batches_completed + 1, "batch": snapshot}
        result = super().preprocess_batch(batch)
        self.phase = "forward" if self.batch_limit is not None else "training_step"
        return result

    def check_loss(self, model, inputs, output):
        if not inputs or not isinstance(inputs[0], dict):
            return
        self.phase = "loss_check"
        loss, items = output
        if not torch.isfinite(loss).all() or not torch.isfinite(items).all():
            raise FloatingPointError("Non-finite loss before backward")
        self.phase = "backward"

    def check_gradients(self):
        self.phase = "gradient_check"
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient before optimizer update: {name}")

    def optimizer_step(self):
        if self.batch_limit is not None:
            self.check_gradients()
        self.phase = "optimizer_step"
        super().optimizer_step()
        self.optimizer_steps += 1

    def finish_batch(self, trainer):
        if self.batch_limit is not None:
            self.check_gradients()
        self.batches_completed += 1
        self.phase = "batch_completed"
        self.last_batch = None
        if self.batch_limit is not None and self.batches_completed >= self.batch_limit:
            if not self.optimizer_steps:
                raise RuntimeError("Diagnostic finished without an optimizer update")
            raise DiagnosticComplete()

    def save_failure(self, directory):
        if getattr(getattr(self, "validator", None), "failure_saved", False):
            return  # Validation already saved its own input; do not overwrite it with a training batch.
        if self.last_batch is not None:
            temporary = directory / "failure_batch.pt.tmp"
            torch.save(self.last_batch, temporary)
            temporary.replace(directory / "failure_batch.pt")
            write_json(directory / "failure_batch.json", {
                **{k: v for k, v in self.last_batch.items() if k != "batch"},
                "phase": self.phase, "optimizer_steps": self.optimizer_steps,
                "image_files": self.last_batch["batch"].get("im_file", []),
                "snapshot": "failure_batch.pt", "input_stage": "augmented CPU batch before GPU preprocessing",
                "weight_reference": "initial_transfer.json; weights/last.pt if an epoch checkpoint exists",
                "replay_limit": "The failing step's weights and RNG state are not captured; replay is approximate."})

    def close_diagnostic(self):
        if self.loss_hook is not None:
            self.loss_hook.remove()
