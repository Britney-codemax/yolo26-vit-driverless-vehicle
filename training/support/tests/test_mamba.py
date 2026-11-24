import unittest

import torch

from fusion.runtime import ARTIFACTS, STUDY_MODELS, bootstrap
bootstrap()
from fusion.mamba import affine_scan, ResidualSS2D
from fusion.models import build_model, make_detector
from test_resume import Resume as _Resume


class Scan(unittest.TestCase):
    def test_four_routes_match_pixelwise_recurrence_on_rectangle(self):
        from torch.nn import functional as F
        base = torch.nn.Identity()
        base.i, base.f = 0, -1
        block = ResidualSS2D(base, 4, 4, 2).double()
        block.alpha.data.fill_(1.)
        x = torch.randn(1, 4, 3, 5, dtype=torch.float64, requires_grad=True)
        feature = F.silu(block.local(block.in_proj(x)))[0]
        rows = [(y, col) for y in range(3) for col in range(5)]
        cols = sorted(rows, key=lambda pos: (pos[1], pos[0]))
        pixels = {pos: [] for pos in rows}
        for k, route in enumerate((rows, cols, rows[::-1], cols[::-1])):
            state = x.new_zeros(4, 2)
            for y, col in route:
                u = feature[:, y, col]
                t, write, read = (block.select[k] @ u).split([1,2,2])
                dt = F.softplus(block.step[k] @ t + block.step_bias[k])
                state = torch.exp(-dt[:, None] * block.log_A[k].exp()) * state + (dt * u)[:, None] * write
                pixels[y, col].append(state @ read + block.skip[k] * u)
        merged = torch.stack([torch.stack(pixels[pos]).mean(0) for pos in rows]).view(1,3,5,4)
        expected = x + block.out_proj(block.norm(merged).permute(0,3,1,2))
        actual = block(x)
        torch.testing.assert_close(actual, expected)
        cotangent = torch.randn_like(x)
        for a, b in zip(torch.autograd.grad(actual, (x, block.select, block.log_A), cotangent),
                        torch.autograd.grad(expected, (x, block.select, block.log_A), cotangent)):
            torch.testing.assert_close(a, b)

    def test_parallel_matches_serial_outputs_and_gradients(self):
        for length in (1, 2, 3, 15, 40, 65):
            decay = torch.rand(2, 3, length, dtype=torch.float64, requires_grad=True)
            drive = torch.randn_like(decay, requires_grad=True)
            state = torch.zeros_like(drive[..., 0])
            values = []
            for i in range(length):
                state = decay[..., i] * state + drive[..., i]
                values.append(state)
            expected = torch.stack(values, -1)
            actual = affine_scan(decay, drive)
            torch.testing.assert_close(actual, expected)
            loss_weight = torch.randn_like(actual)
            def gradients(value):
                return torch.autograd.grad((value * loss_weight).sum(), (decay, drive),
                                           allow_unused=True, materialize_grads=True)
            for left, right in zip(gradients(expected), gradients(actual)):
                torch.testing.assert_close(left, right)

    def test_original_detector_equivalence_and_active_scan(self):
        torch.set_num_threads(2)
        weights = ARTIFACTS / "weights/yolo26s.pt"
        baseline, _ = build_model("baseline", yolo_weights=weights)
        baseline = make_detector(baseline.model.yaml, nc=2, weights=baseline.model).eval()
        sample = torch.randn(2, 3, 64, 96)
        with torch.no_grad():
            expected = baseline(sample)[0]
        for variant in STUDY_MODELS[2:]:
            wrapper, _ = build_model(variant, yolo_weights=weights)
            model = make_detector(wrapper.model.yaml, nc=2, weights=wrapper.model).eval()
            block = next(m for m in model.modules() if isinstance(m, ResidualSS2D))
            with torch.no_grad():
                torch.testing.assert_close(model(sample)[0], expected, atol=0, rtol=0)
                block.alpha.fill_(0.1)
            from types import SimpleNamespace
            from ultralytics.cfg import DEFAULT_CFG_DICT
            model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
            model.train()
            batch = {"img": sample, "batch_idx": torch.tensor([0., 1.]), "cls": torch.tensor([[0.],[1.]]),
                     "bboxes": torch.tensor([[.4,.4,.3,.3],[.6,.6,.2,.2]])}
            loss, _ = model.loss(batch)
            loss.sum().backward()
            for p in (block.select, block.log_A, block.step, block.alpha):
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)


class HybridResume(_Resume):
    variant = "p4_mamba_p5_vit"


del _Resume
