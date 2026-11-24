"""Lightweight four-direction selective state-space branch (VMamba-style SS2D)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def affine_scan(decay, drive):
    """All states h[t] = decay[t] * h[t-1] + drive[t], with h[-1] = 0."""
    length = drive.size(-1)
    if length == 1:
        return drive
    pairs = length // 2
    left = slice(0, 2 * pairs, 2)
    right = slice(1, 2 * pairs, 2)
    odd = affine_scan(decay[..., right] * decay[..., left],
                      drive[..., right] + decay[..., right] * drive[..., left])
    even = torch.cat((drive[..., :1], drive[..., 2::2] + decay[..., 2::2] * odd[..., :(length - 1) // 2]), -1)
    interleaved = torch.stack((even[..., :pairs], odd), -1).flatten(-2)
    return torch.cat((interleaved, even[..., pairs:]), -1) if length % 2 else interleaved


class ResidualSS2D(nn.Module):
    def __init__(self, base, channels, width, state_size):
        super().__init__()
        self.base = base
        self.in_proj = nn.Conv2d(channels, width, 1)
        self.local = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.rank = math.ceil(width / 16)
        self.state_size = state_size
        self.select = nn.Parameter(torch.empty(4, self.rank + 2 * state_size, width))
        self.step = nn.Parameter(torch.empty(4, width, self.rank))
        nn.init.uniform_(self.select, -width ** -0.5, width ** -0.5)
        nn.init.uniform_(self.step, -self.rank ** -0.5, self.rank ** -0.5)
        dt = torch.empty(4, width).uniform_(math.log(.001), math.log(.1)).exp()
        self.step_bias = nn.Parameter(dt.expm1().log())
        self.log_A = nn.Parameter(torch.arange(1, state_size + 1).float().log().repeat(4, width, 1))
        self.skip = nn.Parameter(torch.ones(4, width))
        self.norm = nn.LayerNorm(width)
        self.out_proj = nn.Conv2d(width, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.i, self.f, self.type = base.i, base.f, type(self).__name__
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        base = self.base(x)
        z = F.silu(self.local(self.in_proj(base)))
        b, c, h, w = z.shape
        row = z.flatten(2)
        col = z.transpose(2, 3).flatten(2)
        seq = torch.stack((row, col, row.flip(-1), col.flip(-1)), 1)
        select = torch.einsum("bkcl,kpc->bkpl", seq, self.select)
        steps, write, read = select.split((self.rank, self.state_size, self.state_size), dim=2)
        dt = F.softplus(torch.einsum("bkrl,kcr->bkcl", steps, self.step) + self.step_bias[..., None])
        decay = torch.exp(-dt.unsqueeze(3) * self.log_A.exp()[..., None])
        drive = (dt * seq).unsqueeze(3) * write.unsqueeze(2)
        state = affine_scan(decay, drive)
        y = (state * read.unsqueeze(2)).sum(3) + seq * self.skip[..., None]
        rows = (y[:, 0] + y[:, 2].flip(-1)).reshape(b, c, h, w)
        cols = (y[:, 1] + y[:, 3].flip(-1)).reshape(b, c, w, h).transpose(2, 3)
        branch = self.norm(((rows + cols) / 4).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return base + self.alpha * self.out_proj(branch)
