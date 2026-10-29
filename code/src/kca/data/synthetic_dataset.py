"""Synthetic dataset with motion/geometry ground truth — for CPU end-to-end testing
of the eval harness and table generation (no real data needed).

Each clip: a 'moving' cluster translates over time; a 'static' cluster does not.
The class is decided by motion direction, so a motion-faithful explainer should put
Smot on the moving cluster (moving_gt) and Sgeo on the static one (static_gt).
"""
from __future__ import annotations

import torch

from ..types import Clip
from .base import Batch, Dataset


class SyntheticDataset(Dataset):
    name = "synthetic"
    num_classes = 2

    def __init__(self, n: int = 16, T: int = 8, N: int = 128, grid: int = 3, seed: int = 0):
        super().__init__(grid=grid)
        self.n, self.T, self.N, self.seed = n, T, N, seed

    def __len__(self) -> int:
        return self.n

    def _raw(self, idx: int) -> Batch:
        g = torch.Generator().manual_seed(self.seed * 1000 + idx)
        T, N = self.T, self.N
        pts = torch.rand(1, T, N, 3, generator=g) - 0.5
        moving = (pts[0, 0, :, 0] < 0.0)                     # left half moves
        direction = 1.0 if idx % 2 == 0 else -1.0           # class label
        for t in range(T):
            pts[0, t, :, 0] += moving.float() * (0.12 * t * direction)
        mask = torch.ones(1, T, N, dtype=torch.bool)
        label = torch.tensor([0 if direction > 0 else 1], dtype=torch.long)
        mv = moving.view(1, 1, N).expand(1, T, N).contiguous()
        return Batch(clip=Clip(points=pts, mask=mask, label=label),
                     moving_gt=mv, static_gt=~mv)
