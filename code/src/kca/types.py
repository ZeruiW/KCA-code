"""Core data contracts shared across KCA.

A point-cloud video is a list of per-frame point sets. We keep points in a single
padded tensor with a validity mask so batching and interventions are vectorised.

Shapes (B = batch, T = frames, N = max points/frame, C = channels = 3 + features):
    points : (B, T, N, C)   xyz [+ optional features]
    mask   : (B, T, N) bool  True = real point, False = padding
A *region partition* labels every (t, point) as belonging to one of m regions
(or -1 for "unassigned / padding"):
    regions : (B, T, N) long  region id in [0, m)  or -1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Clip:
    """A batch of point-cloud video clips with an optional region partition."""

    points: torch.Tensor          # (B, T, N, C)
    mask: torch.Tensor            # (B, T, N) bool
    regions: Optional[torch.Tensor] = None  # (B, T, N) long, region id or -1
    label: Optional[torch.Tensor] = None    # (B,) long, GT class (if known)

    @property
    def B(self) -> int: return self.points.shape[0]
    @property
    def T(self) -> int: return self.points.shape[1]
    @property
    def N(self) -> int: return self.points.shape[2]
    @property
    def C(self) -> int: return self.points.shape[3]
    @property
    def device(self) -> torch.device: return self.points.device

    def to(self, device) -> "Clip":
        return Clip(
            points=self.points.to(device),
            mask=self.mask.to(device),
            regions=None if self.regions is None else self.regions.to(device),
            label=None if self.label is None else self.label.to(device),
        )

    def clone(self) -> "Clip":
        return Clip(
            points=self.points.clone(),
            mask=self.mask.clone(),
            regions=None if self.regions is None else self.regions.clone(),
            label=None if self.label is None else self.label.clone(),
        )

    def num_regions(self) -> int:
        if self.regions is None:
            raise ValueError("Clip has no region partition.")
        m = int(self.regions.max().item()) + 1
        return max(m, 0)
