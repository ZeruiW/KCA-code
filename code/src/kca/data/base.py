"""Dataset contract for KCA evaluation.

A dataset yields batches as `Clip` objects (points/mask/regions/label) plus optional
ground-truth localization masks for the motion-vs-geometry test:

    moving_gt : (B, T, N) bool  — points belonging to moving structure (limbs / manipulated
                                  object) -> the target of Smot.
    static_gt : (B, T, N) bool  — static class-defining structure -> the target of Sgeo.

Concrete datasets (MSR-Action3D, NTU, HOI4D, SYNTHIA-4D) subclass this. Regions are
attached here (default: voxel-grid) so the explainer and baselines get a partition.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from ..types import Clip
from .. import regions as rg


@dataclass
class Batch:
    clip: Clip
    moving_gt: Optional[torch.Tensor] = None      # (B,T,N) bool
    static_gt: Optional[torch.Tensor] = None      # (B,T,N) bool


class Dataset(abc.ABC):
    name: str = "abstract"
    num_classes: int = 0

    def __init__(self, grid: int = 4, region_method: str = "voxel", region_k: int = 8):
        self.grid = grid
        self.region_method = region_method
        self.region_k = region_k

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def _raw(self, idx: int) -> Batch:
        """Return a single-sample Batch (B=1) with points/mask/label and optional GT."""

    def get(self, idx: int) -> Batch:
        b = self._raw(idx)
        if b.clip.regions is None:
            b.clip = rg.attach_regions(b.clip, grid=self.grid,
                                       method=self.region_method, k=self.region_k)
        return b

    def batches(self, batch_size: int = 1, limit: Optional[int] = None) -> Iterator[Batch]:
        """Yield single-sample batches (B=1). Region partitions differ per clip, so we
        keep B=1 for correctness; the explainer/metrics are written to vectorise over
        points within a clip, which dominates cost."""
        n = len(self) if limit is None else min(limit, len(self))
        for i in range(n):
            yield self.get(i)
