"""REAL, learnable 4D classification tasks with a controllable discriminative CUE,
for validating KCA on trained models where ground truth is known by construction.

Every clip has K spatial clusters (same layout across classes). One cluster is the
"odd" one, and the class = which cluster is odd. What makes it odd is the CUE:

  cue="motion": the odd cluster TRANSLATES over time; all others static, all same
                shape. Only motion is discriminative -> a correct explanation puts
                mass on Smot at the odd cluster.
  cue="shape" : the odd cluster has a DIFFERENT (larger) shape; nothing moves. Only
                geometry is discriminative -> a correct explanation puts mass on
                Sgeo at the odd cluster.

Training one model per cue and explaining both yields the motion-vs-geometry DOUBLE
DISSOCIATION (the primary, hard-to-game validation of the decomposition): Smot should
dominate on the motion model and Sgeo on the shape model, for the same region type.

GT: odd_gt = points of the odd cluster; the rest is the complement.
"""
from __future__ import annotations

import torch

from ..types import Clip
from .base import Batch, Dataset


class ClusterCueTask(Dataset):
    name = "cluster_cue"

    def __init__(self, n: int = 256, K: int = 3, per: int = 32, T: int = 8,
                 cue: str = "motion", amp: float = 0.35, shape_scale: float = 2.2,
                 jitter: float = 0.05, grid: int = 3, seed: int = 0,
                 region_method: str = "kmeans", region_k: int = 3):
        super().__init__(grid=grid, region_method=region_method, region_k=region_k)
        assert cue in ("motion", "shape")
        self.n, self.K, self.per, self.T, self.cue = n, K, per, T, cue
        self.amp, self.shape_scale, self.jitter, self.seed = amp, shape_scale, jitter, seed
        self.num_classes = K
        self.centers = torch.stack([
            torch.tensor([-0.4 + 0.8 * i / max(K - 1, 1), 0.0, 0.0]) for i in range(K)
        ])

    def __len__(self) -> int:
        return self.n

    def _raw(self, idx: int) -> Batch:
        g = torch.Generator().manual_seed(self.seed * 100003 + idx + (0 if self.cue == "motion" else 7))
        K, per, T = self.K, self.per, self.T
        N = K * per
        cls = idx % K
        base = torch.zeros(T, N, 3)
        cluster_of = torch.zeros(N, dtype=torch.long)
        for c in range(K):
            sl = slice(c * per, (c + 1) * per)
            cluster_of[sl] = c
            spread = 0.18 * (self.shape_scale if (self.cue == "shape" and c == cls) else 1.0)
            local = (torch.rand(per, 3, generator=g) - 0.5) * spread
            center = self.centers[c] + (torch.rand(3, generator=g) - 0.5) * self.jitter
            pts0 = center + local
            for t in range(T):
                p = pts0.clone()
                if self.cue == "motion" and c == cls:
                    p[:, 1] += self.amp * (t / (T - 1))          # odd cluster translates
                base[t, sl] = p
        pts = base.unsqueeze(0)
        mask = torch.ones(1, T, N, dtype=torch.bool)
        label = torch.tensor([cls], dtype=torch.long)
        odd = (cluster_of == cls).view(1, 1, N).expand(1, T, N).contiguous()
        # moving_gt = odd cluster only when motion is the cue (else no moving GT)
        moving = odd if self.cue == "motion" else torch.zeros_like(odd)
        return Batch(clip=Clip(points=pts, mask=mask, label=label),
                     moving_gt=moving if self.cue == "motion" else odd,
                     static_gt=odd if self.cue == "shape" else ~odd)


class MotionDirectionTask(Dataset):
    """Class = the DIRECTION the (single) actor cluster moves: 0:+y 1:-y 2:+x 3:-x.
    A reference cluster is always present (constant geometry). Only motion *direction*
    is discriminative -> tests whether KCA's `displace` (trajectory-reversal) probe is
    sign-sensitive: reversing the actor's trajectory should flip the prediction to the
    OPPOSITE direction (up<->down, left<->right), which `freeze` (motion removal) cannot
    distinguish. opposite(d): 0<->1, 2<->3."""
    name = "motion_direction"
    OPP = {0: 1, 1: 0, 2: 3, 3: 2}
    AXIS = {0: (1, +1.0), 1: (1, -1.0), 2: (0, +1.0), 3: (0, -1.0)}  # (xyz-axis, sign)

    def __init__(self, n=256, per=40, T=8, amp=0.3, jitter=0.05, grid=3, seed=0,
                 region_method="kmeans", region_k=2):
        super().__init__(grid=grid, region_method=region_method, region_k=region_k)
        self.n, self.per, self.T, self.amp, self.jitter, self.seed = n, per, T, amp, jitter, seed
        self.num_classes = 4
        self.centers = torch.tensor([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])  # actor, reference

    def __len__(self):
        return self.n

    def _raw(self, idx):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        per, T = self.per, self.T
        N = 2 * per
        cls = idx % 4
        axis, sign = self.AXIS[cls]
        base = torch.zeros(T, N, 3)
        actor = torch.zeros(N, dtype=torch.bool)
        for c in range(2):
            sl = slice(c * per, (c + 1) * per)
            local = (torch.rand(per, 3, generator=g) - 0.5) * 0.18
            center = self.centers[c] + (torch.rand(3, generator=g) - 0.5) * self.jitter
            pts0 = center + local
            for t in range(T):
                p = pts0.clone()
                if c == 0:                                       # actor moves in class dir
                    # centred traversal: +dir and -dir share the SAME spatial envelope
                    # and differ ONLY in temporal order, so the model must use direction
                    # (not a static envelope) -- and `displace`-reverse flips +dir<->-dir.
                    p[:, axis] += sign * self.amp * (t / (T - 1) - 0.5)
                base[t, sl] = p
            if c == 0:
                actor[sl] = True
        pts = base.unsqueeze(0); mask = torch.ones(1, T, N, dtype=torch.bool)
        return Batch(clip=Clip(points=pts, mask=mask, label=torch.tensor([cls])),
                     moving_gt=actor.view(1, 1, N).expand(1, T, N).contiguous(),
                     static_gt=(~actor).view(1, 1, N).expand(1, T, N).contiguous())


# back-compat alias used by earlier tests/driver
class WhichClusterMoves(ClusterCueTask):
    name = "which_cluster_moves"

    def __init__(self, **kw):
        kw.setdefault("cue", "motion")
        kw.setdefault("region_method", "voxel")
        kw.setdefault("region_k", 8)
        super().__init__(**kw)
