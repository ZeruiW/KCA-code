"""A synthetic backbone with a KNOWN motion/geometry decomposition.

Used to unit-test the method on CPU without GPU or data. Its target read-out is

    phi(X) = a * (#valid points) + b * sum_{t,i} || p_{t,i} - p_{t0,i} ||

so the geometry term is pure presence and the motion term is pure displacement.
This lets us check that KCA recovers the true Sgeo (presence) and Smot (motion)
contributions of each region. It is NOT a real action model — only a test oracle.
"""
from __future__ import annotations

import torch

from ..types import Clip
from .base import Backbone


class SyntheticBackbone(Backbone):
    name = "synthetic"
    has_attention = False

    def __init__(self, a: float = 1.0, b: float = 2.0, t0: int = 0, n_classes: int = 2):
        self.a, self.b, self.t0, self.n_classes = a, b, t0, n_classes

    def _energy(self, clip: Clip) -> torch.Tensor:
        presence = clip.mask.float().sum(dim=(1, 2))           # (B,)
        # motion = sum of CONSECUTIVE-frame point displacement, gated by both frames
        # being valid. Robust to partial point removal (a removed point-frame simply
        # contributes no motion) -- no fixed-anchor artifact under deletion/insertion.
        xyz = clip.points[..., :3]
        step = (xyz[:, 1:] - xyz[:, :-1]).norm(dim=-1)         # (B,T-1,N)
        both = clip.mask[:, 1:] & clip.mask[:, :-1]            # (B,T-1,N)
        motion = (step * both.float()).sum(dim=(1, 2))         # (B,)
        return self.a * presence + self.b * motion             # (B,)

    def logits(self, clip: Clip) -> torch.Tensor:
        e = self._energy(clip)                                 # (B,) class-1 score
        z = torch.zeros_like(e)                                # class-0 = 0
        return torch.stack([z, e], dim=1)                      # (B,2)


class ShapeMotionBackbone(Backbone):
    """Additive test oracle phi = a * shape + b * motion, where geometry is a SHAPE
    descriptor (per-frame point spread) rather than mere presence -- so abl-geo
    (baseline shape + trajectory) genuinely removes it while freeze preserves it.
    Used to verify the order-symmetrized decomposition and that an ADDITIVE model gives
    interaction residual Delta_int = 0 (Prop. interact)."""
    name = "shape_motion"
    has_attention = False

    def __init__(self, a: float = 1.0, b: float = 1.0, n_classes: int = 2):
        self.a, self.b, self.n_classes = a, b, n_classes

    def _energy(self, clip: Clip) -> torch.Tensor:
        xyz = clip.points[..., :3]
        m = clip.mask.float()
        # per-frame shape spread: std of valid points around their centroid, summed over t
        cnt = m.sum(-1, keepdim=True).clamp_min(1)             # (B,T,1)
        cen = (xyz * m.unsqueeze(-1)).sum(2, keepdim=True) / cnt.unsqueeze(-1)
        var = (((xyz - cen) ** 2).sum(-1) * m).sum(-1) / cnt.squeeze(-1)  # (B,T)
        shape = var.sqrt().sum(-1)                             # (B,)
        step = (xyz[:, 1:] - xyz[:, :-1]).norm(dim=-1)
        both = (clip.mask[:, 1:] & clip.mask[:, :-1]).float()
        motion = (step * both).sum(dim=(1, 2))                 # (B,)
        return self.a * shape + self.b * motion

    def logits(self, clip: Clip) -> torch.Tensor:
        e = self._energy(clip)
        return torch.stack([torch.zeros_like(e), e], dim=1)
