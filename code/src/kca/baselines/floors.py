"""Floor baselines (paper Sec. exp-setup): guard against metric gaming.

A real attribution must beat these decisively.
  random        : uniform noise importance.
  geometry_edge : local curvature (surface-edge) saliency — high on geometric edges,
                  ignores motion and the model entirely. A strong-looking but
                  model-blind floor.
"""
from __future__ import annotations

import torch

from ..types import Clip


def random_field(clip: Clip, generator: torch.Generator | None = None) -> torch.Tensor:
    f = torch.rand(clip.mask.shape, generator=generator, device=clip.device)
    return f * clip.mask


def motion_prior(clip: Clip) -> torch.Tensor:
    """Model-BLIND motion saliency: per point, the mean distance to the nearest point of
    the previous and of the next frame (a correspondence-free scene-flow magnitude, valid
    without point identity). Highlights whatever moves, ignoring the model -- it should
    do well on naive 'Smot->moving' pointing but FAIL the motion-only/object-only
    crossover (it cannot tell motion that *matters to the decision* from motion that is
    merely present)."""
    B, T, N, _ = clip.points.shape
    xyz = clip.points[..., :3]
    out = torch.zeros(B, T, N, device=clip.device)
    cnt = torch.zeros(B, T, N, device=clip.device)
    for b in range(B):
        for t in range(T):
            v = clip.mask[b, t]
            if not v.any():
                continue
            for u in (t - 1, t + 1):
                if 0 <= u < T and clip.mask[b, u].any():
                    d = torch.cdist(xyz[b, t][v], xyz[b, u][clip.mask[b, u]]).min(1).values
                    out[b, t, v] += d
                    cnt[b, t, v] += 1
    return out / cnt.clamp_min(1) * clip.mask


def geometry_edge(clip: Clip, knn: int = 8) -> torch.Tensor:
    """Per-point curvature proxy: mean distance to k nearest neighbours in frame.
    High on edges/corners. Model-agnostic, motion-blind."""
    B, T, N, C = clip.points.shape
    xyz = clip.points[..., :3]
    out = torch.zeros(B, T, N, device=clip.device)
    for b in range(B):
        for t in range(T):
            v = clip.mask[b, t]
            p = xyz[b, t][v]                                  # (n,3)
            n = p.shape[0]
            if n <= 1:
                continue
            d = torch.cdist(p, p)                             # (n,n)
            d.fill_diagonal_(float("inf"))
            k = min(knn, n - 1)
            nn = d.topk(k, largest=False).values             # (n,k)
            score = nn.mean(1)
            full = torch.zeros(N, device=clip.device)
            full[v] = score
            out[b, t] = full
    return out * clip.mask
