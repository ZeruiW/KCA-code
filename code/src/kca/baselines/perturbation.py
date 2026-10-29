"""Perturbation/occlusion baselines (paper Sec. exp-setup), model-agnostic.

  point_drop : single-region occlusion saliency (Zheng ICCV'19, lifted to regions) —
               importance of a region = phi drop when it alone is dropped. O(m) passes.
  occam      : random-mask occlusion attribution (Schinagl CVPR'22) — average phi drop
               over many random point masks, weighted by membership. O(M) passes.
  extremal   : extremal-perturbation mask optimization (Fong ICCV'19) — learn a soft
               point mask maximising phi-drop under an area constraint.
"""
from __future__ import annotations

import torch

from ..backbones.base import Backbone
from ..types import Clip
from .. import interventions as itv


@torch.no_grad()
def point_drop(model: Backbone, clip: Clip, target: torch.Tensor,
               baseline: itv.BaselineKind = "bg") -> torch.Tensor:
    """Per-region ablation drop scattered onto points. Requires a region partition."""
    if clip.regions is None:
        raise ValueError("point_drop needs regions")
    B, T, N = clip.mask.shape
    phi = model.phi(clip, target)
    out = torch.zeros(B, T, N, device=clip.device)
    for r in range(clip.num_regions()):
        drop = phi - model.phi(itv.ablate(clip, r, baseline=baseline), target)
        sel = (clip.regions == r) & clip.mask
        out += sel.float() * drop.view(B, 1, 1)
    return out


@torch.no_grad()
def occam(model: Backbone, clip: Clip, target: torch.Tensor,
          n_masks: int = 200, keep_prob: float = 0.5, batch: int = 16,
          generator: torch.Generator | None = None) -> torch.Tensor:
    """OccAM: importance = E_mask[ phi(masked) * 1(point kept) ] normalised.
    Random per-point Bernoulli masks (dropped points removed); O(n_masks) passes."""
    B, T, N, C = clip.points.shape
    weighted = torch.zeros(B, T, N, device=clip.device)
    count = torch.zeros(B, T, N, device=clip.device)
    for s0 in range(0, n_masks, batch):
        S = min(batch, n_masks - s0)
        keep = (torch.rand(S, B, T, N, generator=generator).to(clip.device) < keep_prob) & clip.mask
        pts = clip.points.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(S * B, T, N, C)
        probe = Clip(points=pts, mask=keep.reshape(S * B, T, N), regions=None, label=None)
        score = model.phi(probe, target.repeat(S)).view(S, B, 1, 1)
        weighted += (keep.float() * score).sum(0)
        count += keep.float().sum(0)
    return (weighted / count.clamp_min(1e-6)) * clip.mask


def extremal(model: Backbone, clip: Clip, target: torch.Tensor,
             area: float = 0.1, iters: int = 100, lr: float = 0.1) -> torch.Tensor:
    """Extremal perturbations: optimise a soft per-point mask m in [0,1] to MINIMISE
    the kept-region phi at fixed area (so the masked-out points are the important
    ones), with an area penalty. Returns (1 - m) as importance."""
    B, T, N, C = clip.points.shape
    w = clip.mask.unsqueeze(-1).float()                        # masked-out points shrink onto
    baseline_pts = ((clip.points * w).sum((1, 2), keepdim=True)    # the clip centroid (on-manifold
                    / w.sum((1, 2), keepdim=True).clamp_min(1)).expand_as(clip.points)  # for real nets)
    logits = torch.zeros(B, T, N, device=clip.device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    valid = clip.mask.float()
    for _ in range(iters):
        m = torch.sigmoid(logits) * valid
        pts = m.unsqueeze(-1) * clip.points + (1 - m).unsqueeze(-1) * baseline_pts
        probe = Clip(points=pts, mask=clip.mask, regions=clip.regions, label=clip.label)
        phi = model.phi(probe, target).sum()
        area_pen = ((m.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)) - area).clamp_min(0).sum()
        loss = phi + 10.0 * area_pen          # keep-mask small AND low phi -> kept pts unimportant
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        importance = (1 - torch.sigmoid(logits)) * valid
    return importance.detach()
