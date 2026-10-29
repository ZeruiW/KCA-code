"""Stage 1 proposal: cheap saliency to RANK regions before spending interventions.

Attention backbones supply aggregated spacetime attention via
`Backbone.proposal_saliency`; everything else uses the model-agnostic
gradient x activation fallback here. Attention is used ONLY to rank, never as the
explanation (paper Sec. method-eff).
"""
from __future__ import annotations

import torch

from .backbones.base import Backbone
from .types import Clip


def grad_times_input(model: Backbone, clip: Clip, target: torch.Tensor) -> torch.Tensor:
    """|d phi / d x  *  x| summed over channels -> (B, T, N) non-negative saliency.

    Forces grad locally so it works even inside a torch.no_grad() context (e.g. when
    called from the amortized head's O(1) inference path)."""
    with torch.enable_grad():
        pts = clip.points.detach().clone().requires_grad_(True)
        probe = Clip(points=pts, mask=clip.mask, regions=clip.regions, label=clip.label)
        phi = model.phi(probe, target).sum()
        grad, = torch.autograd.grad(phi, pts)
    sal = (grad * pts).abs().sum(-1)                       # (B,T,N)
    return (sal * clip.mask).detach()


def region_proposal(model: Backbone, clip: Clip, target: torch.Tensor) -> torch.Tensor:
    """Per-region proposal scores (B, m): mean point saliency inside each region."""
    point_sal = model.proposal_saliency(clip, target)
    if point_sal is None:
        if model.supports_grad():
            point_sal = grad_times_input(model, clip, target)
        else:                                              # last-resort: uniform
            point_sal = clip.mask.float()
    m = clip.num_regions()
    B = clip.B
    out = torch.zeros(B, m, device=clip.device)
    for r in range(m):
        sel = (clip.regions == r) & clip.mask             # (B,T,N)
        s = (point_sal * sel).sum(dim=(1, 2))
        cnt = sel.sum(dim=(1, 2)).clamp_min(1)
        out[:, r] = s / cnt
    return out


def topk_regions(scores: torch.Tensor, k: int, n_control: int = 0,
                 generator: torch.Generator | None = None) -> list[int]:
    """Region ids to intervene on: top-k by proposal + a random control set.

    Uses batch-mean ranking (one shared region set per batch keeps interventions
    vectorised). Returns a deduplicated list of region ids.
    """
    mean = scores.mean(0)                                  # (m,)
    m = mean.numel()
    k = min(k, m)
    top = torch.topk(mean, k).indices.tolist()
    chosen = set(top)
    if n_control > 0:
        rest = [r for r in range(m) if r not in chosen]
        if rest:
            idx = torch.randperm(len(rest), generator=generator)[:n_control]
            chosen.update(rest[i] for i in idx.tolist())
    return sorted(chosen)
