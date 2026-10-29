"""Soundness / sanity metrics (paper Sec. exp-ablation).

  Infidelity (Yeh'19): E[(I^T d - (phi(x) - phi(x - d)))^2] over random perturbations
    d; lower = the attribution better predicts the output change. (I = attribution.)
  Max-Sensitivity (Yeh'19): max change in attribution under small input noise; lower
    = more stable.
  Adebayo sanity: caller randomizes the model weights and checks that the
    attribution--original rank correlation collapses toward 0; we provide the
    rank-correlation helper.
"""
from __future__ import annotations

import torch

from ..backbones.base import Backbone
from ..types import Clip


@torch.no_grad()
def infidelity(
    model: Backbone, clip: Clip, field: torch.Tensor, target: torch.Tensor,
    n_pert: int = 20, sigma: float = 0.1, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(B,) infidelity. Perturbation d removes a random point subset (sets to baseline);
    I^T d = summed attribution of the removed points."""
    B, T, N, C = clip.points.shape
    P = T * N
    valid = clip.mask.reshape(B, -1)
    field_flat = field.reshape(B, -1)
    baseline_pts = torch.full_like(clip.points, -1.5)
    phi_full = model.phi(clip, target)

    acc = torch.zeros(B)
    for _ in range(n_pert):
        r = torch.rand(B, P, generator=generator, device=clip.device)
        r[~valid] = 2.0
        drop = r < sigma                                    # removed subset
        It_d = (field_flat * drop).sum(1)                   # predicted change
        keep = valid & ~drop
        pts = torch.where(keep.view(B, T, N).unsqueeze(-1), clip.points, baseline_pts)
        probe = Clip(points=pts, mask=clip.mask & keep.view(B, T, N),
                     regions=clip.regions, label=clip.label)
        actual = phi_full - model.phi(probe, target)        # true change
        acc += (It_d - actual) ** 2
    return acc / n_pert


@torch.no_grad()
def max_sensitivity(
    explain_fn, clip: Clip, field: torch.Tensor,
    n_pert: int = 10, radius: float = 0.02, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(B,) max relative change of the attribution under small input jitter.

    explain_fn: Clip -> (B,T,N) attribution field, so we can re-explain perturbed inputs.
    """
    B = field.shape[0]
    base = field.reshape(B, -1)
    base_norm = base.norm(dim=1).clamp_min(1e-8)
    worst = torch.zeros(B)
    for _ in range(n_pert):
        noise = torch.randn(clip.points.shape, generator=generator, device=clip.device) * radius
        jittered = Clip(points=clip.points + noise, mask=clip.mask,
                        regions=clip.regions, label=clip.label)
        f2 = explain_fn(jittered).reshape(B, -1)
        diff = (f2 - base).norm(dim=1) / base_norm
        worst = torch.maximum(worst, diff)
    return worst


def _avg_rank(x: torch.Tensor) -> torch.Tensor:
    from scipy.stats import rankdata
    return torch.from_numpy(rankdata(x.detach().cpu().double().numpy())).float()


def rank_correlation(field_a: torch.Tensor, field_b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) Spearman rank correlation between two attribution fields over valid points.
    Used for the Adebayo model-randomization test: should collapse to ~0 when the
    backbone's weights are randomized."""
    B = field_a.shape[0]
    out = torch.zeros(B)
    for b in range(B):
        v = mask[b].reshape(-1)
        a = field_a[b].reshape(-1)[v]
        c = field_b[b].reshape(-1)[v]
        if a.numel() < 2:
            continue
        ra = _avg_rank(a)                                   # average ranks: region-constant
        rc = _avg_rank(c)                                   # fields have many ties
        ra = ra - ra.mean(); rc = rc - rc.mean()
        out[b] = (ra * rc).sum() / (ra.norm() * rc.norm() + 1e-8)
    return out
