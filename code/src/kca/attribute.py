"""KCA explainer: the 3-stage algorithm (paper Alg. kca).

  Stage 1  proposal   -> rank regions, pick top-k (+ control)
  Stage 2  sparse interventions -> Smot/Sgeo/Nec on the chosen regions  [O(k) passes]
  (Stage 3 amortization lives in amortized.py — distils a head for O(1) inference.)

Produces per-region scores AND per-point geometry/motion fields by scattering each
region's score onto its member points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import interventions as itv
from . import proposal as prop
from . import scores as sc
from .backbones.base import Backbone
from .types import Clip


@dataclass
class Attribution:
    smot_field: torch.Tensor          # (B, T, N) per-point motion importance
    sgeo_field: torch.Tensor          # (B, T, N) per-point geometry importance
    region_scores: dict[int, sc.RegionScores] = field(default_factory=dict)
    n_forward: int = 0                # forward passes spent (efficiency reporting)

    @property
    def nec_field(self) -> torch.Tensor:
        return self.smot_field + self.sgeo_field

    def mean_abs_interaction(self) -> float:
        """Mean |Delta_int| over scored regions (decomposition-validity diagnostic,
        Prop. interact). Small => motion and geometry act near-additively."""
        ds = [rs.delta_int.abs().mean().item() for rs in self.region_scores.values()
              if rs.delta_int is not None]
        return sum(ds) / len(ds) if ds else 0.0


def explain(
    model: Backbone,
    clip: Clip,
    target: torch.Tensor | None = None,
    k: int = 16,
    n_control: int = 4,
    baseline: itv.BaselineKind = "proxy",
    t0: int = 0,
    symmetric: bool = True,
    generator: torch.Generator | None = None,
    estimator: str = "local",
    n_samples: int = 128,
    q: float = 0.5,
) -> Attribution:
    """Run KCA on a batch. If `target` is None, explains the predicted class.

    `estimator="local"`: contrasts at the observed input. `symmetric=True` (default)
    returns the order-symmetrized decomposition (4 vertices per region, robust to
    motion-geometry non-additivity and reporting Delta_int); `symmetric=False` uses the
    cheaper motion-first telescoping (3 vertices). O(k) passes.
    `estimator="context"`: the same symmetric decomposition averaged over random
    contexts of the other chosen regions (`scores.context_region_scores`, n_samples
    passes, keep-probability q) — robust when a structure spans many regions."""
    if clip.regions is None:
        raise ValueError("explain() needs a region partition; call regions.attach_regions first")
    if target is None:
        target = model.predicted_class(clip)
    if estimator == "context" and clip.B > 1:                  # one clip at a time
        parts = [explain(model, Clip(points=clip.points[b:b + 1], mask=clip.mask[b:b + 1],
                                     regions=clip.regions[b:b + 1],
                                     label=None if clip.label is None else clip.label[b:b + 1]),
                         target[b:b + 1], k=k, n_control=n_control, baseline=baseline, t0=t0,
                         symmetric=symmetric, generator=generator, estimator=estimator,
                         n_samples=n_samples, q=q) for b in range(clip.B)]
        return Attribution(torch.cat([p.smot_field for p in parts]),
                           torch.cat([p.sgeo_field for p in parts]),
                           parts[0].region_scores, n_forward=sum(p.n_forward for p in parts))

    # Stage 1: proposal (1 pass, or 1 backward for grad x act)
    pscores = prop.region_proposal(model, clip, target)
    chosen = prop.topk_regions(pscores, k=k, n_control=n_control, generator=generator)

    # Stage 2: sparse causal interventions, O(k) regions (3 or 4 passes each + shared phi)
    B, T, N = clip.mask.shape
    smot_f = torch.zeros(B, T, N, device=clip.device)
    sgeo_f = torch.zeros(B, T, N, device=clip.device)
    region_scores: dict[int, sc.RegionScores] = {}
    n_fwd = 1
    batched = baseline in ("proxy", "bg") and clip.B == 1 and symmetric
    if estimator == "context":
        if baseline not in ("proxy", "bg"):
            raise ValueError("the context estimator supports the proxy / bg ablations")
        region_scores = sc.context_region_scores(model, clip, chosen, target, n_samples=n_samples,
                                                 q=q, t0=t0, generator=generator, ablate=baseline)
        n_fwd += n_samples
    elif batched:                                   # same numbers, one batched pass
        region_scores = sc.local_region_scores(model, clip, chosen, target, t0=t0, ablate=baseline)
        n_fwd += 3 * len(chosen)
    for r in chosen:
        if estimator == "context" or batched:
            rs = region_scores[r]
        elif symmetric:
            rs = sc.symmetric_region_scores(model, clip, r, target, baseline=baseline, t0=t0)
            n_fwd += 3
        else:
            rs = sc.score_region(model, clip, r, target, baseline=baseline, t0=t0)
            n_fwd += 2
        region_scores[r] = rs
        sel = (clip.regions == r) & clip.mask
        smot_f += sel.float() * rs.smot.view(B, 1, 1)
        sgeo_f += sel.float() * rs.sgeo.view(B, 1, 1)
    return Attribution(smot_f, sgeo_f, region_scores, n_forward=n_fwd)
