"""Causal scores and the motion-vs-geometry decomposition (paper Sec. method-causal,
method-decomp, method-theory).

All scores are differences of the scalar target phi(.) under interventions, so they
are defined for ANY backbone (attention, SSM, conv) — nothing here touches internals.

    Nec(R)  = phi(X) - phi(ablate(R) X)                       (Eq. necessity)
    Suf(R)  = phi(ablate(Omega\\R) X) - phi(X_empty)          (Eq. sufficiency)
    Smot(R) = phi(X) - phi(freeze(R) X)                       (motion necessity)
    Sgeo(R) = phi(freeze(R) X) - phi(ablate(R) X)             (geometry, motion held out)
  =>  Nec(R) = Smot(R) + Sgeo(R)   exactly  (Prop. completeness, by construction)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import interventions as itv
from .backbones.base import Backbone
from .types import Clip


@dataclass
class RegionScores:
    """Per-region causal scores for one batch. Each tensor is (B,) for region R."""

    region_id: int
    necessity: torch.Tensor      # Nec(R)
    smot: torch.Tensor           # Smot(R)
    sgeo: torch.Tensor           # Sgeo(R)
    sufficiency: torch.Tensor | None = None
    delta_int: torch.Tensor | None = None  # interaction residual (symmetrized scoring)

    @property
    def residual(self) -> torch.Tensor:
        """Decomposition residual eps = Nec - (Smot + Sgeo). Must be ~0 by
        construction; we return it as a numerical sanity check."""
        return self.necessity - (self.smot + self.sgeo)


@torch.no_grad()
def score_region(
    model: Backbone,
    clip: Clip,
    region_id: int,
    target: torch.Tensor,
    baseline: itv.BaselineKind = "nn",
    t0: int = 0,
    with_sufficiency: bool = False,
) -> RegionScores:
    """Compute Smot/Sgeo/Nec for one region via the telescoping in Eq. decomp.

    Three forward passes share the original phi:  phi(X), phi(freeze), phi(ablate).
    """
    phi = model.phi(clip, target)                               # phi(X)
    phi_frz = model.phi(itv.freeze(clip, region_id, t0=t0), target)
    phi_abl = model.phi(itv.ablate(clip, region_id, baseline=baseline, t0=t0), target)

    smot = phi - phi_frz                                        # motion alone
    sgeo = phi_frz - phi_abl                                    # then the static shape
    nec = phi - phi_abl                                         # = smot + sgeo

    suff = None
    if with_sufficiency:
        suff = _sufficiency(model, clip, region_id, target, baseline)
    return RegionScores(region_id, nec, smot, sgeo, suff)


@torch.no_grad()
def _sufficiency(model, clip, region_id, target, baseline) -> torch.Tensor:
    """Suf(R) = phi(keep only R) - phi(empty). Ablate the complement of R."""
    # ablate every region except R, then compare to the fully-baselined clip.
    keep = (clip.regions == region_id)
    comp = clip.clone()
    sel = (~keep) & clip.mask
    b = itv.baseline_fill(clip, sel, baseline)
    comp.points = torch.where(sel.unsqueeze(-1), b, comp.points)
    if baseline == "bg":
        comp.mask = comp.mask & ~sel
    phi_only = model.phi(comp, target)

    empty = clip.clone()
    allsel = clip.mask
    b_all = itv.baseline_fill(clip, allsel, baseline)
    empty.points = torch.where(allsel.unsqueeze(-1), b_all, empty.points)
    if baseline == "bg":
        empty.mask = torch.zeros_like(empty.mask)
    phi_empty = model.phi(empty, target)
    return phi_only - phi_empty


@torch.no_grad()
def symmetric_region_scores(
    model: Backbone,
    clip: Clip,
    region_id: int,
    target: torch.Tensor,
    baseline: itv.BaselineKind = "nn",
    t0: int = 0,
    with_sufficiency: bool = False,
) -> RegionScores:
    """Order-symmetrized scores over the FOUR vertices of the (motion, geometry) cube
    (Prop. interact). Both orders are exactly realizable because abl-geo (ablate shape,
    keep trajectory) provides the {motion-kept, geometry-removed} vertex:

        phi      = phi(X)                       (M, G)
        phi_frz  = phi(freeze)                  (-M, G)   motion removed, geometry kept
        phi_geo  = phi(abl-geo)                 (M, -G)   geometry removed, motion kept
        phi_abl  = phi(ablate)                  (-M, -G)

    motion-first:   Smot = phi-phi_frz,    Sgeo = phi_frz-phi_abl
    geometry-first: Sgeo' = phi-phi_geo,   Smot' = phi_geo-phi_abl
    symmetrized:    Sbar_mot=(Smot+Smot')/2, Sbar_geo=(Sgeo+Sgeo')/2  (sum = Nec exactly)
    interaction:    Delta_int = phi - phi_frz - phi_geo + phi_abl  (mixed 2nd difference /
                    Shapley interaction index; single-order scores deviate from the
                    symmetric ones by +/- Delta_int/2; = 0 iff motion & geometry act
                    additively, i.e. Assumption (S)).
    """
    phi = model.phi(clip, target)
    phi_frz = model.phi(itv.freeze(clip, region_id, t0=t0), target)
    phi_geo = model.phi(itv.ablate_keep_trajectory(clip, region_id, baseline=baseline, t0=t0), target)
    phi_abl = model.phi(itv.ablate(clip, region_id, baseline=baseline, t0=t0), target)

    nec = phi - phi_abl
    smot_mf, sgeo_mf = phi - phi_frz, phi_frz - phi_abl       # motion-first
    sgeo_gf, smot_gf = phi - phi_geo, phi_geo - phi_abl       # geometry-first
    smot_bar = 0.5 * (smot_mf + smot_gf)
    sgeo_bar = 0.5 * (sgeo_mf + sgeo_gf)
    delta_int = phi - phi_frz - phi_geo + phi_abl    # mixed 2nd difference (Shapley interaction)

    suff = _sufficiency(model, clip, region_id, target, baseline) if with_sufficiency else None
    return RegionScores(region_id, nec, smot_bar, sgeo_bar, suff, delta_int)


# ---------------------------------------------------------------------------------------
# Context-averaged (Banzhaf-type) estimator — robust to redundancy
# ---------------------------------------------------------------------------------------
# The local scores above are contrasts AT the observed input: if a decision-relevant
# structure is split over several regions that are individually redundant (removing any
# one leaves enough of the others), every local necessity is ~0. The context-averaged
# estimator averages the same contrasts over random CONTEXTS in which the other regions
# are themselves randomly intervened. Each region r carries two binary factors — motion
# kept M_r and geometry kept G_r — i.e. one vertex of its (M, G) cube:
#     (1,1) original   (0,1) freeze   (1,0) abl-geo   (0,0) ablate
# We draw S joint states with M_r, G_r ~ Bernoulli(q) independently, evaluate phi once per
# state (all regions intervened jointly), and fit
#     phi ~ b0 + sum_r [ mu_r M_r + gamma_r G_r + iota_r M_r G_r ]
# by (ridge) least squares. Then, exactly as for the local symmetric scores,
#     Smot(r) = mu_r + iota_r/2,  Sgeo(r) = gamma_r + iota_r/2,  Delta_int(r) = iota_r,
#     Nec(r)  = mu_r + gamma_r + iota_r = Smot(r) + Sgeo(r)          (completeness)
# i.e. the local order-symmetrised decomposition averaged over contexts (it coincides
# with it when phi has no cross-region interactions). Cost: S forward passes in total,
# independent of the number of regions.

STATE_ORIGINAL, STATE_FREEZE, STATE_ABLGEO, STATE_ABLATE = 0, 1, 2, 3


def apply_states(clip: Clip, region_ids, states, baseline: itv.BaselineKind = "proxy",
                 t0: int = 0) -> Clip:
    """Apply one intervention state per region, jointly (regions are disjoint). Reference
    (unvectorised) implementation; the estimators use `interventions.compose_states`."""
    out = clip
    for r, s in zip(region_ids, states):
        s = int(s)
        if s == STATE_FREEZE:
            out = itv.freeze(out, r, t0=t0)
        elif s == STATE_ABLGEO:
            out = itv.ablate_keep_trajectory(out, r, baseline=baseline, t0=t0)
        elif s == STATE_ABLATE:
            out = itv.ablate(out, r, baseline=baseline, t0=t0)
    return out


@torch.no_grad()
def _phi_states(model: Backbone, clip: Clip, views, region_state: torch.Tensor,
                target: torch.Tensor, batch: int, ablate: str = "proxy") -> torch.Tensor:
    out = []
    for s in range(0, region_state.shape[0], batch):
        bc = itv.compose_states(clip, views, region_state[s:s + batch], ablate=ablate)
        out.append(model.phi(bc, target.expand(bc.B)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def local_region_scores(model: Backbone, clip: Clip, region_ids, target: torch.Tensor,
                        t0: int = 0, batch: int = 32, blob: float = 0.0,
                        views=None, ablate: str = "proxy") -> dict[int, RegionScores]:
    """Batched order-symmetrised LOCAL scores, identical to calling
    `symmetric_region_scores(..., baseline=ablate)` region by region (ablate = "proxy"
    or "bg"): the 3k one-region states (freeze / abl-geo / ablate) plus the original go
    through the model in batches."""
    assert clip.B == 1
    region_ids = list(region_ids)
    m = clip.num_regions()
    views = views or itv.region_views(clip, t0=t0, blob=blob)
    st = torch.zeros(1 + 3 * len(region_ids), m, dtype=torch.long)
    for j, r in enumerate(region_ids):
        st[1 + 3 * j, r], st[2 + 3 * j, r], st[3 + 3 * j, r] = STATE_FREEZE, STATE_ABLGEO, STATE_ABLATE
    y = _phi_states(model, clip, views, st, target, batch, ablate)
    phi = y[0]
    out = {}
    dev = clip.device
    for j, r in enumerate(region_ids):
        frz, geo, abl = y[1 + 3 * j], y[2 + 3 * j], y[3 + 3 * j]
        smot = 0.5 * ((phi - frz) + (geo - abl))
        sgeo = 0.5 * ((frz - abl) + (phi - geo))
        out[r] = RegionScores(r, (phi - abl).view(1).to(dev), smot.view(1).to(dev),
                              sgeo.view(1).to(dev), None, (phi - frz - geo + abl).view(1).to(dev))
    return out


@torch.no_grad()
def context_region_scores(model: Backbone, clip: Clip, region_ids, target: torch.Tensor,
                          n_samples: int = 128, q: float = 0.5, t0: int = 0,
                          batch: int = 32, ridge: float = 1e-3, blob: float = 0.0,
                          generator: torch.Generator | None = None,
                          views=None, ablate: str = "proxy") -> dict[int, RegionScores]:
    """Context-averaged Smot/Sgeo/Nec/Delta_int for `region_ids` (B = 1 clip; ablate =
    "proxy" or "bg"). Regions outside `region_ids` stay at their original state."""
    assert clip.B == 1, "context estimator explains one clip at a time"
    region_ids = list(region_ids)
    k, m = len(region_ids), clip.num_regions()
    g = generator or torch.Generator().manual_seed(0)
    Mk = (torch.rand(n_samples, k, generator=g) < q)
    Gk = (torch.rand(n_samples, k, generator=g) < q)
    sub = torch.where(Mk & Gk, STATE_ORIGINAL, torch.where(Gk, STATE_FREEZE,
                      torch.where(Mk, STATE_ABLGEO, STATE_ABLATE)))
    st = torch.zeros(n_samples, m, dtype=torch.long)
    st[:, region_ids] = sub
    views = views or itv.region_views(clip, t0=t0, blob=blob)
    y = _phi_states(model, clip, views, st, target, batch, ablate)   # (S,)
    Mf, Gf = Mk.float(), Gk.float()
    X = torch.cat([torch.ones(n_samples, 1), Mf, Gf, Mf * Gf], 1)   # (S, 3k+1)
    A = X.T @ X + ridge * torch.eye(3 * k + 1)
    A[0, 0] -= ridge                                           # do not shrink the intercept
    coef = torch.linalg.solve(A, X.T @ y)
    mu, gam, iota = coef[1:k + 1], coef[k + 1:2 * k + 1], coef[2 * k + 1:]
    out = {}
    dev = clip.device
    for j, r in enumerate(region_ids):
        smot = (mu[j] + iota[j] / 2).view(1).to(dev)
        sgeo = (gam[j] + iota[j] / 2).view(1).to(dev)
        out[r] = RegionScores(r, smot + sgeo, smot, sgeo, None, iota[j].view(1).to(dev))
    return out
