"""The three spacetime do-operators (paper Sec. method-interventions).

Each takes a Clip and a region id, and returns a NEW intervened Clip. Off-region
points and other context are left untouched (interventional validity, Assumption I).

    ablate   : replace the region's points by a content-free baseline b
               -> removes presence AND motion.
    freeze   : render the region static at its frame-t0 configuration
               -> removes motion, KEEPS geometry. (the novel operator)
    displace : perturb the region's trajectory by xi (reverse / scramble / noise)
               -> counterfactual motion probe.

CORRESPONDENCE-FREE. Real point-cloud videos have no point identity across frames
(every frame is sampled independently), so operators never assume that index i at frame
t and index i at frame t' are the same physical point. They only use the region's
per-frame point SET R_t (a tracked region, `regions.tracked_regions`): e.g. freeze fills
the slots of R_t with points of R_t0. Slots are paired in index order (slot j of R_t <-
point j mod |R_t0| of R_t0), so on data that DOES keep identity (the synthetic tasks) the
result equals the classic index-aligned p_{t,i} <- p_{t0,i} exactly.
"""
from __future__ import annotations

from typing import Literal

import torch

from .types import Clip

BaselineKind = Literal["proxy", "bg", "mean", "nn"]


def _region_sel(clip: Clip, region_id: int) -> torch.Tensor:
    """(B, T, N) bool mask selecting valid points of `region_id`."""
    if clip.regions is None:
        raise ValueError("intervention requires a region partition")
    return (clip.regions == region_id) & clip.mask


def baseline_fill(clip: Clip, sel: torch.Tensor, kind: BaselineKind) -> torch.Tensor:
    """Compute the content-free baseline value(s) `b` for the selected points.

    Returns a (B, T, N, C) tensor; only positions where `sel` is True are used.
    - bg   : push points to a fixed out-of-scene location (scene-background fill).
    - mean : per-clip mean of all valid points (feature mean).
    - nn   : distribution-aware nearest-neighbour fill — copy each ablated point's
             value from a random *valid non-region* point in the same frame.
    """
    B, T, N, C = clip.points.shape
    if kind == "bg":
        # a constant just outside the unit cube the data is normalised to.
        b = torch.full_like(clip.points, -1.5)
        return b
    if kind == "mean":
        valid = clip.mask.unsqueeze(-1)                      # (B,T,N,1)
        s = (clip.points * valid).sum(dim=(1, 2))            # (B,C)
        cnt = valid.sum(dim=(1, 2)).clamp_min(1)             # (B,1)
        mean = (s / cnt).view(B, 1, 1, C)
        return mean.expand(B, T, N, C).clone()
    if kind == "nn":
        # for each frame, donors = valid points NOT in the region; sample with
        # replacement to fill the holes. falls back to mean if a frame has none.
        donor = clip.mask & ~sel                             # (B,T,N)
        out = clip.points.clone()
        idx_all = torch.arange(N, device=clip.device)
        for b in range(B):
            for t in range(T):
                d = idx_all[donor[b, t]]
                if d.numel() == 0:
                    continue
                pick = d[torch.randint(d.numel(), (N,), device=clip.device)]
                out[b, t] = clip.points[b, t, pick]
        return out
    raise ValueError(f"unknown baseline kind {kind!r}")


def ablate(clip: Clip, region_id: int, baseline: BaselineKind = "nn", t0: int = 0) -> Clip:
    """Remove the region's geometry AND motion.

    baseline "proxy": replace the region by its geometry-free proxy (a point mass) held
        STATIC at the region's frame-t0 centroid — the (M=0, G=0) vertex that pairs with
        abl-geo's moving proxy, so every edge of the (M, G) cube changes exactly one
        factor (the default for the decomposition).
    "bg": true removal (points masked off); "mean"/"nn": feature-mean / donor fill.
    """
    if baseline == "proxy":
        return ablate_keep_trajectory(clip, region_id, t0=t0, static=True)
    sel = _region_sel(clip, region_id)                        # (B,T,N)
    b = baseline_fill(clip, sel, baseline)
    out = clip.clone()
    s = sel.unsqueeze(-1)
    out.points = torch.where(s, b, out.points)
    if baseline == "bg":
        # background fill also drops the points from the valid set (true removal)
        out.mask = out.mask & ~sel
    return out


def _fill_from_frames(clip: Clip, region_id: int, src_frame) -> Clip:
    """Fill the slots of R_t, for every frame t, with the points of R_{src_frame(t)}
    (paired in index order, cyclically when the counts differ). A frame where the region
    is absent stays absent; an empty source frame is replaced by the nearest non-empty one."""
    sel = _region_sel(clip, region_id)                        # (B,T,N)
    out = clip.clone()
    B, T = sel.shape[:2]
    for b in range(B):
        idx = [torch.nonzero(sel[b, t]).squeeze(1) for t in range(T)]
        nonempty = [t for t in range(T) if idx[t].numel()]
        if not nonempty:
            continue
        for t in range(T):
            dst = idx[t]
            if dst.numel() == 0:
                continue
            s = src_frame(t)
            if idx[s].numel() == 0:
                s = min(nonempty, key=lambda u: abs(u - s))
            src = idx[s]
            pick = src[torch.arange(dst.numel(), device=clip.device) % src.numel()]
            out.points[b, t, dst] = clip.points[b, s, pick]
    return out


def freeze(clip: Clip, region_id: int, t0: int = 0) -> Clip:
    """Render the region static at its frame-t0 configuration: R_t <- R_{t0} for all t.

    Removes the region's motion (translation, rotation AND deformation) while keeping
    its geometry; correspondence-free (uses only the per-frame point sets of a tracked
    region), reducing to p_{t,i} <- p_{t0,i} when point identity exists.
    """
    return _fill_from_frames(clip, region_id, lambda t: t0)


def _centroids(clip: Clip, sel: torch.Tensor) -> torch.Tensor:
    """(B,T,3) per-frame centroid of the selected points (nan where empty)."""
    w = sel.unsqueeze(-1).float()
    return (clip.points[..., :3] * w).sum(2) / w.sum(2)


def ablate_keep_trajectory(clip: Clip, region_id: int,
                           baseline: BaselineKind = "nn", t0: int = 0,
                           blob: float = 0.0, static: bool = False) -> Clip:
    """abl-geo: remove the region's GEOMETRY but KEEP its motion.

    The region is replaced by a geometry-free proxy carried along its centroid
    trajectory c_t: p_{t,j} <- c_t + sigma * eps_j with slot-fixed isotropic offsets
    eps_j and sigma = blob * (RMS radius of R_t0) / sqrt(3). The default blob=0 is a
    point mass (no shape, no extent); blob=1 keeps the region's size but not its shape.
    This realises the {motion kept, geometry removed} vertex of the (M, G) cube, so the
    order-symmetrised decomposition (Prop. interact) is well defined, and the proxy stays
    where the region is. (`baseline` is accepted for API symmetry.) Correspondence-free.
    `static=True` holds the proxy at the frame-t0 centroid instead (= ablate "proxy").
    """
    sel = _region_sel(clip, region_id)
    out = clip.clone()
    B, T, N = sel.shape
    cen = _centroids(clip, sel)                               # (B,T,3)
    g = torch.Generator(device="cpu").manual_seed(7919 + int(region_id))
    eps = torch.randn(N, 3, generator=g).to(clip.device)      # slot-fixed offsets
    for b in range(B):
        present = [t for t in range(T) if sel[b, t].any()]
        if not present:
            continue
        ta = t0 if sel[b, t0].any() else present[0]
        p0 = clip.points[b, ta, sel[b, ta], :3]
        sigma = blob * (p0 - p0.mean(0)).pow(2).sum(-1).mean().sqrt() / (3 ** 0.5)
        for t in present:
            dst = torch.nonzero(sel[b, t]).squeeze(1)
            c = cen[b, ta] if static else cen[b, t]
            out.points[b, t, dst, :3] = c + sigma * eps[:dst.numel()]
    return out


def displace(
    clip: Clip,
    region_id: int,
    mode: Literal["reverse", "scramble", "noise"] = "reverse",
    sigma: float = 0.05,
    generator: torch.Generator | None = None,
) -> Clip:
    """Counterfactual motion: perturb the region's trajectory, keep per-frame geometry.

    - reverse  : play the region's frames backwards in time (R_t <- R_{T-1-t}).
    - scramble : random temporal permutation of the region's frames.
    - noise    : rigidly translate the region by per-frame gaussian offsets (trajectory
                 noise; its shape in each frame is untouched).
    Correspondence-free like freeze.
    """
    T = clip.T
    if mode == "reverse":
        return _fill_from_frames(clip, region_id, lambda t: T - 1 - t)
    if mode == "scramble":
        perm = torch.randperm(T, generator=generator).tolist()
        return _fill_from_frames(clip, region_id, lambda t: perm[t])
    if mode == "noise":
        sel = _region_sel(clip, region_id)
        out = clip.clone()
        off = torch.randn(clip.B, T, 1, 3, generator=generator).to(clip.device) * sigma
        out.points[..., :3] = torch.where(sel.unsqueeze(-1), clip.points[..., :3] + off, clip.points[..., :3])
        return out
    raise ValueError(f"unknown displace mode {mode!r}")


# ---------------------------------------------------------------------------------------
# Vectorised joint interventions (used by the batched estimators in scores.py)
# ---------------------------------------------------------------------------------------
# freeze / abl-geo / bg-ablate act on disjoint regions slot by slot, and each region's
# new values depend only on that region's own points, so the clip with "every region
# frozen" (resp. abl-geo) holds each region's intervened values in its own slots. Any
# joint state — one of {original, freeze, abl-geo, ablate} per region — is then a
# per-slot selection between three precomputed views, for a whole batch of states at once.

def region_views(clip: Clip, t0: int = 0, blob: float = 0.0) -> tuple[torch.Tensor, ...]:
    """(frozen, ablgeo, proxy-static) point tensors (B,T,N,C): every region frozen /
    replaced by its moving proxy / replaced by its static proxy."""
    frz, geo, abl = clip.clone(), clip.clone(), clip.clone()
    for r in range(clip.num_regions()):
        frz = freeze(frz, r, t0=t0)
        geo = ablate_keep_trajectory(geo, r, t0=t0, blob=blob)
        abl = ablate_keep_trajectory(abl, r, t0=t0, blob=blob, static=True)
    return frz.points, geo.points, abl.points


def compose_states(clip: Clip, views: tuple[torch.Tensor, torch.Tensor],
                   region_state: torch.Tensor, ablate: str = "proxy") -> Clip:
    """Batch of jointly intervened copies of a single clip (B = 1).

    region_state: (S, m) long in {0 original, 1 freeze, 2 abl-geo, 3 ablate}; ablate is
    the static proxy ("proxy", default) or true removal ("bg", points masked off)."""
    assert clip.B == 1
    S = region_state.shape[0]
    reg = clip.regions[0]                                       # (T,N)
    valid = (reg >= 0) & clip.mask[0]
    st = torch.zeros(S, *reg.shape, dtype=torch.long, device=clip.device)
    st[:, valid] = region_state.to(clip.device)[:, reg[valid]]
    frz, geo, abl = views
    pts = clip.points.expand(S, -1, -1, -1)
    pts = torch.where((st == 1).unsqueeze(-1), frz.expand(S, -1, -1, -1), pts)
    pts = torch.where((st == 2).unsqueeze(-1), geo.expand(S, -1, -1, -1), pts)
    if ablate == "proxy":
        pts = torch.where((st == 3).unsqueeze(-1), abl.expand(S, -1, -1, -1), pts)
        mask = clip.mask.expand(S, -1, -1)
    else:
        mask = clip.mask.expand(S, -1, -1) & (st != 3)
    return Clip(points=pts.contiguous(), mask=mask.contiguous(),
                regions=clip.regions.expand(S, -1, -1).contiguous(),
                label=None if clip.label is None else clip.label.expand(S))
