"""Faithfulness metrics (paper Sec. exp-setup, Tab. faithfulness).

All operate through the Backbone interface on a per-point importance field
(B, T, N) and a Clip. Points are removed/inserted by replacing them with the
content-free baseline (mask off), matching the deletion/insertion protocol of
RISE adapted to point clouds (point-drop, Petsiuk'18 / Zheng ICCV'19).

Conventions: Deletion AUC lower is better; Insertion AUC higher is better. The scored
quantity is the target-class PROBABILITY (softmax), as in RISE / ERASER — bounded and
comparable across clips and backbones (`score="logit"` for the raw logit).
"""
from __future__ import annotations

import torch

from ..backbones.base import Backbone
from ..types import Clip


def _order_points(field: torch.Tensor, mask: torch.Tensor, descending: bool) -> torch.Tensor:
    """Flatten (T,N) per clip and return point indices sorted by importance.

    Returns (B, T*N) long index into the flattened (T,N) grid; invalid points are
    pushed to the end so they are removed last / inserted never.
    """
    B = field.shape[0]
    flat = field.reshape(B, -1).clone()
    valid = mask.reshape(B, -1)
    # invalid points: send to the extreme so they sort last regardless of direction
    flat[~valid] = float("-inf") if descending else float("inf")
    return flat.argsort(dim=1, descending=descending)


def _importance_rank(order: torch.Tensor) -> torch.Tensor:
    """(B,P) rank of each flat position: 0 = most important. Invariant across thresholds,
    so callers compute it once."""
    B, P = order.shape
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(P, device=order.device).unsqueeze(0).expand(B, P))
    return rank


def _phi_with_kept(model, clip, target, keep_flat, baseline_pts):
    """phi of a clip where only `keep_flat` (B,T*N bool) points remain; rest -> baseline."""
    B, T, N, C = clip.points.shape
    keep = keep_flat.view(B, T, N)
    pts = torch.where(keep.unsqueeze(-1), clip.points, baseline_pts)
    m = clip.mask & keep
    probe = Clip(points=pts, mask=m, regions=clip.regions, label=clip.label)
    return model.phi(probe, target)


def _score(model, clip: Clip, target: torch.Tensor, score: str = "prob") -> torch.Tensor:
    if score == "logit":
        return model.phi(clip, target)
    return model.logits(clip).softmax(-1).gather(1, target.view(-1, 1)).squeeze(1)


def _phi_keeps(model, clip: Clip, target: torch.Tensor, keeps: torch.Tensor,
               batch: int = 32, score: str = "prob") -> torch.Tensor:
    """phi of the clip with only `keeps` points left, for S keep-sets at once.
    keeps: (S, B, T*N) bool -> (S, B). Removed points are masked off (true removal)."""
    S, B, P = keeps.shape
    T, N, C = clip.T, clip.N, clip.C
    step = max(1, batch // B)
    out = []
    for s0 in range(0, S, step):
        k = keeps[s0:s0 + step]
        s = k.shape[0]
        kk = k.reshape(s * B, T, N)
        pts = clip.points.unsqueeze(0).expand(s, -1, -1, -1, -1).reshape(s * B, T, N, C)
        pts = torch.where(kk.unsqueeze(-1), pts, torch.full_like(pts, -1.5))
        m = clip.mask.unsqueeze(0).expand(s, -1, -1, -1).reshape(s * B, T, N) & kk
        out.append(_score(model, Clip(points=pts, mask=m), target.repeat(s), score).view(s, B))
    return torch.cat(out, 0)


@torch.no_grad()
def deletion_insertion(
    model: Backbone, clip: Clip, field: torch.Tensor, target: torch.Tensor,
    steps: int = 20, score: str = "prob",
) -> dict:
    """Deletion & Insertion AUC. Removes/inserts points in importance order.

    Deletion: start from full clip, remove most-important first -> phi should drop
    fast (low AUC = good). Insertion: start empty, add most-important first -> phi
    should rise fast (high AUC = good). Returns dict with curves and AUCs.
    """
    B = clip.B
    order = _order_points(field, clip.mask, descending=True)     # (B, P) most-imp first
    n_valid = clip.mask.reshape(B, -1).sum(1)                    # (B,)
    valid_flat = clip.mask.reshape(B, -1)
    rank = _importance_rank(order)                              # (B,P) 0 = most important
    fracs = torch.linspace(0, 1, steps + 1)
    removed = torch.stack([rank < (f * n_valid).round().long().unsqueeze(1) for f in fracs])  # (S,B,P)
    del_c = _phi_keeps(model, clip, target, ~removed & valid_flat, score=score).T   # (B, steps+1)
    ins_c = _phi_keeps(model, clip, target, removed & valid_flat, score=score).T
    del_auc = torch.trapz(del_c, fracs.to(del_c), dim=1)
    ins_auc = torch.trapz(ins_c, fracs.to(ins_c), dim=1)
    return {"deletion_auc": del_auc, "insertion_auc": ins_auc,
            "deletion_curve": del_c, "insertion_curve": ins_c, "fracs": fracs}


@torch.no_grad()
def aopc_comp_suff(
    model: Backbone, clip: Clip, field: torch.Tensor, target: torch.Tensor,
    percents=(1, 5, 10, 20, 50), score: str = "prob",
) -> dict:
    """Comprehensiveness & Sufficiency in AOPC form (ERASER, DeYoung'20).

    Comprehensiveness = mean over k% of [phi(X) - phi(X without top-k%)]  (higher better)
    Sufficiency       = mean over k% of [phi(X) - phi(X with only top-k%)] (lower better)
    """
    B = clip.B
    order = _order_points(field, clip.mask, descending=True)
    n_valid = clip.mask.reshape(B, -1).sum(1)
    valid_flat = clip.mask.reshape(B, -1)
    rank = _importance_rank(order)
    phi_full = _score(model, clip, target, score)
    top = torch.stack([(rank < (pc / 100.0 * n_valid).round().long().unsqueeze(1)) & valid_flat
                       for pc in percents])                                # (S,B,P)
    phi_wo = _phi_keeps(model, clip, target, valid_flat & ~top, score=score)   # (S,B)
    phi_only = _phi_keeps(model, clip, target, top, score=score)
    return {"comprehensiveness": (phi_full - phi_wo).mean(0),
            "sufficiency": (phi_full - phi_only).mean(0)}


@torch.no_grad()
def sensitivity_n(
    model: Backbone, clip: Clip, field: torch.Tensor, target: torch.Tensor,
    n_subsets: int = 50, subset_frac: float = 0.1,
    generator: torch.Generator | None = None, score: str = "prob",
) -> torch.Tensor:
    """Sensitivity-n (Ancona'18): Pearson corr between summed attribution of a random
    point subset and the actual phi drop when that subset is removed. Returns (B,)."""
    B, T, N, C = clip.points.shape
    P = T * N
    valid_flat = clip.mask.reshape(B, -1)
    field_flat = field.reshape(B, -1)
    phi_full = _score(model, clip, target, score)
    r = torch.rand(n_subsets, B, P, generator=generator).to(clip.device)
    r[:, ~valid_flat] = 2.0
    subset = r < subset_frac                                             # (S,B,P)
    attr_sums = (field_flat.unsqueeze(0) * subset).sum(-1).T             # (B,S)
    phi_drops = (phi_full.unsqueeze(0) - _phi_keeps(model, clip, target, valid_flat & ~subset, score=score)).T
    a = attr_sums - attr_sums.mean(1, keepdim=True)
    d = phi_drops - phi_drops.mean(1, keepdim=True)
    num = (a * d).sum(1)
    den = a.norm(dim=1) * d.norm(dim=1) + 1e-8
    return num / den
