"""Stage 3: the amortized explainer head (paper Sec. method-eff, Alg. line 'optional').

Distills a light MLP g_psi that regresses (Sgeo, Smot) per region from cheap
per-region features (proposal saliency + simple geometry/motion summaries), trained
on Stage-2 exact scores. Inference is then one forward pass + g_psi, i.e. O(1) extra
cost instead of O(k) interventions.

The head is backbone-agnostic: features are read through the Backbone interface and
the region partition, never from architecture internals.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbones.base import Backbone
from .types import Clip
from . import proposal as prop
from . import scores as sc
from . import interventions as itv


def region_features(model: Backbone, clip: Clip, target: torch.Tensor,
                    use_internal: bool = True) -> torch.Tensor:
    """(m, F) per-region descriptors for the amortized head. Always includes 4 cheap
    input descriptors [mean proposal saliency, log #point-frames, mean displacement,
    spatial extent]; when the backbone exposes internal `point_features`, the per-region
    mean of those is concatenated (the paper's 'distil from internal features' — far more
    predictive of necessity than input descriptors alone)."""
    pscore = prop.region_proposal(model, clip, target)         # (B, m)
    m = clip.num_regions()
    xyz = clip.points[..., :3]
    step = (xyz[:, 1:] - xyz[:, :-1]).norm(dim=-1)             # (B,T-1,N)
    both = (clip.mask[:, 1:] & clip.mask[:, :-1]).float()
    feats = []
    for r in range(m):
        sel = (clip.regions == r) & clip.mask                  # (B,T,N)
        cnt = sel.sum().clamp_min(1).float()
        sel_pair = sel[:, 1:] & clip.mask[:, :-1]
        disp = (step * both * sel_pair.float()).sum() / cnt
        p = xyz[sel]                                            # (n,3)
        extent = (p.max(0).values - p.min(0).values).norm() if p.numel() else torch.tensor(0.)
        feats.append(torch.stack([pscore[:, r].mean(), cnt.log1p(), disp, extent]))
    feats = torch.stack(feats)                                 # (m, 4)
    if use_internal and hasattr(model, "point_features"):
        pf = model.point_features(clip)                        # (B,T,N,D) internal
        D = pf.shape[-1]
        internal = torch.zeros(m, D, device=clip.device)
        for r in range(m):
            sel = (clip.regions == r) & clip.mask
            if sel.any():
                internal[r] = pf[sel].mean(0)
        feats = torch.cat([feats, internal], dim=1)            # (m, 4+D)
    return feats


class AmortizedHead(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),                              # -> (Sgeo, Smot)
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)                                 # (m, 2)


def distill(model: Backbone, dataset, target_fn=None, k: int = 16,
            epochs: int = 200, lr: float = 1e-3, limit: int | None = None,
            baseline: itv.BaselineKind = "nn", use_internal: bool = True) -> AmortizedHead:
    """Train g_psi on Stage-2 exact (Sgeo, Smot) labels over a dataset split.

    For each clip: compute features for all regions and exact scores for the top-k
    proposed regions; regress features->scores. The head's input width is inferred from
    the features (4 input descriptors, +D internal if the backbone exposes them).
    """
    X, Y = [], []
    for batch in dataset.batches(limit=limit):
        clip = batch.clip
        target = (batch.clip.label if target_fn is None else target_fn(model, clip))
        if target is None:
            target = model.predicted_class(clip)
        feats = region_features(model, clip, target, use_internal=use_internal)
        pscores = prop.region_proposal(model, clip, target)
        chosen = prop.topk_regions(pscores, k=k, n_control=0)
        for r in chosen:
            rs = sc.score_region(model, clip, r, target, baseline=baseline)
            X.append(feats[r])
            Y.append(torch.stack([rs.sgeo.mean(), rs.smot.mean()]))
    if not X:
        raise RuntimeError("no training pairs for amortized head")
    # features/labels are constant training data — detach the proposal autograd graph
    X = torch.stack(X).detach(); Y = torch.stack(Y).detach()
    head = AmortizedHead(in_dim=X.shape[1])
    head._use_internal = use_internal                          # remembered for inference
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((head(X) - Y) ** 2).mean()
        loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def explain_amortized(model: Backbone, clip: Clip, head: AmortizedHead,
                      target: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """O(1) inference: one proposal pass + head -> per-point (Sgeo, Smot) fields."""
    if target is None:
        target = model.predicted_class(clip)
    feats = region_features(model, clip, target, use_internal=getattr(head, "_use_internal", True))
    pred = head(feats)                                         # (m,2) -> (Sgeo, Smot)
    B, T, N = clip.mask.shape
    sgeo_f = torch.zeros(B, T, N); smot_f = torch.zeros(B, T, N)
    for r in range(clip.num_regions()):
        sel = ((clip.regions == r) & clip.mask).float()
        sgeo_f += sel * pred[r, 0]
        smot_f += sel * pred[r, 1]
    return sgeo_f, smot_f
