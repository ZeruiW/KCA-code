"""Ground-truth localization metrics (paper Sec. exp-mpg, Tab. mpg).

Given a per-point importance field and a per-point binary GT mask of the
"should-be-important" structure (moving limbs for Smot, static class-defining
structure for Sgeo), measure whether mass lands on the GT region.

  Pointing Game (Zhang'18): does the single most-important point fall on GT?
  EBPG (energy-based PG, Score-CAM/Wang'20): fraction of total positive
    attribution energy that lies inside the GT mask.
"""
from __future__ import annotations

import torch


def pointing_game(field: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) in {0,1}: 1 if the argmax-importance valid point is inside GT."""
    B = field.shape[0]
    f = field.reshape(B, -1).clone()
    f[~mask.reshape(B, -1)] = float("-inf")
    hit_idx = f.argmax(1)                                  # (B,)
    gt_flat = gt.reshape(B, -1)
    return gt_flat.gather(1, hit_idx.view(-1, 1)).squeeze(1).float()


def ebpg(field: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) energy-based pointing game: positive attribution energy inside GT / total."""
    B = field.shape[0]
    f = field.reshape(B, -1).clamp_min(0)
    f = f * mask.reshape(B, -1).float()
    gt_flat = gt.reshape(B, -1).float()
    inside = (f * gt_flat).sum(1)
    total = f.sum(1).clamp_min(1e-8)
    return inside / total
