"""Cross-backbone rank consistency (paper Sec. exp-cross, Tab. cross).

Kendall's tau between the per-method ranking of a faithfulness score on two
different backbones. The model-agnostic headline: KCA's ranking of regions/methods
is consistent across architecturally distinct backbones.
"""
from __future__ import annotations

import torch


def kendall_tau(x: torch.Tensor, y: torch.Tensor) -> float:
    """Kendall's tau-b between two 1-D score vectors (same length, paired)."""
    x = x.flatten(); y = y.flatten()
    n = x.numel()
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (x[i] - x[j]).item()
            dy = (y[i] - y[j]).item()
            s = dx * dy
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else float("nan")
