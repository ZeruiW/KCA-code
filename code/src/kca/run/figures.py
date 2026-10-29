"""Render the paper's figures from harness outputs (replaces the \\figph placeholders).

  deletion_insertion_curves : Fig. A — target-class confidence vs fraction removed
                              (deletion, faster drop better) / inserted (faster rise).
  motion_geometry_overlay   : Fig. B — Smot/Sgeo coloured point-cloud overlay for one
                              clip at a few timesteps.

matplotlib is imported lazily so the core library and tests need no plotting deps.
Install on the GPU/analysis host: pip install matplotlib
"""
from __future__ import annotations

import os

import torch


def deletion_insertion_curves(curves_by_method: dict, fracs, out_path: str, title: str = ""):
    """curves_by_method: {label: {'deletion': (steps+1,), 'insertion': (steps+1,)}} as
    1-D tensors/arrays (already averaged over clips). Writes a 2-panel PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = torch.as_tensor(fracs).cpu().numpy()
    fig, (axd, axi) = plt.subplots(1, 2, figsize=(7, 3))
    for label, c in curves_by_method.items():
        axd.plot(fracs, torch.as_tensor(c["deletion"]).cpu().numpy(), label=label)
        axi.plot(fracs, torch.as_tensor(c["insertion"]).cpu().numpy(), label=label)
    axd.set_title("Deletion (↓ faster better)"); axd.set_xlabel("fraction removed")
    axi.set_title("Insertion (↑ faster better)"); axi.set_xlabel("fraction inserted")
    axd.set_ylabel("target-class confidence")
    axi.legend(fontsize=6, loc="lower right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def motion_geometry_overlay(clip, smot_field, sgeo_field, out_path: str,
                            timesteps=None, b: int = 0):
    """Scatter the clip's points coloured by Smot (top row) and Sgeo (bottom row) across
    a few timesteps. clip: Clip; fields: (B,T,N)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = clip.T
    timesteps = timesteps or list(range(0, T, max(1, T // 4)))[:4]
    fig, axes = plt.subplots(2, len(timesteps), figsize=(3 * len(timesteps), 6))
    if len(timesteps) == 1:
        axes = axes.reshape(2, 1)
    for col, t in enumerate(timesteps):
        m = clip.mask[b, t]
        xy = clip.points[b, t, m, :2].cpu().numpy()
        for row, (field, name) in enumerate([(smot_field, "$S_{mot}$"), (sgeo_field, "$S_{geo}$")]):
            c = field[b, t, m].cpu().numpy()
            ax = axes[row, col]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="hot", s=6)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name)
            if row == 0:
                ax.set_title(f"t={t}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path
