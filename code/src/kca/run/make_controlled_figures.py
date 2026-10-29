"""Render REAL figures from the controlled experiment (trained 4D net) into paper/fig/.

Produces, from an actually-trained model (no placeholders):
  controlled_curves.pdf   deletion/insertion curves, KCA vs baselines
  controlled_overlay.pdf  Smot/Sgeo per-point overlay on a clip across time
  controlled_dissoc.pdf   motion-vs-geometry double-dissociation bars

Run:  PYTHONPATH=src python -m kca.run.make_controlled_figures --out ../paper/fig
"""
from __future__ import annotations

import argparse
import os

import torch

from ..data.motion_task import ClusterCueTask
from ..run.train_backbone import train
from .. import attribute, methods
from ..metrics import faithfulness as F
from . import figures


def _avg_curves(bb, ds, method_names, n, steps=16):
    acc = {m: {"deletion": None, "insertion": None} for m in method_names}
    fracs = None
    cnt = 0
    for i in range(min(n, len(ds))):
        b = ds.get(i); clip = b.clip; tgt = clip.label
        for m in method_names:
            f = methods.run_method(m, bb, clip, tgt) if m != "kca" else attribute.explain(bb, clip, tgt, k=12).nec_field
            if f is None:
                continue
            di = F.deletion_insertion(bb, clip, f, tgt, steps=steps)
            fracs = di["fracs"]
            for key in ("deletion", "insertion"):
                c = di[f"{key}_curve"].mean(0)
                acc[m][key] = c if acc[m][key] is None else acc[m][key] + c
        cnt += 1
    for m in method_names:
        for key in ("deletion", "insertion"):
            if acc[m][key] is not None:
                acc[m][key] = acc[m][key] / cnt
    return acc, fracs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="../paper/fig")
    p.add_argument("--n", type=int, default=16)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    # train the motion model + render curves and overlay
    tr = ClusterCueTask(n=180, K=3, cue="motion", seed=0, region_method="tracked", region_k=3)
    va = ClusterCueTask(n=60, K=3, cue="motion", seed=1000, region_method="tracked", region_k=3)
    bb = train(tr, epochs=14, seed=0, verbose=False)

    labels = {"kca": "KCA", "occam": "OccAM", "integrated_gradients": "Integrated Grad.",
              "geometry_edge": "Geometry-edge", "random": "Random"}
    curves, fracs = _avg_curves(bb, va, list(labels), args.n)
    cb = {labels[m]: curves[m] for m in labels if curves[m]["deletion"] is not None}
    figures.deletion_insertion_curves(cb, fracs, os.path.join(args.out, "controlled_curves.pdf"),
                                      title="Controlled motion task (trained 4D net)")

    b0 = va.get(0); clip = b0.clip; tgt = clip.label
    attr = attribute.explain(bb, clip, tgt, k=12)
    figures.motion_geometry_overlay(clip, attr.smot_field, attr.sgeo_field,
                                    os.path.join(args.out, "controlled_overlay.pdf"))

    # double dissociation bars
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fracs_by_cue = {}
    for cue in ("motion", "shape"):
        t2 = ClusterCueTask(n=120, K=3, cue=cue, seed=0, region_method="tracked", region_k=3)
        v2 = ClusterCueTask(n=60, K=3, cue=cue, seed=1000, region_method="tracked", region_k=3)
        m2 = train(t2, epochs=14, seed=0, verbose=False)
        sm = sg = 0.0
        for i in range(min(args.n, len(v2))):
            b = v2.get(i); clip = b.clip; tgt = clip.label
            a = attribute.explain(m2, clip, tgt, k=12)
            odd = (b.moving_gt if cue == "motion" else b.static_gt).bool()
            sm += a.smot_field.clamp_min(0)[odd].sum().item()
            sg += a.sgeo_field.clamp_min(0)[odd].sum().item()
        fracs_by_cue[cue] = sm / (sm + sg + 1e-8)
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    cues = list(fracs_by_cue)
    ax.bar(cues, [fracs_by_cue[c] for c in cues], color=["#d1495b", "#30638e"])
    ax.set_ylabel(r"motion share $S_{mot}/(S_{mot}{+}S_{geo})$ on odd cluster")
    ax.set_ylim(0, 1); ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.set_title("Motion-vs-geometry dissociation")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "controlled_dissoc.pdf"), bbox_inches="tight")
    print("[written] controlled_curves.pdf, controlled_overlay.pdf, controlled_dissoc.pdf in", args.out)
    print("dissociation:", {k: round(v, 3) for k, v in fracs_by_cue.items()})


if __name__ == "__main__":
    main()
