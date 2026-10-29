"""Deeper analyses on a trained controlled net: per-method localization, the proposal
budget (k) efficiency/faithfulness tradeoff, baseline-fill sweep, and an amortized-head
optimization sweep. Saves JSON + figures to results/analysis/ for the results writeup.

Run:  PYTHONPATH=src python -m kca.run.analysis_extra --out results/analysis
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..data.motion_task import ClusterCueTask
from ..run.train_backbone import train
from .. import attribute, methods, amortized as A
from ..metrics import faithfulness as F, localization as L, soundness as S


def _agg(v):
    t = torch.tensor(v, dtype=torch.float)
    return {"mean": t.mean().item(), "std": t.std(unbiased=False).item(), "n": len(v)}


def localization_all_methods(bb, ds, n):
    """EBPG + Pointing-Game of every method's field vs the moving-cluster GT.
    For KCA we use the motion field Smot (the quantity meant to localize motion)."""
    names = ["random", "geometry_edge", "grad_x_input", "integrated_gradients",
             "grad_cam_3d", "occam", "motion_prior", "staa4d", "kca_motion"]
    acc = {m: {"ebpg": [], "pg": []} for m in names}
    for i in range(n):
        b = ds.get(i); clip = b.clip; tgt = clip.label; gt = b.moving_gt
        for m in names:
            f = methods.run_method(m, bb, clip, tgt)
            if f is None:
                continue
            acc[m]["ebpg"].append(L.ebpg(f, gt, clip.mask).mean().item())
            acc[m]["pg"].append(L.pointing_game(f, gt, clip.mask).mean().item())
    return {m: {"ebpg": _agg(a["ebpg"]), "pg": _agg(a["pg"])} for m, a in acc.items() if a["ebpg"]}


def k_sweep(bb, ds, n, ks=(4, 8, 12, 16, 24, 32)):
    """Proposal budget k -> faithfulness (Deletion-AUC), #forward passes, latency.
    Shows the O(k) efficiency knob and where faithfulness saturates."""
    out = {}
    for k in ks:
        dels, fwds, lats = [], [], []
        for i in range(n):
            b = ds.get(i); clip = b.clip; tgt = clip.label
            t0 = time.time()
            attr = attribute.explain(bb, clip, tgt, k=k, n_control=2)
            lats.append(time.time() - t0)
            fwds.append(attr.n_forward)
            dels.append(F.deletion_insertion(bb, clip, attr.nec_field, tgt, steps=12)["deletion_auc"].mean().item())
        out[k] = {"deletion": _agg(dels), "n_forward": _agg(fwds), "latency_s": _agg(lats)}
    return out


def baseline_sweep(bb, ds, n):
    out = {}
    for bl in ("bg", "mean", "nn"):
        dels = []
        for i in range(n):
            b = ds.get(i); clip = b.clip; tgt = clip.label
            attr = attribute.explain(bb, clip, tgt, k=12, baseline=bl)
            dels.append(F.deletion_insertion(bb, clip, attr.nec_field, tgt, steps=12)["deletion_auc"].mean().item())
        out[bl] = _agg(dels)
    return out


def amortized_sweep(bb, train_ds, va, n, configs):
    """Optimize the amortized head. KEY comparison: input-only features (4 scalars) vs
    +internal backbone features (the paper's 'distil from internal features'). Also vary
    distillation set size/epochs. Reports held-out fidelity (rank-corr to exact)."""
    out = {}
    for use_internal in (False, True):
        tag = "internal" if use_internal else "input_only"
        for (lim, ep) in configs:
            head = A.distill(bb, train_ds, k=12, epochs=ep, limit=lim, baseline="nn",
                             use_internal=use_internal)
            fid, dam, dex = [], [], []
            for i in range(n):
                b = va.get(i); clip = b.clip; tgt = clip.label
                ex = attribute.explain(bb, clip, tgt, k=12).nec_field
                sg, sm = A.explain_amortized(bb, clip, head, tgt)
                fid.append(S.rank_correlation(ex, sg + sm, clip.mask).mean().item())
                dam.append(F.deletion_insertion(bb, clip, sg + sm, tgt, steps=10)["deletion_auc"].mean().item())
                dex.append(F.deletion_insertion(bb, clip, ex, tgt, steps=10)["deletion_auc"].mean().item())
            out[f"{tag}_lim{lim}_ep{ep}"] = {"features": tag, "limit": lim, "epochs": ep,
                                             "fidelity_rankcorr": _agg(fid),
                                             "deletion_amortized": _agg(dam), "deletion_exact": _agg(dex)}
    return out


def _bar(d, key, ylabel, title, path, sort=True):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    items = [(m, d[m][key]["mean"], d[m][key]["std"]) for m in d]
    if sort:
        items.sort(key=lambda kv: kv[1])
    labels = [m for m, _, _ in items]
    vals = [v for _, v, _ in items]; errs = [e for _, _, e in items]
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#d1495b" if "kca" in m else "#9aa5b1" for m in labels]
    ax.barh(labels, vals, xerr=errs, color=colors)
    ax.set_xlabel(ylabel); ax.set_title(title); fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _ksweep_fig(ks, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = sorted(ks); xs = k
    deldata = [ks[i]["deletion"]["mean"] for i in k]
    fwd = [ks[i]["n_forward"]["mean"] for i in k]
    fig, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot(xs, deldata, "o-", color="#d1495b", label="Deletion-AUC")
    ax1.set_xlabel("proposal budget $k$"); ax1.set_ylabel("Deletion-AUC ↓", color="#d1495b")
    ax2 = ax1.twinx(); ax2.plot(xs, fwd, "s--", color="#30638e", label="#forward")
    ax2.set_ylabel("#forward passes", color="#30638e")
    ax1.set_title("Proposal budget: faithfulness vs cost"); fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/analysis")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--fig", default="results/analysis/fig")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    tr = ClusterCueTask(n=200, K=3, cue="motion", seed=0, region_method="kmeans", region_k=3)
    va = ClusterCueTask(n=80, K=3, cue="motion", seed=1000, region_method="kmeans", region_k=3)
    print("training reference motion model...")
    bb = train(tr, epochs=16, seed=0, verbose=False)

    print("1) per-method localization"); loc = localization_all_methods(bb, va, args.n)
    print("2) k-sweep");                  ks = k_sweep(bb, va, args.n)
    print("3) baseline-fill sweep");      bls = baseline_sweep(bb, va, args.n)
    print("4) amortized optimization");   am = amortized_sweep(
        bb, tr, va, args.n, configs=[(20, 120), (60, 200), (120, 300), (200, 400)])

    res = {"localization_all": loc, "k_sweep": ks, "baseline_sweep": bls, "amortized_sweep": am}
    with open(os.path.join(args.out, "analysis.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    _bar(loc, "ebpg", "EBPG ↑ (mass on moving GT)", "Per-method motion localization",
         os.path.join(args.fig, "localization_bars.pdf"))
    _ksweep_fig(ks, os.path.join(args.fig, "k_sweep.pdf"))

    # console summary
    print("\n-- localization EBPG (top) --")
    for m, v in sorted(loc.items(), key=lambda kv: -kv[1]["ebpg"]["mean"])[:5]:
        print(f"   {m:20s} EBPG {v['ebpg']['mean']:.3f}  PG {v['pg']['mean']:.3f}")
    print("-- amortized fidelity vs distillation budget --")
    for cfg, v in am.items():
        print(f"   {cfg:14s} rankcorr {v['fidelity_rankcorr']['mean']:.3f}  "
              f"del amort {v['deletion_amortized']['mean']:.2f} vs exact {v['deletion_exact']['mean']:.2f}")
    print(f"\n[written] {args.out}/analysis.json + figures in {args.fig}")


if __name__ == "__main__":
    main()
