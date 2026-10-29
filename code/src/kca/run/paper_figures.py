"""Render the paper's real-data figures (PDF) from the benchmark outputs.

  fig/msr_curves.pdf      mean deletion / insertion curves on P4Transformer (per method)
  fig/msr_qual.pdf        Smot / Sgeo overlays on real MSR clips, per backbone (also the teaser)
  fig/dissociation.pdf    motion share on the cue structure: controlled task + real crossover
  fig/sanity.pdf          Adebayo cascading randomisation (rank correlation vs blocks randomised)

Run:  python -m kca.run.paper_figures --results results/msr --controlled results/controlled \
          --data <npz dir> --fig ../paper/fig
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ..data.msr_action3d import ACTIONS, MSRAction3D

BB_NAMES = {"p4transformer": "P4Transformer", "pstnet": "PSTNet", "mamba4d": "Mamba4D"}

CURVE_METHODS = [("random", "Random", "#bbbbbb", ":"), ("grad_x_input", r"Grad$\times$Input", "#8c6d31", "-"),
                 ("integrated_gradients", "Int. Grad.", "#bd9e39", "-"), ("grad_cam_3d", "3D Grad-CAM", "#e7ba52", "-"),
                 ("point_drop", "Region occl.", "#6b6ecf", "-"), ("occam", "OccAM", "#9c9ede", "-"),
                 ("extremal", "Extremal", "#637939", "-"), ("attn_rollout", "Rollout", "#b5cf6b", "-"),
                 ("kca", "KCA (local)", "#d62728", "-"), ("kca_ctx", "KCA (context)", "#ff7f0e", "--")]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "pdf.fonttype": 42})
    return plt


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def fig_curves(recs, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for key, ax, title in (("deletion_curve", axes[0], "Deletion (lower is better)"),
                           ("insertion_curve", axes[1], "Insertion (higher is better)")):
        for m, lab, col, ls in CURVE_METHODS:
            cs = [r["methods"][m][key] for r in recs if m in r["methods"] and key in r["methods"][m]]
            if not cs:
                continue
            c = np.mean(np.array(cs), 0)
            x = np.linspace(0, 1, len(c))
            ax.plot(x, c, ls, color=col, lw=1.6 if m.startswith("kca") else 1.0, label=lab)
        ax.set_xlabel("fraction of points removed" if key.startswith("del") else "fraction of points inserted")
        ax.set_ylabel("target-class probability")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].legend(ncol=2, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _fields(ds, rec_by_clip, name, method="kca"):
    i = [os.path.basename(f) for f in ds.files].index(name)
    b = ds.get(i)
    r = rec_by_clip[name]["methods"][method]
    reg = b.clip.regions[0]
    sm, sg = torch.zeros(reg.shape), torch.zeros(reg.shape)
    for rid, (a, g) in enumerate(zip(r["region_smot"], r["region_sgeo"])):
        sm[reg == rid] = a
        sg[reg == rid] = g
    return b, sm, sg


def fig_qual(results, data, path, clips, backbones, frames=(2, 12, 21)):
    plt = _plt()
    dss = {}                                         # one loader per points-per-frame value
    rows = [(bb, c) for c in clips for bb in backbones]
    fig, axes = plt.subplots(len(rows), 2 * len(frames), figsize=(1.05 * 2 * len(frames), (1.55 if len(rows) < 4 else 1.25) * len(rows)),
                             squeeze=False)
    for ri, (bb, name) in enumerate(rows):
        recs = {r["clip"]: r for r in load(os.path.join(results, bb, "raw.jsonl"))}
        if name not in recs or "kca" not in recs[name]["methods"]:
            for ax in axes[ri]:
                ax.axis("off")
            continue
        npts = recs[name].get("num_points", 1024)  # the clip the benchmark explained
        if npts not in dss:
            dss[npts] = MSRAction3D(data, "test", num_points=npts, region_method="tracked", region_k=12)
        b, sm, sg = _fields(dss[npts], recs, name)
        P = b.clip.points[0].numpy()
        pos = np.concatenate([sm.clamp_min(0).flatten().numpy(), sg.clamp_min(0).flatten().numpy()])
        vmax = max(float(np.percentile(pos[pos > 0], 95)) if (pos > 0).any() else 1.0, 1e-6)
        for fi, t in enumerate(frames):
            for k, (f, cmap, lab) in enumerate(((sm, "Reds", r"$S_{mot}$"), (sg, "Blues", r"$S_{geo}$"))):
                ax = axes[ri, 2 * fi + k]
                v = np.clip(f[t].clamp_min(0).numpy() / vmax, 0, 1)
                ax.scatter(P[t, :, 0], -P[t, :, 1], c="#d9d9d9", s=1.0, linewidths=0, rasterized=True)
                hot = v > 0.02
                ax.scatter(P[t, hot, 0], -P[t, hot, 1], c=v[hot], cmap=cmap, vmin=-0.3, vmax=1.0, s=1.6,
                           linewidths=0, rasterized=True)
                ax.set_aspect("equal"); ax.axis("off")
                if ri == 0:
                    ax.set_title(f"{lab}, t={t}")
        axes[ri, 0].text(-0.15, 0.5, f"{ACTIONS[b.clip.label.item()]}\n({BB_NAMES.get(bb, bb)})", transform=axes[ri, 0].transAxes,
                         rotation=90, va="center", ha="right", fontsize=6.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_dissociation(controlled, crossover, path, extras=()):
    """Motion share on the cue structure, class defined by motion (red) vs by shape (blue):
    the controlled task and the real-data crossover on each backbone (local estimator;
    95 % bootstrap CIs over clips on real data). extras: [(crossover.json, backbone label)]."""
    plt = _plt()
    red, blue = "#d62728", "#1f77b4"
    groups = []                                     # (label, motion-task share, shape-task share, ci_m, ci_g)
    if controlled and "motion" in json.load(open(controlled)):
        c = json.load(open(controlled))
        groups.append(("controlled", c["motion"]["odd_motion_fraction_pooled"], c["shape"]["odd_motion_fraction_pooled"], None, None))
    for f, lab in ([(crossover, "P4Transf.")] if crossover else []) + list(extras):
        x = json.load(open(f))
        if not all("kca" in x.get(t, {}) for t in ("motion", "geometry")):
            continue
        ci = x.get("ci_kca", {})
        groups.append((lab, x["motion"]["kca"]["arm_motion_share"], x["geometry"]["kca"]["arm_motion_share"],
                       ci.get("motion"), ci.get("geometry")))
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    w = 0.36
    for i, (lab, vm, vg, cm, cg) in enumerate(groups):
        for dx, v, ci, col in ((-w / 2, vm, cm, red), (w / 2, vg, cg, blue)):
            err = None if ci is None else [[v - ci[0]], [ci[1] - v]]
            ax.bar(i + dx, v, w * 0.95, color=col, yerr=err, capsize=2, error_kw={"lw": 0.8})
            if v < 0.02:                                  # a zero bar is a result, not a missing value
                ax.text(i + dx, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=col)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.axvline(0.5, color="#999999", lw=0.6, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("motion share on\nthe cue structure")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=red, label="class defined by motion"), Patch(color=blue, label="class defined by shape")],
              frameon=False, fontsize=6.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_sanity(ablation, path):
    plt = _plt()
    s = json.load(open(ablation)).get("sanity")
    if not s:
        return
    x = [r["blocks_randomized"] for r in s]
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    ax.plot([0] + x, [1] + [r["kca_rankcorr"] for r in s], "o-", color="#d62728", label="KCA")
    ax.plot([0] + x, [1] + [r["gradxinput_rankcorr"] for r in s], "s--", color="#8c6d31", label=r"Grad$\times$Input")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("blocks re-initialised (head first)"); ax.set_ylabel("rank corr. to trained")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/msr")
    p.add_argument("--controlled", default="results/controlled/crossover.json")
    p.add_argument("--data", required=True)
    p.add_argument("--fig", default="../paper/fig")
    p.add_argument("--clips", default="a09_s06_e01_sdepth.npz,a01_s06_e01_sdepth.npz,a14_s06_e01_sdepth.npz")  # first = teaser
    a = p.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    main_raw = os.path.join(a.results, "p4transformer", "raw.jsonl")
    if os.path.exists(main_raw):
        fig_curves(load(main_raw), os.path.join(a.fig, "msr_curves.pdf"))
    bbs = [b for b in ("p4transformer", "pstnet", "mamba4d") if os.path.exists(os.path.join(a.results, b, "raw.jsonl"))]
    if bbs:
        fig_qual(a.results, a.data, os.path.join(a.fig, "msr_qual.pdf"), a.clips.split(","), bbs)
        fig_qual(a.results, a.data, os.path.join(a.fig, "msr_teaser.pdf"), a.clips.split(",")[:1], bbs[:1],
                 frames=(2, 8, 14, 21))
    cx = os.path.join(a.results, "crossover", "crossover.json")
    extras = [(os.path.join(a.results, sub, "crossover.json"), lab)
              for sub, lab in (("crossover_pstnet", "PSTNet"), ("crossover_mamba4d", "Mamba4D"))]
    fig_dissociation(a.controlled if os.path.exists(a.controlled) else None, cx if os.path.exists(cx) else None,
                     os.path.join(a.fig, "dissociation.pdf"), [(f, lab) for f, lab in extras if os.path.exists(f)])
    ab = os.path.join(a.results, "ablation", "ablation.json")
    if os.path.exists(ab):
        fig_sanity(ab, os.path.join(a.fig, "sanity.pdf"))
    print("[figures] ->", a.fig)


if __name__ == "__main__":
    main()
