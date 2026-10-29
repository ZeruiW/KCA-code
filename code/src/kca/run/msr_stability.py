"""Explanation stability under sensor re-sampling (MSR-Action3D).

Depth and LiDAR sensors re-sample the surface in every frame, so a deployed model never sees
the same points twice. An explanation that will be acted on should not change when the same
scene is sampled again. For each clip we draw R independent point samplings of the same 24
frames (sampling seed r = 0..R-1; r = 0 is the benchmark's sampling), explain the model's
prediction on each with every method, transfer each map onto the points of sampling 0
(nearest neighbour within the same frame), and record the Spearman rank correlation with the
sampling-0 map (point level). Because a region-constant map can be ranked differently when
region boundaries move by a few points, we also record a part-level correlation: each map is
averaged over the points of each of the 19 skeleton bones (the body part a practitioner reads
off a map), and the two 19-vectors are rank-correlated. Both are reported. Finally, as a
mitigation that needs only forward passes, the maps of samplings {0,1} and {2,3} are averaged
(on the sampling-0 points) and the two averages are correlated in the same two ways. Method randomness (OccAM masks, KCA's context draws, ...) is seeded per clip
exactly as in `msr_benchmark`, so only the point sampling changes between runs.

Per clip and sampling we also record the predicted class and, for KCA, the whole-body motion
share of the positive importance, Smot+ / (Smot+ + Sgeo+).

Records go to <out>/stability.jsonl (one line per clip; resumable); tables and numbers are
made by `kca.run.paper_tables`.

Run:  python -m kca.run.msr_stability --backbone p4transformer --ckpt <last.pth> \
          --data <npz dir> --out results/msr_full/stability --points 2048 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from .. import attribute, methods as M
from ..backbones.official import OfficialBackbone
from ..data.msr_action3d import BONES, MSRAction3D, nearest_bone
from ..metrics import soundness as S
from .msr_benchmark import KCA, METHOD_KW, stratified

DEFAULT_METHODS = "random,motion_prior,grad_x_input,integrated_gradients,point_drop,occam,attn_rollout,kca,kca_ctx"


def transfer(field: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """field (T,N) on points src (T,N,3) -> values at points dst (T,N,3), nearest neighbour per frame."""
    out = torch.empty(dst.shape[:2], dtype=field.dtype, device=field.device)
    for t in range(dst.shape[0]):
        out[t] = field[t, torch.cdist(dst[t], src[t]).argmin(1)]
    return out


def bone_labels(ds, i, clip) -> np.ndarray:
    """(T,N) nearest-bone index of every point of a sampled clip (raw units, as the GT)."""
    d = np.load(ds.files[i], allow_pickle=True)
    J = d["joints"][ds._window(len(d["point_clouds"]))]
    pts = clip.points[0].cpu().numpy() * 300.0
    return np.stack([nearest_bone(pts[t], J[t]) for t in range(len(J))])


def part_vector(field: torch.Tensor, lab: np.ndarray) -> np.ndarray:
    f = field.detach().cpu().numpy().reshape(-1)
    lab = lab.reshape(-1)
    return np.array([f[lab == b].mean() if (lab == b).any() else np.nan for b in range(len(BONES))])


def part_corr(u: np.ndarray, v: np.ndarray) -> float:
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 3 or np.ptp(u[ok]) == 0 or np.ptp(v[ok]) == 0:
        return float("nan")
    return float(spearmanr(u[ok], v[ok]).statistic)


def motion_share(at) -> float:
    sm = at.smot_field.clamp_min(0).sum().item()
    sg = at.sgeo_field.clamp_min(0).sum().item()
    return sm / (sm + sg) if sm + sg > 0 else float("nan")


def explain_all(model, clip, target, names, i, regions):
    fields, shares = {}, {}
    for m in names:
        torch.manual_seed(1000 + i)                            # as in msr_benchmark
        if m in KCA:
            at = attribute.explain(model, clip, target, k=regions, n_control=0,
                                   generator=torch.Generator().manual_seed(i), **KCA[m])
            fields[m] = at.nec_field.detach()
            shares[m] = motion_share(at)
        else:
            f = M.run_method(m, model, clip, target, **METHOD_KW.get(m, {}))
            if f is not None:
                fields[m] = f.detach().to(clip.points.device)
    return fields, shares


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="p4transformer")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--samplings", type=int, default=5)
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--regions", type=int, default=12)
    p.add_argument("--points", type=int, default=1024, help="points per frame (use the model's training value)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-clips", type=int, default=None, help="smoke test: stop after this many clips")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    names = a.methods.split(",")
    model = OfficialBackbone(a.backbone, a.ckpt, repo=a.repo, device=a.device)
    dss = [MSRAction3D(a.data, "test", num_points=a.points, region_method="tracked", region_k=a.regions, seed=r)
           for r in range(a.samplings)]
    idxs = stratified(dss[0], a.per_class)[:a.max_clips]
    path = os.path.join(a.out, "stability.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["clip"] for l in fh if l.strip()}
    print(f"[stability/{a.backbone}] {len(idxs)} clips x {a.samplings} samplings ({len(done)} done), methods={names}",
          flush=True)
    with open(path, "a") as fh:
        for n, i in enumerate(idxs):
            name = os.path.basename(dss[0].files[i])
            if name in done:
                continue
            t0 = time.time()
            ref = None
            on_ref = {m: [] for m in names}                                   # every sampling's map on sampling-0 points
            rec = {"clip": name, "label": dss[0]._label(dss[0].files[i]), "num_points": a.points,
                   "samplings": a.samplings, "pred": [], "rank_corr": {m: [] for m in names}, "part_corr": {m: [] for m in names},
                   "avg2_rank_corr": {}, "avg2_part_corr": {},
                   "motion_share": {m: [] for m in names if m in KCA}}
            for r, ds in enumerate(dss):
                clip = ds.get(i).clip
                with torch.no_grad():
                    target = model.logits(clip).argmax(1)
                rec["pred"].append(int(target))
                fields, shares = explain_all(model, clip, target, names, i, a.regions)
                for m, v in shares.items():
                    rec["motion_share"][m].append(v)
                pts = clip.points[0].to(clip.points.device)
                lab = bone_labels(ds, i, clip)
                parts = {m: part_vector(f[0], lab) for m, f in fields.items()}
                if r == 0:
                    ref = (pts, {m: f[0] for m, f in fields.items()}, clip.mask, parts, lab)
                    for m, f in fields.items():
                        on_ref[m].append(f[0])
                    continue
                for m, f in fields.items():
                    if m not in ref[1]:
                        continue
                    moved = transfer(f[0], pts, ref[0])                       # this sampling's map on sampling-0 points
                    on_ref[m].append(moved)
                    rec["rank_corr"][m].append(S.rank_correlation(ref[1][m][None], moved[None], ref[2]).item())
                    rec["part_corr"][m].append(part_corr(ref[3][m], parts[m]))
            for m, maps in on_ref.items():
                if len(maps) >= 4:
                    u, v = (maps[0] + maps[1]) / 2, (maps[2] + maps[3]) / 2
                    rec["avg2_rank_corr"][m] = S.rank_correlation(u[None], v[None], ref[2]).item()
                    rec["avg2_part_corr"][m] = part_corr(part_vector(u, ref[4]), part_vector(v, ref[4]))
            rec["time_s"] = time.time() - t0
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            mean = {m: round(float(np.nanmean(v)), 2) for m, v in rec["rank_corr"].items() if v}
            part = {m: round(float(np.nanmean(v)), 2) for m, v in rec["part_corr"].items() if v}
            print(f"  [{n + 1}/{len(idxs)}] {name} preds {rec['pred']} point {mean} part {part} ({rec['time_s']:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
