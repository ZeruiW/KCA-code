"""LEGACY orchestrator (kept for the synthetic --dry-run wiring check). The real-data
paper numbers are produced by kca.run.msr_benchmark / msr_crossover / msr_ablation and
kca.run.paper_tables (per-clip records, Wilcoxon significance) — see SETUP.md.

CVPR-scale benchmark orchestrator: one entry point that fills ALL paper tables.

Given trained checkpoints for the three backbones, it produces:
  Tab.1 main faithfulness (P4Transformer / MSR), all methods, mean±std over R runs + sig
  Tab.2 cross-backbone (PSTNet | P4Transformer | Mamba4D) Del/Ins + Kendall tau
  Tab.3 motion-vs-geometry localization (needs GT dataset, e.g. NTU/HOI4D)
  Tab.4 ablation + efficiency (intervention set, baseline, k, exact vs amortized; latency)
and writes the .tex bodies + a master JSON + a CONCLUSIONS scaffold.

Config is a JSON/dict mapping backbone -> {class import, checkpoint, repo}. Example
(edit paths, then run ONE command on the GPU host):

  python -m kca.run.cvpr_benchmark --config configs/msr.json --out results/cvpr

A --dry-run uses the synthetic backbone/dataset to validate the whole matrix on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .. import methods, attribute
from ..run import evaluate as E
from ..run import make_tables as T
from . import run_experiments as RX


def _load_backbone(spec, num_classes):
    if spec["backbone"] == "synthetic":
        from ..backbones.synthetic import SyntheticBackbone
        return SyntheticBackbone()
    ck = spec.get("checkpoint")
    return RX.build_backbone(spec["backbone"], os.path.expanduser(ck) if ck else None, num_classes,
                             repo=spec.get("repo"), device=spec.get("device"))


def _load_dataset(spec):
    if spec["dataset"] == "synthetic":
        from ..data.synthetic_dataset import SyntheticDataset
        return SyntheticDataset(n=spec.get("limit", 16), grid=3)
    from ..data.msr_action3d import MSRAction3D
    return MSRAction3D(os.path.expanduser(spec["data_root"]), split=spec.get("split", "test"),
                       frames=spec.get("frames", 24), num_points=spec.get("points", 1024),
                       region_method="tracked", region_k=spec.get("regions", 12))


def multi_run(model, ds, names, runs, steps, limit, do_loc=False):
    """Repeat evaluate() with different metric RNG -> mean±std over `runs` (the paper's
    'mean±std over 10 runs'). Returns {method:{metric:{mean,std,n}}} and per-run raw."""
    raw = []
    for r in range(runs):
        torch.manual_seed(1000 + r)
        raw.append(E.evaluate(model, ds, names, steps=steps, limit=limit, do_localization=do_loc))
    out = {}
    for m in names:
        if any("na" in rr[m] for rr in raw):
            out[m] = {"na": True}; continue
        out[m] = {}
        for c in raw[0][m]:
            vals = torch.tensor([rr[m][c]["mean"] for rr in raw])
            out[m][c] = {"mean": vals.mean().item(), "std": vals.std(unbiased=False).item(), "n": runs}
    return out, raw


def table1(cfg, out, runs, steps, limit):
    spec = cfg["tab1"]
    ds = _load_dataset(spec); model = _load_backbone(spec, ds.num_classes)
    names = [k for k, _ in RX.FAITHFULNESS_ROWS if k != "kca_amortized"]
    res, _ = multi_run(model, ds, names, runs, steps, limit)
    body = T.latex_table_body(res, [r for r in RX.FAITHFULNESS_ROWS if r[0] != "kca_amortized"],
                              RX.FAITHFULNESS_COLS)
    T.save(res, out, [r for r in RX.FAITHFULNESS_ROWS if r[0] != "kca_amortized"],
           RX.FAITHFULNESS_COLS, stem="tab1_faithfulness")
    print("[Tab.1] written"); return res


def table2(cfg, out, runs, steps, limit):
    specs = cfg["tab2"]            # list of backbone specs (PSTNet, P4T, Mamba4D)
    names = ["integrated_gradients", "occam", "extremal", "attn_rollout", "staa4d", "kca"]
    by_bb = {}
    for spec in specs:
        ds = _load_dataset(spec); model = _load_backbone(spec, ds.num_classes)
        by_bb[spec["backbone"]], _ = multi_run(model, ds, names, runs, steps, limit)
    cons = E.kendall_consistency(by_bb, metric="deletion_auc")
    with open(os.path.join(out, "tab2_cross.json"), "w") as fh:
        json.dump({"per_backbone": by_bb, "kendall": cons}, fh, indent=2)
    print("[Tab.2] Kendall tau:", cons["tau"]); return by_bb, cons


def table3(cfg, out, runs, steps, limit):
    spec = cfg["tab3"]            # GT-localization dataset (NTU/HOI4D)
    ds = _load_dataset(spec); model = _load_backbone(spec, ds.num_classes)
    names = ["grad_cam_3d", "occam", "motion_prior", "kca_motion"]
    res, _ = multi_run(model, ds, names, runs, steps, limit, do_loc=True)
    rows = [("grad_cam_3d", "3D Grad-CAM"), ("occam", "OccAM"),
            ("motion_prior", "Motion-prior (floor)"), ("kca_motion", r"\textbf{KCA}")]
    T.save(res, out, rows, ["pg_moving", "ebpg_moving"], stem="tab3_localization")
    print("[Tab.3] written"); return res


def table4(cfg, out, steps, limit):
    """Ablation + efficiency: intervention set, baseline, k, exact vs amortized; latency."""
    spec = cfg["tab4"]; ds = _load_dataset(spec); model = _load_backbone(spec, ds.num_classes)
    rep = {}
    def faith(field_fn):
        r = E.evaluate_field_fn(model, ds, field_fn, steps=steps, limit=limit)
        return r.get("deletion_auc", {}).get("mean", float("nan"))
    for bl in ("bg", "mean", "nn"):
        rep[f"baseline_{bl}"] = faith(lambda m, c, t, bl=bl: attribute.explain(m, c, t, baseline=bl).nec_field)
    for k in (8, 16, 32):
        rep[f"k_{k}"] = faith(lambda m, c, t, k=k: attribute.explain(m, c, t, k=k).nec_field)
    for sym in (True, False):
        rep[f"symmetric_{sym}"] = faith(lambda m, c, t, s=sym: attribute.explain(m, c, t, symmetric=s).nec_field)
    # efficiency: latency + #forward of exact KCA vs OccAM
    b0 = ds.get(0); clip = b0.clip; tgt = clip.label
    t0 = time.time(); a = attribute.explain(model, clip, tgt); lat_kca = time.time() - t0
    rep["kca_exact"] = {"latency_s": lat_kca, "n_forward": a.n_forward}
    t0 = time.time(); _ = methods.run_method("occam", model, clip, tgt); rep["occam_latency_s"] = time.time() - t0
    with open(os.path.join(out, "tab4_ablation.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    print("[Tab.4] efficiency: KCA %.3fs (%d fwd) vs OccAM %.3fs" %
          (lat_kca, a.n_forward, rep["occam_latency_s"])); return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="JSON with tab1..tab4 backbone/dataset specs")
    p.add_argument("--out", default="results/cvpr")
    p.add_argument("--runs", type=int, default=10)        # mean±std over runs (paper: 10)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.dry_run or not args.config:
        s = {"backbone": "synthetic", "dataset": "synthetic", "limit": 8}
        cfg = {"tab1": s, "tab2": [s, s], "tab3": s, "tab4": s}
        args.runs = min(args.runs, 2); args.limit = args.limit or 6
    else:
        cfg = json.load(open(args.config))
    table1(cfg, args.out, args.runs, args.steps, args.limit)
    table2(cfg, args.out, args.runs, args.steps, args.limit)
    if "tab3" in cfg:
        table3(cfg, args.out, args.runs, args.steps, args.limit)
    table4(cfg, args.out, args.steps, args.limit)
    print(f"\n[done] all tables -> {args.out}/")


if __name__ == "__main__":
    main()
