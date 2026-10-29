"""One-command experiment driver: backbone + dataset -> evaluate -> paper-ready LaTeX.

Row labels / method order / metric columns match paper/sec/4_experiments.tex exactly,
so the emitted .tex can be pasted straight over the \\tbd cells.

CPU smoke test (no GPU/data):
    python -m kca.run.run_experiments --table faithfulness --backbone synthetic \
        --dataset synthetic --limit 6 --steps 8 --out out_demo

GPU runs (after wrapping a backbone and placing data; see SETUP.md):
    python -m kca.run.run_experiments --table faithfulness \
        --backbone p4transformer --checkpoint ckpt/p4t_msr/best.pth \
        --dataset msr --data-root data/msr --out out --amortized
    python -m kca.run.run_experiments --table cross \
        --backbones pstnet,p4transformer,mamba4d \
        --checkpoints ckpt/pst.pth,ckpt/p4t.pth,ckpt/mamba.pth --dataset msr --data-root data/msr
"""
from __future__ import annotations

import argparse
import os

from .. import methods as M
from . import evaluate as E
from . import make_tables as T


# ----- table specifications (mirror the paper tables) ------------------------------
FAITHFULNESS_ROWS = [
    ("random", "Random (floor)"), ("geometry_edge", "Geometry-edge (floor)"),
    ("grad_x_input", r"Grad$\times$Input"),
    ("smoothgrad", r"SmoothGrad~\cite{smilkov2017smoothgrad}"),
    ("integrated_gradients", r"Integrated Grad.~\cite{sundararajan2017ig}"),
    ("grad_cam_3d", r"3D Grad-CAM~\cite{selvaraju2017gradcam}"),
    ("point_drop", r"Point-drop~\cite{zheng2019pointcloudsaliency}"),
    ("occam", r"OccAM~\cite{schinagl2022occam}"),
    ("extremal", r"Extremal Pert.~\cite{fong2019extremal}"),
    ("attn_rollout", r"Attn. Rollout~\cite{abnar2020rollout}"),
    ("transformer_attr", r"Transformer Attr.~\cite{chefer2021transformer}"),
    ("staa4d", r"STAA-4D (ablation)~\cite{wang2025staa}"),
    ("kca", r"\textbf{\method} (exact)"),
    ("kca_amortized", r"\textbf{\method} (amortized)"),
]
FAITHFULNESS_COLS = ["deletion_auc", "insertion_auc", "comprehensiveness", "sufficiency", "sensitivity_n"]
COL_NAMES = {"deletion_auc": "Del.", "insertion_auc": "Ins.", "comprehensiveness": "Comp.",
             "sufficiency": "Suff.", "sensitivity_n": "Sens-$n$"}

LOCALIZATION_ROWS = [
    ("grad_cam_3d", r"3D Grad-CAM~\cite{selvaraju2017gradcam}"),
    ("occam", r"OccAM~\cite{schinagl2022occam}"),
    ("kca_motion", r"\textbf{\method}"),
]
LOCALIZATION_COLS = ["pg_moving", "ebpg_moving"]


# ----- factories (lazy import so synthetic CPU runs need no cuda/repos) ------------
def build_dataset(args):
    if args.dataset == "synthetic":
        from ..data.synthetic_dataset import SyntheticDataset
        return SyntheticDataset(n=args.limit or 16, grid=args.grid)
    if args.dataset == "msr":
        from ..data.msr_action3d import MSRAction3D
        return MSRAction3D(args.data_root, split=args.split, grid=args.grid)
    raise ValueError(f"unknown dataset {args.dataset}")


def build_backbone(name, checkpoint, num_classes, repo=None, device=None, **cfg):
    if name == "synthetic":
        from ..backbones.synthetic import SyntheticBackbone
        return SyntheticBackbone(a=1.0, b=2.0)
    if name in ("p4transformer", "mamba4d", "pstnet"):
        from ..backbones.official import OfficialBackbone
        return OfficialBackbone(name, checkpoint, num_classes=num_classes, repo=repo, device=device, **cfg)
    raise ValueError(f"unknown backbone {name}")


def _amortized_field_fn(model, dataset, args):
    """Train the Stage-3 head on a split, return a field_fn yielding its necessity field."""
    from .. import amortized as A
    head = A.distill(model, dataset, k=args.k, epochs=args.amortized_epochs,
                     limit=args.amortized_train_limit, baseline=args.baseline)

    def fn(m, clip, target):
        sgeo, smot = A.explain_amortized(m, clip, head, target)
        return sgeo + smot
    return fn


def run_faithfulness(args):
    ds = build_dataset(args)
    model = build_backbone(args.backbone, args.checkpoint, ds.num_classes)
    method_keys = [k for k, _ in FAITHFULNESS_ROWS if k != "kca_amortized"]
    res = E.evaluate(model, ds, method_keys, steps=args.steps, limit=args.limit)
    if args.amortized:
        res["kca_amortized"] = E.evaluate_field_fn(
            model, ds, _amortized_field_fn(model, ds, args), steps=args.steps, limit=args.limit)
        rows = FAITHFULNESS_ROWS
    else:
        rows = [r for r in FAITHFULNESS_ROWS if r[0] != "kca_amortized"]
    body = T.save(res, args.out, rows, FAITHFULNESS_COLS, stem="tab1_faithfulness")
    print(T.header_row(FAITHFULNESS_COLS, COL_NAMES)); print(body)
    print(f"\n[written] {args.out}/tab1_faithfulness.tex (+ .json)")


def run_localization(args):
    ds = build_dataset(args)
    model = build_backbone(args.backbone, args.checkpoint, ds.num_classes)
    res = E.evaluate(model, ds, ["grad_cam_3d", "occam", "kca_motion"],
                     steps=args.steps, limit=args.limit, do_localization=True)
    body = T.save(res, args.out, LOCALIZATION_ROWS, LOCALIZATION_COLS, stem="tab3_localization")
    print(body); print(f"\n[written] {args.out}/tab3_localization.tex")


def run_cross(args):
    backbones = args.backbones.split(",")
    ckpts = (args.checkpoints.split(",") if args.checkpoints else [None] * len(backbones))
    method_keys = ["integrated_gradients", "occam", "extremal", "attn_rollout", "staa4d", "kca"]
    results_by_bb = {}
    for bb, ck in zip(backbones, ckpts):
        ds = build_dataset(args)
        model = build_backbone(bb, ck, ds.num_classes)
        results_by_bb[bb] = E.evaluate(model, ds, method_keys, steps=args.steps, limit=args.limit)
    cons = E.kendall_consistency(results_by_bb, metric="deletion_auc")
    import json
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tab2_cross.json"), "w") as fh:
        json.dump({"per_backbone": results_by_bb, "kendall": cons}, fh, indent=2)
    print("Cross-backbone Del/Ins per method:")
    for m in method_keys:
        cells = []
        for bb in backbones:
            r = results_by_bb[bb].get(m, {})
            cells.append("n/a" if "na" in r or "deletion_auc" not in r
                         else f"{r['deletion_auc']['mean']:.1f}/{r['insertion_auc']['mean']:.1f}")
        print(f"  {m:22s} " + "  ".join(f"{bb}:{c}" for bb, c in zip(backbones, cells)))
    print("Kendall tau (deletion ranking across backbones):", cons["tau"])
    print(f"\n[written] {args.out}/tab2_cross.json")


def main():
    p = argparse.ArgumentParser(description="KCA experiment driver")
    p.add_argument("--table", choices=["faithfulness", "cross", "localization"], required=True)
    p.add_argument("--backbone", default="synthetic")
    p.add_argument("--backbones", default="synthetic")     # cross only, comma-sep
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoints", default=None)          # cross only, comma-sep
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--baseline", default="nn", choices=["bg", "mean", "nn"])
    p.add_argument("--amortized", action="store_true")
    p.add_argument("--amortized-epochs", type=int, default=200)
    p.add_argument("--amortized-train-limit", type=int, default=None)
    p.add_argument("--out", default="out")
    args = p.parse_args()

    if args.table == "faithfulness":
        run_faithfulness(args)
    elif args.table == "localization":
        run_localization(args)
    elif args.table == "cross":
        run_cross(args)


if __name__ == "__main__":
    main()
