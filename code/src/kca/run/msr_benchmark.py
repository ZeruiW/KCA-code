"""Real-data benchmark on MSR-Action3D: every method x every metric, per clip.

For one trained backbone (OfficialBackbone checkpoint) and a stratified subset of the test
split (`--per-class` clips per action, in file order), explain the model's PREDICTED
class with each method and record, per clip:
  faithfulness   Deletion/Insertion AUC, Comprehensiveness/Sufficiency (AOPC), Sensitivity-n
  localisation   Pointing Game / EBPG of the map vs the skeleton moving-GT (and, for KCA,
                 of Smot vs moving-GT and Sgeo vs static-GT)
  decomposition  (KCA) Smot/Sgeo totals, motion share on moving-GT, |Delta_int|/|Nec|
  cost           wall-clock seconds (and forward passes for KCA)
Raw per-clip records go to <out>/raw.jsonl (resumable); tables are made by
`kca.run.paper_tables`.

Run:  python -m kca.run.msr_benchmark --backbone p4transformer --ckpt <last.pth> \
          --data <npz dir> --out results/msr/p4transformer
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .. import attribute, methods as M
from ..backbones.official import OfficialBackbone
from ..data.msr_action3d import MSRAction3D
from ..metrics import faithfulness as F, localization as L

ALL = ["random", "geometry_edge", "motion_prior", "grad_x_input", "smoothgrad",
       "integrated_gradients", "grad_cam_3d", "point_drop", "occam", "extremal",
       "attn_rollout", "transformer_attr", "staa4d", "kca", "kca_ctx"]
KCA = {"kca": dict(estimator="local"), "kca_ctx": dict(estimator="context", n_samples=256, q=0.5)}
# per-method budgets (CPU-scale; same for every backbone)
METHOD_KW = {"occam": dict(n_masks=128), "extremal": dict(iters=30, lr=0.2)}


def stratified(ds: MSRAction3D, per_class: int) -> list[int]:
    seen: dict[int, int] = {}
    out = []
    for i, f in enumerate(ds.files):
        c = ds._label(f)
        if seen.get(c, 0) < per_class:
            seen[c] = seen.get(c, 0) + 1
            out.append(i)
    return out


def faith(model, clip, field, target, steps, metrics="all"):
    di = F.deletion_insertion(model, clip, field, target, steps=steps)
    out = {"deletion_auc": di["deletion_auc"].item(), "insertion_auc": di["insertion_auc"].item(),
           "deletion_curve": di["deletion_curve"][0].tolist(), "insertion_curve": di["insertion_curve"][0].tolist()}
    if metrics == "delins":
        return out
    ao = F.aopc_comp_suff(model, clip, field, target)
    sn = F.sensitivity_n(model, clip, field, target, n_subsets=30, generator=torch.Generator().manual_seed(0))
    return {**out, "comprehensiveness": ao["comprehensiveness"].item(),
            "sufficiency": ao["sufficiency"].item(), "sensitivity_n": sn.item()}


def loc(field, gt, mask, prefix=""):
    if gt is None or not gt.any():
        return {}
    return {prefix + "pg": L.pointing_game(field, gt, mask).item(),
            prefix + "ebpg": L.ebpg(field, gt, mask).item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--methods", default="all")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--metrics", default="all", choices=["all", "delins"])
    p.add_argument("--regions", type=int, default=12)
    p.add_argument("--points", type=int, default=1024, help="points per frame (use the model's training value)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    names = ALL if a.methods == "all" else a.methods.split(",")
    model = OfficialBackbone(a.backbone, a.ckpt, repo=a.repo, device=a.device)
    ds = MSRAction3D(a.data, "test", num_points=a.points, region_method="tracked", region_k=a.regions)
    idxs = stratified(ds, a.per_class)
    raw_path = os.path.join(a.out, "raw.jsonl")
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path) as fh:
            done = {json.loads(l)["clip"] for l in fh if l.strip()}
    print(f"[{a.backbone}] {len(idxs)} clips ({len(done)} done), methods={names}", flush=True)
    with open(raw_path, "a") as fh:
        for n, i in enumerate(idxs):
            name = os.path.basename(ds.files[i])
            if name in done:
                continue
            b = ds.get(i)
            clip = b.clip
            with torch.no_grad():
                logits = model.logits(clip)
            target = logits.argmax(1)
            rec = {"clip": name, "label": int(clip.label), "pred": int(target), "num_points": a.points, "methods": {}}
            for m in names:
                torch.manual_seed(1000 + i)
                t0 = time.time()
                extra = {}
                if m in KCA:
                    at = attribute.explain(model, clip, target, k=a.regions, n_control=0,
                                           generator=torch.Generator().manual_seed(i), **KCA[m])
                    field = at.nec_field
                    dt = time.time() - t0
                    rs = list(at.region_scores.values())
                    nec = torch.stack([r.necessity for r in rs]).abs().mean().item()
                    dint = torch.stack([r.delta_int for r in rs]).abs().mean().item()
                    mv = b.moving_gt
                    sm_mv = at.smot_field[mv].sum().item()
                    sg_mv = at.sgeo_field[mv].sum().item()
                    extra = {"n_forward": at.n_forward, "smot_total": at.smot_field.sum().item(),
                             "sgeo_total": at.sgeo_field.sum().item(),
                             "smot_moving": sm_mv, "sgeo_moving": sg_mv,
                             "abs_dint_over_abs_nec": dint / max(nec, 1e-8),
                             "region_smot": [r.smot.item() for r in rs],
                             "region_sgeo": [r.sgeo.item() for r in rs],
                             **loc(at.smot_field, b.moving_gt, clip.mask, "smot_"),
                             **loc(at.sgeo_field, b.static_gt, clip.mask, "sgeo_")}
                else:
                    field = M.run_method(m, model, clip, target, **METHOD_KW.get(m, {}))
                    dt = time.time() - t0
                    if field is None:
                        rec["methods"][m] = {"na": True}
                        continue
                field = field.detach().to(clip.points.device)
                rec["methods"][m] = {"time_s": dt, **faith(model, clip, field, target, a.steps, a.metrics),
                                     **loc(field, b.moving_gt, clip.mask), **extra}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            tot = sum(v.get("time_s", 0) for v in rec["methods"].values())
            print(f"  [{n + 1}/{len(idxs)}] {name} label {rec['label']} pred {rec['pred']} "
                  f"(explain {tot:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
