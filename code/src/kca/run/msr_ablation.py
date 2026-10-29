"""Ablation, efficiency and sanity studies of KCA on a real backbone (MSR-Action3D).

  ablation   region count m, ablation vertex (proxy / true removal), estimator (local /
             context with S samples and keep-probability q), symmetric vs motion-first,
             anchor frame t0 of freeze / the static proxy (first vs middle frame)
  sanity     Adebayo cascading randomisation: re-initialise the backbone from the head
             down; the attribution's rank correlation with the trained model's must fall
  displace   on the top-Smot region: effect of reversing its trajectory vs freezing it
Writes <out>/ablation.json.

Run:  python -m kca.run.msr_ablation --ckpt <p4t last.pth> --data <npz dir> --out results/msr/ablation
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from .. import attribute, interventions as itv, methods as M
from ..backbones.official import OfficialBackbone
from ..data.msr_action3d import MSRAction3D
from ..metrics import faithfulness as F, localization as L, soundness as S
from .msr_benchmark import stratified


def run_variant(model, ds, idxs, m, **kw):
    rows = []
    for i in idxs:
        ds.region_k = m
        b = ds.get(i)
        clip = b.clip
        with torch.no_grad():
            tgt = model.logits(clip).argmax(1)
        t0 = time.time()
        at = attribute.explain(model, clip, tgt, k=m, n_control=0, generator=torch.Generator().manual_seed(i), **kw)
        dt = time.time() - t0
        di = F.deletion_insertion(model, clip, at.nec_field, tgt, steps=20)
        rs = list(at.region_scores.values())
        nec = torch.stack([r.necessity for r in rs]).abs().mean().item()
        dint = torch.stack([r.delta_int for r in rs]).abs().mean().item() if rs[0].delta_int is not None else float("nan")
        mv = b.moving_gt
        rows.append({"deletion_auc": di["deletion_auc"].item(), "insertion_auc": di["insertion_auc"].item(),
                     "smot_ebpg": L.ebpg(at.smot_field, mv, clip.mask).item(),
                     "smot_pg": L.pointing_game(at.smot_field, mv, clip.mask).item(),
                     "smot_moving": at.smot_field.clamp_min(0)[mv].sum().item(),       # positive parts
                     "sgeo_moving": at.sgeo_field.clamp_min(0)[mv].sum().item(),
                     "dint_over_nec": dint / max(nec, 1e-8), "time_s": dt, "n_forward": at.n_forward})
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k not in ("smot_moving", "sgeo_moving")}
    sm, sg = sum(r["smot_moving"] for r in rows), sum(r["sgeo_moving"] for r in rows)
    out["motion_share_moving"] = sm / (sm + sg) if sm + sg else float("nan")
    return out


def randomize_from_top(net, stage):
    """Re-initialise the last `stage` parameterised blocks (head first), Adebayo-style."""
    blocks = []
    if hasattr(net, "mlp_head"):
        blocks.append(net.mlp_head)
    for name in ("transformer", "mambaBlocks"):
        if hasattr(net, name):
            layers = getattr(net, name).layers
            blocks += list(layers)[::-1]
    for name in ("pos_embedding", "tube_embedding"):
        if hasattr(net, name):
            blocks.append(getattr(net, name))
    g = torch.Generator().manual_seed(123)
    for blk in blocks[:stage]:
        for p in blk.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g).to(p.device) * (p.std() if p.numel() > 1 and p.std() > 0 else 0.02))
    return len(blocks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--backbone", default="p4transformer")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--parts", default="ablation,sanity,displace")
    p.add_argument("--device", default="cpu")
    p.add_argument("--points", type=int, default=1024, help="points per frame (use the model's training value)")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "ablation.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    model = OfficialBackbone(a.backbone, a.ckpt, device=a.device)
    ds = MSRAction3D(a.data, "test", num_points=a.points, region_method="tracked", region_k=12)
    idxs = stratified(ds, a.per_class)

    def save():
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)

    parts = a.parts.split(",")
    if "ablation" in parts:
        variants = {}
        for m in (6, 12, 24):
            variants[f"local_m{m}"] = (m, dict(estimator="local"))
            variants[f"ctx_m{m}"] = (m, dict(estimator="context", n_samples=256))
        variants["local_m12_bg"] = (12, dict(estimator="local", baseline="bg"))
        variants["ctx_m12_bg"] = (12, dict(estimator="context", n_samples=256, baseline="bg"))
        variants["local_m12_nn"] = (12, dict(estimator="local", baseline="nn"))
        variants["local_m12_motionfirst"] = (12, dict(estimator="local", symmetric=False))
        for S_ in (64, 128, 512):
            variants[f"ctx_m12_S{S_}"] = (12, dict(estimator="context", n_samples=S_))
        variants["ctx_m12_q0.75"] = (12, dict(estimator="context", n_samples=256, q=0.75))
        variants["local_m12_t0mid"] = (12, dict(estimator="local", t0=12))          # anchor frame: middle, not first
        variants["ctx_m12_t0mid"] = (12, dict(estimator="context", n_samples=256, t0=12))
        res.setdefault("ablation", {})
        for name, (m, kw) in variants.items():
            if name in res["ablation"]:
                continue
            res["ablation"][name] = run_variant(model, ds, idxs, m, **kw)
            print(name, {k: round(v, 3) for k, v in res["ablation"][name].items()}, flush=True)
            save()
        ds.region_k = 12

    if "sanity" in parts and "sanity" not in res:
        base = {}
        sub = idxs[:20]
        for i in sub:
            clip = ds.get(i).clip
            tgt = model.logits(clip).argmax(1)
            base[i] = (tgt, attribute.explain(model, clip, tgt, k=12, n_control=0).nec_field,
                       M.run_method("grad_x_input", model, clip, tgt))
        n_blocks = randomize_from_top(copy.deepcopy(model.net).cpu(), 0)
        curve = []
        for stage in range(1, n_blocks + 1):
            rnd = OfficialBackbone(a.backbone, net=copy.deepcopy(model.net).cpu(), num_classes=model.num_classes, device=a.device)
            rnd.preproc = model.preproc
            randomize_from_top(rnd.net, stage)
            rk, rg = [], []
            for i in sub:
                clip = ds.get(i).clip
                tgt, f_k, f_g = base[i]
                rk.append(S.rank_correlation(f_k, attribute.explain(rnd, clip, tgt, k=12, n_control=0).nec_field, clip.mask).item())
                rg.append(S.rank_correlation(f_g, M.run_method("grad_x_input", rnd, clip, tgt), clip.mask).item())
            curve.append({"blocks_randomized": stage, "kca_rankcorr": float(np.mean(rk)), "gradxinput_rankcorr": float(np.mean(rg))})
            print("sanity", curve[-1], flush=True)
        res["sanity"] = curve
        save()

    if "displace" in parts and "displace" not in res:
        rows = []
        for i in idxs:
            b = ds.get(i)
            clip = b.clip
            with torch.no_grad():
                lg = model.logits(clip)
            tgt = lg.argmax(1)
            at = attribute.explain(model, clip, tgt, k=12, n_control=0)
            r = max(at.region_scores, key=lambda k: at.region_scores[k].smot.item())
            with torch.no_grad():
                phi = lg.gather(1, tgt.view(-1, 1)).item()
                lf = model.logits(itv.freeze(clip, r))
                lr = model.logits(itv.displace(clip, r, mode="reverse"))
            rows.append({"freeze_drop": phi - lf[0, tgt].item(), "reverse_drop": phi - lr[0, tgt].item(),
                         "freeze_flip": float(lf.argmax(1).item() != tgt.item()),
                         "reverse_flip": float(lr.argmax(1).item() != tgt.item())})
        res["displace"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print("displace", res["displace"], flush=True)
        save()


if __name__ == "__main__":
    main()
