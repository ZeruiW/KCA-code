"""Real-data motion/geometry crossover (double dissociation) on MSR-Action3D.

Two binary tasks are built from real MSR clips; in both, the two classes differ ONLY in
one physical structure — the subject's right arm (shoulder -> hand, from the skeleton):
  motion task    class 1 = original clip; class 0 = the same clip with the arm's motion
                 slowed to `alpha` x speed (its frames replayed from t0 + alpha (t - t0)):
                 the arm's geometry is real, only its motion differs.
  geometry task  class 1 = original; class 0 = the same clip with the arm scaled radially
                 about its bones by `beta` (every frame; beta < 1 thins it): its motion is
                 intact, only its geometry differs.
Explained class: the one whose defining property is PRESENT and removable by KCA's
interventions — the original class on original clips in the motion task (freeze removes
its normal-speed motion), the thickened class on thickened clips in the geometry task
(the proxy removes its thick shape). Variants kept for the record: explaining the
original class in the geometry task (its evidence is the ABSENCE of the cue; the shares
then depend on the sign convention; crossover_beta2/3.json), and thinning the arm
(beta = 0.3), which the P4Transformer could not learn (chance accuracy; crossover_thin.json).
`--backbone pstnet|mamba4d` repeats the test on another official architecture.
(Slowing, not freezing, keeps the motion cue distinct from KCA's own freeze operator;
the whole arm is edited so it stays attached at the shoulder.) A P4Transformer is
trained per task; KCA explains held-out clips and we measure, on the arm points,
    motion share = Smot / (Smot + Sgeo).
A correct decomposition gives a high share on the motion model and a low share on the
geometry model FOR THE SAME REGION; the model-blind motion prior flags the arm as moving
in both tasks and cannot make the distinction.

Run:  python -m kca.run.msr_crossover --data <npz dir> --out results/msr/crossover
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import attribute, regions as rg
from ..backbones import cpu_ops
from ..backbones.official import OfficialBackbone, build_net, preprocess
from ..baselines import floors
from ..data.msr_action3d import BONES, bone_motion, nearest_bone
from ..metrics import localization as L
from ..types import Clip
from .train_official import REDUCED

ARM_BONES = [i for i, b in enumerate(BONES) if b in [(1, 8), (8, 10), (10, 12)]]   # r-shoulder -> r-hand


class ArmClips:
    """Centred 24-frame clips with per-point arm masks (nearest bone in ARM_BONES)."""

    def __init__(self, root, train, T=24, N=1024, seed=0, min_arm_share=0.3):
        rng = np.random.RandomState(seed)
        self.items = []
        for fn in sorted(glob.glob(os.path.join(root, "*.npz"))):
            subject = int(os.path.basename(fn).split("_")[1][1:])
            if (subject <= 5) != train:
                continue
            d = np.load(fn, allow_pickle=True)
            pcs, J = d["point_clouds"], d["joints"]
            if len(pcs) < T:
                continue
            s = (len(pcs) - T) // 2
            idx = list(range(s, s + T))
            Jc = J[idx]
            mot = bone_motion(Jc)
            if mot[ARM_BONES].sum() < min_arm_share * mot.sum():
                continue                                    # keep clips whose arm really moves
            pts = np.stack([p[rng.choice(len(p), N, replace=len(p) < N)] for p in (pcs[i] for i in idx)])
            lab = np.stack([nearest_bone(pts[t], Jc[t]) for t in range(T)])
            self.items.append((pts.astype(np.float32), Jc.astype(np.float32), np.isin(lab, ARM_BONES), lab))

    def slowed(self, k, alpha):
        pts, J, arm, _ = self.items[k]
        out = pts.copy()
        for t in range(len(pts)):
            s = int(round(alpha * t))
            src = pts[s][arm[s]]
            dst = np.nonzero(arm[t])[0]
            if len(src) and len(dst):
                out[t, dst] = src[np.arange(len(dst)) % len(src)]
        return out

    def thickened(self, k, beta):
        pts, J, arm, lab = self.items[k]
        out = pts.copy()
        for t in range(len(pts)):
            for bi in ARM_BONES:
                sel = arm[t] & (lab[t] == bi)
                if not sel.any():
                    continue
                a, b = J[t, BONES[bi][0]], J[t, BONES[bi][1]]
                ab = b - a
                p = pts[t, sel]
                u = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
                proj = a + u[:, None] * ab
                out[t, sel] = proj + beta * (p - proj)
        return out

    def sample(self, k, task, positive, alpha, beta):
        if positive:
            return self.items[k][0]
        return self.slowed(k, alpha) if task == "motion" else self.thickened(k, beta)


def train_task(task, tr, te, a, device, cfg):
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    net = build_net(a.backbone, 2, a.repo, **cfg)
    pe = getattr(net, "pos_embedding", None)
    if isinstance(pe, torch.nn.Conv1d) and pe.in_channels == 4:     # time-channel init, as in train_official
        with torch.no_grad():
            pe.weight[:, 3] *= 1.0 / 12
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    items = [(k, pos) for k in range(len(tr.items)) for pos in (0, 1)]
    n_it = a.epochs * ((len(items) + 7) // 8)
    it = 0
    for ep in range(a.epochs):
        net.train()
        rng.shuffle(items)
        for s in range(0, len(items), 8):
            ch = items[s:s + 8]
            x = np.stack([tr.sample(k, task, pos, a.alpha, a.beta) for k, pos in ch]) / 300.0
            x = x * rng.uniform(0.9, 1.1, size=(len(ch), 1, 1, 3))
            x = preprocess(torch.from_numpy(x).float(), {"center": True}).to(device)
            y = torch.tensor([pos for _, pos in ch], device=device)
            for g in opt.param_groups:
                g["lr"] = 3e-4 * 0.5 * (1 + np.cos(np.pi * it / n_it))
            with cpu_ops.device_compat():
                loss = F.cross_entropy(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
    net.eval()
    correct = 0
    with torch.no_grad(), cpu_ops.device_compat():
        for k in range(len(te.items)):
            for pos in (0, 1):
                x = preprocess(torch.from_numpy(te.sample(k, task, pos, a.alpha, a.beta)[None] / 300.0).float(),
                               {"center": True}).to(device)
                correct += int(net(x).argmax(1).item() == pos)
    return net.cpu(), correct / (2 * len(te.items))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--n-eval", type=int, default=40)
    p.add_argument("--regions", type=int, default=12)
    p.add_argument("--device", default=None, help="training device")
    p.add_argument("--explain-device", default="cpu")
    p.add_argument("--backbone", default="p4transformer", choices=["p4transformer", "pstnet", "mamba4d"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--train-only", action="store_true", help="train and save both task models, then stop")
    p.add_argument("--tasks", default="motion,geometry", help="subset to run; results merge into crossover.json")
    p.add_argument("--seed", type=int, default=0, help="training seed of the two task models")
    p.add_argument("--eval-only", action="store_true", help="reuse <out>/<task>.pth")
    p.add_argument("--geo-explain", default="modified", choices=["modified", "original"],
                   help="geometry task: explain the class whose defining shape is PRESENT "
                        "(modified = the thickened clips, class 0) or the original class")
    p.add_argument("--scale", default="reduced", choices=["reduced", "official"], help="task-model width")
    p.add_argument("--points", type=int, default=1024, help="points per frame")
    p.add_argument("--reuse", action="store_true", help="load <out>/<task>.pth when present instead of retraining")
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = REDUCED[a.backbone] if a.scale == "reduced" else {}
    tr, te = ArmClips(a.data, True, N=a.points), ArmClips(a.data, False, N=a.points)
    print(f"arm-moving clips: train {len(tr.items)} test {len(te.items)}", flush=True)
    res = {"alpha": a.alpha, "beta": a.beta, "n_train": len(tr.items), "n_test": len(te.items),
           "geo_explain": a.geo_explain, "seed": a.seed, "backbone": a.backbone, "scale": a.scale,
           "points": a.points}
    tasks = a.tasks.split(",")
    out_json = os.path.join(a.out, "crossover.json")
    if set(tasks) != {"motion", "geometry"} and os.path.exists(out_json):
        res = {**json.load(open(out_json)), **res}          # keep the other task's results
    for task in tasks:
        t0 = time.time()
        ck = os.path.join(a.out, f"{task}.pth")
        if (a.eval_only or a.reuse) and os.path.exists(ck):
            saved = torch.load(ck, map_location="cpu", weights_only=False)
            net = build_net(a.backbone, 2, a.repo, **saved.get("cfg", REDUCED[a.backbone]))
            net.load_state_dict(saved["model"]); net.eval()
            acc = saved["test_acc"]
        else:
            net, acc = train_task(task, tr, te, a, device, cfg)
            torch.save({"model": net.state_dict(), "test_acc": acc, "backbone": a.backbone, "cfg": cfg}, ck)
            print(f"[{task}] trained: test acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if a.train_only:
            continue
        model = OfficialBackbone(a.backbone, net=net, num_classes=2, device=a.explain_device)
        model.preproc = {"center": True}
        rec = {"test_acc": acc, "train_s": time.time() - t0}
        for est, kw in (("kca", dict(estimator="local")), ("kca_ctx", dict(estimator="context", n_samples=256))):
            sm = sg = smp = sgp = 0.0
            ebpg, per_clip = [], []
            for k in range(min(a.n_eval, len(te.items))):
                pts, J, arm, _ = te.items[k]
                tgt = torch.tensor([1])                      # evidence for "original"
                if task == "geometry" and a.geo_explain == "modified":
                    pts, tgt = te.sample(k, task, False, a.alpha, a.beta), torch.tensor([0])
                clip = Clip(points=torch.from_numpy(pts[None] / 300.0).float(),
                            mask=torch.ones(1, *pts.shape[:2], dtype=torch.bool))
                clip = rg.attach_regions(clip, method="tracked", k=a.regions)
                at = attribute.explain(model, clip, tgt, k=a.regions, n_control=0,
                                       generator=torch.Generator().manual_seed(k), **kw)
                g = torch.from_numpy(arm)[None]
                sm += at.smot_field[g].sum().item()
                sg += at.sgeo_field[g].sum().item()
                a_, b_ = at.smot_field.clamp_min(0)[g].sum().item(), at.sgeo_field.clamp_min(0)[g].sum().item()
                smp += a_
                sgp += b_
                per_clip.append((a_, b_))
                ebpg.append(L.ebpg(at.nec_field, g, clip.mask).item())
            rec[est] = {"arm_motion_share": smp / (smp + sgp) if smp + sgp else float("nan"),   # positive parts
                        "arm_motion_share_signed": sm / (sm + sg) if sm + sg else float("nan"),
                        "arm_smot": sm, "arm_sgeo": sg, "nec_ebpg_arm": float(np.mean(ebpg)),
                        "per_clip_pos": per_clip}
        mp = []
        for k in range(min(a.n_eval, len(te.items))):
            pts, J, arm, _ = te.items[k]
            clip = Clip(points=torch.from_numpy(pts[None] / 300.0).float(),
                        mask=torch.ones(1, *pts.shape[:2], dtype=torch.bool))
            mp.append(L.ebpg(floors.motion_prior(clip), torch.from_numpy(arm)[None], clip.mask).item())
        rec["motion_prior_ebpg_arm"] = float(np.mean(mp))
        res[task] = rec
        if "motion" in res and "geometry" in res:          # bootstrap CIs over clips (pooled share)
            rng = np.random.RandomState(0)
            for est in ("kca", "kca_ctx"):
                pm = np.array(res["motion"][est]["per_clip_pos"]); pg = np.array(res["geometry"][est]["per_clip_pos"])
                share = lambda x: x[:, 0].sum() / max(x.sum(), 1e-12)
                bm, bg = [], []
                for _ in range(2000):
                    bm.append(share(pm[rng.randint(0, len(pm), len(pm))]))
                    bg.append(share(pg[rng.randint(0, len(pg), len(pg))]))
                bm, bg = np.array(bm), np.array(bg)
                res[f"ci_{est}"] = {"motion": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                    "geometry": [float(np.percentile(bg, 2.5)), float(np.percentile(bg, 97.5))],
                                    "gap": [float(np.percentile(bm - bg, 2.5)), float(np.percentile(bm - bg, 97.5))]}
        print(f"[{task}] acc {acc:.3f} | KCA arm motion share {rec['kca']['arm_motion_share']:.3f} "
              f"(ctx {rec['kca_ctx']['arm_motion_share']:.3f}) | motion-prior EBPG(arm) {rec['motion_prior_ebpg_arm']:.2f}",
              flush=True)
        if not a.train_only:
            with open(out_json, "w") as fh:
                json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
