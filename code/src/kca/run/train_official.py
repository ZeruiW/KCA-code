"""Train an OFFICIAL backbone (P4Transformer / PSTNet / Mamba4D) on MSR-Action3D — random
per-axis scaling in [0.9, 1.1], coordinates / 300, subjects <= 5 train / > 5 test (the
official data protocol) — on CUDA, or on CPU / Apple MPS through the pure-PyTorch ops in
`kca.backbones.cpu_ops`.

`--opt sgd` is the official optimiser recipe (SGD 0.9, wd 1e-4, warm-up + step decay; it
needs the official ~17k iterations). `--opt adamw` (default) converges in the few
thousand iterations a laptop affords (AdamW 3e-4, warm-up + cosine).

Two changes make the from-scratch runs leave the initial plateau within a few hundred
iterations (measured, 2026-09-26): P4Transformer/Mamba4D embed ABSOLUTE (x, y, z, t)
with t = 1..L, so at initialisation the time channel dominates every token (|pos| ~ 34
vs |features| ~ 7) and the token max-pool picks tokens by frame index, not content.
  --t-init-scale   rescales the INITIAL weights of that time channel (default 1/L; the
                   architecture is unchanged),
  --center         subtracts each clip's centroid (a preprocessing step stored in the
                   checkpoint and re-applied by OfficialBackbone at explanation time).
`--scale reduced` keeps each official architecture but shrinks its width/depth (REDUCED)
so all three train on a laptop; `--scale official` uses the published configs.

Checkpoint (loadable by `OfficialBackbone(name, checkpoint)`):
    {'model': state_dict, 'cfg': model kwargs, 'num_classes', 'epoch', 'acc', 'train_args'}

Run:  python -m kca.run.train_official --backbone p4transformer --data <npz dir> --out <dir>
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

from ..backbones import cpu_ops
from ..backbones.official import build_net, preprocess

REDUCED = {
    "p4transformer": dict(spatial_stride=16, dim=256, depth=4, heads=4, dim_head=64, mlp_dim=512),
    "pstnet": dict(),
    "mamba4d": dict(spatial_stride=16, dim=256, mlp_dim=512, depth_mamba_inter=2, depth_mamba_intra=1),
}


class MSRVideos:
    """All videos of one split, pre-loaded (each frame capped at `max_pts` points)."""

    def __init__(self, root: str, train: bool, max_pts: int = 2048, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.videos, self.labels, self.names = [], [], []
        for fn in sorted(glob.glob(os.path.join(root, "*.npz"))):
            base = os.path.basename(fn)
            subject = int(base.split("_")[1][1:])
            if (subject <= 5) != train:
                continue
            pcs = np.load(fn, allow_pickle=True)["point_clouds"]
            frames = []
            for p in pcs:
                if len(p) > max_pts:
                    p = p[rng.choice(len(p), max_pts, replace=False)]
                frames.append(p.astype(np.float32))
            self.videos.append(frames)
            self.labels.append(int(base.split("_")[0][1:]) - 1)
            self.names.append(base)
        self.num_classes = 20

    def clip(self, vi: int, t0: int, T: int, N: int, rng, scale_aug: bool) -> np.ndarray:
        out = np.empty((T, N, 3), np.float32)
        for j, p in enumerate(self.videos[vi][t0:t0 + T]):
            if len(p) >= N:
                sel = rng.choice(len(p), N, replace=False)
            else:                                          # official cyclic repeat
                rep, res = divmod(N, len(p))
                sel = np.concatenate([np.arange(len(p))] * rep + [rng.choice(len(p), res, replace=False)])
            out[j] = p[sel]
        if scale_aug:
            out = out * rng.uniform(0.9, 1.1, size=3).astype(np.float32)
        return out / 300.0


def evaluate(net, data: MSRVideos, T: int, N: int, device, clips_per_video: int = 8, seed: int = 0,
             preproc: dict | None = None):
    """Video-level accuracy: mean logits over evenly spaced clips of each test video."""
    rng = np.random.RandomState(seed)
    net.eval()
    correct = total = 0
    with torch.no_grad(), cpu_ops.device_compat():
        for vi, frames in enumerate(data.videos):
            if len(frames) < T:
                continue
            starts = np.unique(np.linspace(0, len(frames) - T, clips_per_video).round().astype(int))
            x = torch.from_numpy(np.stack([data.clip(vi, s, T, N, rng, False) for s in starts])).to(device)
            pred = net(preprocess(x, preproc or {})).mean(0).argmax().item()
            correct += int(pred == data.labels[vi])
            total += 1
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=["p4transformer", "pstnet", "mamba4d"])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--scale", default="reduced", choices=["reduced", "official"])
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--points", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr", type=float, default=None, help="default 1e-3 (adamw) / 1e-2 (sgd)")
    p.add_argument("--wd", type=float, default=None, help="default 0.05 (adamw) / 1e-4 (sgd)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--milestones", type=float, nargs=2, default=[0.6, 0.85],
                   help="sgd step decay, fractions of --epochs")
    p.add_argument("--center", type=int, default=1, help="subtract each clip's centroid (1/0)")
    p.add_argument("--t-init-scale", type=float, default=None,
                   help="initial scale of the pos-embedding time channel (default 1/L)")
    p.add_argument("--clips-per-video", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/state.pth (written with every evaluation, removed at the end)")
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train.log"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    cfg = REDUCED[a.backbone] if a.scale == "reduced" else {}
    tr, te = MSRVideos(a.data, True, seed=a.seed), MSRVideos(a.data, False, seed=a.seed)
    net = build_net(a.backbone, tr.num_classes, a.repo, **cfg)
    pe = getattr(net, "pos_embedding", None)
    if isinstance(pe, torch.nn.Conv1d) and pe.in_channels == 4:
        L = len(range(0, a.frames, cfg.get("temporal_stride", 2)))
        a.t_init_scale = 1.0 / L if a.t_init_scale is None else a.t_init_scale
        with torch.no_grad():
            pe.weight[:, 3] *= a.t_init_scale
    preproc = {"center": bool(a.center)}
    net = net.to(device)
    nparam = sum(q.numel() for q in net.parameters()) / 1e6
    say(f"[{a.backbone}] device={device} cfg={cfg} params={nparam:.2f}M train={len(tr.videos)} "
        f"test={len(te.videos)} T={a.frames} N={a.points}")
    if a.opt == "sgd":
        a.lr, a.wd = a.lr or 1e-2, 1e-4 if a.wd is None else a.wd
        opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    else:
        a.lr, a.wd = a.lr or 3e-4, 0.05 if a.wd is None else a.wd
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    m1, m2 = (int(f * a.epochs) for f in a.milestones)
    eligible = [i for i, v in enumerate(tr.videos) if len(v) >= a.frames]

    def lr_at(ep, it, n_it):
        if ep < a.warmup:
            return a.lr * (0.01 + 0.99 * (ep * n_it + it) / (a.warmup * n_it))
        if a.opt == "sgd":
            return a.lr * (0.1 ** ((ep >= m1) + (ep >= m2)))
        prog = (ep - a.warmup + it / n_it) / max(1, a.epochs - a.warmup)
        return a.lr * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * prog)))

    best, acc, start = 0.0, 0.0, 0
    state_path = os.path.join(a.out, "state.pth")
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        best, acc, start = st["best"], st["acc"], st["epoch"]
        rng.set_state(st["rng"])
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        say(f"[resume] from epoch {start}")
    for ep in range(start, a.epochs):
        net.train()
        items = [(vi, rng.randint(0, len(tr.videos[vi]) - a.frames + 1))
                 for vi in eligible for _ in range(a.clips_per_video)]
        rng.shuffle(items)
        n_it = (len(items) + a.bs - 1) // a.bs
        t0, tot, cor, seen = time.time(), 0.0, 0, 0
        for it in range(n_it):
            chunk = items[it * a.bs:(it + 1) * a.bs]
            x = torch.from_numpy(np.stack([tr.clip(vi, s, a.frames, a.points, rng, True) for vi, s in chunk])).to(device)
            y = torch.tensor([tr.labels[vi] for vi, _ in chunk], device=device)
            for g in opt.param_groups:
                g["lr"] = lr_at(ep, it, n_it)
            with cpu_ops.device_compat():
                out = net(preprocess(x, preproc))
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(chunk); cor += (out.argmax(1) == y).sum().item(); seen += len(chunk)
        msg = f"ep {ep + 1:3d}/{a.epochs} loss {tot / seen:.3f} train-acc {cor / seen:.3f} lr {opt.param_groups[0]['lr']:.4g} ({time.time() - t0:.0f}s)"
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            acc = evaluate(net, te, a.frames, a.points, device, preproc=preproc)
            msg += f"  test-video-acc {acc:.3f}"
            ck = {"model": net.state_dict(), "cfg": cfg, "preproc": preproc, "num_classes": tr.num_classes,
                  "epoch": ep + 1, "acc": acc, "train_args": vars(a)}
            torch.save(ck, os.path.join(a.out, "last.pth"))
            if acc >= best:
                best = acc
                torch.save(ck, os.path.join(a.out, "best.pth"))
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best,
                        "acc": acc, "rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       state_path)
        say(msg)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"backbone": a.backbone, "last_acc": acc, "best_acc": best, "params_M": nparam,
                   "cfg": cfg, "preproc": preproc, "args": vars(a)}, fh, indent=2)
    if os.path.exists(state_path):
        os.remove(state_path)
    say(f"[done] last-epoch test acc {acc:.3f} (best {best:.3f}) -> {a.out}")


if __name__ == "__main__":
    main()
