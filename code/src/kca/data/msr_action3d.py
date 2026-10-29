"""MSR-Action3D loader, matching the canonical hehefan P4Transformer/PSTNet format.

Reference: github.com/hehefan/P4Transformer  datasets/msr.py
  - one .npz per video, key 'point_clouds', shape (num_frames,) of (num_pts_t, 3)
  - filenames like 'a01_s01_e01_sdepth.npz'; subject id from '_sNN_'
  - standard split: subject id <= 5 -> train (270 videos), > 5 -> test (297)
  - per-frame random sampling of num_points (cyclic repeat if fewer); coords / 300
  - 20 action classes; label = aNN - 1
Frames are sampled INDEPENDENTLY, so there is no point identity across frames: regions
default to the correspondence-free `tracked` supervoxels.

Ground truth (Tab. mpg). `kca.data.prepare_msr` also stores the 20-joint skeleton in the
same coordinates ('joints'). Every point is assigned to its nearest bone segment; a bone
is MOVING if its midpoint path length over the clip is >= `moving_frac` x the clip's
maximum, STATIC if <= `static_frac` x maximum. moving_gt / static_gt are the points of
moving / static bones. Sampling is seeded per clip index, so a clip is reproducible.

This is the explanation *subject* loader (no augmentation — we explain fixed inputs).
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch

from ..types import Clip
from .base import Batch, Dataset

_SUBJ = re.compile(r"_s(\d+)_")
_ACTION = re.compile(r"a(\d+)_")
_COORD_NORM = 300.0

ACTIONS = ["high arm wave", "horizontal arm wave", "hammer", "hand catch", "forward punch",
           "high throw", "draw x", "draw tick", "draw circle", "hand clap", "two hand wave",
           "side-boxing", "bend", "forward kick", "side kick", "jogging", "tennis swing",
           "tennis serve", "golf swing", "pick up & throw"]

BONES = [(2, 19), (2, 0), (2, 1), (0, 7), (1, 8), (7, 9), (8, 10), (9, 11), (10, 12),
         (2, 3), (3, 6), (6, 4), (6, 5), (4, 13), (5, 14), (13, 15), (14, 16), (15, 17), (16, 18)]


def nearest_bone(points: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """points (N,3), joints (20,3) -> (N,) index of the nearest bone segment."""
    a = joints[[b[0] for b in BONES]]                          # (nb,3)
    b = joints[[b[1] for b in BONES]]
    ab = b - a
    t = np.clip(((points[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9), 0, 1)
    proj = a[None] + t[..., None] * ab[None]                   # (N,nb,3)
    return np.linalg.norm(points[:, None] - proj, axis=-1).argmin(1)


def bone_motion(joints: np.ndarray) -> np.ndarray:
    """joints (T,20,3) -> (nb,) path length of each bone midpoint over the clip."""
    mid = 0.5 * (joints[:, [b[0] for b in BONES]] + joints[:, [b[1] for b in BONES]])
    return np.linalg.norm(np.diff(mid, axis=0), axis=-1).sum(0)


class MSRAction3D(Dataset):
    name = "msr_action3d"
    num_classes = 20

    def __init__(self, root: str, split: str = "test", frames: int = 24,
                 num_points: int = 1024, frame_interval: int = 1, grid: int = 4,
                 region_method: str = "tracked", region_k: int = 12, seed: int = 0,
                 moving_frac: float = 0.5, static_frac: float = 0.2):
        super().__init__(grid=grid, region_method=region_method, region_k=region_k)
        self.root, self.split = root, split
        self.frames, self.num_points, self.frame_interval = frames, num_points, frame_interval
        self.seed, self.moving_frac, self.static_frac = seed, moving_frac, static_frac
        files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not files:
            raise FileNotFoundError(
                f"No .npz under {root!r}. Build them with `python -m kca.data.prepare_msr` "
                "from the raw MSR-Action3D depth + skeleton files.")
        self.files = [f for f in files if self._in_split(f) and self._long_enough(f)]

    def _in_split(self, path: str) -> bool:
        m = _SUBJ.search(os.path.basename(path))
        if not m:
            return True
        subj = int(m.group(1))
        return (subj <= 5) if self.split == "train" else (subj > 5)

    def _long_enough(self, path: str) -> bool:
        with np.load(path, allow_pickle=True) as d:
            return len(d["point_clouds"]) >= (self.frames - 1) * self.frame_interval + 1

    def __len__(self) -> int:
        return len(self.files)

    def _label(self, path: str) -> int:
        m = _ACTION.search(os.path.basename(path))
        return int(m.group(1)) - 1 if m else 0

    def _window(self, F: int) -> list[int]:
        span = (self.frames - 1) * self.frame_interval + 1
        start = max(0, (F - span) // 2)                        # centred clip
        return [min(F - 1, start + i * self.frame_interval) for i in range(self.frames)]

    def _raw(self, idx: int) -> Batch:
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        pcs = data["point_clouds"]
        idxs = self._window(len(pcs))
        rng = np.random.RandomState(self.seed * 100003 + idx)
        out = np.zeros((self.frames, self.num_points, 3), dtype=np.float32)
        for j, fi in enumerate(idxs):
            p = pcs[fi]
            n = p.shape[0]
            if n >= self.num_points:
                sel = rng.choice(n, self.num_points, replace=False)
            else:                                   # cyclic repeat (official behaviour)
                rep, res = divmod(self.num_points, n)
                sel = np.concatenate([np.arange(n)] * rep + [rng.choice(n, res, replace=False)])
            out[j] = p[sel]
        moving = static = None
        if "joints" in data.files:
            J = data["joints"][idxs]                               # (T,20,3)
            mot = bone_motion(J)
            mov_b = mot >= self.moving_frac * mot.max()
            sta_b = mot <= self.static_frac * mot.max()
            lab = np.stack([nearest_bone(out[t], J[t]) for t in range(self.frames)])   # (T,N)
            moving = torch.from_numpy(mov_b[lab]).unsqueeze(0)
            static = torch.from_numpy(sta_b[lab]).unsqueeze(0)
        pts = torch.from_numpy(out / _COORD_NORM).unsqueeze(0)          # (1,T,N,3)
        mask = torch.ones(pts.shape[:3], dtype=torch.bool)
        label = torch.tensor([self._label(path)], dtype=torch.long)
        return Batch(clip=Clip(points=pts, mask=mask, label=label), moving_gt=moving, static_gt=static)
