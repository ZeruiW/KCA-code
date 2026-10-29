"""Convert raw MSR-Action3D (depth .bin + 20-joint skeleton .txt) into the canonical
point-cloud-video .npz used by P4Transformer / PSTNet / Mamba4D, plus skeleton ground
truth in the SAME coordinates (used for motion-localization GT, Tab. mpg).

Depth -> points follows MeteorNet `action_cls/preprocess_file.py` exactly (focal 280,
principal point at the image centre, z = depth value), so the `point_clouds` key is
byte-compatible with the official loaders (which later divide by 300).

Skeleton: the 20-joint files store (u, v, d, conf) per joint with (u, v) in depth-image
pixels. `d` uses a different depth unit than the depth maps, so the joint depth is taken
from the depth map around (u, v) (median of the non-zero window, i.e. the body surface),
falling back to d / r_t with a per-frame unit ratio r_t estimated from the joints that
did hit the silhouette.

Raw data: MeteorNet's MSR-Action3D zip (Depth.rar + MSRAction3DSkeleton(20joints).rar).

Run:  python -m kca.data.prepare_msr --depth <Depth dir> --skeleton <skeleton dir> --out <npz dir>
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import struct

import numpy as np

FOCAL = 280.0
WIN = 3                     # half-window (pixels) for the joint surface-depth lookup

# MSR-Action3D joint order (0-indexed): 0 l-shoulder 1 r-shoulder 2 neck 3 spine 4 l-hip
# 5 r-hip 6 hip-centre 7 l-elbow 8 r-elbow 9 l-wrist 10 r-wrist 11 l-hand 12 r-hand
# 13 l-knee 14 r-knee 15 l-ankle 16 r-ankle 17 l-foot 18 r-foot 19 head
BONES = [(2, 19), (2, 0), (2, 1), (0, 7), (1, 8), (7, 9), (8, 10), (9, 11), (10, 12),
         (2, 3), (3, 6), (6, 4), (6, 5), (4, 13), (5, 14), (13, 15), (14, 16), (15, 17), (16, 18)]


def read_depth(fn: str) -> np.ndarray:
    with open(fn, "rb") as f:
        nf, w, h = struct.unpack("<LLL", f.read(12))
        return np.frombuffer(f.read(), dtype="<u4").reshape(nf, h, w).astype(np.float64)


def depth_to_points(dm: np.ndarray) -> np.ndarray:
    """MeteorNet formula. dm: (H, W) -> (n, 3) float32 over the non-zero pixels."""
    h, w = dm.shape
    yy, xx = np.nonzero(dm > 0)
    z = dm[yy, xx]
    x = (xx - w / 2) / FOCAL * z
    y = (yy - h / 2) / FOCAL * z
    return np.stack([x, y, z], -1).astype(np.float32)


def joints_to_points(dm: np.ndarray, sk: np.ndarray) -> np.ndarray:
    """sk: (20, 4) (u, v, d, conf) for one frame -> (20, 3) in point-cloud coordinates."""
    h, w = dm.shape
    zs = np.full(len(sk), np.nan)
    for j, (u, v, _, _) in enumerate(sk):
        ui, vi = int(round(u)), int(round(v))
        win = dm[max(0, vi - WIN):vi + WIN + 1, max(0, ui - WIN):ui + WIN + 1]
        nz = win[win > 0]
        if nz.size:
            zs[j] = np.median(nz)
    hit = ~np.isnan(zs)
    ratio = np.median(sk[hit, 2] / zs[hit]) if hit.any() else 1.0
    zs[~hit] = sk[~hit, 2] / ratio
    x = (sk[:, 0] - w / 2) / FOCAL * zs
    y = (sk[:, 1] - h / 2) / FOCAL * zs
    return np.stack([x, y, zs], -1).astype(np.float32)


def convert_one(args) -> tuple[str, int, float]:
    depth_fn, skel_dir, out_dir = args
    base = os.path.basename(depth_fn).replace("_sdepth.bin", "")
    depth = read_depth(depth_fn)
    sk = np.loadtxt(os.path.join(skel_dir, f"{base}_skeleton.txt")).reshape(-1, 20, 4)
    T = min(len(depth), len(sk))
    clouds = np.empty(T, dtype=object)
    joints = np.zeros((T, 20, 3), np.float32)
    for t in range(T):
        clouds[t] = depth_to_points(depth[t])
        joints[t] = joints_to_points(depth[t], sk[t])
    # alignment diagnostic: median distance joint -> nearest point (depth units, ~5 mm)
    d = [np.sqrt(((clouds[t][None] - joints[t][:, None]) ** 2).sum(-1)).min(1)
         for t in range(0, T, max(1, T // 4))]
    np.savez_compressed(os.path.join(out_dir, f"{base}_sdepth.npz"), point_clouds=clouds,
                        joints=joints, joint_conf=sk[:T, :, 3].astype(np.float32))
    return base, T, float(np.median(np.concatenate(d)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", required=True, help="dir with aNN_sNN_eNN_sdepth.bin")
    p.add_argument("--skeleton", required=True, help="dir with aNN_sNN_eNN_skeleton.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.depth, "*_sdepth.bin")))
    with mp.Pool(a.workers) as pool:
        res = pool.map(convert_one, [(f, a.skeleton, a.out) for f in files])
    align = np.array([r[2] for r in res])
    print(f"converted {len(res)} videos -> {a.out}; frames {min(r[1] for r in res)}-"
          f"{max(r[1] for r in res)}; joint->surface median dist {np.median(align):.1f} "
          f"(p95 {np.percentile(align, 95):.1f}) depth units")


if __name__ == "__main__":
    main()
