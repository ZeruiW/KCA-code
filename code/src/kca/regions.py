"""Spacetime region partition (paper Sec. method-setup).

KCA scores *regions* R: 3D sub-structures tracked over a time window. The canonical
choice in the paper is spatio-temporal supervoxels.

  tracked_regions   CORRESPONDENCE-FREE (use this on real data): k-means per frame,
                    each frame initialised from the previous frame's centres, so region
                    r follows the same structure through time without point identity.
  voxel_grid_regions / kmeans_regions
                    assign by the FRAME-0 position and broadcast by point index, i.e.
                    they ASSUME persistent point identity across frames. That holds for
                    the synthetic controlled tasks but NOT for real point-cloud video
                    (MSR/NTU frames are sampled independently), where they would scatter
                    a "region" randomly over the body at t > 0.
"""
from __future__ import annotations

import torch

from .types import Clip


def voxel_grid_regions(clip: Clip, grid: int = 4) -> torch.Tensor:
    """Partition points into a `grid^3` spatial voxel grid, anchored at frame 0.

    A point's region is decided by its frame-0 position (identity assumed stable
    across frames), so freeze/displace act on a coherent tracked structure. Empty
    voxels are skipped and ids are made contiguous. Returns (B, T, N) long, -1 = pad.
    """
    B, T, N, _ = clip.points.shape
    xyz0 = clip.points[:, 0, :, :3]                               # (B,N,3) anchor frame
    out = torch.full((B, T, N), -1, dtype=torch.long, device=clip.device)
    for b in range(B):
        valid = clip.mask[b, 0]                                   # (N,)
        if valid.sum() == 0:
            continue
        p = xyz0[b][valid]                                        # (n,3)
        lo, hi = p.min(0).values, p.max(0).values
        span = (hi - lo).clamp_min(1e-6)
        cell = ((p - lo) / span * grid).clamp(max=grid - 1e-4).long()  # (n,3)
        flat = cell[:, 0] * grid * grid + cell[:, 1] * grid + cell[:, 2]
        # make ids contiguous in [0, m)
        uniq, inv = torch.unique(flat, return_inverse=True)
        ids_full = torch.full((N,), -1, dtype=torch.long, device=clip.device)
        ids_full[valid] = inv
        # broadcast frame-0 identity to every frame, keeping -1 on padded slots
        out[b] = torch.where(clip.mask[b], ids_full.unsqueeze(0).expand(T, N),
                             torch.full_like(out[b], -1))
    return out


def _fps(points: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest-point sampling indices (deterministic init for k-means)."""
    n = points.shape[0]
    sel = [0]
    d = torch.full((n,), float("inf"), device=points.device)
    for _ in range(1, min(k, n)):
        d = torch.minimum(d, (points - points[sel[-1]]).norm(dim=1))
        sel.append(int(d.argmax().item()))
    return torch.tensor(sel, device=points.device)


def kmeans_regions(clip: Clip, k: int = 8, iters: int = 12) -> torch.Tensor:
    """Spatio-temporal supervoxels: k-means on frame-0 points, broadcast to all frames
    by point identity (so a region is a coherent 3D structure tracked over time). This
    is the paper's intended region type; the voxel grid is a cruder fallback.
    Returns (B,T,N) long, -1 = pad."""
    B, T, N, _ = clip.points.shape
    out = torch.full((B, T, N), -1, dtype=torch.long, device=clip.device)
    for b in range(B):
        v = clip.mask[b, 0]
        p = clip.points[b, 0, :, :3][v]                       # (n,3)
        if p.shape[0] == 0:
            continue
        kk = min(k, p.shape[0])
        centers = p[_fps(p, kk)]
        for _ in range(iters):
            a = torch.cdist(p, centers).argmin(1)             # (n,)
            for j in range(kk):
                m = a == j
                if m.any():
                    centers[j] = p[m].mean(0)
        ids = torch.full((N,), -1, dtype=torch.long, device=clip.device)
        ids[v] = a
        out[b] = torch.where(clip.mask[b], ids.unsqueeze(0).expand(T, N), torch.full_like(out[b], -1))
    return out


def _kmeans(p: torch.Tensor, centers: torch.Tensor, iters: int):
    a = torch.cdist(p, centers).argmin(1)
    for _ in range(iters):
        for j in range(centers.shape[0]):
            m = a == j
            if m.any():
                centers[j] = p[m].mean(0)
        a = torch.cdist(p, centers).argmin(1)
    return a, centers


def tracked_regions(clip: Clip, k: int = 8, iters: int = 8) -> torch.Tensor:
    """Correspondence-free spatio-temporal supervoxels (tracked k-means).

    Frame 0: k-means with farthest-point initialisation started from the lowest point
    (argmin of the vertical coordinate, so the partition does not depend on point order);
    frame t > 0: k-means initialised at frame t-1's centres, so cluster r tracks the same
    structure. A centre whose cluster empties keeps its last position. Returns (B,T,N)
    long, -1 = pad."""
    B, T, N, _ = clip.points.shape
    out = torch.full((B, T, N), -1, dtype=torch.long, device=clip.device)
    for b in range(B):
        centers = None
        for t in range(T):
            v = clip.mask[b, t]
            p = clip.points[b, t, :, :3][v]
            if p.shape[0] == 0:
                continue
            if centers is None:
                kk = min(k, p.shape[0])
                start = int(p[:, 1].argmax())                  # image y points down: max y = lowest
                order = torch.cat([p[start:start + 1], p])
                centers = order[_fps(order, kk)].clone()
            a, centers = _kmeans(p, centers, iters if t == 0 else max(2, iters // 2))
            ids = torch.full((N,), -1, dtype=torch.long, device=clip.device)
            ids[v] = a
            out[b, t] = ids
    return out


def attach_regions(clip: Clip, grid: int = 4, method: str = "voxel", k: int = 8) -> Clip:
    c = clip.clone()
    if method == "tracked":
        c.regions = tracked_regions(clip, k=k)
    elif method == "kmeans":
        c.regions = kmeans_regions(clip, k=k)
    else:
        c.regions = voxel_grid_regions(clip, grid=grid)
    return c
