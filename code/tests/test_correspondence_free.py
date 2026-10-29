"""Correspondence-free interventions: real point-cloud video has NO point identity across
frames, so KCA must not depend on point order within a frame.

Verifiable success criteria:
  1. Per-frame permutation invariance: shuffling every frame's point order (and
     re-deriving tracked regions) leaves the KCA region scores unchanged.
  2. On data WITHOUT identity (each frame re-sampled from a moving surface):
       freeze   -> the region's centroid is static at its t0 position,
       displace -> the region's centroid trajectory is reversed,
       abl-geo  -> the centroid trajectory is kept, the shape (spread) removed,
     and off-region points are untouched.
  3. Control: the classic index-aligned freeze (p_{t,i} <- p_{t0,i}) FAILS criterion 2 on
     such data — the reason the operators had to change.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kca import interventions as itv, regions as rg, scores as sc
from kca.backbones.simple4d import Simple4DBackbone, Simple4DNet
from kca.data.motion_task import ClusterCueTask
from kca.types import Clip


def _permute_frames(clip: Clip, g: torch.Generator) -> Clip:
    B, T, N = clip.mask.shape
    pts, msk = clip.points.clone(), clip.mask.clone()
    for t in range(T):
        p = torch.randperm(N, generator=g)
        pts[:, t], msk[:, t] = clip.points[:, t, p], clip.mask[:, t, p]
    return Clip(points=pts, mask=msk, label=clip.label)


def test_permutation_invariance():
    torch.manual_seed(0)
    model = Simple4DBackbone(Simple4DNet(3, dim=32, heads=4, layers=1, grid=3).eval(), 3)
    clip = ClusterCueTask(n=6, K=3, cue="motion", seed=3).get(1).clip
    clip = Clip(points=clip.points, mask=clip.mask, label=clip.label)
    shuf = _permute_frames(clip, torch.Generator().manual_seed(5))
    tgt = torch.tensor([1])
    out = []
    for c in (clip, shuf):
        c = rg.attach_regions(c, method="tracked", k=3)
        out.append(torch.stack([torch.cat([rs.smot, rs.sgeo, rs.delta_int]) for rs in
                                (sc.symmetric_region_scores(model, c, r, tgt, baseline="bg")
                                 for r in range(c.num_regions()))]))
    assert out[0].shape == out[1].shape
    assert torch.allclose(out[0], out[1], atol=1e-4), (out[0], out[1])
    print("OK  per-frame point shuffling leaves KCA scores unchanged (max diff %.1e)"
          % (out[0] - out[1]).abs().max().item())


def _no_identity_clip(T=8, n_arm=60, n_body=120, seed=0):
    """A static 'body' blob plus an 'arm' segment translating in +y; EVERY frame is
    re-sampled independently from the surfaces (no point identity, like a depth camera)."""
    g = torch.Generator().manual_seed(seed)
    frames = []
    for t in range(T):
        body = torch.randn(n_body, 3, generator=g) * 0.08
        s = torch.rand(n_arm, 1, generator=g)
        arm = torch.cat([0.4 + 0.3 * s, 0.1 * t + 0.02 * torch.randn(n_arm, 1, generator=g),
                         0.02 * torch.randn(n_arm, 1, generator=g)], 1)
        frames.append(torch.cat([body, arm])[torch.randperm(n_arm + n_body, generator=g)])
    pts = torch.stack(frames).unsqueeze(0)
    return Clip(points=pts, mask=torch.ones(1, T, n_arm + n_body, dtype=torch.bool))


def _centroid_track(clip, sel):
    w = sel.unsqueeze(-1).float()
    return ((clip.points * w).sum(2) / w.sum(2).clamp_min(1))[0]      # (T, 3)


def test_operators_without_identity():
    clip = rg.attach_regions(_no_identity_clip(), method="tracked", k=2)
    cen = [_centroid_track(clip, clip.regions == r) for r in range(2)]
    arm = int(torch.stack([c[:, 1].max() - c[:, 1].min() for c in cen]).argmax())
    sel = clip.regions == arm
    orig = _centroid_track(clip, sel)
    assert orig[-1, 1] - orig[0, 1] > 0.5, orig                         # the arm moves

    frz = itv.freeze(clip, arm, t0=0)
    c = _centroid_track(frz, sel)
    assert (c - c[0]).abs().max() < 0.03, c                            # static ...
    assert (c[0] - orig[0]).abs().max() < 1e-6                         # ... at its t0 place
    assert torch.equal(frz.points[~sel], clip.points[~sel])            # off-region untouched

    rev = itv.displace(clip, arm, mode="reverse")
    c = _centroid_track(rev, sel)
    assert (c[:, 1] - orig.flip(0)[:, 1]).abs().max() < 0.03, (c, orig)

    geo = itv.ablate_keep_trajectory(clip, arm)
    c = _centroid_track(geo, sel)
    assert (c - orig).abs().max() < 1e-5                               # trajectory kept
    spread = (geo.points[0][sel[0]] - c.repeat_interleave(sel[0].sum(1), 0)).norm(dim=-1)
    assert spread.max() < 1e-5                                         # shape removed

    # control: index-aligned freeze does NOT make the region static without identity
    naive = clip.clone()
    s = sel.unsqueeze(-1)
    naive.points = torch.where(s, clip.points[:, :1].expand_as(clip.points), clip.points)
    c = _centroid_track(naive, sel)
    assert (c - c[0]).abs().max() > 0.1, c
    print("OK  freeze/displace/abl-geo act on the tracked region without point identity; "
          "index-aligned freeze fails the same check (drift %.2f)" % (c - c[0]).abs().max())


if __name__ == "__main__":
    test_permutation_invariance()
    test_operators_without_identity()
    print("\nALL CORRESPONDENCE-FREE TESTS PASSED")
