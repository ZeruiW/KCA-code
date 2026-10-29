"""Verify KCA's core math against a synthetic oracle (paper Eq. decomp, Prop. complete).

Goals (verifiable success criteria):
  1. Completeness: eps = Nec - (Smot + Sgeo) == 0 exactly, for every region.
  2. Motion localization: a region that MOVES gets Smot > Sgeo; a region that is
     present-but-static gets Sgeo > Smot. (oracle has a=presence, b=motion weights)
  3. Recovered magnitudes match the oracle's closed form.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kca.types import Clip
from kca import scores as sc, regions as rg, interventions as itv
from kca.backbones.synthetic import SyntheticBackbone


def make_clip():
    """2 regions: region A (left half) MOVES along x; region B (right half) is STATIC.
    Single batch, T=4 frames, N=8 points (4 per region)."""
    torch.manual_seed(0)
    T, N = 4, 8
    base = torch.zeros(1, T, N, 3)
    # region A: points 0..3 at x in [-0.4..-0.1], translate +x over time (MOTION)
    for i in range(4):
        base[0, :, i, 0] = -0.4 + 0.1 * i
        base[0, :, i, 1] = 0.1 * i
        for t in range(T):
            base[0, t, i, 0] += 0.2 * t            # moves in x
    # region B: points 4..7 at x in [0.1..0.4], NO motion (STATIC, present)
    for i in range(4):
        base[0, :, 4 + i, 0] = 0.1 + 0.1 * i
        base[0, :, 4 + i, 1] = 0.1 * i
    mask = torch.ones(1, T, N, dtype=torch.bool)
    # region partition by hand: 0 for A, 1 for B (stable across t)
    regions = torch.zeros(1, T, N, dtype=torch.long)
    regions[:, :, 4:] = 1
    return Clip(points=base, mask=mask, regions=regions)


def test_completeness_and_localization():
    clip = make_clip()
    model = SyntheticBackbone(a=1.0, b=2.0)
    target = torch.tensor([1])

    sA = sc.score_region(model, clip, region_id=0, target=target, baseline="bg")
    sB = sc.score_region(model, clip, region_id=1, target=target, baseline="bg")

    # 1. completeness: residual == 0
    assert torch.allclose(sA.residual, torch.zeros(1), atol=1e-5), sA.residual
    assert torch.allclose(sB.residual, torch.zeros(1), atol=1e-5), sB.residual

    # 2. localization: motion is attributed to the MOVING region, not the static one.
    #    A moves -> Smot(A) > 0;  B static -> Smot(B) ~ 0;  both present -> Sgeo > 0.
    assert sA.smot.item() > 0, sA.smot
    assert sB.smot.item() < 1e-5, sB.smot          # static region carries no motion mass
    assert sA.smot.item() > sB.smot.item() + 1.0, (sA.smot, sB.smot)
    assert sA.sgeo.item() > 0 and sB.sgeo.item() > 0, (sA.sgeo, sB.sgeo)

    # 3. closed-form check. presence counts point-frames (T*points_in_region):
    #    Sgeo(R) = a * (#points_R * T) = 1.0 * (4 * 4) = 16.0
    #    Smot(A) = b * (consecutive-frame motion of A): step 0.2/frame over T-1=3
    #      steps * 4 points = 2.4 ; Smot = 2.0 * 2.4 = 4.8
    assert abs(sA.sgeo.item() - 16.0) < 1e-4, sA.sgeo
    assert abs(sB.sgeo.item() - 16.0) < 1e-4, sB.sgeo
    assert abs(sA.smot.item() - 4.8) < 1e-4, sA.smot
    print("OK  Smot(A)=%.3f Sgeo(A)=%.3f | Smot(B)=%.3f Sgeo(B)=%.3f | eps=%.2e"
          % (sA.smot, sA.sgeo, sB.smot, sB.sgeo, sA.residual.abs().item()))


def test_explain_fields_and_completeness():
    """End-to-end explain() produces fields whose sum equals per-region necessity."""
    clip = rg.attach_regions(make_clip(), grid=2)
    model = SyntheticBackbone(a=1.0, b=2.0)
    from kca import attribute
    attr = attribute.explain(model, clip, k=8, n_control=0, baseline="bg")
    # every region's residual ~ 0
    for r, rs in attr.region_scores.items():
        assert torch.allclose(rs.residual, torch.zeros(1), atol=1e-5), (r, rs.residual)
    assert attr.smot_field.shape == clip.mask.shape
    print("OK  explain(): %d regions scored, %d forward passes, fields %s"
          % (len(attr.region_scores), attr.n_forward, tuple(attr.smot_field.shape)))


if __name__ == "__main__":
    test_completeness_and_localization()
    test_explain_fields_and_completeness()
    print("\nALL CORE TESTS PASSED")
