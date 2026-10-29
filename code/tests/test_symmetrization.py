"""Verify the order-symmetrized decomposition and interaction residual (Prop. interact).

Verifiable success criteria:
  1. abl-geo realizes the {motion-kept, geometry-removed} vertex: it keeps the region's
     motion but collapses its shape (on the shape/motion oracle).
  2. Completeness ALWAYS holds for the symmetrized scores: Sbar_mot + Sbar_geo = Nec.
  3. ADDITIVE model -> Delta_int ~ 0 (a rigid-moving region under the additive
     shape+motion oracle).
  4. NON-additive model -> Delta_int != 0 (an oracle with an explicit motion x geometry
     product term), and completeness still holds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kca.types import Clip
from kca import scores as sc, interventions as itv
from kca.backbones.base import Backbone
from kca.backbones.synthetic import ShapeMotionBackbone


def rigid_moving_clip():
    """One region (all points), a fixed random shape rigidly translating in +x over t."""
    g = torch.Generator().manual_seed(1)
    T, N = 5, 16
    shape0 = (torch.rand(N, 3, generator=g) - 0.5) * 0.4         # the region's shape at t0
    pts = torch.zeros(1, T, N, 3)
    for t in range(T):
        pts[0, t] = shape0
        pts[0, t, :, 0] += 0.2 * t                               # RIGID translation
    mask = torch.ones(1, T, N, dtype=torch.bool)
    regions = torch.zeros(1, T, N, dtype=torch.long)            # single region id 0
    return Clip(points=pts, mask=mask, regions=regions)


def test_ablgeo_keeps_motion_removes_shape():
    clip = rigid_moving_clip()
    m = ShapeMotionBackbone(a=1.0, b=1.0)
    tgt = torch.tensor([1])
    phi = m.phi(clip, tgt).item()
    phi_frz = m.phi(itv.freeze(clip, 0), tgt).item()           # motion removed, shape KEPT
    phi_geo = m.phi(itv.ablate_keep_trajectory(clip, 0, baseline="bg"), tgt).item()  # shape removed, motion KEPT
    phi_abl = m.phi(itv.ablate(clip, 0, baseline="bg"), tgt).item()
    # ablate removes everything; freeze retains the shape term; abl-geo retains the motion term
    assert abs(phi_abl) < 1e-5, phi_abl
    assert phi_frz > 1e-3, phi_frz                              # shape survives freeze
    assert phi_geo > 1e-3, phi_geo                              # motion survives abl-geo
    # additivity: phi(X) == phi(freeze) + phi(abl-geo) - phi(ablate) for an additive model
    assert abs(phi - (phi_frz + phi_geo - phi_abl)) < 1e-4, (phi, phi_frz, phi_geo, phi_abl)
    print("OK  vertices | phi=%.3f = frz(shape)=%.3f + geo(motion)=%.3f - abl=%.3f"
          % (phi, phi_frz, phi_geo, phi_abl))


def test_symmetrized_completeness_and_additive_zero_interaction():
    clip = rigid_moving_clip()
    m = ShapeMotionBackbone(a=1.0, b=1.0)
    tgt = torch.tensor([1])
    rs = sc.symmetric_region_scores(m, clip, 0, tgt, baseline="bg")
    # 2. completeness of symmetrized scores
    assert torch.allclose(rs.smot + rs.sgeo, rs.necessity, atol=1e-5), (rs.smot, rs.sgeo, rs.necessity)
    # 3. additive oracle + rigid motion -> interaction ~ 0
    assert rs.delta_int.abs().item() < 1e-4, rs.delta_int
    print("OK  symmetrized completeness holds; additive Delta_int=%.2e (~0)" % rs.delta_int.abs().item())


class InteractingBackbone(Backbone):
    """phi = motion * shape (pure interaction) -> Delta_int must be NON-zero."""
    name = "interacting"; has_attention = False
    def logits(self, clip):
        xyz = clip.points[..., :3]; m = clip.mask.float()
        cnt = m.sum(-1, keepdim=True).clamp_min(1)
        cen = (xyz * m.unsqueeze(-1)).sum(2, keepdim=True) / cnt.unsqueeze(-1)
        shape = ((((xyz - cen) ** 2).sum(-1) * m).sum(-1) / cnt.squeeze(-1)).sqrt().sum(-1)
        step = (xyz[:, 1:] - xyz[:, :-1]).norm(dim=-1)
        both = (clip.mask[:, 1:] & clip.mask[:, :-1]).float()
        motion = (step * both).sum(dim=(1, 2))
        e = motion * shape                                      # interaction term
        return torch.stack([torch.zeros_like(e), e], dim=1)


def test_interacting_nonzero_but_complete():
    clip = rigid_moving_clip()
    m = InteractingBackbone()
    tgt = torch.tensor([1])
    rs = sc.symmetric_region_scores(m, clip, 0, tgt, baseline="bg")
    assert torch.allclose(rs.smot + rs.sgeo, rs.necessity, atol=1e-5)   # completeness still exact
    assert rs.delta_int.abs().item() > 1e-3, rs.delta_int               # interaction detected
    print("OK  interacting oracle | completeness exact, Delta_int=%.3f (!=0, detected)" % rs.delta_int.item())


if __name__ == "__main__":
    test_ablgeo_keeps_motion_removes_shape()
    test_symmetrized_completeness_and_additive_zero_interaction()
    test_interacting_nonzero_but_complete()
    print("\nALL SYMMETRIZATION TESTS PASSED")
