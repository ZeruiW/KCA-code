"""Registry mapping method name -> attribution field (B,T,N) or None ('n/a').

Unifies KCA and every baseline behind one signature so the eval harness and table
generator can loop over them. A method returning None is reported as 'n/a' (e.g.
attention methods on attention-free backbones).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .backbones.base import Backbone
from .types import Clip
from . import attribute
from .baselines import floors, gradient, perturbation, attention as attn

Method = Callable[[Backbone, Clip, torch.Tensor], Optional[torch.Tensor]]


def _kca(model, clip, target):
    return attribute.explain(model, clip, target).nec_field


def _kca_ctx(model, clip, target):
    """Context-averaged KCA (robust when a structure spans many regions)."""
    return attribute.explain(model, clip, target, estimator="context", n_samples=256).nec_field


def _kca_motion(model, clip, target):
    return attribute.explain(model, clip, target).smot_field


def _kca_geometry(model, clip, target):
    return attribute.explain(model, clip, target).sgeo_field


REGISTRY: dict[str, Method] = {
    # floors
    "random": lambda m, c, t: floors.random_field(c),
    "geometry_edge": lambda m, c, t: floors.geometry_edge(c),
    "motion_prior": lambda m, c, t: floors.motion_prior(c),
    # gradient family
    "grad_x_input": gradient.grad_x_input,
    "smoothgrad": gradient.smoothgrad,
    "integrated_gradients": gradient.integrated_gradients,
    "grad_cam_3d": gradient.grad_cam_3d,
    # perturbation
    "point_drop": perturbation.point_drop,
    "occam": perturbation.occam,
    "extremal": perturbation.extremal,
    # attention-specific (None on non-attention backbones)
    "attn_rollout": attn.rollout,
    "transformer_attr": attn.transformer_attr,
    "staa4d": attn.staa4d,
    # ours
    "kca": _kca,
    "kca_ctx": _kca_ctx,
    "kca_motion": _kca_motion,
    "kca_geometry": _kca_geometry,
}


def run_method(name: str, model: Backbone, clip: Clip, target: torch.Tensor,
               **kw) -> Optional[torch.Tensor]:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](model, clip, target, **kw)
