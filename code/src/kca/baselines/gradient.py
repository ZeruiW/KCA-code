"""Gradient-family baselines (paper Sec. exp-setup), all model-agnostic given a
differentiable backbone. Each returns a (B,T,N) per-point importance field.

  grad_x_input        : |grad . x| over channels
  smoothgrad          : grad averaged over noisy inputs (Smilkov'17)
  integrated_gradients: path integral from baseline to input (Sundararajan'17)
  grad_cam_3d         : input-space Grad-CAM proxy (no feature maps via the generic
                        interface, so we use grad-weighted input activations)
"""
from __future__ import annotations

import torch

from ..backbones.base import Backbone
from ..types import Clip


def _grad(model: Backbone, pts: torch.Tensor, clip: Clip, target: torch.Tensor) -> torch.Tensor:
    pts = pts.detach().clone().requires_grad_(True)
    probe = Clip(points=pts, mask=clip.mask, regions=clip.regions, label=clip.label)
    phi = model.phi(probe, target).sum()
    g, = torch.autograd.grad(phi, pts)
    return g


def grad_x_input(model: Backbone, clip: Clip, target: torch.Tensor) -> torch.Tensor:
    g = _grad(model, clip.points, clip, target)
    return (g * clip.points).abs().sum(-1) * clip.mask


def smoothgrad(model: Backbone, clip: Clip, target: torch.Tensor,
               n: int = 25, sigma: float = 0.1, generator=None) -> torch.Tensor:
    acc = torch.zeros(clip.mask.shape, device=clip.device)
    for _ in range(n):
        noise = torch.randn(clip.points.shape, generator=generator, device=clip.device) * sigma
        g = _grad(model, clip.points + noise, clip, target)
        acc += g.abs().sum(-1)
    return (acc / n) * clip.mask


def _centroid_baseline(clip: Clip) -> torch.Tensor:
    """Every point collapsed onto the clip's valid centroid: the content-free point-cloud
    reference (a far-away constant would be off-manifold for real backbones)."""
    w = clip.mask.unsqueeze(-1).float()
    c = (clip.points * w).sum((1, 2), keepdim=True) / w.sum((1, 2), keepdim=True).clamp_min(1)
    return c.expand_as(clip.points).clone()


def integrated_gradients(model: Backbone, clip: Clip, target: torch.Tensor,
                         steps: int = 32, baseline_val: float | None = None) -> torch.Tensor:
    base = _centroid_baseline(clip) if baseline_val is None else torch.full_like(clip.points, baseline_val)
    total = torch.zeros(clip.points.shape, device=clip.device)
    for s in range(1, steps + 1):
        alpha = s / steps
        interp = base + alpha * (clip.points - base)
        total += _grad(model, interp, clip, target)
    avg_grad = total / steps
    ig = (clip.points - base) * avg_grad
    return ig.abs().sum(-1) * clip.mask


def grad_cam_3d(model: Backbone, clip: Clip, target: torch.Tensor) -> torch.Tensor:
    """3D Grad-CAM. Backbones exposing their spacetime tokens (`grad_cam`, e.g. the
    official P4Transformer / Mamba4D tube embedding) get token-level Grad-CAM scattered
    to points; otherwise an input-space proxy (ReLU of grad-weighted input activations)."""
    fn = getattr(model, "grad_cam", None)
    if fn is not None:
        out = fn(clip, target)
        if out is not None:
            return out
    g = _grad(model, clip.points, clip, target)
    w = g.mean(dim=(1, 2), keepdim=True)                      # (B,1,1,C) channel weights
    cam = (w * clip.points).sum(-1).clamp_min(0)              # (B,T,N)
    return cam * clip.mask
