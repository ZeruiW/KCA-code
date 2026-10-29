"""Attention/transformer-specific baselines (paper Sec. exp-setup).

UNDEFINED for non-attention backbones (PSTNet, Mamba4D) -> return None, which the
table generator renders as "n/a" (a result, not a gap).

  rollout            : attention rollout (Abnar'20) — supplied by the backbone.
  transformer_attr   : Chefer'21 attribution — supplied by the backbone if available.
  staa4d             : the KCA proposal field used AS the explanation (the
                       attention-only ablation that isolates the causal stage's gain).
"""
from __future__ import annotations

from typing import Optional

import torch

from ..backbones.base import Backbone
from ..types import Clip


def staa4d(model: Backbone, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
    """STAA-4D ablation: the attention proposal as the explanation itself.
    Returns None on attention-free backbones (-> 'n/a')."""
    if not model.has_attention:
        return None
    return model.proposal_saliency(clip, target)


def rollout(model: Backbone, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
    if not model.has_attention:
        return None
    fn = getattr(model, "attention_rollout", None)
    return fn(clip, target) if fn is not None else model.proposal_saliency(clip, target)


def transformer_attr(model: Backbone, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
    if not model.has_attention:
        return None
    fn = getattr(model, "transformer_attribution", None)
    return fn(clip, target) if fn is not None else None
