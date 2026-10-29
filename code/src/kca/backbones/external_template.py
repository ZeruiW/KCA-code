"""TEMPLATE for wrapping a real 4D backbone behind the KCA Backbone interface.

Copy this to p4transformer.py / mamba4d.py / pstnet.py on the GPU host and fill the
three marked spots. KCA, every baseline, and every metric then work unchanged.

The ONLY things a wrapper must provide:
  1. logits(clip)            : Clip -> (B, num_classes)   [required]
  2. proposal_saliency(...)  : attention map for attn models; None -> grad×act fallback
  3. supports_grad()         : True if logits is differentiable w.r.t. clip.points

Input mapping: our Clip.points is (B,T,N,3) with a (B,T,N) validity mask. The hehefan
repos expect (B, T, N, 3) float tensors already (coords/300-normalised) — usually a
direct pass-through. Mamba4D follows the same convention.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..types import Clip
from .base import Backbone


class ExternalBackbone(Backbone):
    name = "external"            # set to 'p4transformer' / 'mamba4d' / 'pstnet'
    has_attention = False        # True ONLY for P4Transformer

    def __init__(self, checkpoint: str, device: str = "cuda", num_classes: int = 20):
        self.device = device
        self.num_classes = num_classes
        # --- FILL 1: build the model and load the trained checkpoint ---------------
        #   from <repo>.models import P4Transformer            # or Mamba4D / PSTNet
        #   self.net = P4Transformer(radius=..., nsamples=..., num_classes=num_classes)
        #   sd = torch.load(checkpoint, map_location=device)
        #   self.net.load_state_dict(sd["model"] if "model" in sd else sd)
        #   self.net.eval().to(device)
        raise NotImplementedError("fill the model construction for this backbone")

    def _to_model_input(self, clip: Clip) -> torch.Tensor:
        """Clip (B,T,N,3) -> the tensor the repo's forward expects. Usually identity;
        zero-out padded points so they don't perturb the model."""
        x = clip.points[..., :3].clone()
        x[~clip.mask] = 0.0
        return x.to(self.device)

    def logits(self, clip: Clip) -> torch.Tensor:
        x = self._to_model_input(clip)
        # --- FILL 2: call the repo forward, return (B, num_classes) ---------------
        #   return self.net(x)
        raise NotImplementedError

    def proposal_saliency(self, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.has_attention:
            return None          # -> KCA uses the grad×activation fallback
        # --- FILL 3 (P4Transformer only): aggregate spacetime attention to (B,T,N) --
        #   register a forward hook on the attention blocks, run logits(clip),
        #   average heads/layers, scatter token attention back to points.
        raise NotImplementedError

    def supports_grad(self) -> bool:
        return True              # all three repos are differentiable w.r.t. input xyz
