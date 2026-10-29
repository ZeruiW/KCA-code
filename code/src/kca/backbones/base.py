"""The single interface KCA explains through.

The entire method (interventions, scores, decomposition) and every metric talk to
the model ONLY via this interface, so the same code explains an attention model
(P4Transformer), a state-space model (Mamba4D), and a convolution model (PSTNet)
without modification. This is the model-agnostic guarantee of the paper
(Sec. method-inst).
"""
from __future__ import annotations

import abc
from typing import Optional

import torch

from ..types import Clip


class Backbone(abc.ABC):
    """Wraps a trained 4D action-recognition model.

    Subclasses adapt a concrete checkpoint (P4Transformer / Mamba4D / PSTNet) to
    this contract. KCA never touches architecture internals except through
    `proposal_saliency`, which is allowed to use attention *only to rank regions*,
    never as the explanation itself.
    """

    #: human-readable id, e.g. "p4transformer"
    name: str = "abstract"
    #: whether the backbone exposes attention maps (controls the default proposal)
    has_attention: bool = False

    @abc.abstractmethod
    def logits(self, clip: Clip) -> torch.Tensor:
        """Forward pass. Returns (B, num_classes) class logits.

        Must accept arbitrary point/mask tensors (intervened inputs included) and
        be deterministic given its weights (set eval mode, no dropout).
        """

    def phi(self, clip: Clip, target: torch.Tensor) -> torch.Tensor:
        """The scalar target read-out phi(X) = f(X)_c per the paper (Eq. necessity).

        target: (B,) long class indices. Returns (B,) the target-class logit.
        """
        out = self.logits(clip)                      # (B, K)
        return out.gather(1, target.view(-1, 1)).squeeze(1)

    def predicted_class(self, clip: Clip) -> torch.Tensor:
        """argmax class, used when no explicit target is given."""
        return self.logits(clip).argmax(dim=1)

    # --- optional hooks used by gradient baselines & the attention proposal ---

    def proposal_saliency(self, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
        """Cheap per-point saliency to RANK regions in KCA Stage 1.

        Returns (B, T, N) non-negative scores, or None to signal "use the
        backbone-agnostic grad x activation fallback" (handled by the explainer).
        Attention backbones override this with aggregated spacetime attention.
        """
        return None

    def supports_grad(self) -> bool:
        """True if `logits` is differentiable w.r.t. `clip.points` (for IG, grad-cam,
        and the grad x activation proposal). State-space/conv models that are
        differentiable should return True; non-differentiable read-outs return False.
        """
        return True
