"""P4Transformer (attention backbone). Repo: github.com/hehefan/P4Transformer
(models/msr.py, class P4Transformer; official MSR config radius 0.7, nsamples 32, 24 frames).

Thin alias of `OfficialBackbone("p4transformer", ...)` — see official.py:
    bb = P4TransformerBackbone("ckpt/p4t/last.pth")            # our or official checkpoints
"""
from __future__ import annotations

from .official import OfficialBackbone


class P4TransformerBackbone(OfficialBackbone):
    def __init__(self, checkpoint=None, num_classes=20, repo=None, device=None, **cfg):
        super().__init__("p4transformer", checkpoint, num_classes=num_classes, repo=repo,
                         device=device, **cfg)
