"""PSTNet (convolution backbone, no attention).
Repo: github.com/hehefan/Point-Spatio-Temporal-Convolution
(models/sequence_classification.py, class MSRAction; official MSR config radius 0.5, nsamples 9).

Thin alias of `OfficialBackbone("pstnet", ...)` — see official.py.
"""
from __future__ import annotations

from .official import OfficialBackbone


class PSTNetBackbone(OfficialBackbone):
    def __init__(self, checkpoint=None, num_classes=20, repo=None, device=None, **cfg):
        super().__init__("pstnet", checkpoint, num_classes=num_classes, repo=repo, device=device, **cfg)
