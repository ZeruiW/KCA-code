"""Mamba4D (state-space backbone, NO attention). Repo: github.com/IRMVLab/Mamba4D
(models/msr.py, class MAMBA4D; official MSR config radius 0.7, nsamples 32, 24 frames, intra-Mamba on —
the repo's `--intra False` path does not construct).

Thin alias of `OfficialBackbone("mamba4d", ...)` — see official.py. KCA explains this
attention-free SSM with the same code; attention baselines are n/a here.
"""
from __future__ import annotations

from .official import OfficialBackbone


class Mamba4DBackbone(OfficialBackbone):
    def __init__(self, checkpoint=None, num_classes=20, repo=None, device=None, **cfg):
        super().__init__("mamba4d", checkpoint, num_classes=num_classes, repo=repo, device=device, **cfg)
