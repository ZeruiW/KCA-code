"""A small but REAL trainable attention-based 4D point-cloud-video classifier.

Architecture (a minimal P4Transformer-style model):
  per-point MLP -> per-frame voxel-grid pooling into spacetime tokens
  -> Transformer encoder (self-attention, attention maps readable) -> mean-pool -> head.

It is a genuine differentiable network (not a hand-coded oracle), so training it and
then explaining it validates KCA on a learned model. It exposes attention for the
proposal stage (has_attention=True), so the attention baselines (rollout/STAA-4D) and
the attention-proposal path are exercised too.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import Clip
from .base import Backbone


class _Block(nn.Module):
    """Pre-norm transformer block that stores its attention weights."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.attn: Optional[torch.Tensor] = None  # (B, heads, L, L), last forward

    def forward(self, x, key_pad):                     # x:(B,L,D) key_pad:(B,L) True=pad
        B, L, D = x.shape
        h = self.h
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(B, L, h, D // h).transpose(1, 2)    # (B,h,L,d)
        k = k.view(B, L, h, D // h).transpose(1, 2)
        v = v.view(B, L, h, D // h).transpose(1, 2)
        score = (q @ k.transpose(-2, -1)) / (D // h) ** 0.5   # (B,h,L,L)
        score = score.masked_fill(key_pad[:, None, None, :], float("-inf"))
        attn = score.softmax(dim=-1)
        self.attn = attn.detach()
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class Simple4DNet(nn.Module):
    def __init__(self, num_classes: int, dim: int = 64, heads: int = 4, layers: int = 2,
                 grid: int = 3, lo: float = -0.7, hi: float = 0.7):
        super().__init__()
        self.grid, self.lo, self.hi, self.dim = grid, lo, hi, dim
        self.cells = grid ** 3
        self.max_T = 32
        self.enc = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.spatial_emb = nn.Parameter(torch.zeros(self.cells, dim))
        self.temporal_emb = nn.Parameter(torch.zeros(self.max_T, dim))  # PER-FRAME (order matters)
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.normal_(self.spatial_emb, std=0.02)
        nn.init.normal_(self.temporal_emb, std=0.02)

    def _cell_id(self, xyz):                            # xyz:(B,T,N,3) -> (B,T,N) long
        g = self.grid
        c = ((xyz - self.lo) / (self.hi - self.lo) * g).floor().clamp(0, g - 1).long()
        return c[..., 0] * g * g + c[..., 1] * g + c[..., 2]

    def tokens(self, clip: Clip):
        """Return token features (B, T*cells, D), key-pad mask (B, T*cells), and the
        (B,T,N) token id each point maps to (for attention->point scatter)."""
        B, T, N, _ = clip.points.shape
        feat = self.enc(clip.points[..., :3])          # (B,T,N,D)
        m = clip.mask.float().unsqueeze(-1)            # (B,T,N,1)
        cid = self._cell_id(clip.points[..., :3])      # (B,T,N)
        cid = cid.clamp(0, self.cells - 1)
        # scatter-mean point features into (B,T,cells,D)
        tok = torch.zeros(B, T, self.cells, self.dim, device=clip.device)
        cnt = torch.zeros(B, T, self.cells, 1, device=clip.device)
        idx = cid.unsqueeze(-1).expand(-1, -1, -1, self.dim)
        tok.scatter_add_(2, idx, feat * m)
        cnt.scatter_add_(2, cid.unsqueeze(-1), m)
        tok = tok / cnt.clamp_min(1e-6)
        tok = (tok + self.spatial_emb.view(1, 1, self.cells, self.dim)
               + self.temporal_emb[:T].view(1, T, 1, self.dim))     # per-frame temporal pos
        key_pad = (cnt.squeeze(-1) < 0.5)              # (B,T,cells) empty cell = pad
        tok = tok.reshape(B, T * self.cells, self.dim)
        key_pad = key_pad.reshape(B, T * self.cells)
        # global token id per point = t*cells + cid
        ttid = (torch.arange(T, device=clip.device).view(1, T, 1) * self.cells + cid)
        return tok, key_pad, ttid                      # tok,(B,L,D); key_pad,(B,L); ttid,(B,T,N)

    def forward(self, clip: Clip, return_tokens: bool = False):
        tok, key_pad, ttid = self.tokens(clip)
        x = tok
        for blk in self.blocks:
            x = blk(x, key_pad)
        x = self.norm(x)
        x = x.masked_fill(key_pad.unsqueeze(-1), 0.0)
        pooled = x.sum(1) / (~key_pad).float().sum(1, keepdim=True).clamp_min(1)
        logits = self.head(pooled)
        if return_tokens:
            return logits, key_pad, ttid
        return logits


class Simple4DBackbone(Backbone):
    name = "simple4d"
    has_attention = True

    def __init__(self, net: Simple4DNet, num_classes: int):
        self.net = net.eval()
        self.num_classes = num_classes

    def logits(self, clip: Clip) -> torch.Tensor:
        return self.net(clip)

    @torch.no_grad()
    def point_features(self, clip: Clip) -> torch.Tensor:
        """(B,T,N,D) per-point INTERNAL encoder features — the representation the
        amortized head distils from (paper Sec. method-eff). Richer than input-derived
        descriptors, so it lets the O(1) head better approximate the exact causal field."""
        return self.net.enc(clip.points[..., :3]) * clip.mask.unsqueeze(-1)

    def proposal_saliency(self, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
        """Aggregate last-layer attention to a per-point score (attention used ONLY to
        propose). Importance of a key token = mean attention it receives; scatter to
        the points that map to that token."""
        with torch.no_grad():
            _, key_pad, ttid = self.net(clip, return_tokens=True)
            attn = self.net.blocks[-1].attn                # (B,h,L,L)
            if attn is None:
                return None
            key_imp = attn.mean(1).mean(1)                 # (B,L) mean over heads & queries
        B, T, N = ttid.shape
        flat = key_imp.gather(1, ttid.reshape(B, -1)).reshape(B, T, N)
        return flat * clip.mask
