"""Build the OFFICIAL backbones (P4Transformer, PSTNet, Mamba4D) from their cloned repos
and wrap them behind the `Backbone` interface — on CUDA with the compiled ops, or on
CPU / Apple MPS via the pure-PyTorch stand-ins in `cpu_ops`.

Verified against the repos' train-msr.py (2026-09-26):
  P4Transformer  models/msr.py                     class P4Transformer   radius 0.7  nsamples 32
  PSTNet         models/sequence_classification.py class MSRAction       radius 0.5  nsamples 9
  Mamba4D        models/msr.py                     class MAMBA4D         radius 0.7  nsamples 32
Every repo has a top-level `models` / `modules` layout with clashing bare module names
(`point_4d_convolution`, `pointnet2_utils`, ...), so each repo is imported in isolation
(its bare-name modules are evicted from `sys.modules` after loading).

Input contract of all three forwards: (B, T, N, 3) with a FIXED N per frame, coordinates
divided by 300 (the official loaders). A KCA `Clip` may mark points invalid
(`mask=False`, i.e. removed by an intervention or a deletion metric); `OfficialBackbone`
turns that into a fixed-N cloud the network can read by re-sampling the frame's
remaining valid points (the official loaders' own cyclic-repeat rule for short frames).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import torch

from ..types import Clip
from . import cpu_ops
from .base import Backbone

DEFAULT_EXT = os.path.expanduser(os.environ.get("KCA_EXT", "~/kca_work/ext"))

SPECS = {
    "p4transformer": dict(repo="P4Transformer", file="models/msr.py", cls="P4Transformer",
                          modules=("modules", "modules-pytorch-1.8.1"), has_attention=True,
                          defaults=dict(radius=0.7, nsamples=32, spatial_stride=32,
                                        temporal_kernel_size=3, temporal_stride=2, emb_relu=False,
                                        dim=1024, depth=5, heads=8, dim_head=128, mlp_dim=2048)),
    "pstnet": dict(repo="Point-Spatio-Temporal-Convolution", file="models/sequence_classification.py",
                   cls="MSRAction", modules=("modules",), has_attention=False,
                   defaults=dict(radius=0.5, nsamples=9)),
    "mamba4d": dict(repo="Mamba4D", file="models/msr.py", cls="MAMBA4D", modules=("modules",),
                    has_attention=False,
                    defaults=dict(radius=0.7, nsamples=32, spatial_stride=32, temporal_kernel_size=3,
                                  temporal_stride=2, emb_relu=False, dim=1024, mlp_dim=2048,
                                  depth_mamba_inter=4, rms_norm=False, drop_out_in_block=0.1,
                                  drop_path=0.1, depth_mamba_intra=12, intra=True)),
}

_LOADED: dict[str, object] = {}


def _import_repo(name: str, repo_dir: str):
    """Import one repo's model module in isolation and return it (cached)."""
    key = f"_kca_official_{name}"
    if key in _LOADED:
        return _LOADED[key]
    spec = SPECS[name]
    if not torch.cuda.is_available() or os.environ.get("KCA_PURE_OPS") == "1":   # pure ops on CUDA too (Colab)
        cpu_ops.install()
    mod_dir = next((os.path.join(repo_dir, m) for m in spec["modules"]
                    if os.path.isdir(os.path.join(repo_dir, m))), None)
    if mod_dir is None:
        raise FileNotFoundError(f"{repo_dir}: none of {spec['modules']} found")
    before = set(sys.modules)
    sys.path.insert(0, mod_dir)
    try:
        s = importlib.util.spec_from_file_location(key, os.path.join(repo_dir, spec["file"]))
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    finally:
        sys.path.remove(mod_dir)
        root = os.path.abspath(repo_dir)
        for k in set(sys.modules) - before:                    # evict clashing bare names
            f = getattr(sys.modules[k], "__file__", None) or ""
            if k != key and os.path.abspath(f).startswith(root):
                sys.modules.pop(k, None)
    _LOADED[key] = m
    return m


def build_net(name: str, num_classes: int = 20, repo: Optional[str] = None, **cfg) -> torch.nn.Module:
    spec = SPECS[name]
    repo_dir = repo or os.path.join(DEFAULT_EXT, spec["repo"])
    m = _import_repo(name, repo_dir)
    kw = {**spec["defaults"], **cfg, "num_classes": num_classes}
    return getattr(m, spec["cls"])(**kw)


def preprocess(x: torch.Tensor, preproc: dict) -> torch.Tensor:
    """Network-input normalisation stored with a checkpoint (applied identically in
    training and explanation): center -> subtract each clip's centroid."""
    if preproc.get("center"):
        x = x - x.reshape(x.shape[0], -1, x.shape[-1]).mean(1)[:, None, None, :]
    return x


def resample_to_fixed(points: torch.Tensor, mask: torch.Tensor,
                      fallback: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B,T,N,3) + validity mask -> (B,T,N,3) where every slot holds a VALID point of the
    same frame (invalid slots re-sampled cyclically from the frame's valid points, as the
    official loaders do for short frames). A frame with no valid point is filled with
    `fallback` (B,3) (default: the clip's valid centroid), i.e. an 'empty' frame."""
    B, T, N, C = points.shape
    if bool(mask.all()):
        return points
    out = points.clone()
    if fallback is None:
        w = mask.unsqueeze(-1).float()
        fallback = (points * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1)
    ar = torch.arange(N, device=points.device)
    for b in range(B):
        for t in range(T):
            v = mask[b, t]
            nv = int(v.sum())
            if nv == N:
                continue
            if nv == 0:
                out[b, t] = fallback[b].view(1, C).expand(N, C)
                continue
            src = ar[v]
            out[b, t] = points[b, t, src[ar % nv]]
    return out


class OfficialBackbone(Backbone):
    """Official model + checkpoint behind the KCA Backbone interface."""

    def __init__(self, name: str, checkpoint: Optional[str] = None, num_classes: int = 20,
                 repo: Optional[str] = None, device: Optional[str] = None,
                 net: Optional[torch.nn.Module] = None, **cfg):
        self.name = name
        self.has_attention = SPECS[name]["has_attention"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = dict(cfg)
        self.preproc: dict = {}
        if net is None:
            if checkpoint is not None:
                ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
                ck_cfg = ck.get("cfg")
                if ck_cfg is None and "args" in ck:          # an official train-msr.py checkpoint
                    args = vars(ck["args"]) if hasattr(ck["args"], "__dict__") else dict(ck["args"])
                    ck_cfg = {k: args[k] for k in SPECS[name]["defaults"] if k in args}
                self.cfg = {**(ck_cfg or {}), **cfg}
                self.preproc = ck.get("preproc", {})
                num_classes = ck.get("num_classes", num_classes)
                net = build_net(name, num_classes, repo, **self.cfg)
                sd = ck.get("model", ck.get("state_dict", ck))
                sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
                net.load_state_dict(sd, strict=True)          # never silently random
            else:
                net = build_net(name, num_classes, repo, **self.cfg)
        self.net = net.eval().to(self.device)
        self.num_classes = num_classes
        self._attn: list[torch.Tensor] = []
        self._record_attn = False            # only the attention methods record (memory!)
        self._tube: Optional[tuple] = None
        if self.has_attention:
            self._patch_attention()
        self._hook_tube()

    # ---- forward ---------------------------------------------------------------------
    def _input(self, clip: Clip) -> torch.Tensor:
        x = resample_to_fixed(clip.points[..., :3], clip.mask)
        return preprocess(x, self.preproc).to(self.device)

    def logits(self, clip: Clip) -> torch.Tensor:
        with cpu_ops.device_compat():
            out = self.net(self._input(clip))
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out.to(clip.points.device)

    def supports_grad(self) -> bool:
        return True

    # ---- tube-token geometry (for Grad-CAM / attention -> point scatter) --------------
    def _hook_tube(self):
        emb = getattr(self.net, "tube_embedding", None)
        if emb is None:
            return

        def hook(_m, _inp, out):
            self._tube = out                                   # (xyzs (B,L,n,3), feats)
        emb.register_forward_hook(hook)

    def token_point_map(self, clip: Clip) -> Optional[torch.Tensor]:
        """(B,T,N) token id (0..L*n-1) of the nearest tube anchor for every point, using
        the anchors of the last forward; frame t maps to the temporally nearest anchor
        frame (P4DConv anchors sit at padded frames k//2, k//2+s, ... minus the left pad)."""
        if self._tube is None:
            return None
        axyz = self._tube[0].detach().to(clip.points.device)  # (B, L, n, 3)
        B, L, n, _ = axyz.shape
        T = clip.T
        emb = self.net.tube_embedding
        k, s = emb.temporal_kernel_size, emb.temporal_stride
        pad0 = emb.temporal_padding[0]
        centers = torch.tensor([k // 2 + s * l - pad0 for l in range(L)])
        tl = torch.stack([(centers - t).abs().argmin() for t in range(T)])   # (T,)
        pts = clip.points[..., :3]
        ids = torch.empty(B, T, clip.N, dtype=torch.long, device=pts.device)
        for t in range(T):
            a = axyz[:, tl[t]]                                  # (B, n, 3)
            nn_ = torch.cdist(pts[:, t], a).argmin(-1)          # (B, N)
            ids[:, t] = tl[t] * n + nn_
        return ids

    # ---- attention (P4Transformer only) -------------------------------------------
    def _patch_attention(self):
        """Wrap each official Attention module so its softmax map is recorded (the
        official forward discards it). Numerically identical to the original forward."""
        from einops import rearrange

        store = self._attn
        owner = self

        def make(mod):
            def forward(x):
                h = mod.heads
                q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in mod.to_qkv(x).chunk(3, dim=-1))
                attn = (torch.einsum("bhid,bhjd->bhij", q, k) * mod.scale).softmax(dim=-1)
                if owner._record_attn:
                    if attn.requires_grad:
                        attn.retain_grad()
                    store.append(attn)
                out = rearrange(torch.einsum("bhij,bhjd->bhid", attn, v), "b h n d -> b n (h d)")
                return mod.to_out(out)
            return forward

        for m in self.net.modules():
            if type(m).__name__ == "Attention" and hasattr(m, "to_qkv"):
                m.forward = make(m)

    def _scatter_tokens(self, tok_imp: torch.Tensor, clip: Clip) -> torch.Tensor:
        ids = self.token_point_map(clip)
        B, T, N = ids.shape
        return tok_imp.to(ids.device).gather(1, ids.reshape(B, -1)).reshape(B, T, N) * clip.mask

    def proposal_saliency(self, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
        """Mean attention each token RECEIVES in the last layer, scattered to points."""
        if not self.has_attention:
            return None
        a = self._record(clip)[-1].mean(1).mean(1)             # (B, L)
        return self._scatter_tokens(a, clip)

    def _record(self, clip: Clip) -> list[torch.Tensor]:
        """One no-grad forward with attention recording switched on."""
        self._attn.clear()
        self._record_attn = True
        try:
            with torch.no_grad():
                self.logits(clip)
        finally:
            self._record_attn = False
        out = list(self._attn)
        self._attn.clear()
        return out

    def attention_rollout(self, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
        """Abnar & Zuidema (2020): product of (A + I)/2 over layers; no CLS token (the
        head max-pools tokens), so token importance = mean rollout column."""
        if not self.has_attention:
            return None
        R = None
        for A in self._record(clip):
            A = A.mean(1)
            A = 0.5 * A + 0.5 * torch.eye(A.shape[-1], device=A.device)
            A = A / A.sum(-1, keepdim=True)
            R = A if R is None else A @ R
        return self._scatter_tokens(R.mean(1), clip)

    def transformer_attribution(self, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
        """Chefer et al. (2021, generic attention explainability): R <- R + E_h[(grad A ⊙ A)^+] R."""
        if not self.has_attention:
            return None
        self._attn.clear()
        self._record_attn = True
        try:
            with torch.enable_grad():
                x = self._input(clip).detach()
                with cpu_ops.device_compat():
                    out = self.net(x)
                out.gather(1, target.to(out.device).view(-1, 1)).sum().backward()
        finally:
            self._record_attn = False
        attns, R = list(self._attn), None
        self._attn.clear()
        for A in attns:
            cam = (A.grad * A).clamp_min(0).mean(1)            # (B, L, L)
            I = torch.eye(cam.shape[-1], device=cam.device).expand_as(cam)
            R = I.clone() if R is None else R
            R = R + cam @ R
        self.net.zero_grad(set_to_none=True)
        return self._scatter_tokens(R.mean(1), clip)

    def grad_cam(self, clip: Clip, target: torch.Tensor) -> Optional[torch.Tensor]:
        """3D Grad-CAM on the tube-embedding tokens (channel weights = mean gradient),
        scattered to points via the nearest anchor."""
        if getattr(self.net, "tube_embedding", None) is None:
            return None
        feats = {}

        def hook(_m, _inp, out):
            f = out[1]
            f.retain_grad()
            feats["f"] = f
        h = self.net.tube_embedding.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                with cpu_ops.device_compat():
                    out = self.net(self._input(clip).detach())
                out.gather(1, target.to(out.device).view(-1, 1)).sum().backward()
        finally:
            h.remove()
        f = feats["f"]                                         # (B, L, C, n) or (B, L, n, C)
        g = f.grad
        if f.shape[2] != self._tube[0].shape[2]:               # (B, L, C, n) -> (B, L, n, C)
            f, g = f.transpose(2, 3), g.transpose(2, 3)
        w = g.mean(dim=(1, 2), keepdim=True)                   # (B, 1, 1, C)
        cam = (w * f).sum(-1).clamp_min(0)                     # (B, L, n)
        self.net.zero_grad(set_to_none=True)
        return self._scatter_tokens(cam.reshape(cam.shape[0], -1).detach(), clip)
