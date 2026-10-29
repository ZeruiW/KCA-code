"""Pure-PyTorch stand-ins for the CUDA-only extensions imported by the OFFICIAL 4D
backbones (P4Transformer, PSTNet, Mamba4D), so their unmodified model code runs on CPU
or Apple MPS (and GPU-trained checkpoints load unchanged).

`install()` registers these fake modules in `sys.modules`:
  pointnet2_utils, pointnet2_ops.pointnet2_utils   furthest_point_sample, gather_operation,
                                                    ball_query, grouping_operation,
                                                    three_nn, three_interpolate
  knn_cuda                                          KNN
  mamba_ssm.modules.mamba_simple                    Mamba (same parameter names as mamba_ssm)
  mamba_ssm.ops.triton.layernorm                    RMSNorm, layer_norm_fn, rms_norm_fn
`device_compat()` patches `Tensor.get_device` while a forward runs: the official code
does `torch.zeros(..., device=x.get_device())`, which is -1 (invalid) off-CUDA.

Semantics follow the CUDA kernels: FPS starts at index 0; ball query returns the first
`nsample` in-radius indices in index order, padded with the first hit (0 if none).
(Unlike the official FPS kernel, no point near the origin is skipped.)

On a CUDA host, `install()` also enables two optional fast paths with the SAME semantics
(`cuda_fast`: a compiled FPS kernel and mamba_ssm's selective-scan kernel), each checked
against the loops below before use; KCA_FAST_OPS=0 disables them.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


FAST: dict = {"fps": None, "scan": None}     # CUDA fast paths, set by cuda_fast.prepare()


# ---------------------------------------------------------------------------------------
# pointnet2 ops
# ---------------------------------------------------------------------------------------
def furthest_point_sample(xyz: torch.Tensor, npoint) -> torch.Tensor:
    """xyz (B, N, 3) -> (B, npoint) int32 indices, seeded at index 0 like the CUDA op."""
    npoint = int(npoint)
    if xyz.is_cuda and FAST["fps"] is not None:
        return FAST["fps"](xyz.detach(), npoint)
    B, N, _ = xyz.shape
    x = xyz.detach()
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = far
        c = x[ar, far].unsqueeze(1)                              # (B, 1, 3)
        dist = torch.minimum(dist, ((x - c) ** 2).sum(-1))
        far = dist.argmax(1)
    return idx.int()


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features (B, C, N), idx (B, M) -> (B, C, M)."""
    idx = idx.long()
    return features.gather(2, idx.unsqueeze(1).expand(-1, features.shape[1], -1))


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """xyz (B, N, 3), new_xyz (B, M, 3) -> (B, M, nsample) int32."""
    B, N, _ = xyz.shape
    d2 = torch.cdist(new_xyz.detach(), xyz.detach()) ** 2       # (B, M, N)
    inside = d2 < radius * radius
    order = torch.where(inside, torch.arange(N, device=xyz.device), N)
    k = min(nsample, N)
    first = order.topk(k, dim=2, largest=False, sorted=True).values  # smallest indices first
    if k < nsample:
        first = torch.cat([first, first[..., -1:].expand(-1, -1, nsample - k)], 2)
    has = first[..., :1] < N
    pad = torch.where(has, first[..., :1], torch.zeros_like(first[..., :1]))
    first = torch.where(first < N, first, pad.expand_as(first))
    return first.int()


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features (B, C, N), idx (B, M, K) -> (B, C, M, K)."""
    B, C, N = features.shape
    _, M, K = idx.shape
    flat = idx.long().reshape(B, 1, M * K).expand(-1, C, -1)
    return features.gather(2, flat).reshape(B, C, M, K)


def three_nn(unknown: torch.Tensor, known: torch.Tensor):
    """unknown (B, n, 3), known (B, m, 3) -> dist (B, n, 3) [euclidean], idx (B, n, 3)."""
    d2 = torch.cdist(unknown, known) ** 2
    val, idx = d2.topk(3, dim=2, largest=False)
    return val.clamp_min(0).sqrt(), idx.int()


def three_interpolate(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """features (B, C, m), idx/weight (B, n, 3) -> (B, C, n)."""
    B, C, m = features.shape
    n = idx.shape[1]
    g = features.gather(2, idx.long().reshape(B, 1, n * 3).expand(-1, C, -1)).reshape(B, C, n, 3)
    return (g * weight.unsqueeze(1)).sum(-1)


class KNN(nn.Module):
    """knn_cuda.KNN stand-in: KNN(k, transpose_mode=True)(ref (B,N,D), query (B,M,D))."""

    def __init__(self, k: int, transpose_mode: bool = True):
        super().__init__()
        self.k, self.transpose_mode = int(k), transpose_mode

    def forward(self, ref, query):
        if not self.transpose_mode:
            ref, query = ref.transpose(1, 2), query.transpose(1, 2)
        d = torch.cdist(query, ref)
        dist, idx = d.topk(self.k, dim=2, largest=False)
        if not self.transpose_mode:
            dist, idx = dist.transpose(1, 2), idx.transpose(1, 2)
        return dist, idx


# ---------------------------------------------------------------------------------------
# Mamba (selective state-space) — reference semantics of mamba_ssm's `selective_scan_ref`
# ---------------------------------------------------------------------------------------
def selective_scan(u, delta, A, Bm, Cm, D):
    """u, delta (b, l, d); A (d, n); Bm, Cm (b, l, n); D (d,) -> y (b, l, d).
    h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t ;  y_t = <h_t, C_t> + D u_t."""
    if u.is_cuda and FAST["scan"] is not None:
        return FAST["scan"](u, delta, A, Bm, Cm, D)
    b, l, d = u.shape
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    # unbind (not per-step slicing): slicing's backward materialises a full-size gradient
    # every step, which makes the backward O(L^2)
    for dt, ut, bt, ct in zip(delta.unbind(1), u.unbind(1), Bm.unbind(1), Cm.unbind(1)):
        dt = dt.unsqueeze(-1)                                   # (b, d, 1)
        h = torch.exp(dt * A) * h + (dt * ut.unsqueeze(-1)) * bt.unsqueeze(1)
        ys.append((h * ct.unsqueeze(1)).sum(-1))
    return torch.stack(ys, 1) + u * D


class Mamba(nn.Module):
    """Drop-in for mamba_ssm.modules.mamba_simple.Mamba (same parameters & init)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,
                 layer_idx=None, device=None, dtype=None):
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, d_state, d_conv, expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states, inference_params=None):
        # KCA_MAMBA_CHECKPOINT=1: recompute this layer in backward instead of storing its
        # activations (identical numbers; official-width Mamba4D otherwise needs ~50 GB at bs 16)
        if os.environ.get("KCA_MAMBA_CHECKPOINT") == "1" and torch.is_grad_enabled() and hidden_states.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, hidden_states, use_reentrant=False)
        return self._forward(hidden_states)

    def _forward(self, hidden_states):
        L = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)
        x = self.act(self.conv1d(x.transpose(1, 2))[..., :L]).transpose(1, 2)
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        y = selective_scan(x, delta, -torch.exp(self.A_log.float()), Bm, Cm, self.D)
        return self.out_proj(y * self.act(z))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(x, self.weight, self.bias, residual=residual, prenorm=prenorm,
                           residual_in_fp32=residual_in_fp32, eps=self.eps)


def _norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False,
             eps=1e-6, rms=False, **_):
    if residual is not None:
        x = x + residual
    res = x
    if rms:
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        y = y + bias if bias is not None else y
    else:
        y = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return (y, res) if prenorm else y


def layer_norm_fn(x, weight, bias, **kw):
    return _norm_fn(x, weight, bias, rms=False, **kw)


def rms_norm_fn(x, weight, bias, **kw):
    return _norm_fn(x, weight, bias, rms=True, **kw)


# ---------------------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------------------
def _module(name: str, **attrs) -> types.ModuleType:
    m = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def install() -> None:
    """Register the stand-ins (idempotent). Call BEFORE importing an official repo."""
    if torch.cuda.is_available():
        from . import cuda_fast
        cuda_fast.prepare()
    ops = dict(furthest_point_sample=furthest_point_sample, gather_operation=gather_operation,
               ball_query=ball_query, grouping_operation=grouping_operation,
               three_nn=three_nn, three_interpolate=three_interpolate)
    _module("pointnet2_utils", **ops)
    p2 = _module("pointnet2_ops")
    p2.pointnet2_utils = _module("pointnet2_ops.pointnet2_utils", **ops)
    _module("knn_cuda", KNN=KNN)
    ms = _module("mamba_ssm")
    mods = _module("mamba_ssm.modules")
    mods.mamba_simple = _module("mamba_ssm.modules.mamba_simple", Mamba=Mamba)
    ms.modules = mods
    opsm = _module("mamba_ssm.ops")
    tri = _module("mamba_ssm.ops.triton")
    tri.layernorm = _module("mamba_ssm.ops.triton.layernorm", RMSNorm=RMSNorm,
                            layer_norm_fn=layer_norm_fn, rms_norm_fn=rms_norm_fn)
    opsm.triton = tri
    ms.ops = opsm


@contextlib.contextmanager
def device_compat():
    """Make `Tensor.get_device()` return a torch.device while official forwards run."""
    orig = torch.Tensor.get_device
    torch.Tensor.get_device = lambda self: self.device
    try:
        yield
    finally:
        torch.Tensor.get_device = orig
