"""Optional CUDA fast paths for two loops in `cpu_ops`, with the SAME semantics:

  fps    furthest point sampling as ONE kernel (cpu_ops runs one Python step per sampled
         point: ~33k steps per PSTNet forward at 2048 points). Unlike the official
         pointnet2 kernel it does not skip points near the origin, and ties go to the
         lowest index (torch.argmax), so it selects what `cpu_ops.furthest_point_sample` does.
  scan   the Mamba selective scan through mamba_ssm's compiled `selective_scan_cuda`
         (cpu_ops loops over the sequence and stores every state for backward, which at
         Mamba4D's official width does not fit in GPU memory).

`prepare()` builds / imports each one, checks it against the cpu_ops loop on random
inputs and only then enables it (cpu_ops.FAST). Anything that fails falls back to the
loop with a message. KCA_FAST_OPS=0 disables both. Called by `cpu_ops.install()` on CUDA.
"""
from __future__ import annotations

import os
import sys

import torch

from . import cpu_ops

STATUS: dict = {}

_FPS_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

constexpr int BS = 512;

__global__ void kca_fps_kernel(int n, int m, const float* __restrict__ xyz,
                               float* __restrict__ temp, int* __restrict__ out) {
  __shared__ float dists[BS];
  __shared__ int dists_i[BS];
  const int b = blockIdx.x, tid = threadIdx.x;
  xyz += (long long)b * n * 3;
  temp += (long long)b * n;
  out += (long long)b * m;
  for (int k = tid; k < n; k += BS) temp[k] = __int_as_float(0x7f800000);   // +inf, as the loop's init
  if (tid == 0) out[0] = 0;
  int old = 0;
  __syncthreads();
  for (int j = 1; j < m; j++) {
    const float x1 = xyz[old * 3], y1 = xyz[old * 3 + 1], z1 = xyz[old * 3 + 2];
    float best = -1.f;
    int besti = 0;
    for (int k = tid; k < n; k += BS) {
      const float dx = __fsub_rn(xyz[k * 3], x1);
      const float dy = __fsub_rn(xyz[k * 3 + 1], y1);
      const float dz = __fsub_rn(xyz[k * 3 + 2], z1);
      const float d = __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
      const float d2 = fminf(d, temp[k]);
      temp[k] = d2;
      if (d2 > best) { best = d2; besti = k; }
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
      if (tid < s) {
        const float v1 = dists[tid], v2 = dists[tid + s];
        const int i1 = dists_i[tid], i2 = dists_i[tid + s];
        if (v2 > v1 || (v2 == v1 && i2 < i1)) { dists[tid] = v2; dists_i[tid] = i2; }
      }
      __syncthreads();
    }
    old = dists_i[0];
    if (tid == 0) out[j] = old;
    __syncthreads();
  }
}

torch::Tensor fps(torch::Tensor xyz, int64_t m) {
  TORCH_CHECK(xyz.is_cuda() && xyz.dim() == 3 && xyz.size(2) == 3, "fps: xyz must be a CUDA (B, N, 3) tensor");
  const c10::cuda::CUDAGuard guard(xyz.device());
  auto x = xyz.to(torch::kFloat).contiguous();
  const int64_t B = x.size(0), n = x.size(1);
  auto out = torch::zeros({B, m}, x.options().dtype(torch::kInt));
  if (B == 0 || m == 0 || n == 0) return out;
  auto temp = torch::empty({B, n}, x.options());
  kca_fps_kernel<<<B, BS, 0, at::cuda::getCurrentCUDAStream()>>>(
      (int)n, (int)m, x.data_ptr<float>(), temp.data_ptr<float>(), out.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


def _build_fps():
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name="kca_fps_v1", cpp_sources="torch::Tensor fps(torch::Tensor xyz, int64_t m);",
                      cuda_sources=_FPS_CU, functions=["fps"], extra_cuda_cflags=["-O3"], verbose=False)
    return lambda xyz, m: mod.fps(xyz, int(m))


def _check_fps(fast) -> None:
    """Every pick is a farthest point from the points picked before it (up to float
    rounding), starting at index 0, on data with exact duplicates (cyclic-repeat padding);
    STATUS records how often the picks equal the cpu_ops loop's."""
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.rand(3, 1500, 3, device="cuda", generator=g)
    x = torch.cat([x, x[:, :548]], 1)                       # duplicates, N = 2048
    for m in (64, 1024):
        got = fast(x, m)
        assert got.dtype == torch.int32 and got.shape == (3, m) and bool((got[:, 0] == 0).all())
        got = got.long()
        mind = torch.full(x.shape[:2], float("inf"), device="cuda")
        for j in range(1, m):
            prev = x.gather(1, got[:, j - 1, None, None].expand(-1, 1, 3))
            mind = torch.minimum(mind, ((x - prev) ** 2).sum(-1))
            pick = mind.gather(1, got[:, j, None]).squeeze(1)
            assert bool((pick >= mind.max(1).values * (1 - 1e-5)).all()), f"fps pick {j} is not a farthest point"
        ref = cpu_ops.furthest_point_sample(x, m).long()
        same = (x.gather(1, ref[..., None].expand(-1, -1, 3)) == x.gather(1, got[..., None].expand(-1, -1, 3))).all(-1)
        STATUS[f"fps_same_as_loop_m{m}"] = round(same.float().mean().item(), 4)


class _SelectiveScan(torch.autograd.Function):
    """selective_scan_cuda fwd/bwd exactly as mamba_ssm's SelectiveScanFn calls them
    (no z, no delta_bias, no softplus; B and C variable with one group)."""

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D):
        out, x, *_ = _SSC.fwd(u, delta, A, B, C, D, None, None, False)
        ctx.save_for_backward(u, delta, A, B, C, D, x)
        return out

    @staticmethod
    def backward(ctx, dout):
        u, delta, A, B, C, D, x = ctx.saved_tensors
        du, ddelta, dA, dB, dC, dD, *_ = _SSC.bwd(u, delta, A, B, C, D, None, None, dout.contiguous(),
                                                  x, None, None, False, False)
        return du, ddelta, dA, dB, dC, dD


_SSC = None


def _build_scan():
    global _SSC
    import selective_scan_cuda                    # the compiled extension of the mamba_ssm wheel
    _SSC = selective_scan_cuda

    def scan(u, delta, A, Bm, Cm, D):             # cpu_ops layout: u, delta (b,l,d); Bm, Cm (b,l,n)
        f = lambda t: t.float().transpose(1, 2).contiguous()
        y = _SelectiveScan.apply(f(u), f(delta), A.float().contiguous(), f(Bm).unsqueeze(1),
                                 f(Cm).unsqueeze(1), D.float().contiguous())
        return y.transpose(1, 2).to(u.dtype)
    return scan


def _check_scan(fast) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    b, l, d, n = 2, 96, 32, 16
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    ins = [r(b, l, d), torch.nn.functional.softplus(r(b, l, d)) * 0.1, -torch.rand(d, n, device="cuda", generator=g) * 4,
           r(b, l, n), r(b, l, n), r(d)]
    outs = []
    for fn in (cpu_ops.selective_scan, fast):
        xs = [t.clone().requires_grad_() for t in ins]
        y = fn(*xs)
        (y * torch.cos(torch.arange(y.numel(), device="cuda").view_as(y).float())).sum().backward()
        outs.append([y.detach()] + [t.grad for t in xs])
    for a, c in zip(*outs):
        err = (a - c).abs().max().item() / max(a.abs().max().item(), 1e-6)
        assert err < 1e-3, f"selective scan differs from the reference loop (rel err {err:.2e})"


def prepare() -> dict:
    """Enable the fast paths that build and pass their check (idempotent)."""
    if STATUS or not torch.cuda.is_available():
        return STATUS
    if os.environ.get("KCA_FAST_OPS", "1") == "0":
        STATUS.update(fps="off (KCA_FAST_OPS=0)", scan="off (KCA_FAST_OPS=0)")
    for key, build, check in (("fps", _build_fps, _check_fps), ("scan", _build_scan, _check_scan)):
        if key in STATUS:
            continue
        try:
            fn = build()
            check(fn)
            cpu_ops.FAST[key] = fn
            STATUS[key] = "on"
        except Exception as e:                      # fall back to the cpu_ops loop
            cpu_ops.FAST[key] = None
            STATUS[key] = f"off ({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''})"
    same = STATUS.get("fps_same_as_loop_m1024")
    print(f"[kca.cuda_fast] fps: {STATUS['fps']}" + (f" (picks equal to the loop's: {same:.2%})" if same is not None else "")
          + f" | scan: {STATUS['scan']}", file=sys.stderr, flush=True)
    return STATUS
