"""Quantized weights on the GPU and the kernels that use them.

Q8_0 (GGUF: blocks of 32 int8 values with one fp16 scale, 34 bytes) is
repacked on the GPU into a contiguous int8 matrix `qs` [N, K] and a scale
matrix `d` [N, K/32], which Triton loads with simple 2D tiles.

QLinear(x) with x [M, K] float32 returns [M, N] float32:
  M == 1  Triton GEMV that reads the int8 weights directly (decode path);
  M > 1   dequantize to a bf16 scratch buffer + oneDNN matmul (prefill path).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .weight import _Scratch

QK8_0 = 32


def q8_0_to_device(raw: np.ndarray, n: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw Q8_0 bytes of an [n, k] matrix (row-major, 34-byte blocks) -> (qs int8 [n, k], d fp16 [n, k/32])."""
    assert k % QK8_0 == 0 and raw.size == n * k // QK8_0 * 34, (raw.shape, n, k)
    blocks = torch.from_numpy(np.array(raw, copy=True)).to(device).view(n, k // QK8_0, 34)
    d = blocks[:, :, :2].contiguous().view(torch.float16).view(n, k // QK8_0)
    qs = blocks[:, :, 2:].contiguous().view(torch.int8).view(n, k)
    return qs, d


_GEMV_CONFIGS = [triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw)
                 for bn, bk, nw in [(4, 512, 4), (8, 256, 4), (8, 512, 4), (8, 512, 8),
                                    (16, 256, 4), (16, 256, 8), (16, 512, 8), (32, 256, 8)]]


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _q8_gemv(W, S, X, Y, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = rn < N
    rn64 = rn.to(tl.int64)
    nb = K // 32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        m2 = nm[:, None] & (rk < K)[None, :]
        x = tl.load(X + rk, mask=rk < K, other=0.0)
        w = tl.load(W + rn64[:, None] * K + rk[None, :], mask=m2, other=0).to(tl.float32)
        s = tl.load(S + rn64[:, None] * nb + (rk // 32)[None, :], mask=m2, other=0.0).to(tl.float32)
        acc += tl.sum(w * s * x[None, :], axis=1)
    tl.store(Y + rn, acc, mask=nm)


@triton.jit
def _q8_dequant(W, S, O, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    m2 = (rn < N)[:, None] & (rk < K)[None, :]
    rn64 = rn.to(tl.int64)
    w = tl.load(W + rn64[:, None] * K + rk[None, :], mask=m2, other=0).to(tl.float32)
    s = tl.load(S + rn64[:, None] * (K // 32) + (rk // 32)[None, :], mask=m2, other=0.0).to(tl.float32)
    tl.store(O + rn64[:, None] * K + rk[None, :], (w * s).to(tl.bfloat16), mask=m2)


class QLinear:
    """y = x @ W^T for a Q8_0 weight W [N, K] held as int8 + fp16 block scales.

    A QWeight (C2) with the one repacked layout, used for dense Q8_0 weights.
    """

    qtype = "Q8_0"

    def __init__(self, qs: torch.Tensor, d: torch.Tensor):
        self.qs, self.d = qs, d
        self.N, self.K = qs.shape

    @classmethod
    def from_q8_0(cls, raw: np.ndarray, n: int, k: int, device) -> "QLinear":
        return cls(*q8_0_to_device(raw, n, k, device))

    @property
    def nbytes(self) -> int:
        return self.qs.numel() + self.d.numel() * 2

    def dequant(self, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.qs.device)
        grid = (triton.cdiv(self.N, 32), triton.cdiv(self.K, 256))
        _q8_dequant[grid](self.qs, self.d, out, self.N, self.K, BLOCK_N=32, BLOCK_K=256, num_warps=8)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        if M == 1:
            x = x.reshape(self.K).float().contiguous()
            y = torch.empty(1, self.N, dtype=torch.float32, device=x.device)
            _q8_gemv[lambda meta: (triton.cdiv(self.N, meta["BLOCK_N"]),)](
                self.qs, self.d, x, y, self.N, self.K)
            return y
        w = self.dequant(_Scratch.get(self.N * self.K, self.qs.device).view(self.N, self.K))
        return F.linear(x.to(torch.bfloat16), w).float()

    def rows_dequant(self, idx: torch.Tensor) -> torch.Tensor:
        """Dequantized rows idx (embedding lookup), float32 [len(idx), K]."""
        q = self.qs[idx].float().view(-1, self.K // QK8_0, QK8_0)
        return (q * self.d[idx].float()[:, :, None]).view(-1, self.K)


def _selftest():
    import time
    dev = torch.device("xpu", 0)
    torch.manual_seed(0)
    for n, k in [(8192, 5376), (5376, 10752), (21504, 5376), (131072, 5376)]:
        w = torch.randn(n, k, device=dev) * 0.02
        d = (w.view(n, k // 32, 32).abs().amax(-1) / 127).clamp(min=1e-8).half()
        qs = torch.round(w.view(n, k // 32, 32) / d.float()[..., None]).clamp(-127, 127).to(torch.int8).view(n, k)
        lin = QLinear(qs, d)
        wref = lin.dequant().float()
        x = torch.randn(1, k, device=dev)
        y = lin(x)
        ref = x @ wref.T
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        for _ in range(3):
            lin(x)
        torch.xpu.synchronize()
        t = time.perf_counter()
        it = 50
        for _ in range(it):
            lin(x)
        torch.xpu.synchronize()
        dt = (time.perf_counter() - t) / it
        x8 = torch.randn(8, k, device=dev)
        err8 = ((lin(x8) - x8 @ wref.T).abs().max() / (x8 @ wref.T).abs().max()).item()
        print(f"Q8_0 GEMV {n:6d}x{k:5d}: {lin.nbytes / dt / 1e9:6.1f} GB/s  {dt * 1e6:7.1f} us  "
              f"rel err {err:.2e}  (M=8 path rel err {err8:.2e})  cfg {_q8_gemv.best_config}")
        del w, qs, d, lin, wref


if __name__ == "__main__":
    _selftest()
