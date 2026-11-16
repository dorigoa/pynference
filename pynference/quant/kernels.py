"""Triton kernels on GGUF-native quantized blocks (S9).

All kernels read the GGUF byte layout of the 32-element block types (Q4_0,
Q4_1, Q5_0, Q5_1, IQ4_NL and native Q8_0, which is used for expert stacks)
directly, so weights stay byte-for-byte as on disk (contract C2/C5). The
formulas mirror `dequantize_row_*` in ggml-quants.c.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .types import TYPES, row_bytes

# Fields of one GGUF block, in bytes/words (a word = 2 bytes):
# BB  block bytes, WPB = BB // 2 words per block, QS byte offset of the
# quantized data, MS word offset of m (None: none), QHB byte offset of the
# 4 qh bytes (None: none). The 5th bit of element e is bit e & 7 of the qh
# byte e >> 3 (little-endian), i.e. the same bit e of the uint32 that
# ggml-quants.c loads as two uint16 words; byte access keeps the Triton
# load pattern identical to the nibble bytes (much faster than uint16
# gathers with 32-bit shifts on Intel XPU: Q5_0 GEMV was 24 GB/s vs 193).
_LAYOUT = {
    "Q4_0":   dict(BB=18, WPB=9,  QS=2, MS=None, QHB=None),
    "Q4_1":   dict(BB=20, WPB=10, QS=4, MS=1,    QHB=None),
    "Q5_0":   dict(BB=22, WPB=11, QS=6, MS=None, QHB=2),
    "Q5_1":   dict(BB=24, WPB=12, QS=8, MS=1,    QHB=4),
    "IQ4_NL": dict(BB=18, WPB=9,  QS=2, MS=None, QHB=None),
    "Q8_0":   dict(BB=34, WPB=17, QS=2, MS=None, QHB=None),
}


@triton.jit
def _iq4nl(nib):
    # kvalues_iq4nl from ggml-common.h
    v = tl.where(nib == 0, -127.0, 113.0)
    v = tl.where(nib == 1, -104.0, v)
    v = tl.where(nib == 2, -83.0, v)
    v = tl.where(nib == 3, -65.0, v)
    v = tl.where(nib == 4, -49.0, v)
    v = tl.where(nib == 5, -35.0, v)
    v = tl.where(nib == 6, -22.0, v)
    v = tl.where(nib == 7, -10.0, v)
    v = tl.where(nib == 8, 1.0, v)
    v = tl.where(nib == 9, 13.0, v)
    v = tl.where(nib == 10, 25.0, v)
    v = tl.where(nib == 11, 38.0, v)
    v = tl.where(nib == 12, 53.0, v)
    v = tl.where(nib == 13, 69.0, v)
    v = tl.where(nib == 14, 89.0, v)
    return tl.where(nib == 15, 113.0, v)


@triton.jit
def _decode(RAW8, RAWI, W16, off8, b, e, RB16,
            Q8_0: tl.constexpr, HAS_M: tl.constexpr, HAS_QH: tl.constexpr, IQ4: tl.constexpr,
            BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
            MS: tl.constexpr, QHB: tl.constexpr):
    """Float32 values of elements e (0..31) of block b, rows at byte offset off8."""
    off16 = off8 >> 1
    d = tl.load(W16 + off16 + b * WPB).to(tl.float16, bitcast=True).to(tl.float32)
    if Q8_0:
        q = tl.load(RAWI + off8 + b * BB + QS + e).to(tl.float32)
        return q * d
    byte = tl.load(RAW8 + off8 + b * BB + QS + (e % 16)).to(tl.int32)
    nib = (byte >> tl.where(e >= 16, 4, 0)) & 0x0F
    if HAS_QH:
        qb = tl.load(RAW8 + off8 + b * BB + QHB + (e >> 3)).to(tl.int32)
        nib = nib | (((qb >> (e & 7)) & 1) << 4)
    v = nib.to(tl.float32)
    if IQ4:
        return _iq4nl(nib) * d
    if HAS_M:
        m = tl.load(W16 + off16 + b * WPB + MS).to(tl.float16, bitcast=True).to(tl.float32)
        return v * d + m
    if HAS_QH:  # Q5_0
        return (v - 16.0) * d
    return (v - 8.0) * d  # Q4_0


def _jit_args(qtype: str) -> dict:
    L = _LAYOUT[qtype]
    return dict(Q8_0=qtype == "Q8_0", HAS_M=L["MS"] is not None, HAS_QH=L["QHB"] is not None,
                IQ4=qtype == "IQ4_NL", BB=L["BB"], WPB=L["WPB"], QS=L["QS"],
                MS=L["MS"] if L["MS"] is not None else -1, QHB=L["QHB"] if L["QHB"] is not None else -1)


@triton.jit
def _dequant_kernel(RAW8, RAWI, W16, OUT, IDX, M, K, RB8, RB16,
                    Q8_0: tl.constexpr, HAS_M: tl.constexpr, HAS_QH: tl.constexpr, IQ4: tl.constexpr,
                    BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
                    MS: tl.constexpr, QHB: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, HAS_IDX: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    nm = rn < M
    if HAS_IDX:
        src = tl.load(IDX + rn, mask=nm, other=0).to(tl.int64)
    else:
        src = rn.to(tl.int64)
    src = tl.where(nm, src, 0)
    off8 = (src * RB8)[:, None]
    b = (rk // 32)[None, :]
    e = (rk % 32)[None, :]
    v = _decode(RAW8, RAWI, W16, off8, b, e, RB16, Q8_0, HAS_M, HAS_QH, IQ4, BB, WPB, QS, MS, QHB)
    m2 = nm[:, None] & (rk < K)[None, :]
    tl.store(OUT + rn.to(tl.int64)[:, None] * K + rk[None, :], v, mask=m2)


_GEMV_CONFIGS = [triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw)
                 for bn, bk, nw in [(4, 512, 4), (8, 256, 4), (8, 512, 4), (8, 512, 8),
                                    (16, 256, 4), (16, 512, 8), (32, 256, 8)]]


# the type constexprs must be in the autotune key: the same (N, K) shape is
# benchmarked with different configs for the nibble types vs Q5_0 vs Q8_0
_GEMV_KEY = ["N", "K", "Q8_0", "HAS_M", "HAS_QH", "IQ4", "BB"]


@triton.autotune(configs=_GEMV_CONFIGS, key=_GEMV_KEY)
@triton.jit
def _gemv_kernel(RAW8, RAWI, W16, X, Y, N, K, RB8, RB16,
                 Q8_0: tl.constexpr, HAS_M: tl.constexpr, HAS_QH: tl.constexpr, IQ4: tl.constexpr,
                 BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
                 MS: tl.constexpr, QHB: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = rn < N
    off8 = tl.where(nm, rn, 0).to(tl.int64)[:, None] * RB8
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < K
        x = tl.load(X + rk, mask=mk, other=0.0)
        b = (rk // 32)[None, :]
        e = (rk % 32)[None, :]
        w = _decode(RAW8, RAWI, W16, off8, b, e, RB16, Q8_0, HAS_M, HAS_QH, IQ4, BB, WPB, QS, MS, QHB)
        acc += tl.sum(tl.where(nm[:, None] & mk[None, :], w, 0.0) * x[None, :], axis=1)
    tl.store(Y + rn, acc, mask=nm)


@triton.autotune(configs=_GEMV_CONFIGS, key=_GEMV_KEY)
@triton.jit
def _gemv_indexed_kernel(RAW8, RAWI, W16, X, SLOTS, OUT, N, K, RB8, RB16, SLOT_BYTES,
                         Q8_0: tl.constexpr, HAS_M: tl.constexpr, HAS_QH: tl.constexpr, IQ4: tl.constexpr,
                         BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
                         MS: tl.constexpr, QHB: tl.constexpr,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    re = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    slot = tl.load(SLOTS + re).to(tl.int64)
    off8 = slot * SLOT_BYTES + rn.to(tl.int64)[:, None] * RB8
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < K
        x = tl.load(X + re.to(tl.int64) * K + rk, mask=mk, other=0.0)
        b = (rk // 32)[None, :]
        e = (rk % 32)[None, :]
        w = _decode(RAW8, RAWI, W16, off8, b, e, RB16, Q8_0, HAS_M, HAS_QH, IQ4, BB, WPB, QS, MS, QHB)
        acc += tl.sum(tl.where(mk[None, :], w, 0.0) * x[None, :], axis=1)
    tl.store(OUT + re.to(tl.int64) * N + rn, acc)


def _views(raw: torch.Tensor):
    return raw, raw.view(torch.int8), raw.view(torch.uint16)


def dequant_native(qtype: str, raw: torch.Tensor, out: torch.Tensor) -> None:
    """raw [N, row_bytes] uint8 GGUF -> out [N, K] (bf16 or float32)."""
    n, rb = raw.shape
    k = out.shape[-1]
    w8, wi, w16 = _views(raw)
    grid = (triton.cdiv(n, 32), triton.cdiv(k, 256))
    _dequant_kernel[grid](w8, wi, w16, out, out, n, k, rb, rb // 2,
                          BLOCK_N=32, BLOCK_K=256, HAS_IDX=False, **_jit_args(qtype), num_warps=8)


def rows_dequant_native(qtype: str, raw: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> None:
    """Dequant rows idx (int32/int64, [M]) to float32 out [M, K]."""
    n, rb = raw.shape
    m, k = out.shape
    w8, wi, w16 = _views(raw)
    grid = (triton.cdiv(m, 32), triton.cdiv(k, 256))
    _dequant_kernel[grid](w8, wi, w16, out, idx, m, k, rb, rb // 2,
                          BLOCK_N=32, BLOCK_K=256, HAS_IDX=True, **_jit_args(qtype), num_warps=8)


def gemv_native(qtype: str, raw: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
    """y [1, N] float32 = W @ x ; raw [N, row_bytes], x [K] float32 contiguous."""
    n, rb = raw.shape
    k = x.numel()
    w8, wi, w16 = _views(raw)
    _gemv_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)](
        w8, wi, w16, x, y, n, k, rb, rb // 2, **_jit_args(qtype))


def gemv_indexed_native(qtype: str, w_stack: torch.Tensor, x: torch.Tensor,
                        slots: torch.Tensor, out: torch.Tensor) -> None:
    """out [E, N] = W[slots[e]] @ x[e]; w_stack [S, N, row_bytes] GGUF, x [E, K]."""
    s, n, rb = w_stack.shape
    e, k = x.shape
    w8, wi, w16 = _views(w_stack.reshape(s * n, rb))
    grid = lambda meta: (e, triton.cdiv(n, meta["BLOCK_N"]))
    _gemv_indexed_kernel[grid](w8, wi, w16, x, slots, out, n, k, rb, rb // 2, n * rb,
                               **_jit_args(qtype))


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _gemv_half_kernel(W, X, Y, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = rn < N
    off = tl.where(nm, rn, 0).to(tl.int64)[:, None] * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < K
        x = tl.load(X + rk, mask=mk, other=0.0)
        w = tl.load(W + off + rk[None, :], mask=nm[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(Y + rn, acc, mask=nm)


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _gemv_indexed_half_kernel(W, X, SLOTS, OUT, N, K, SLOT_NK,
                              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    re = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    slot = tl.load(SLOTS + re).to(tl.int64)
    off = slot * SLOT_NK + rn.to(tl.int64)[:, None] * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < K
        x = tl.load(X + re.to(tl.int64) * K + rk, mask=mk, other=0.0)
        w = tl.load(W + off + rk[None, :], mask=mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + re.to(tl.int64) * N + rn, acc)


def gemv_half(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
    """F16/BF16 weight [N, K] 16-bit, x [K] float32, y [1, N] float32."""
    n, k = w.shape
    _gemv_half_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)](w, x, y, n, k)


def gemv_indexed_half(w_stack: torch.Tensor, x: torch.Tensor, slots: torch.Tensor,
                      out: torch.Tensor) -> None:
    """F16/BF16 expert stack [S, N, K] 16-bit, x [E, K], out [E, N]."""
    s, n, k = w_stack.shape
    e, _ = x.shape
    _gemv_indexed_half_kernel[lambda meta: (e, triton.cdiv(n, meta["BLOCK_N"]))](
        w_stack, x, slots, out, n, k, n * k)
