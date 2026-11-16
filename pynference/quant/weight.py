"""QWeight: one quantized weight [N, K] on the device, with GEMV and dequant.

Weights stay in the native GGUF layout (contract C2), so expert stacks are
copied from RAM byte for byte. Q8_0 is the one dense type with a repacked
layout (`QLinear` in q8_0.py). The Triton kernels used here live in
`kernels.py`, and `q8_0.py` pulls in Triton too; both imports are lazy so
that CPU-only paths (gguf reading, tests, moe) never load Triton at import
time (tests/cpu/test_quant_decode_interp.py needs to set TRITON_INTERPRET
before Triton is first imported).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .types import TYPES, check_split, row_bytes


class _Scratch:
    """One growable bf16 buffer per device for dequantized prefill weights."""
    bufs: dict = {}

    @classmethod
    def get(cls, numel: int, device) -> torch.Tensor:
        b = cls.bufs.get(device)
        if b is None or b.numel() < numel:
            b = cls.bufs[device] = torch.empty(numel, dtype=torch.bfloat16, device=device)
        return b[:numel]


class QWeight:
    """y = x @ W^T for a quantized W [N, K]. Subclasses pick the layout."""

    qtype: str
    N: int
    K: int

    @property
    def nbytes(self) -> int:
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x [M, K] float32 -> [M, N] float32 (C2: GEMV for M == 1, dequant + bf16 matmul otherwise)."""
        raise NotImplementedError

    def dequant(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """[N, K] bf16."""
        raise NotImplementedError

    def rows_dequant(self, idx: torch.Tensor) -> torch.Tensor:
        """Rows idx as float32 [len(idx), K] (embedding lookups)."""
        raise NotImplementedError


class NativeQWeight(QWeight):
    """Weight stored as raw GGUF bytes [N, row_bytes] on the device."""

    def __init__(self, qtype: str, raw: torch.Tensor, n: int, k: int):
        assert raw.dtype == torch.uint8 and raw.shape == (n, row_bytes(qtype, k))
        self.qtype, self.raw, self.N, self.K = qtype, raw, n, k

    @property
    def nbytes(self) -> int:
        return self.raw.nbytes

    def dequant(self, out: torch.Tensor | None = None) -> torch.Tensor:
        from . import kernels
        if out is None:
            out = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.raw.device)
        kernels.dequant_native(self.qtype, self.raw, out)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from . import kernels
        if x.shape[-1] != self.K:
            raise ValueError(f"x has K={x.shape[-1]}, weight has K={self.K}")
        if x.numel() == self.K:  # decode path
            y = torch.empty(1, self.N, dtype=torch.float32, device=x.device)
            kernels.gemv_native(self.qtype, self.raw, x.reshape(-1).float().contiguous(), y)
            return y
        w = self.dequant(_Scratch.get(self.N * self.K, self.raw.device).view(self.N, self.K))
        return F.linear(x.to(torch.bfloat16), w).float()

    def rows_dequant(self, idx: torch.Tensor) -> torch.Tensor:
        from . import kernels
        out = torch.empty(idx.numel(), self.K, dtype=torch.float32, device=self.raw.device)
        kernels.rows_dequant_native(self.qtype, self.raw, idx, out)
        return out


class HalfQWeight(QWeight):
    """F16/BF16 weight on the device [N, K] as a 16-bit float tensor."""

    def __init__(self, qtype: str, w: torch.Tensor, n: int, k: int):
        assert qtype in ("F16", "BF16")
        assert w.dtype == (torch.float16 if qtype == "F16" else torch.bfloat16)
        self.qtype, self.w, self.N, self.K = qtype, w, n, k

    @property
    def nbytes(self) -> int:
        return self.w.nbytes

    def dequant(self, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.empty(self.N, self.K, dtype=torch.bfloat16, device=self.w.device)
        return out.copy_(self.w)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from . import kernels
        if x.numel() == self.K:
            y = torch.empty(1, self.N, dtype=torch.float32, device=x.device)
            kernels.gemv_half(self.w, x.reshape(-1).float().contiguous(), y)
            return y
        return F.linear(x.to(torch.bfloat16), self.w.to(torch.bfloat16)).float()

    def rows_dequant(self, idx: torch.Tensor) -> torch.Tensor:
        return self.w[idx].float()


class F32QWeight(QWeight):
    """F32 weight on the device: plain torch (C2)."""

    qtype = "F32"

    def __init__(self, w: torch.Tensor, n: int, k: int):
        assert w.dtype == torch.float32
        self.w, self.N, self.K = w, n, k

    @property
    def nbytes(self) -> int:
        return self.w.nbytes

    def dequant(self, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            return self.w.to(torch.bfloat16)
        return out.copy_(self.w)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().reshape(-1, self.K) @ self.w.T

    def rows_dequant(self, idx: torch.Tensor) -> torch.Tensor:
        return self.w[idx]


def make_qweight(raw: np.ndarray, qtype: str, n: int, k: int, device) -> QWeight:
    """Raw GGUF bytes of an [n, k] matrix (row-major) -> QWeight on device."""
    check_split(qtype, k)
    rb = row_bytes(qtype, k)
    if raw.size != n * rb:
        raise ValueError(f"{qtype}: expected {n * rb} bytes for {n}x{k}, got {raw.size}")
    t = TYPES[qtype]
    if qtype == "Q8_0":
        from .q8_0 import QLinear, q8_0_to_device  # lazy: pulls in Triton
        return QLinear(*q8_0_to_device(raw, n, k, device))
    dev_bytes = torch.from_numpy(np.ascontiguousarray(raw)).to(device)
    if qtype == "F32":
        return F32QWeight(dev_bytes.view(torch.float32).view(n, k), n, k)
    if qtype == "F16":
        return HalfQWeight(qtype, dev_bytes.view(torch.float16).view(n, k), n, k)
    if qtype == "BF16":
        return HalfQWeight(qtype, dev_bytes.view(torch.bfloat16).view(n, k), n, k)
    return NativeQWeight(qtype, dev_bytes.view(n, rb), n, k)


def gemv_indexed(qtype: str, w_stack: torch.Tensor, x: torch.Tensor,
                 slots: torch.Tensor, out: torch.Tensor) -> None:
    """out[e] = W[slots[e]] @ x[e] on expert stacks (C2/C5).

    w_stack: [S, N, row_bytes] uint8 GGUF (or [S, N, K] for 16-bit/F32),
    x: [E, K] float32, slots: [E] int32, out: [E, N] float32.
    """
    if qtype == "F32":
        for e in range(slots.shape[0]):
            out[e].copy_(w_stack[slots[e]].T @ x[e])
        return
    from . import kernels
    if qtype in ("F16", "BF16"):
        kernels.gemv_indexed_half(w_stack, x, slots, out)
    else:
        kernels.gemv_indexed_native(qtype, w_stack, x, slots, out)
