"""CPU check of the Triton native-block decode, via the Triton interpreter.

The GPU job validates these kernels, but a nibble/qh index typo costs a whole
GPU job (S9-1 failed 19 tests exactly that way). Under TRITON_INTERPRET=1 the
same Triton kernels run on CPU tensors, so this test compares dequant_native
element-for-element with the gguf-py reference (quant.ref) for every block-32
type, on every element index, with the real GGUF block layouts.

The env var must be set before Triton is imported anywhere, hence the
module-level assignment before importing pynference.quant.kernels (no other
module in tests/cpu imports Triton).
"""
import os

os.environ["TRITON_INTERPRET"] = "1"

import numpy as np
import torch

from pynference.quant import kernels
from pynference.quant.ref import ref_dequant
from pynference.quant.testing import make_test_blocks

TYPES = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL", "Q8_0"]
# K-quants (S10): k must be a whole number of 256-element super-blocks, and 512
# covers every element index of the super-block layout twice
KTYPES = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]


def test_dequant_native_interpreted():
    n, k = 4, 64  # two blocks per row, all 32 element indices covered
    for qtype in TYPES:
        raw = make_test_blocks(qtype, n, k, seed=7)
        ref = ref_dequant(raw, qtype)
        out = torch.empty(n, k, dtype=torch.float32)
        kernels.dequant_native(qtype, torch.from_numpy(np.ascontiguousarray(raw)), out)
        bad = ~np.isclose(out.numpy(), ref, rtol=1e-4, atol=0)
        assert not bad.any(), f"{qtype}: {bad.sum()}/{bad.size} wrong elements"


def test_dequant_kquant_interpreted():
    n, k = 2, 512
    for qtype in KTYPES:
        raw = make_test_blocks(qtype, n, k, seed=7)
        ref = ref_dequant(raw, qtype)
        out = torch.empty(n, k, dtype=torch.float32)
        kernels.dequant_native(qtype, torch.from_numpy(np.ascontiguousarray(raw)), out)
        bad = ~np.isclose(out.numpy(), ref, rtol=1e-4, atol=0)
        assert not bad.any(), f"{qtype}: {bad.sum()}/{bad.size} wrong elements"
        ref = ref_dequant(raw, qtype)
        out = torch.empty(n, k, dtype=torch.float32)
        kernels.dequant_native(qtype, torch.from_numpy(np.ascontiguousarray(raw)), out)
        bad = ~np.isclose(out.numpy(), ref, rtol=1e-4, atol=0)
        assert not bad.any(), f"{qtype}: {bad.sum()}/{bad.size} wrong elements"


def _views(t):
    return t, t.view(torch.int8), t.view(torch.uint16)


def test_gemv_interpreted_tail():
    # BLOCK_K=512 with k<512 forces the clamped k-tail, n=3 with BLOCK_N=4
    # forces masked rows: both must agree with the dequant reference
    for qtype in TYPES + KTYPES:
        k = 64 if qtype in TYPES else 256
        n = 3
        raw_np = make_test_blocks(qtype, n, k, seed=11)
        ref = ref_dequant(raw_np, qtype)
        x_np = np.linspace(-1.0, 1.0, k, dtype=np.float32)
        rb = raw_np.shape[1]
        w8, wi, w16 = _views(torch.from_numpy(np.ascontiguousarray(raw_np)))
        y = torch.empty(1, n, dtype=torch.float32)
        kernels._gemv_kernel.fn[(1,)](w8, wi, w16, torch.from_numpy(x_np), y,
                                      n, k, rb, rb // 2,
                                      BLOCK_N=4, BLOCK_K=512, num_warps=4,
                                      **kernels._jit_args(qtype))
        bad = ~np.isclose(y[0].numpy(), ref @ x_np, rtol=1e-4, atol=1e-6)
        assert not bad.any(), f"{qtype} gemv tail: {bad.sum()}/{n} wrong rows"


def test_gemv_indexed_interpreted_tail():
    # same tail/row masking for the indexed kernel: slots [1, 0] on a stack
    # of two experts, one python-loop row per expert to compare against
    for qtype in TYPES + KTYPES:
        k = 64 if qtype in TYPES else 256
        n, e = 3, 2
        raws = [make_test_blocks(qtype, n, k, seed=11 + s) for s in range(e)]
        refs = [ref_dequant(r, qtype) for r in raws]
        rb = raws[0].shape[1]
        xs = np.stack([np.linspace(-1.0, 1.0, k, dtype=np.float32),
                       np.linspace(1.0, -1.0, k, dtype=np.float32)])
        stack = torch.from_numpy(np.ascontiguousarray(np.stack(raws))).reshape(e * n, rb)
        w8, wi, w16 = _views(stack)
        out = torch.empty(e, n, dtype=torch.float32)
        kernels._gemv_indexed_kernel.fn[(e, 1)](
            w8, wi, w16, torch.from_numpy(xs), torch.tensor([1, 0], dtype=torch.int32),
            out, n, k, rb, rb // 2, n * rb,
            BLOCK_N=4, BLOCK_K=512, num_warps=4, **kernels._jit_args(qtype))
        want = np.stack([refs[1] @ xs[0], refs[0] @ xs[1]])
        bad = ~np.isclose(out.numpy(), want, rtol=1e-4, atol=1e-6)
        assert not bad.any(), f"{qtype} gemv idx tail: {bad.sum()}/{want.size} wrong"
