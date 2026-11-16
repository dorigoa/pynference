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
