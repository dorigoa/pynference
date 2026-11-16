"""S9 step 3: TYPES registry, gguf-py reference, random test blocks, make_qweight.

The hand-written numpy dequant below encodes the formulas of
`dequantize_row_*` in ggml-quants.c; gguf-py must agree with it.
"""
import numpy as np
import pytest

from pynference.quant import (F32QWeight, HalfQWeight, NativeQWeight, TYPES,
                              QLinear, check_split, make_qweight, row_bytes)
from pynference.quant.ref import ref_dequant
from pynference.quant.testing import make_test_blocks
from pynference.quant.types import QType

import torch

N, K = 3, 128

# kvalues_iq4nl from ggml-common.h
KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.int8)


def _blocks(raw, n, bb):
    return raw.reshape(n, -1, bb)


def _scales(blk):
    return blk[..., :2].copy().view(np.float16).astype(np.float32)[..., 0]


def _nibbles(blk, off):
    qs = blk[..., off:off + 16]
    out = np.zeros(qs.shape[:2] + (32,), np.uint8)
    out[..., :16] = qs & 0x0F
    out[..., 16:] = qs >> 4
    return out


def hand_dequant(raw, qtype, n, k):
    if qtype == "Q4_0":
        blk = _blocks(raw, n, 18)
        return (_nibbles(blk, 2).astype(np.int32) - 8) * _scales(blk)[..., None]
    if qtype == "Q4_1":
        blk = _blocks(raw, n, 20)
        m = blk[..., 2:4].copy().view(np.float16).astype(np.float32)[..., 0]
        return _nibbles(blk, 4) * _scales(blk)[..., None] + m[..., None]
    if qtype in ("Q5_0", "Q5_1"):
        bb, qoff, off = (22, 2, 6) if qtype == "Q5_0" else (24, 4, 8)
        blk = _blocks(raw, n, bb)
        qh = blk[..., qoff:qoff + 4].copy().view(np.uint32)      # [n, nb, 1]
        # ggml-quants.c: bit e of qh is the 5th bit of element e (j and j+12 masks)
        fifth = ((qh >> np.arange(32, dtype=np.uint32)) & np.uint32(1)).astype(np.uint8) * 0x10
        v = _nibbles(blk, off) | fifth
        if qtype == "Q5_0":
            return (v.astype(np.int32) - 16) * _scales(blk)[..., None]
        m = blk[..., 2:4].copy().view(np.float16).astype(np.float32)[..., 0]
        return v * _scales(blk)[..., None] + m[..., None]
    if qtype == "Q8_0":
        blk = _blocks(raw, n, 34)
        return blk[..., 2:].view(np.int8) * _scales(blk)[..., None]
    if qtype == "IQ4_NL":
        blk = _blocks(raw, n, 18)
        return KV_IQ4NL[_nibbles(blk, 2)] * _scales(blk)[..., None]
    raise AssertionError(qtype)


def test_types_match_gguf_py():
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    for name, t in TYPES.items():
        q = GGMLQuantizationType(t.ggml_id)
        assert q.name == name
        assert (t.block, t.block_bytes) == GGML_QUANT_SIZES[q]
        assert isinstance(t, QType)


def test_row_bytes_and_check_split():
    assert row_bytes("Q4_0", 5376) == 5376 // 32 * 18
    assert row_bytes("Q8_0", 5376) == 5376 // 32 * 34
    assert row_bytes("IQ4_NL", 3072) == 3072 // 32 * 18
    assert row_bytes("F16", 100) == 200
    with pytest.raises(ValueError):
        check_split("Q4_0", 33)
    with pytest.raises(ValueError):
        row_bytes("Q5_0", 31)


@pytest.mark.parametrize("qtype", ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL"])
def test_blocks_valid_and_ref_matches_hand(qtype):
    raw = make_test_blocks(qtype, N, K, seed=7)
    assert raw.shape == (N, row_bytes(qtype, K)) and raw.dtype == np.uint8
    words = raw.view(np.uint16)
    wpb = row_bytes(qtype, K) // 2 // (K // 32)
    d = words[:, 0::wpb].copy().view(np.float16).astype(np.float32)
    assert (d >= 1e-3).all() and (d <= 1e-1).all()  # scales in the required range
    ref = ref_dequant(raw, qtype)
    assert ref.shape == (N, K) and ref.dtype == np.float32 and np.isfinite(ref).all()
    assert np.allclose(ref, hand_dequant(raw, qtype, N, K).reshape(N, K), rtol=1e-6, atol=0)


@pytest.mark.parametrize("qtype", ["F32", "F16", "BF16"])
def test_float_blocks(qtype):
    raw = make_test_blocks(qtype, N, K, seed=3)
    assert raw.shape == (N, row_bytes(qtype, K))
    ref = ref_dequant(raw, qtype)
    assert ref.shape == (N, K)
    assert (ref > 0).all() and (ref >= 1e-3 - 1e-4).all() and (ref <= 1e-1 + 1e-3).all()


def test_make_qweight_native():
    raw = make_test_blocks("Q4_0", N, K, seed=1)
    w = make_qweight(raw, "Q4_0", N, K, torch.device("cpu"))
    assert isinstance(w, NativeQWeight)
    assert (w.qtype, w.N, w.K) == ("Q4_0", N, K)
    assert w.nbytes == raw.nbytes
    assert torch.equal(w.raw, torch.from_numpy(raw))  # bytes kept as-is
    with pytest.raises(ValueError):
        make_qweight(raw, "Q4_0", N, K + 1, torch.device("cpu"))


def test_make_qweight_all_types():
    cpu = torch.device("cpu")
    w = make_qweight(make_test_blocks("Q8_0", N, K, seed=2), "Q8_0", N, K, cpu)
    assert isinstance(w, QLinear) and w.qtype == "Q8_0" and (w.N, w.K) == (N, K)
    w = make_qweight(make_test_blocks("F16", N, K, seed=2), "F16", N, K, cpu)
    assert isinstance(w, HalfQWeight) and w.w.dtype == torch.float16
    w = make_qweight(make_test_blocks("BF16", N, K, seed=2), "BF16", N, K, cpu)
    assert isinstance(w, HalfQWeight) and w.w.dtype == torch.bfloat16
    w = make_qweight(make_test_blocks("F32", N, K, seed=2), "F32", N, K, cpu)
    assert isinstance(w, F32QWeight)
    x = torch.rand(2, K)
    assert torch.allclose(w(x), x @ w.w.T)  # F32 path is plain torch (C2)
    assert w.dequant().dtype == torch.bfloat16
    assert w.rows_dequant(torch.tensor([0, 2])).shape == (2, K)
