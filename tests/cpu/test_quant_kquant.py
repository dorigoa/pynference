"""S10: K-quant block layouts checked against a hand-written numpy decode.

gguf-py is the reference for the Triton kernel, but the element -> (sub-block,
byte, bit) mappings of the K-quants are easy to get wrong in the same way twice
(S9 lesson: an index typo costs a whole GPU job). The numpy decode below is
written straight from `dequantize_row_q4_K` / `dequantize_row_q5_K` in
ggml-quants.c, so a mismatch between it and gguf-py pins the layout down.
"""
import numpy as np
import pytest

from pynference.quant import TYPES, check_split, row_bytes
from pynference.quant.ref import ref_dequant
from pynference.quant.testing import make_test_blocks

N, K = 3, 512  # two super-blocks per row


def _fp16(blk, byte):
    return blk[..., byte:byte + 2].copy().view(np.float16).astype(np.float32)[..., 0]


def hand_kquant(raw, qtype, n, k):
    """x = d*sc*q - dmin*mn per 32-element sub-block (Q4_K, Q5_K)."""
    bb = TYPES[qtype].block_bytes
    blk = raw.reshape(n, -1, bb)
    e = np.arange(256)
    s = e >> 5                                    # sub-block of the element
    S = blk[..., 4:16]                            # 12 scale/min bytes
    s0 = S[..., s]
    s4 = S[..., s + 4]
    sm = S[..., np.where(s < 4, s, s - 4)]        # only read when s >= 4
    # ggml get_scale_min_k4: 6-bit scale and min packed over 3 bytes
    sc = np.where(s < 4, s0 & 0x3F, (s4 & 0xF) | ((sm >> 6) << 4))
    mn = np.where(s < 4, s4 & 0x3F, (s4 >> 4) | ((s0 >> 6) << 4))
    qoff = 16 if qtype == "Q4_K" else 48           # d, dmin, 12 scale bytes, [qh], qs
    qb = blk[..., qoff + ((s >> 1) << 5) + (e & 31)]
    q = (qb >> ((s & 1) * 4)) & 0xF
    if qtype == "Q5_K":                            # 5th bit: bit s of qh byte e&31
        qh = blk[..., 16 + (e & 31)]
        q = q | (((qh >> s) & 1) << 4)
    d = _fp16(blk, 0)[..., None]
    dmin = _fp16(blk, 2)[..., None]
    return (d * sc * q - dmin * mn).reshape(n, k).astype(np.float32)


@pytest.mark.parametrize("qtype", ["Q4_K", "Q5_K"])
def test_kquant_layout(qtype):
    raw = make_test_blocks(qtype, N, K, seed=11)
    assert raw.shape == (N, row_bytes(qtype, K))
    assert TYPES[qtype].block == 256 and row_bytes(qtype, 512) == 2 * TYPES[qtype].block_bytes
    ref = ref_dequant(raw, qtype)
    assert ref.shape == (N, K) and np.isfinite(ref).all()
    assert np.allclose(ref, hand_kquant(raw, qtype, N, K), rtol=1e-6, atol=0)


@pytest.mark.parametrize("qtype", ["Q4_K", "Q5_K"])
def test_kquant_scales_in_range(qtype):
    """make_test_blocks must keep both fp16 fields of every super-block finite."""
    raw = make_test_blocks(qtype, N, K, seed=5)
    blk = raw.reshape(N, -1, TYPES[qtype].block_bytes)
    for byte in (0, 2):
        v = _fp16(blk, byte)
        assert (v >= 1e-3).all() and (v <= 1e-1).all()


def test_kquant_split():
    with pytest.raises(ValueError):
        check_split("Q4_K", 257)
    with pytest.raises(ValueError):
        row_bytes("Q5_K", 128)
