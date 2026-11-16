"""S11: MXFP4, IQ4_XS and Q2_0 block layouts checked against numpy decodes.

Same pattern as test_quant_kquant.py (S10 lesson): gguf-py is the GPU-kernel
reference, so a second decode written straight from `dequantize_row_mxfp4`,
`dequantize_row_iq4_xs` and `dequantize_row_q2_0` in ggml-quants.c pins the
byte/bit layout down against it.
"""
import numpy as np
import pytest

from pynference.quant import TYPES, check_split, row_bytes
from pynference.quant.ref import ref_dequant
from pynference.quant.testing import make_test_blocks

N, K = 3, 512

KVALUES_FP4 = np.array((0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12),
                       dtype=np.int8)
KVALUES_IQ4NL = np.array((-127, -104, -83, -65, -49, -35, -22, -10,
                          1, 13, 25, 38, 53, 69, 89, 113), dtype=np.int8)


def _fp16(blk, byte):
    return blk[..., byte:byte + 2].copy().view(np.float16).astype(np.float32)[..., 0]


def hand_mxfp4(raw, n, k):
    """x = 2**(e-128) * kvalues_fp4[nib] (dequantize_row_mxfp4).

    The scale is ggml_e8m0_to_fp32_half: the gguf-py bit trick, equal to
    2**(e-128) for every e in 0..255 (e=0,1 are denormal mantissas).
    """
    blk = raw.reshape(n, -1, 17)
    e = blk[..., 0].astype(np.uint32)
    bits = np.where(e < 2, np.uint32(0x00200000) << e, (e - np.uint32(1)) << np.uint32(23))
    d = bits[..., None].view(np.float32)
    qs = blk[..., 1:17]
    x = np.empty((n, qs.shape[1], 32), dtype=np.int8)
    x[..., :16] = KVALUES_FP4[qs & 0xF]      # low nibbles are the first half
    x[..., 16:] = KVALUES_FP4[qs >> 4]
    return (d * x.astype(np.float32)).reshape(n, k).astype(np.float32)


def hand_iq4_xs(raw, n, k):
    """x = d*(ls-32)*kvalues_iq4nl[q], 8 sub-blocks of 32 (dequantize_row_iq4_xs)."""
    blk = raw.reshape(n, -1, TYPES["IQ4_XS"].block_bytes)
    e = np.arange(256)
    ib = e >> 5                                   # sub-block of the element
    sh = blk[..., 2:4].copy().view(np.uint16)[..., 0].astype(np.int32)[..., None]  # scales_h
    sl = blk[..., 4:8]                                # scales_l (4 bytes)
    ls = ((sl[..., ib >> 1] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
    qb = blk[..., 8 + ((e >> 5) << 4) + (e & 15)]
    nib = (qb >> (((e >> 4) & 1) * 4)) & 0xF
    d = _fp16(blk, 0)[..., None]
    return (d * (ls - 32) * KVALUES_IQ4NL[nib]).reshape(n, k).astype(np.float32)


def hand_q2_0(raw, n, k):
    """x = (q-1)*d, 2-bit values 4 per byte (dequantize_row_q2_0)."""
    blk = raw.reshape(n, -1, TYPES["Q2_0"].block_bytes)
    e = np.arange(64)
    q = (blk[..., 2 + (e >> 2)] >> ((e % 4) * 2)) & 3
    d = _fp16(blk, 0)[..., None]
    return ((q.astype(np.int32) - 1) * d).reshape(n, k).astype(np.float32)


HAND = {"MXFP4": hand_mxfp4, "IQ4_XS": hand_iq4_xs, "Q2_0": hand_q2_0}


@pytest.mark.parametrize("qtype", list(HAND))
def test_layout_vs_ggufpy(qtype):
    raw = make_test_blocks(qtype, N, K, seed=11)
    assert raw.shape == (N, row_bytes(qtype, K))
    ref = ref_dequant(raw, qtype)
    assert ref.shape == (N, K) and np.isfinite(ref).all()
    assert np.allclose(ref, HAND[qtype](raw, N, K), rtol=1e-6, atol=0)


def test_mxfp4_registry_and_odd_rows():
    t = TYPES["MXFP4"]
    assert (t.ggml_id, t.block, t.block_bytes) == (39, 32, 17)
    # an odd row_bytes (17 blocks of 17 bytes) still splits and sizes right
    assert row_bytes("MXFP4", 544) == 17 * 17
    with pytest.raises(ValueError):
        check_split("MXFP4", 33)


def test_mxfp4_scales_in_range():
    """make_test_blocks keeps every E8M0 byte of every block in [118, 138]."""
    raw = make_test_blocks("MXFP4", N, 17 * 32, seed=5)
    e = raw.reshape(N, -1, 17)[..., 0]
    assert (e >= 118).all() and (e <= 138).all()
