"""S10: K-quant block layouts checked against a hand-written numpy decode.

gguf-py is the reference for the Triton kernel, but the element -> (sub-block,
byte, bit) mappings of the K-quants are easy to get wrong in the same way twice
(S9 lesson: an index typo costs a whole GPU job). The numpy decode below is
written straight from `dequantize_row_q2_K` ... `q6_K` in ggml-quants.c, so a
mismatch between it and gguf-py pins the layout down.
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


def hand_q6(raw, n, k):
    """x = d*sc_int8*((ql-nibble | qh 2 bits<<4) - 32) (Q6_K).

    ggml-quants.c loops over 32 bytes l of ql twice (l and l+32) and takes the
    low nibbles for p<2, the high ones for p>=2 (p = (e>>5)&3); the 2 high bits
    of the value come from byte l of qh at shift 2*p.
    """
    blk = raw.reshape(n, -1, TYPES["Q6_K"].block_bytes)
    e = np.arange(256)
    p = (e >> 5) & 3
    ql = blk[..., ((e >> 7) << 6) + ((p & 1) << 5) + (e & 31)]
    qh = blk[..., 128 + ((e >> 7) << 5) + (e & 31)]
    q6 = ((ql >> ((p >> 1) * 4)) & 0xF) | (((qh >> (p * 2)) & 3) << 4)
    sc = blk[..., 192:208].view(np.int8)[..., e >> 4]     # signed scale per 16
    d = _fp16(blk, 208)[..., None]
    return (d * sc * (q6 - 32)).reshape(n, k).astype(np.float32)


def hand_q2(raw, n, k):
    """x = d*(sc&15)*q - dmin*(sc>>4), 16 sub-blocks of 16 (Q2_K)."""
    blk = raw.reshape(n, -1, TYPES["Q2_K"].block_bytes)
    e = np.arange(256)
    sc = blk[..., e >> 4]                                # scales @0..15
    qb = blk[..., 16 + ((e >> 7) << 5) + (e & 31)]       # qs @16..79
    q = (qb >> (2 * ((e >> 5) & 3))) & 3
    d = _fp16(blk, 80)[..., None]
    dmin = _fp16(blk, 82)[..., None]
    return (d * (sc & 0xF) * q - dmin * (sc >> 4)).reshape(n, k).astype(np.float32)


def hand_q3(raw, n, k):
    """x = d*(sc-32)*(q-4), +4 where the hmask bit is NOT set (Q3_K)."""
    blk = raw.reshape(n, -1, TYPES["Q3_K"].block_bytes)
    e = np.arange(256)
    s = e >> 4                                           # 16 sub-blocks of 16
    S = blk[..., 96:108]                                 # 12 bytes, 6 bits each
    lo = S[..., s & 7]                                   # low nibble, high nibble from s=8
    hi = S[..., 8 + (s & 3)]                             # 2 high bits, 4 per byte
    sc = ((lo >> np.where(s >= 8, 4, 0)) & 0xF) | (((hi >> (2 * (s >> 2))) & 3) << 4)
    qb = blk[..., 32 + ((e >> 7) << 5) + (e & 31)]       # qs @32..95
    q = (qb >> (2 * ((e >> 5) & 3))) & 3
    hm = blk[..., e & 31]                                # hmask @0..31
    hb = (hm >> (e >> 5)) & 1
    d = _fp16(blk, 108)[..., None]
    return (d * (sc.astype(np.int32) - 32) * (q + 4 * hb - 4)).reshape(n, k).astype(np.float32)


def hand(raw, qtype, n, k):
    if qtype == "Q2_K":
        return hand_q2(raw, n, k)
    if qtype == "Q3_K":
        return hand_q3(raw, n, k)
    return hand_q6(raw, n, k) if qtype == "Q6_K" else hand_kquant(raw, qtype, n, k)


# fp16 scale fields of each type, as byte offsets inside one super-block
_FP16_BYTES = {"Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2),
               "Q6_K": (208,)}


@pytest.mark.parametrize("qtype", ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"])
def test_kquant_layout(qtype):
    raw = make_test_blocks(qtype, N, K, seed=11)
    assert raw.shape == (N, row_bytes(qtype, K))
    assert TYPES[qtype].block == 256 and row_bytes(qtype, 512) == 2 * TYPES[qtype].block_bytes
    ref = ref_dequant(raw, qtype)
    assert ref.shape == (N, K) and np.isfinite(ref).all()
    assert np.allclose(ref, hand(raw, qtype, N, K), rtol=1e-6, atol=0)


@pytest.mark.parametrize("qtype", ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"])
def test_kquant_scales_in_range(qtype):
    """make_test_blocks must keep every fp16 field of every super-block finite."""
    raw = make_test_blocks(qtype, N, K, seed=5)
    blk = raw.reshape(N, -1, TYPES[qtype].block_bytes)
    for byte in _FP16_BYTES[qtype]:
        v = _fp16(blk, byte)
        assert (v >= 1e-3).all() and (v <= 1e-1).all()


def test_kquant_split():
    with pytest.raises(ValueError):
        check_split("Q4_K", 257)
    with pytest.raises(ValueError):
        row_bytes("Q5_K", 128)
