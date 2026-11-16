"""S12: IQ3_XXS and IQ2_S block layouts checked against numpy decodes.

Same pattern as test_quant_s11.py (S10 lesson): gguf-py is the GPU-kernel
reference, so a second decode written straight from `dequantize_row_iq3_xxs`
and `dequantize_row_iq2_s` in ggml-quants.c pins the byte/bit layout down
against it. The grid tables come from the generated quant/iq_grids.py.
"""
import numpy as np
import pytest

from pynference.quant import TYPES, check_split, row_bytes
from pynference.quant.iq_grids import IQ2S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS
from pynference.quant.ref import ref_dequant
from pynference.quant.testing import make_test_blocks

N, K = 3, 512

_IQ3XXS_BYTES = np.array(IQ3XXS_GRID, dtype=np.uint32).view(np.uint8)
_IQ2S_BYTES = np.array(IQ2S_GRID, dtype=np.uint64).view(np.uint8)
_KSIGNS = np.array(KSIGNS_IQ2XS, dtype=np.int32)


def _fp16(blk, byte):
    return blk[..., byte:byte + 2].copy().view(np.float16).astype(np.float32)[..., 0]


def hand_iq3_xxs(raw, n, k):
    """x = db * grid_byte * sign (dequantize_row_iq3_xxs).

    8 sub-blocks of 32; aux32 (SCL=66 + 4*sub) packs the 4-bit db scale at
    bits 28..31 and seven 7-bit sign picks; element e: l = (e>>3)&3 picks the
    grid-index byte qs[2l] (e&7 < 4) or qs[2l+1] and the sign pick l; the grid
    byte is IQ3XXS_GRID[idx] byte e&3; the sign bit is bit e&7 of ksigns.
    """
    blk = raw.reshape(n, -1, TYPES["IQ3_XXS"].block_bytes)
    e = np.arange(256)
    ib, l, jj = e >> 5, (e >> 3) & 3, e & 7
    d = _fp16(blk, 0)[..., None]
    so = 66 + ib * 4
    aux = (blk[..., so].astype(np.uint32)
           | (blk[..., so + 1].astype(np.uint32) << np.uint32(8))
           | (blk[..., so + 2].astype(np.uint32) << np.uint32(16))
           | (blk[..., so + 3].astype(np.uint32) << np.uint32(24)))
    db = d * (0.5 + (aux >> np.uint32(28))) * 0.5
    qb = blk[..., 2 + ib * 8 + 2 * l + (jj >> 2)].astype(np.int32)  # *4 overflows uint8
    gb = _IQ3XXS_BYTES[qb * 4 + (jj & 3)].astype(np.float32)
    sg = _KSIGNS[(aux >> (7 * l)) & 127]
    sgn = np.where(((sg >> jj) & 1) == 1, -1.0, 1.0)
    return (db * gb * sgn).reshape(n, k).astype(np.float32)


def hand_iq2_s(raw, n, k):
    """x = db * grid_byte * sign (dequantize_row_iq2_s).

    8 sub-blocks of 32; element e: ib = e>>5, l = (e>>3)&3, jj = e&7. The
    10-bit grid index is qs[4ib+l] | 2 bits of qh[ib] (at shift 8-2*l, the
    same as (qh>>2l)&3 at bit 8); db = d*(0.5 + nibble)*0.25 with the low
    nibble of scales[ib] for l<2 and the high one for l>=2; the sign bit is
    bit jj of qs[32 + 4ib + l] (kmask_iq2xs[jj] = 1 << jj).
    """
    blk = raw.reshape(n, -1, TYPES["IQ2_S"].block_bytes)
    e = np.arange(256)
    ib, l, jj = e >> 5, (e >> 3) & 3, e & 7
    d = _fp16(blk, 0)[..., None]
    qi = (blk[..., 2 + ib * 4 + l].astype(np.int32)
          | (((blk[..., 66 + ib].astype(np.int32) >> (2 * l)) & 3) << 8))
    gb = _IQ2S_BYTES[qi * 8 + jj].astype(np.float32)
    sc = blk[..., 74 + ib].astype(np.int32)
    nib = np.where(l < 2, sc & 0xF, sc >> 4)
    db = d * (0.5 + nib) * 0.25
    sg = blk[..., 34 + ib * 4 + l].astype(np.int32)
    sgn = np.where(((sg >> jj) & 1) == 1, -1.0, 1.0)
    return (db * gb * sgn).reshape(n, k).astype(np.float32)


HAND = {"IQ3_XXS": hand_iq3_xxs, "IQ2_S": hand_iq2_s}


@pytest.mark.parametrize("qtype", list(HAND))
def test_layout_vs_ggufpy(qtype):
    raw = make_test_blocks(qtype, N, K, seed=12)
    assert raw.shape == (N, row_bytes(qtype, K))
    ref = ref_dequant(raw, qtype)
    assert ref.shape == (N, K) and np.isfinite(ref).all()
    assert np.allclose(ref, HAND[qtype](raw, N, K), rtol=1e-6, atol=0)


def test_iq3_xxs_registry():
    t = TYPES["IQ3_XXS"]
    assert (t.ggml_id, t.block, t.block_bytes) == (18, 256, 98)
    assert row_bytes("IQ3_XXS", 512) == 2 * 98
    with pytest.raises(ValueError):
        check_split("IQ3_XXS", 257)


def test_iq2_s_registry():
    t = TYPES["IQ2_S"]
    assert (t.ggml_id, t.block, t.block_bytes) == (22, 256, 82)
    assert row_bytes("IQ2_S", 512) == 2 * 82
    with pytest.raises(ValueError):
        check_split("IQ2_S", 257)


def test_iq3_xxs_scale_in_range():
    """make_test_blocks keeps every fp16 d in [1e-3, 1e-1]."""
    raw = make_test_blocks("IQ3_XXS", N, K, seed=6)
    d = raw.reshape(N, -1, 98)[..., 0:2].copy().view(np.float16).astype(np.float32)
    assert (d >= 1e-3).all() and (d <= 1e-1).all()
