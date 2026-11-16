"""S11 step 1: the MXFP4 block layout checked against a numpy decode.

Same pattern as test_quant_kquant.py (S10 lesson): gguf-py is the GPU-kernel
reference, so a second decode written straight from `dequantize_row_mxfp4` in
ggml-quants.c pins the byte/bit layout down against it.
"""
import numpy as np
import pytest

from pynference.quant import TYPES, check_split, row_bytes
from pynference.quant.ref import ref_dequant
from pynference.quant.testing import make_test_blocks

N, K = 3, 512

KVALUES_FP4 = np.array((0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12),
                       dtype=np.int8)


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


def test_layout_vs_ggufpy():
    raw = make_test_blocks("MXFP4", N, K, seed=11)
    assert raw.shape == (N, row_bytes("MXFP4", K))
    ref = ref_dequant(raw, "MXFP4")
    assert ref.shape == (N, K) and np.isfinite(ref).all()
    assert np.allclose(ref, hand_mxfp4(raw, N, K), rtol=1e-6, atol=0)


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
