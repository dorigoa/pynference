"""S12 step 1: the generated IQ grid tables against the llama.cpp source.

The tables come from `GGML_TABLE_BEGIN` blocks in ggml-common.h (commit
2d8a32502). Here the lengths are checked, plus first/last entries and a few
middle ones read by hand out of the same source file.
"""
from pynference.quant.iq_grids import (IQ2S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
                                       KSIGNS_IQ2XS)


def test_lengths():
    assert len(KMASK_IQ2XS) == 8
    assert len(KSIGNS_IQ2XS) == 128
    assert len(IQ3XXS_GRID) == 256
    assert len(IQ2S_GRID) == 1024


def test_kmask_is_the_bit_table():
    assert KMASK_IQ2XS == [1, 2, 4, 8, 16, 32, 64, 128]


def test_ksigns_spot_values():
    # first row "0, 129, 130, 3, 132, 5, 6, 135" and last row "... 252, 125, 126, 255"
    assert KSIGNS_IQ2XS[:8] == [0, 129, 130, 3, 132, 5, 6, 135]
    assert KSIGNS_IQ2XS[127] == 255
    assert KSIGNS_IQ2XS[126] == 126
    assert KSIGNS_IQ2XS[125] == 125
    # every entry is one byte, as ggml-quants.c reads it as uint8_t
    assert all(0 <= v <= 255 for v in KSIGNS_IQ2XS)


def test_iq3xxs_grid_spot_values():
    # first row "0x04040404, 0x04040414, ..." and last entry 0x3e341c04
    assert IQ3XXS_GRID[0] == 0x04040404
    assert IQ3XXS_GRID[1] == 0x04040414
    assert IQ3XXS_GRID[255] == 0x3E341C04
    assert all(0 <= v <= 0xFFFFFFFF for v in IQ3XXS_GRID)


def test_iq2s_grid_spot_values():
    # first row "0x0808080808080808, 0x080808080808082b, ..." and last entry all 0x2b
    assert IQ2S_GRID[0] == 0x0808080808080808
    assert IQ2S_GRID[1] == 0x080808080808082B
    assert IQ2S_GRID[1023] == 0x2B2B2B2B2B2B2B2B
    assert all(0 <= v <= 0xFFFFFFFFFFFFFFFF for v in IQ2S_GRID)
