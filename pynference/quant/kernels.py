"""Triton kernels on GGUF-native quantized blocks (S9).

All kernels read the GGUF byte layout of the 32-element block types (Q4_0,
Q4_1, Q5_0, Q5_1, IQ4_NL and native Q8_0, which is used for expert stacks)
directly, so weights stay byte-for-byte as on disk (contract C2/C5). The
formulas mirror `dequantize_row_*` in ggml-quants.c.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .iq_grids import IQ3XXS_GRID, IQ2S_GRID, KSIGNS_IQ2XS
from .types import TYPES, row_bytes

# Device tables for the grid types (S12): a grid entry packs its values in its
# bytes little-endian (4 for iq3xxs_grid, 8 for iq2s_grid), so the byte view of
# the uint32/uint64 arrays is exactly the (entry, value) byte map the kernels
# index. ksigns_iq2xs is the sign-bit helper of ggml-quants.c.
_TABLES: dict = {}


def _iq_tables(qtype: str, device):
    """(GRID, KSGN) uint8 device tensors for grid types; (None, None) otherwise."""
    if qtype not in ("IQ3_XXS", "IQ2_S"):
        return None, None
    key = (qtype, str(device))
    t = _TABLES.get(key)
    if t is None:
        if qtype == "IQ3_XXS":
            grid = np.array(IQ3XXS_GRID, dtype=np.uint32).view(np.uint8)
        else:
            grid = np.array(IQ2S_GRID, dtype=np.uint64).view(np.uint8)
        t = (torch.from_numpy(grid.copy()).to(device),
             torch.from_numpy(np.array(KSIGNS_IQ2XS, dtype=np.uint8)).to(device))
        _TABLES[key] = t
    return t

# Fields of one GGUF block, in bytes/words (a word = 2 bytes):
# BB  block bytes, WPB = BB // 2 words per block, QS byte offset of the
# quantized data, MS word offset of m (None: none), QHB byte offset of the
# 4 qh bytes (None: none). The 5th bit of element e is bit e & 7 of the qh
# byte e >> 3 (little-endian), i.e. the same bit e of the uint32 that
# ggml-quants.c loads as two uint16 words; byte access keeps the Triton
# load pattern identical to the nibble bytes (much faster than uint16
# gathers with 32-bit shifts on Intel XPU: Q5_0 GEMV was 24 GB/s vs 193).
# K-quant super-blocks add: BS elements per block (256), DW/DMW word offsets
# of d/dmin, SCL byte offset of the scale bytes, HMB byte offset of the
# Q3_K hmask (None: none). QHB is the qh byte offset for Q5_K (16) and the
# 2-bit high part bytes of Q6_K (128). Q4_K/Q5_K order is d, dmin, 12 scale
# bytes, then qs (Q5_K: qh 32 bytes before qs). Keep every weight/scale load
# byte wide in the nibble pattern (S9 note: mixed-width gathers compile badly).
_LAYOUT = {
    "Q4_0":   dict(BB=18, WPB=9,  QS=2, MS=None, QHB=None),
    "Q4_1":   dict(BB=20, WPB=10, QS=4, MS=1,    QHB=None),
    "Q5_0":   dict(BB=22, WPB=11, QS=6, MS=None, QHB=2),
    "Q5_1":   dict(BB=24, WPB=12, QS=8, MS=1,    QHB=4),
    "IQ4_NL": dict(BB=18, WPB=9,  QS=2, MS=None, QHB=None),
    "Q8_0":   dict(BB=34, WPB=17, QS=2, MS=None, QHB=None),
    # MXFP4 (S11): one E8M0 exponent byte, then 16 nibble bytes in E2M1
    # (kvalues_fp4, doubled). BB=17 is odd, so the fp16 word view is not
    # usable: the scale is loaded as a byte and turned into 2**(e-128).
    "MXFP4":  dict(BB=17, WPB=8,  QS=1, MS=None, QHB=None),
    # Q2_0 (S11): d fp16 + 16 bytes of 2-bit values, 64 elements per block
    # (BS=64: the kernel must not walk blocks in 32-element steps)
    "Q2_0":   dict(BB=18, WPB=9,  QS=2, MS=None, QHB=None, BS=64),
    "Q2_K":   dict(BB=84,  WPB=42,  QS=16, MS=None, QHB=None,
                   BS=256, DW=40,  DMW=41, SCL=0,   HMB=None),
    "Q3_K":   dict(BB=110, WPB=55,  QS=32, MS=None, QHB=None,
                   BS=256, DW=54,  DMW=None, SCL=96, HMB=0),
    "Q4_K":   dict(BB=144, WPB=72,  QS=16, MS=None, QHB=None,
                   BS=256, DW=0,   DMW=1,  SCL=4,   HMB=None),
    "Q5_K":   dict(BB=176, WPB=88,  QS=48, MS=None, QHB=16,
                   BS=256, DW=0,   DMW=1,  SCL=4,   HMB=None),
    "Q6_K":   dict(BB=210, WPB=105, QS=0,  MS=None, QHB=128,
                   BS=256, DW=104, DMW=None, SCL=192, HMB=None),
    # IQ4_XS (S11): d fp16, scales_h uint16 (word 1), scales_l 4 bytes, qs 128
    # bytes; 8 sub-blocks of 32 share a 6-bit scale = low nibble of scales_l
    # (shifted) | 2 bits of scales_h at 2*sub-block, value via kvalues_iq4nl.
    "IQ4_XS": dict(BB=136, WPB=68, QS=8,  MS=None, QHB=None,
                   BS=256, DW=0,   DMW=None, SCL=4,  HMB=None),
    # IQ3_XXS (S12): d fp16, 96 qs bytes (64 grid indices, then 32 scale+sign
    # bytes: 8 groups of 4, aux32 = d-scale 4 bits + 7x7 sign picks); per
    # element: grid byte of IQ3XXS_GRID[idx] and ksigns bit (dequantize_row_iq3_xxs)
    "IQ3_XXS": dict(BB=98,  WPB=49,  QS=2,  MS=None, QHB=None,
                    BS=256, DW=0,   DMW=None, SCL=66, HMB=None),
    # IQ2_S (S12): d fp16, qs 64 bytes (first 32: 8-bit grid index at
    # 4*ib32+l, last 32 at QS+32: signs, bit j of the byte = kmask bit j),
    # qh 8 bytes (2 high index bits per l), scales 8 bytes (two 4-bit db
    # scales per 32-element sub-block); value IQ2S_GRID[idx] byte j (dequantize_row_iq2_s)
    "IQ2_S":   dict(BB=82,  WPB=41,  QS=2,  MS=None, QHB=66,
                    BS=256, DW=0,   DMW=None, SCL=74, HMB=None),
}


@triton.jit
def _iq4nl(nib):
    # kvalues_iq4nl from ggml-common.h
    v = tl.where(nib == 0, -127.0, 113.0)
    v = tl.where(nib == 1, -104.0, v)
    v = tl.where(nib == 2, -83.0, v)
    v = tl.where(nib == 3, -65.0, v)
    v = tl.where(nib == 4, -49.0, v)
    v = tl.where(nib == 5, -35.0, v)
    v = tl.where(nib == 6, -22.0, v)
    v = tl.where(nib == 7, -10.0, v)
    v = tl.where(nib == 8, 1.0, v)
    v = tl.where(nib == 9, 13.0, v)
    v = tl.where(nib == 10, 25.0, v)
    v = tl.where(nib == 11, 38.0, v)
    v = tl.where(nib == 12, 53.0, v)
    v = tl.where(nib == 13, 69.0, v)
    v = tl.where(nib == 14, 89.0, v)
    return tl.where(nib == 15, 113.0, v)


@triton.jit
def _fp4kv(nib):
    # kvalues_fp4 from ggml-common.h (e2m1 values, doubled), used by MXFP4:
    # magnitudes for nib & 7 are 0,1,2,3,4,6,8,12 and nib >= 8 negates
    m = nib & 7
    v = m.to(tl.float32)
    v = tl.where(m == 5, 6.0, v)
    v = tl.where(m == 6, 8.0, v)
    v = tl.where(m == 7, 12.0, v)
    return tl.where(nib >= 8, -v, v)


@triton.jit
def _decode_k(RAW8, RAWI, W16, GRID, KSGN, off8, b, e,
              KQ: tl.constexpr, BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
              QHB: tl.constexpr, DW: tl.constexpr, DMW: tl.constexpr,
              SCL: tl.constexpr, HMB: tl.constexpr):
    """Float32 values of elements e (0..255) of one K-quant super-block b.

    Mirrors dequantize_row_q2_K ... dequantize_row_q6_K in ggml-quants.c:
    g16 = e >> 4 is the 16-element group (scale index for Q2_K/Q3_K/Q6_K),
    g8 = e >> 5 the 32-element group (scale index for Q4_K/Q5_K).
    """
    off16 = off8 >> 1
    d = tl.load(W16 + off16 + b * WPB + DW).to(tl.float16, bitcast=True).to(tl.float32)
    if KQ == "IQ3_XXS":  # x = db * grid_byte * sign, db = d*(1/2 + 4-bit scale)*1/2
        # (dequantize_row_iq3_xxs): aux32 at SCL + 4*sub-block, elements of
        # sub-block ib: l = (e>>3)&3 picks the qs byte and the 7 sign bits,
        # jj = e&7 the grid byte (first 4 from qs[2l], last 4 from qs[2l+1])
        ib = e >> 5
        l = (e >> 3) & 3
        jj = e & 7
        so = off8 + b * BB + SCL + ib * 4
        aux = (tl.load(RAW8 + so).to(tl.int32)
               | (tl.load(RAW8 + so + 1).to(tl.int32) << 8)
               | (tl.load(RAW8 + so + 2).to(tl.int32) << 16)
               | (tl.load(RAW8 + so + 3).to(tl.int32) << 24))
        db = d * (0.5 + ((aux >> 28) & 0xF)) * 0.5
        qb = tl.load(RAW8 + off8 + b * BB + QS + ib * 8 + 2 * l + (jj >> 2)).to(tl.int32)
        g = tl.load(GRID + qb * 4 + (jj & 3)).to(tl.float32)
        sg = tl.load(KSGN + ((aux >> (7 * l)) & 127)).to(tl.int32)
        return db * g * tl.where(((sg >> jj) & 1) == 1, -1.0, 1.0)
    if KQ == "IQ2_S":  # x = db * grid_byte * sign (dequantize_row_iq2_s):
        # element e: ib = e>>5 sub-block, l = (e>>3)&3 group of 8, jj = e&7;
        # 10-bit grid index = qs[4ib+l] | 2 bits of qh[ib] at 2*l; db from the
        # scales nibble (low for l<2); sign bit jj of qs[32 + 4ib + l]
        # (kmask_iq2xs[jj] = 1 << jj)
        ib = e >> 5
        l = (e >> 3) & 3
        jj = e & 7
        qi = (tl.load(RAW8 + off8 + b * BB + QS + ib * 4 + l).to(tl.int32)
              | ((tl.load(RAW8 + off8 + b * BB + QHB + ib).to(tl.int32) >> (2 * l)) & 3) << 8)
        g = tl.load(GRID + qi * 8 + jj).to(tl.float32)
        sc = tl.load(RAW8 + off8 + b * BB + SCL + ib).to(tl.int32)
        nib = (sc >> tl.where(l < 2, 0, 4)) & 0xF
        db = d * (0.5 + nib) * 0.25
        sg = tl.load(RAW8 + off8 + b * BB + QS + 32 + ib * 4 + l).to(tl.int32)
        return db * g * tl.where(((sg >> jj) & 1) == 1, -1.0, 1.0)
    if KQ == "IQ4_XS":  # x = d*(ls-32)*kvalues_iq4nl[q], ls 6-bit per 32 (dequantize_row_iq4_xs)
        ib = e >> 5
        sh = tl.load(W16 + off16 + b * WPB + 1).to(tl.int32)          # scales_h (bytes 2-3)
        sl = tl.load(RAW8 + off8 + b * BB + SCL + (ib >> 1)).to(tl.int32)
        ls = ((sl >> ((ib & 1) * 4)) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
        qb = tl.load(RAW8 + off8 + b * BB + QS + (ib << 4) + (e & 15)).to(tl.int32)
        nib = (qb >> (((e >> 4) & 1) * 4)) & 0x0F
        return d * (ls - 32).to(tl.float32) * _iq4nl(nib)
    g8 = e >> 5
    g16 = e >> 4
    if KQ == "Q2_K":  # 16 sub-blocks of 16: x = d*(sc&15)*q - dmin*(sc>>4)
        sc = tl.load(RAW8 + off8 + b * BB + SCL + g16).to(tl.int32)
        qb = tl.load(RAW8 + off8 + b * BB + QS + ((e >> 7) << 5) + (e & 31)).to(tl.int32)
        q = (qb >> ((g8 & 3) * 2)) & 3
        dmin = tl.load(W16 + off16 + b * WPB + DMW).to(tl.float16, bitcast=True).to(tl.float32)
        return d * (sc & 0xF) * q - dmin * (sc >> 4)
    if KQ == "Q3_K":  # x = d*(sc-32)*(q-4), plus +4 where the hmask bit is set
        s = g16  # 6-bit scales: low nibble from byte s&7 (shift 4 if s>=8),
        # 2 high bits from byte 8+(s&3) at shift 2*(s>>2)
        lo = tl.load(RAW8 + off8 + b * BB + SCL + (s & 7)).to(tl.int32)
        hi = tl.load(RAW8 + off8 + b * BB + SCL + 8 + (s & 3)).to(tl.int32)
        sc = ((lo >> tl.where(s >= 8, 4, 0)) & 0xF) | (((hi >> (2 * (s >> 2))) & 3) << 4)
        qb = tl.load(RAW8 + off8 + b * BB + QS + ((e >> 7) << 5) + (e & 31)).to(tl.int32)
        q = (qb >> ((g8 & 3) * 2)) & 3
        hm = tl.load(RAW8 + off8 + b * BB + HMB + (e & 31)).to(tl.int32)
        hb = (hm >> g8) & 1  # bit g8 of hmask byte e&31 (gguf-py: offset when 0)
        return d * (sc - 32) * (q + 4 * hb - 4)
    if KQ == "Q6_K":  # x = d*sc_i8*((ql-nibble | qh2bits<<4) - 32)
        # within a 128-element half: 32 bytes of ql, then the other 32; low
        # nibbles for p<2, high nibbles for p>=2, where p = (e>>5)&3
        n3 = (e >> 5) & 3
        lo = tl.load(RAW8 + off8 + b * BB + QS + ((e >> 7) << 6)
                     + ((n3 & 1) << 5) + (e & 31)).to(tl.int32)
        hi = tl.load(RAW8 + off8 + b * BB + QHB + ((e >> 7) << 5) + (e & 31)).to(tl.int32)
        q6 = ((lo >> ((n3 >> 1) * 4)) & 0xF) | (((hi >> (n3 * 2)) & 3) << 4)
        sc = tl.load(RAWI + off8 + b * BB + SCL + g16).to(tl.float32)
        return d * sc * (q6 - 32.0)
    # Q4_K and Q5_K: 8 sub-blocks of 32, scales and mins at 6 bits (12 bytes):
    # s<4: sc=S[s]&63, mn=S[s+4]&63; s>=4: sc=(S[s+4]&0xF)|(S[s-4]>>6<<4),
    # mn=(S[s+4]>>4)|(S[s]>>6<<4). All byte loads stay inside the 12 bytes
    # (tl.where only clamps the index of the load whose result is unused).
    s = g8
    s0 = tl.load(RAW8 + off8 + b * BB + SCL + s).to(tl.int32)
    s4 = tl.load(RAW8 + off8 + b * BB + SCL + s + 4).to(tl.int32)
    sm = tl.load(RAW8 + off8 + b * BB + SCL + tl.where(s < 4, s, s - 4)).to(tl.int32)
    sc = tl.where(s < 4, s0 & 0x3F, (s4 & 0xF) | ((sm >> 6) << 4))
    mn = tl.where(s < 4, s4 & 0x3F, (s4 >> 4) | ((s0 >> 6) << 4))
    qb = tl.load(RAW8 + off8 + b * BB + QS + ((s >> 1) << 5) + (e & 31)).to(tl.int32)
    q = (qb >> ((s & 1) * 4)) & 0xF
    if KQ == "Q5_K":  # 5th bit: bit s of qh byte e&31
        qh = tl.load(RAW8 + off8 + b * BB + QHB + (e & 31)).to(tl.int32)
        q = q | (((qh >> s) & 1) << 4)
    dmin = tl.load(W16 + off16 + b * WPB + DMW).to(tl.float16, bitcast=True).to(tl.float32)
    return d * sc * q.to(tl.float32) - dmin * mn.to(tl.float32)


@triton.jit
def _decode(RAW8, RAWI, W16, GRID, KSGN, off8, b, e, RB16,
            Q8_0: tl.constexpr, HAS_M: tl.constexpr, HAS_QH: tl.constexpr, IQ4: tl.constexpr,
            Q2_0: tl.constexpr, MXFP4: tl.constexpr,
            BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
            MS: tl.constexpr, QHB: tl.constexpr,
            KQ: tl.constexpr, DW: tl.constexpr, DMW: tl.constexpr,
            SCL: tl.constexpr, HMB: tl.constexpr):
    """Float32 values of elements e (0..BS-1) of block b, rows at byte offset off8."""
    if KQ != "":
        return _decode_k(RAW8, RAWI, W16, GRID, KSGN, off8, b, e, KQ, BB, WPB, QS, QHB, DW, DMW, SCL, HMB)
    if MXFP4:
        # d = ggml_e8m0_to_fp32_half(e): bit-for-bit the gguf-py trick, i.e.
        # 2**(e-128) for every e in 0..255 (e=0,1 are denormal mantissas)
        eb = tl.load(RAW8 + off8 + b * BB).to(tl.uint32)
        bits = tl.where(eb < 2, 0x00200000 << eb, (eb - 1) << 23)
        d = bits.to(tl.float32, bitcast=True)
        byte = tl.load(RAW8 + off8 + b * BB + QS + (e % 16)).to(tl.int32)
        nib = (byte >> tl.where(e >= 16, 4, 0)) & 0x0F
        return _fp4kv(nib) * d
    off16 = off8 >> 1
    if MXFP4:
        # Tracing only: MXFP4 returned above, but the Triton frontend keeps
        # tracing past that constexpr return and would type-check this load.
        # With odd row_bytes _views passes W16 as uint8: 8->16-bit bitcast fail.
        d = 1.0
    else:
        d = tl.load(W16 + off16 + b * WPB).to(tl.float16, bitcast=True).to(tl.float32)
    if Q8_0:
        q = tl.load(RAWI + off8 + b * BB + QS + e).to(tl.float32)
        return q * d
    if Q2_0:  # 2-bit values, 4 per byte: 00=-1, 01=0, 10=+1, 11=+2 times d
        qb = tl.load(RAW8 + off8 + b * BB + QS + (e >> 2)).to(tl.int32)
        q = (qb >> ((e & 3) * 2)) & 3
        return (q - 1).to(tl.float32) * d
    byte = tl.load(RAW8 + off8 + b * BB + QS + (e % 16)).to(tl.int32)
    nib = (byte >> tl.where(e >= 16, 4, 0)) & 0x0F
    if HAS_QH:
        qb = tl.load(RAW8 + off8 + b * BB + QHB + (e >> 3)).to(tl.int32)
        nib = nib | (((qb >> (e & 7)) & 1) << 4)
    v = nib.to(tl.float32)
    if IQ4:
        return _iq4nl(nib) * d
    if HAS_M:
        m = tl.load(W16 + off16 + b * WPB + MS).to(tl.float16, bitcast=True).to(tl.float32)
        return v * d + m
    if HAS_QH:  # Q5_0
        return (v - 16.0) * d
    return (v - 8.0) * d  # Q4_0


def _jit_args(qtype: str) -> dict:
    L = _LAYOUT[qtype]
    return dict(Q8_0=qtype == "Q8_0", HAS_M=L["MS"] is not None, HAS_QH=L["QHB"] is not None,
                IQ4=qtype == "IQ4_NL", Q2_0=qtype == "Q2_0", MXFP4=qtype == "MXFP4",
                BB=L["BB"], WPB=L["WPB"], QS=L["QS"],
                MS=L["MS"] if L["MS"] is not None else -1, QHB=L["QHB"] if L["QHB"] is not None else -1,
                BS=L.get("BS", 32), KQ=qtype if L.get("BS", 32) == 256 else "",
                DW=L.get("DW", 0), DMW=L["DMW"] if L.get("DMW") is not None else -1,
                SCL=L.get("SCL", 0), HMB=L["HMB"] if L.get("HMB") is not None else -1)


@triton.jit
def _dequant_kernel(RAW8, RAWI, W16, GRID, KSGN, OUT, IDX, M, K, RB8, RB16,
                    Q8_0: tl.constexpr, HAS_M: tl.constexpr, HAS_QH: tl.constexpr, IQ4: tl.constexpr,
                    Q2_0: tl.constexpr, MXFP4: tl.constexpr,
                    BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
                    MS: tl.constexpr, QHB: tl.constexpr,
                    KQ: tl.constexpr, DW: tl.constexpr, DMW: tl.constexpr,
                    SCL: tl.constexpr, HMB: tl.constexpr, BS: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, HAS_IDX: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    nm = rn < M
    if HAS_IDX:
        src = tl.load(IDX + rn, mask=nm, other=0).to(tl.int64)
    else:
        src = rn.to(tl.int64)
    src = tl.where(nm, src, 0)
    off8 = (src * RB8)[:, None]
    b = (rk // BS)[None, :]
    e = (rk % BS)[None, :]
    v = _decode(RAW8, RAWI, W16, GRID, KSGN, off8, b, e, RB16, Q8_0, HAS_M, HAS_QH, IQ4, Q2_0, MXFP4,
                BB, WPB, QS, MS, QHB, KQ, DW, DMW, SCL, HMB)
    m2 = nm[:, None] & (rk < K)[None, :]
    tl.store(OUT + rn.to(tl.int64)[:, None] * K + rk[None, :], v, mask=m2)


# 16-warp and BLOCK_K=1024 configs added for S10: the 2-bit Q2_K GEMV is
# compute-bound (3.05 elements/byte, its 14.5 MB working set fits in L2),
# so more warps and a wider k-tile help where the nibble types do not need them
# S18 step 3: the indexed MXFP4 experts (N=K=2880) want a wider grid when few
# experts run per token (small E), so small-BLOCK_N variants were added; the
# microbench in the S18-3 job reports which one the autotune keeps.
_GEMV_CONFIGS = [triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw)
                 for bn, bk, nw in [(4, 512, 4), (8, 256, 4), (8, 512, 4), (8, 512, 8),
                                    (16, 256, 4), (16, 512, 8), (32, 256, 8),
                                    (4, 1024, 8), (8, 1024, 8), (8, 1024, 16),
                                    (16, 512, 16), (16, 1024, 16), (32, 512, 16),
                                    (2, 512, 4), (4, 512, 8), (8, 256, 8),
                                    (16, 256, 8), (32, 1024, 16), (2, 4096, 4)]]


# the type constexprs must be in the autotune key: the same (N, K) shape is
# benchmarked with different configs for the nibble types vs Q5_0 vs Q8_0
_GEMV_KEY = ["N", "K", "Q8_0", "HAS_M", "HAS_QH", "IQ4", "Q2_0", "MXFP4", "BB", "KQ"]


@triton.autotune(configs=_GEMV_CONFIGS, key=_GEMV_KEY)
@triton.jit
def _gemv_kernel(RAW8, RAWI, W16, GRID, KSGN, X, Y, N, K, RB8, RB16,
                 Q8_0: tl.constexpr, HAS_M: tl.constexpr, HAS_QH: tl.constexpr, IQ4: tl.constexpr,
                 Q2_0: tl.constexpr, MXFP4: tl.constexpr,
                 BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
                 MS: tl.constexpr, QHB: tl.constexpr,
                 KQ: tl.constexpr, DW: tl.constexpr, DMW: tl.constexpr,
                 SCL: tl.constexpr, HMB: tl.constexpr, BS: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = rn < N
    off8 = tl.where(nm, rn, 0).to(tl.int64)[:, None] * RB8
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < K
        x = tl.load(X + rk, mask=mk, other=0.0)
        b = (rk // BS)[None, :]
        e = (rk % BS)[None, :]
        w = _decode(RAW8, RAWI, W16, GRID, KSGN, off8, b, e, RB16, Q8_0, HAS_M, HAS_QH, IQ4, Q2_0, MXFP4,
                    BB, WPB, QS, MS, QHB, KQ, DW, DMW, SCL, HMB)
        acc += tl.sum(tl.where(nm[:, None] & mk[None, :], w, 0.0) * x[None, :], axis=1)
    tl.store(Y + rn, acc, mask=nm)


@triton.autotune(configs=_GEMV_CONFIGS, key=_GEMV_KEY)
@triton.jit
def _gemv_indexed_kernel(RAW8, RAWI, W16, GRID, KSGN, X, SLOTS, OUT, N, K, RB8, RB16, SLOT_BYTES,
                         Q8_0: tl.constexpr, HAS_M: tl.constexpr, HAS_QH: tl.constexpr, IQ4: tl.constexpr,
                         Q2_0: tl.constexpr, MXFP4: tl.constexpr,
                         BB: tl.constexpr, WPB: tl.constexpr, QS: tl.constexpr,
                         MS: tl.constexpr, QHB: tl.constexpr,
                         KQ: tl.constexpr, DW: tl.constexpr, DMW: tl.constexpr,
                         SCL: tl.constexpr, HMB: tl.constexpr, BS: tl.constexpr,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    re = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    # rows past N (when N % BLOCK_N != 0) load from row 0 and are not stored,
    # so no slot ever writes into the next expert's OUT rows
    mr = rn < N
    rn_ok = tl.where(mr, rn, 0)
    slot = tl.load(SLOTS + re).to(tl.int64)
    off8 = slot * SLOT_BYTES + rn_ok.to(tl.int64)[:, None] * RB8
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < K
        x = tl.load(X + re.to(tl.int64) * K + rk, mask=mk, other=0.0)
        b = (rk // BS)[None, :]
        e = (rk % BS)[None, :]
        w = _decode(RAW8, RAWI, W16, GRID, KSGN, off8, b, e, RB16, Q8_0, HAS_M, HAS_QH, IQ4, Q2_0, MXFP4,
                    BB, WPB, QS, MS, QHB, KQ, DW, DMW, SCL, HMB)
        acc += tl.sum(tl.where(mk[None, :], w, 0.0) * x[None, :], axis=1)
    tl.store(OUT + re.to(tl.int64) * N + rn, acc, mask=mr)


def _views(raw: torch.Tensor):
    if raw.shape[-1] % 2:  # MXFP4 row_bytes can be odd: no uint16 view, W16 unused
        return raw, raw.view(torch.int8), raw
    return raw, raw.view(torch.int8), raw.view(torch.uint16)


def _iq_ptrs(qtype: str, raw: torch.Tensor):
    """Device grid/ksigns tensors for grid types, dummy (raw) pointers otherwise."""
    g, k = _iq_tables(qtype, raw.device)
    return raw if g is None else g, raw if k is None else k


def dequant_native(qtype: str, raw: torch.Tensor, out: torch.Tensor) -> None:
    """raw [N, row_bytes] uint8 GGUF -> out [N, K] (bf16 or float32)."""
    n, rb = raw.shape
    k = out.shape[-1]
    w8, wi, w16 = _views(raw)
    (g, sg) = _iq_ptrs(qtype, raw)
    grid = (triton.cdiv(n, 32), triton.cdiv(k, 256))
    _dequant_kernel[grid](w8, wi, w16, g, sg, out, out, n, k, rb, rb // 2,
                          BLOCK_N=32, BLOCK_K=256, HAS_IDX=False, **_jit_args(qtype), num_warps=8)


def rows_dequant_native(qtype: str, raw: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> None:
    """Dequant rows idx (int32/int64, [M]) to float32 out [M, K]."""
    n, rb = raw.shape
    m, k = out.shape
    w8, wi, w16 = _views(raw)
    (g, sg) = _iq_ptrs(qtype, raw)
    grid = (triton.cdiv(m, 32), triton.cdiv(k, 256))
    _dequant_kernel[grid](w8, wi, w16, g, sg, out, idx, m, k, rb, rb // 2,
                          BLOCK_N=32, BLOCK_K=256, HAS_IDX=True, **_jit_args(qtype), num_warps=8)


def gemv_native(qtype: str, raw: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
    """y [1, N] float32 = W @ x ; raw [N, row_bytes], x [K] float32 contiguous."""
    n, rb = raw.shape
    k = x.numel()
    w8, wi, w16 = _views(raw)
    (g, sg) = _iq_ptrs(qtype, raw)
    _gemv_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)](
        w8, wi, w16, g, sg, x, y, n, k, rb, rb // 2, **_jit_args(qtype))


def gemv_indexed_native(qtype: str, w_stack: torch.Tensor, x: torch.Tensor,
                        slots: torch.Tensor, out: torch.Tensor) -> None:
    """out [E, N] = W[slots[e]] @ x[e]; w_stack [S, N, row_bytes] GGUF, x [E, K]."""
    s, n, rb = w_stack.shape
    e, k = x.shape
    w8, wi, w16 = _views(w_stack.reshape(s * n, rb))
    (g, sg) = _iq_ptrs(qtype, w_stack.reshape(s * n, rb))
    grid = lambda meta: (e, triton.cdiv(n, meta["BLOCK_N"]))
    _gemv_indexed_kernel[grid](w8, wi, w16, g, sg, x, slots, out, n, k, rb, rb // 2, n * rb,
                               **_jit_args(qtype))


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _gemv_half_kernel(W, X, Y, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = rn < N
    off = tl.where(nm, rn, 0).to(tl.int64)[:, None] * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < K
        x = tl.load(X + rk, mask=mk, other=0.0)
        w = tl.load(W + off + rk[None, :], mask=nm[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(Y + rn, acc, mask=nm)


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _gemv_indexed_half_kernel(W, X, SLOTS, OUT, N, K, SLOT_NK,
                              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    re = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    slot = tl.load(SLOTS + re).to(tl.int64)
    off = slot * SLOT_NK + rn.to(tl.int64)[:, None] * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < K
        x = tl.load(X + re.to(tl.int64) * K + rk, mask=mk, other=0.0)
        w = tl.load(W + off + rk[None, :], mask=mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + re.to(tl.int64) * N + rn, acc)


def gemv_half(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
    """F16/BF16 weight [N, K] 16-bit, x [K] float32, y [1, N] float32."""
    n, k = w.shape
    _gemv_half_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)](w, x, y, n, k)


def gemv_indexed_half(w_stack: torch.Tensor, x: torch.Tensor, slots: torch.Tensor,
                      out: torch.Tensor) -> None:
    """F16/BF16 expert stack [S, N, K] 16-bit, x [E, K], out [E, N]."""
    s, n, k = w_stack.shape
    e, _ = x.shape
    _gemv_indexed_half_kernel[lambda meta: (e, triton.cdiv(n, meta["BLOCK_N"]))](
        w_stack, x, slots, out, n, k, n * k)
