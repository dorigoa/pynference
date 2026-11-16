"""Random valid GGUF blocks for quantization tests (S9).

`make_test_blocks` fills a weight with random bytes, then replaces every
fp16/bf16 scale field (and the plain values of the 16-bit types) with a
finite fp16-range value in [1e-3, 1e-1]: GEMV against fp32 stays
well-conditioned and no scale is zero or denormal.
"""
from __future__ import annotations

import numpy as np

from .types import TYPES, row_bytes

# fp16 scale fields, as word index (2 bytes each) inside one GGUF block
_SCALE_WORDS = {
    "Q4_0": (0,),
    "Q4_1": (0, 1),
    "Q5_0": (0,),
    "Q5_1": (0, 1),
    "Q8_0": (0,),
    "IQ4_NL": (0,),
    # K-quant super-blocks: d (and dmin) fp16 words inside the 256-element block
    "Q2_K": (40, 41),
    "Q3_K": (54,),
    "Q4_K": (0, 1),
    "Q5_K": (0, 1),
    "Q6_K": (104,),
}


def _uniform_f16(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.uniform(1e-3, 1e-1, shape).astype(np.float16)


def make_test_blocks(qtype: str, n: int, k: int, seed: int = 0) -> np.ndarray:
    """Random GGUF bytes [n, row_bytes] for a weight of n rows and k columns."""
    rng = np.random.default_rng(seed)
    rb = row_bytes(qtype, k)
    t = TYPES[qtype]

    if qtype == "F32":
        return rng.uniform(1e-3, 1e-1, (n, k)).astype(np.float32).view(np.uint8).reshape(n, rb).copy()
    if qtype == "F16":
        return _uniform_f16(rng, (n, k)).view(np.uint8).reshape(n, rb).copy()
    if qtype == "BF16":
        f32 = rng.uniform(1e-3, 1e-1, (n, k)).astype(np.float32).view(np.uint32)
        return (f32 >> 16).astype(np.uint16).view(np.uint8).reshape(n, rb).copy()

    if qtype == "MXFP4":
        # row_bytes is odd when the row holds an odd number of blocks, so the
        # uint16 view below is not usable; only the E8M0 byte matters and it is
        # kept in [118, 138] (d = 2**(e-128) in [2^-10, 2^10], well-conditioned
        # for GEMV against fp32). The 16 nibble bytes of each block stay random.
        raw = rng.integers(0, 256, (n, rb), dtype=np.uint8)
        nblocks = k // t.block
        e_pos = np.arange(nblocks) * t.block_bytes     # the E8M0 byte of each block
        raw[:, e_pos] = rng.integers(118, 139, (n, nblocks), dtype=np.uint8)
        return raw

    raw = rng.integers(0, 256, (n, rb), dtype=np.uint8)
    words = raw.view(np.uint16)          # [n, rb // 2], rb is even for all block types
    nblocks = k // t.block
    wpb = rb // 2 // nblocks             # words per block
    for w in _SCALE_WORDS[qtype]:
        words[:, w::wpb] = _uniform_f16(rng, (n, nblocks)).view(np.uint16)
    return raw
