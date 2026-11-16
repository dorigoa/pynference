"""Registry of quantization types (plan 3.7) and block-split helpers (C2).

Only the types implemented so far are listed; later quantization sections add
the rest. GGUF ids come from `gguf.constants.GGMLQuantizationType`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QType:
    name: str
    ggml_id: int
    block: int         # elements per GGUF block
    block_bytes: int   # bytes per GGUF block


TYPES: dict[str, QType] = {t.name: t for t in [
    QType("F32", 0, 1, 4),
    QType("F16", 1, 1, 2),
    QType("Q4_0", 2, 32, 18),
    QType("Q4_1", 3, 32, 20),
    QType("Q5_0", 6, 32, 22),
    QType("Q5_1", 7, 32, 24),
    QType("Q8_0", 8, 32, 34),
    QType("BF16", 30, 1, 2),
    QType("IQ4_NL", 20, 32, 18),
    QType("Q2_K", 10, 256, 84),
    QType("Q3_K", 11, 256, 110),
    QType("Q4_K", 12, 256, 144),
    QType("Q5_K", 13, 256, 176),
    QType("Q6_K", 14, 256, 210),
    QType("MXFP4", 39, 32, 17),
]}


def check_split(qtype: str, k: int) -> None:
    """Raise ValueError if k is not a whole number of GGUF blocks."""
    t = TYPES[qtype]
    if k % t.block:
        raise ValueError(f"{qtype}: k={k} is not a multiple of the block size {t.block}")


def row_bytes(qtype: str, k: int) -> int:
    """Bytes of one GGUF row holding k elements of qtype."""
    t = TYPES[qtype]
    check_split(qtype, k)
    return k // t.block * t.block_bytes
