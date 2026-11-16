"""Numerical reference for quantized rows: numpy dequantization via gguf-py.

gguf-py (the GGUF python reader used by the converters) is the reference the
GPU kernels are checked against; see reports/S9.md for which types it
supports.
"""
from __future__ import annotations

import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

from .types import TYPES


def ref_dequant(raw: np.ndarray, qtype: str) -> np.ndarray:
    """GGUF bytes [..., row_bytes] uint8 -> float32 [..., k]."""
    t = TYPES[qtype]
    return dequantize(np.ascontiguousarray(raw), GGMLQuantizationType(t.ggml_id))
