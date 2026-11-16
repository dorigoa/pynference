"""Quantized weights and the kernels that use them (plan contract C2).

The GGUF layout of each type is the native kernel layout, so expert weights
are copied from RAM byte for byte (C5). Q8_0 dense weights are an exception:
they are repacked for faster GEMV (see `q8_0`).
"""
from __future__ import annotations

from .types import QType, TYPES, check_split, row_bytes
from .weight import (F32QWeight, HalfQWeight, NativeQWeight, QWeight,
                     gemv_indexed, make_qweight)

__all__ = ["QType", "TYPES", "check_split", "row_bytes", "QWeight",
           "NativeQWeight", "HalfQWeight", "F32QWeight", "make_qweight",
           "gemv_indexed", "QLinear", "q8_0_to_device"]


def __getattr__(name):
    """Lazy access to the repacked Q8_0 path: q8_0.py imports Triton, and
    importing it at package level would load Triton before the interpret-mode
    test (tests/cpu/test_quant_decode_interp.py) can set TRITON_INTERPRET."""
    if name in ("QLinear", "q8_0_to_device"):
        from . import q8_0
        return getattr(q8_0, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
