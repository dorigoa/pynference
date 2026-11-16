"""Quantized weights and the kernels that use them (plan contract C2).

The GGUF layout of each type is the native kernel layout, so expert weights
are copied from RAM byte for byte (C5). Q8_0 dense weights are an exception:
they are repacked for faster GEMV (see `q8_0`).
"""
from __future__ import annotations

from .q8_0 import QLinear, q8_0_to_device
from .types import QType, TYPES, check_split, row_bytes
from .weight import (F32QWeight, HalfQWeight, NativeQWeight, QWeight,
                     gemv_indexed, make_qweight)

__all__ = ["QType", "TYPES", "check_split", "row_bytes", "QWeight",
           "NativeQWeight", "HalfQWeight", "F32QWeight", "make_qweight",
           "gemv_indexed", "QLinear", "q8_0_to_device"]
