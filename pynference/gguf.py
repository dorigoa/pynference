"""GGUF reader: metadata, tensor directory and zero-copy access to tensor bytes.

The file is memory-mapped; `GGUFFile.raw(name)` returns a numpy uint8 view of a
tensor's bytes, and `rows(name, r0, r1)` the bytes of a row range (rows are
contiguous along the innermost dimension, ne[0], for every ggml type).
Split models (`*-00001-of-0000N.gguf`) are opened as one.
"""
from __future__ import annotations

import glob
import mmap
import os
import re
import struct
from dataclasses import dataclass

import numpy as np

# GGML tensor types: name -> (id, elements per block, bytes per block)
GGML_TYPES = {
    "F32": (0, 1, 4), "F16": (1, 1, 2), "Q4_0": (2, 32, 18), "Q4_1": (3, 32, 20),
    "Q5_0": (6, 32, 22), "Q5_1": (7, 32, 24), "Q8_0": (8, 32, 34), "Q8_1": (9, 32, 36),
    "Q2_K": (10, 256, 84), "Q3_K": (11, 256, 110), "Q4_K": (12, 256, 144),
    "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210), "Q8_K": (15, 256, 292),
    "IQ2_XXS": (16, 256, 66), "IQ2_XS": (17, 256, 74), "IQ3_XXS": (18, 256, 98),
    "IQ1_S": (19, 256, 50), "IQ4_NL": (20, 32, 18), "IQ3_S": (21, 256, 110),
    "IQ2_S": (22, 256, 82), "IQ4_XS": (23, 256, 136), "I8": (24, 1, 1), "I16": (25, 1, 2),
    "I32": (26, 1, 4), "I64": (27, 1, 8), "F64": (28, 1, 8), "IQ1_M": (29, 256, 56),
    "BF16": (30, 1, 2), "MXFP4": (39, 32, 17),
}
TYPE_BY_ID = {v[0]: k for k, v in GGML_TYPES.items()}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_NP_SCALAR = {0: "<u1", 1: "<i1", 2: "<u2", 3: "<i2", 4: "<u4", 5: "<i4", 6: "<f4", 7: "?",
              10: "<u8", 11: "<i8", 12: "<f8"}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]   # ggml order: shape[0] is the innermost (contiguous) dim
    type: str
    file: int                # index into GGUFFile._maps
    offset: int              # absolute byte offset in that file
    nbytes: int

    @property
    def n_rows(self) -> int:
        n = 1
        for d in self.shape[1:]:
            n *= d
        return n

    @property
    def row_bytes(self) -> int:
        _, blk, bb = GGML_TYPES[self.type]
        assert self.shape[0] % blk == 0, (self.name, self.shape, self.type)
        return self.shape[0] // blk * bb


class _Reader:
    def __init__(self, buf: memoryview):
        self.buf, self.pos = buf, 0

    def scalar(self, fmt: str):
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def string(self) -> str:
        n = self.scalar("<Q")
        s = bytes(self.buf[self.pos:self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def value(self, t: int):
        if t in _SCALAR:
            return self.scalar(_SCALAR[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.scalar("<I"), self.scalar("<Q")
            if et in _NP_SCALAR:
                dt = np.dtype(_NP_SCALAR[et])
                arr = np.frombuffer(self.buf, dtype=dt, count=n, offset=self.pos).copy()
                self.pos += n * dt.itemsize
                return arr
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")


class GGUFFile:
    def __init__(self, path: str):
        m = re.match(r"(.*-)\d{5}-of-(\d{5})\.gguf$", path)
        paths = sorted(glob.glob(m.group(1) + "*-of-" + m.group(2) + ".gguf")) if m else [path]
        self.paths = paths
        self.kv: dict = {}
        self.tensors: dict[str, TensorInfo] = {}
        self._files, self._maps = [], []
        for fi, p in enumerate(paths):
            f = open(p, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._files.append(f)
            self._maps.append(mm)
            r = _Reader(memoryview(mm))
            if r.buf[:4].tobytes() != b"GGUF":
                raise ValueError(f"{p}: not a GGUF file")
            r.pos = 4
            version, n_tensors, n_kv = r.scalar("<I"), r.scalar("<Q"), r.scalar("<Q")
            if version < 2:
                raise ValueError(f"{p}: GGUF v{version} not supported")
            kv = {}
            for _ in range(n_kv):
                k = r.string()
                kv[k] = r.value(r.scalar("<I"))
            infos = []
            for _ in range(n_tensors):
                name = r.string()
                nd = r.scalar("<I")
                shape = tuple(r.scalar("<Q") for _ in range(nd))
                infos.append((name, shape, TYPE_BY_ID[r.scalar("<I")], r.scalar("<Q")))
            align = int(kv.get("general.alignment", 32))
            start = (r.pos + align - 1) // align * align
            for name, shape, typ, off in infos:
                _, blk, bb = GGML_TYPES[typ]
                n = 1
                for d in shape:
                    n *= d
                self.tensors[name] = TensorInfo(name, shape, typ, fi, start + off, n // blk * bb)
            for k, v in kv.items():
                self.kv.setdefault(k, v)
        self.arch = self.kv.get("general.architecture", "")

    def get(self, key: str, default=None):
        """Metadata value; a bare key is looked up under the architecture prefix first."""
        if self.arch and f"{self.arch}.{key}" in self.kv:
            return self.kv[f"{self.arch}.{key}"]
        return self.kv.get(key, default)

    def raw(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        return np.frombuffer(self._maps[t.file], dtype=np.uint8, count=t.nbytes, offset=t.offset)

    def rows(self, name: str, r0: int, r1: int) -> np.ndarray:
        """Bytes of rows [r0, r1) as a (r1 - r0, row_bytes) uint8 view."""
        t = self.tensors[name]
        rb = t.row_bytes
        return np.frombuffer(self._maps[t.file], dtype=np.uint8, count=(r1 - r0) * rb,
                             offset=t.offset + r0 * rb).reshape(r1 - r0, rb)

    def close(self):
        for m in self._maps:
            m.close()
        for f in self._files:
            f.close()


def main():
    import sys
    from collections import Counter
    g = GGUFFile(sys.argv[1])
    print(f"{len(g.paths)} file(s), arch={g.arch}, {len(g.tensors)} tensors, "
          f"{sum(t.nbytes for t in g.tensors.values()) / 2**30:.2f} GiB")
    for k, v in g.kv.items():
        if isinstance(v, np.ndarray) and v.size > 64 or isinstance(v, list) and len(v) > 64:
            v = f"<array of {len(v)}>"
        elif isinstance(v, str) and len(v) > 200:
            v = f"<string of {len(v)} chars>"
        print(f"  {k} = {v}")
    print("  types:", dict(Counter(t.type for t in g.tensors.values())))
    for t in g.tensors.values():
        if not t.name.startswith("blk.") or t.name.startswith(("blk.0.", "blk.5.")):
            print(f"    {t.name:36s} {str(t.shape):24s} {t.type}")


if __name__ == "__main__":
    main()
