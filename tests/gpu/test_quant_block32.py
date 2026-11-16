"""S9 step 5: GPU tests for the 32-element block kernels in native GGUF layout.

Dequant against gguf-py (quant/ref.py) on random blocks and on the first 256
rows of real tensors (Q4_0 from Qwen3.8-27B-Q4_0.gguf, Q5_0 and IQ4_NL from
Flash-Next IQ4_XS, read-only). GEMV against fp32 with relative error <= 1e-4,
indexed GEMV against a python loop, bandwidth on 8192x5376, F16/BF16 GEMV.
Q8_0 here is the native (expert-stack) layout, not the repacked QLinear.

Run only inside a GPU job (2.4); skipped when the GPUs are hidden.
"""
import os

import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

import numpy as np  # noqa: E402

from pynference.gguf import GGUFFile  # noqa: E402
from pynference.quant import (NativeQWeight, gemv_indexed, kernels,  # noqa: E402
                              make_qweight, row_bytes)
from pynference.quant.ref import ref_dequant  # noqa: E402
from pynference.quant.testing import make_test_blocks  # noqa: E402

DEV = torch.device("xpu", 0)
NATIVE = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL", "Q8_0"]
FLASH = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
QWEN4 = "/nvme/gguf/Qwen3.8-27B-Q4_0.gguf"
REAL = [("Q4_0", QWEN4), ("Q5_0", FLASH), ("IQ4_NL", FLASH)]
REL_TOL = 1e-4
MIN_GBPS = 100.0


def _native(qtype, raw, n, k):
    dev = torch.from_numpy(np.ascontiguousarray(raw).copy()).to(DEV)
    return NativeQWeight(qtype, dev.view(n, row_bytes(qtype, k)), n, k)


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("qtype", NATIVE)
def test_dequant_random(qtype):
    n, k = 96, 384
    raw = make_test_blocks(qtype, n, k, seed=7)
    ref = torch.from_numpy(ref_dequant(raw, qtype)).to(DEV)
    w = _native(qtype, raw, n, k)
    out = torch.empty(n, k, dtype=torch.float32, device=DEV)
    kernels.dequant_native(qtype, w.raw, out)
    err = _rel(out, ref)
    idx = torch.randperm(n, device=DEV)[:16]
    err_rows = _rel(w.rows_dequant(idx.to(torch.int32)), ref[idx])
    print(f"RES {qtype} deq={err:.2e} deq_rows={err_rows:.2e}")
    assert err < REL_TOL and err_rows < REL_TOL


@pytest.mark.parametrize("qtype", NATIVE)
def test_gemv_random(qtype):
    n, k = 2048, 2048
    raw = make_test_blocks(qtype, n, k, seed=8)
    ref = torch.from_numpy(ref_dequant(raw, qtype)).to(DEV)
    w = _native(qtype, raw, n, k)
    x = torch.randn(1, k, device=DEV)
    err = _rel(w(x), x @ ref.T)
    print(f"RES {qtype} gemv={err:.2e}")
    assert err < REL_TOL


@pytest.mark.parametrize("qtype", NATIVE)
def test_gemv_indexed_random(qtype):
    s, n, k, e = 8, 512, 512, 4
    raw = np.stack([make_test_blocks(qtype, n, k, seed=100 + i) for i in range(s)])
    ref = torch.from_numpy(ref_dequant(raw, qtype)).to(DEV)  # [S, N, K]
    stack = torch.from_numpy(np.ascontiguousarray(raw)).to(DEV)
    torch.manual_seed(2)
    slots = torch.randint(0, s, (e,), dtype=torch.int32, device=DEV)
    x = torch.randn(e, k, device=DEV)
    out = torch.zeros(e, n, dtype=torch.float32, device=DEV)
    gemv_indexed(qtype, stack, x, slots, out)
    ref_loop = torch.stack([x[i] @ ref[slots[i].long()].T for i in range(e)])
    err = _rel(out, ref_loop)
    print(f"RES {qtype} gemv_idx={err:.2e}")
    assert err < REL_TOL


@pytest.mark.parametrize("qtype,path", REAL, ids=[q for q, _ in REAL])
def test_real_tensor(qtype, path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    f = GGUFFile(path)
    # Q5_0 in Flash-Next appears only on 3D expert tensors (n_experts, n, k):
    # its 2D "rows" are the flattened view, which is what f.rows returns too
    info = next((t for t in f.tensors.values()
                 if t.type == qtype and len(t.shape) in (2, 3)), None)
    if info is None:
        pytest.skip(f"no real {qtype} tensor in {path}")
    n, k = min(256, info.n_rows), info.shape[0]
    raw = f.rows(info.name, 0, n)
    ref = torch.from_numpy(ref_dequant(raw, qtype)).to(DEV)
    w = _native(qtype, raw, n, k)
    out = torch.empty(n, k, dtype=torch.float32, device=DEV)
    kernels.dequant_native(qtype, w.raw, out)
    deq = _rel(out, ref)
    x = torch.randn(1, k, device=DEV)
    gemv = _rel(w(x), x @ ref.T)
    print(f"RES {qtype} real {info.name} {n}x{k} deq={deq:.2e} gemv={gemv:.2e}")
    assert deq < REL_TOL and gemv < REL_TOL


@pytest.mark.parametrize("qtype", ["F16", "BF16"])
def test_gemv_half(qtype):
    n, k = 512, 512
    raw = make_test_blocks(qtype, n, k, seed=11)
    w = make_qweight(raw, qtype, n, k, DEV)
    ref = torch.from_numpy(ref_dequant(raw, qtype)).to(DEV)
    x = torch.randn(1, k, device=DEV)
    err = _rel(w(x), x @ ref.T)
    print(f"RES {qtype} gemv={err:.2e}")
    assert err < REL_TOL


@pytest.mark.parametrize("qtype", NATIVE)
def test_bandwidth(qtype):
    import time
    n, k = 8192, 5376
    raw = make_test_blocks(qtype, n, k, seed=5)
    w = _native(qtype, raw, n, k)
    x = torch.randn(k, device=DEV)
    y = torch.empty(1, n, dtype=torch.float32, device=DEV)
    for _ in range(3):
        kernels.gemv_native(qtype, w.raw, x, y)
    torch.xpu.synchronize()
    it = 20
    t = time.perf_counter()
    for _ in range(it):
        kernels.gemv_native(qtype, w.raw, x, y)
    torch.xpu.synchronize()
    gbps = w.nbytes / ((time.perf_counter() - t) / it) / 1e9
    print(f"RES {qtype} gbps={gbps:.1f} ({w.nbytes / 1e6:.1f} MB)")
    assert gbps > MIN_GBPS
