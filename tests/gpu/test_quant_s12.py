"""S12 step 4: GPU tests for the IQ3_XXS and IQ2_S kernels (native layout).

Same contract as test_quant_s11.py: dequant against gguf-py (quant/ref.py)
on random blocks and on real tensor rows, GEMV against fp32 with relative
error <= 1e-4, indexed GEMV against a python loop, bandwidth on 8192x5376.
Real tensors come from a scan of the Flash-Next shards: both types exist
only as 3-D MoE expert stacks (ffn_up/ffn_gate_exps, shape (2560, 640, 512),
k = 2560 a whole number of 256-element blocks), read as flat rows.

Run only inside a GPU job (2.4); skipped when the GPUs are hidden.
"""
import os

import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

import numpy as np  # noqa: E402

from pynference.gguf import GGUFFile  # noqa: E402
from pynference.quant import NativeQWeight, gemv_indexed, kernels, row_bytes  # noqa: E402
from pynference.quant.ref import ref_dequant  # noqa: E402
from pynference.quant.testing import make_test_blocks  # noqa: E402

DEV = torch.device("xpu", 0)
KTYPES = ["IQ3_XXS", "IQ2_S"]
FLASH = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
# (qtype, file, tensor name) from the S12 scan; k is info.shape[0] = 2560,
# a whole number of 256-element blocks (the 3-D expert stack read as flat rows)
REAL = [
    ("IQ3_XXS", FLASH, "blk.0.ffn_up_exps.weight"),     # (2560, 640, 512)
    ("IQ2_S", FLASH, "blk.8.ffn_up_exps.weight"),       # (2560, 640, 512)
]
REL_TOL = 1e-4
# per-type GEMV floor, bytes read / s on 8192x5376. The plan sets no bandwidth
# contract (the card only asks to measure), so these just catch real
# regressions: low 40 GB/s floors on the first pass, fixed after the job
# measures (S10/S11 pattern).
MIN_GBPS = {"IQ3_XXS": 40.0, "IQ2_S": 40.0}


def _native(qtype, raw, n, k):
    dev = torch.from_numpy(np.ascontiguousarray(raw).copy()).to(DEV)
    return NativeQWeight(qtype, dev.view(n, row_bytes(qtype, k)), n, k)


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("qtype", KTYPES)
def test_dequant_random(qtype):
    n, k = 96, 512  # two 256-element super-blocks per row
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


@pytest.mark.parametrize("qtype", KTYPES)
def test_gemv_random(qtype):
    n, k = 2048, 2048
    raw = make_test_blocks(qtype, n, k, seed=8)
    ref = torch.from_numpy(ref_dequant(raw, qtype)).to(DEV)
    w = _native(qtype, raw, n, k)
    x = torch.randn(1, k, device=DEV)
    err = _rel(w(x), x @ ref.T)
    print(f"RES {qtype} gemv={err:.2e}")
    assert err < REL_TOL


@pytest.mark.parametrize("qtype", KTYPES)
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


@pytest.mark.parametrize("qtype,path,name", REAL, ids=[q for q, _, _ in REAL])
def test_real_tensor(qtype, path, name):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    f = GGUFFile(path)
    info = f.tensors.get(name)
    if info is None or info.type != qtype:
        pytest.skip(f"{name} ({qtype}) not in {path}")
    n, k = min(256, info.n_rows), info.shape[0]
    raw = f.rows(name, 0, n)
    ref = torch.from_numpy(ref_dequant(raw, qtype)).to(DEV)
    w = _native(qtype, raw, n, k)
    out = torch.empty(n, k, dtype=torch.float32, device=DEV)
    kernels.dequant_native(qtype, w.raw, out)
    deq = _rel(out, ref)
    x = torch.randn(1, k, device=DEV)
    gemv = _rel(w(x), x @ ref.T)
    print(f"RES {qtype} real {name} {n}x{k} deq={deq:.2e} gemv={gemv:.2e}")
    assert deq < REL_TOL and gemv < REL_TOL


@pytest.mark.parametrize("qtype", KTYPES)
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
    assert gbps > MIN_GBPS[qtype]
