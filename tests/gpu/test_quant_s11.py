"""S11 step 4: GPU tests for the MXFP4, IQ4_XS and Q2_0 kernels (native layout).

Same contract as test_quant_kquant.py (S10): dequant against gguf-py
(quant/ref.py) on random blocks and on real tensor rows, GEMV against fp32
with relative error <= 1e-4, indexed GEMV against a python loop, bandwidth
on 8192x5376. Real tensors come from the step 4 scan of the Flash-Next
IQ4_XS shards: IQ4_XS token_embd.weight and a Q2_0 expert stack (3-D
tensor, read as rows); MXFP4 exists in no local file, so it is covered by
the random cases only (as the S11 card allows).

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
KTYPES = ["MXFP4", "IQ4_XS", "Q2_0"]
FLASH = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
# (qtype, file, tensor name) from the scan; k is info.shape[0], a multiple of
# the block size (Q2_0: the 3-D expert stack read as flat rows, k=640)
REAL = [
    ("IQ4_XS", FLASH, "token_embd.weight"),            # (2560, 248320)
    ("Q2_0", FLASH, "blk.8.ffn_down_exps.weight"),     # (640, 2560, 512)
]
REL_TOL = 1e-4
# per-type GEMV floor, bytes read / s on 8192x5376. The plan sets no bandwidth
# contract (the card only asks to measure; C9 goals are non-blocking), so this
# first pass keeps a deliberately low floor that only catches real regressions;
# job S11-1 reports the measured values (RES ... gbps=) and the floors are
# then fixed per type, as done in S10 after jobs S10-1/2/3.
MIN_GBPS = {q: 40.0 for q in KTYPES}


def _native(qtype, raw, n, k):
    dev = torch.from_numpy(np.ascontiguousarray(raw).copy()).to(DEV)
    return NativeQWeight(qtype, dev.view(n, row_bytes(qtype, k)), n, k)


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("qtype", KTYPES)
def test_dequant_random(qtype):
    n, k = 96, 512  # two blocks per row for IQ4_XS, 16 for MXFP4/Q2_0
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


def test_dequant_odd_rows_mxfp4():
    """17 blocks per row -> row_bytes = 17*17 = 289, odd: no uint16 view path."""
    n, k = 16, 544
    raw = make_test_blocks("MXFP4", n, k, seed=9)
    assert row_bytes("MXFP4", k) % 2 == 1
    ref = torch.from_numpy(ref_dequant(raw, "MXFP4")).to(DEV)
    w = _native("MXFP4", raw, n, k)
    out = torch.empty(n, k, dtype=torch.float32, device=DEV)
    kernels.dequant_native("MXFP4", w.raw, out)
    err = _rel(out, ref)
    print(f"RES MXFP4 deq_odd_rows={err:.2e}")
    assert err < REL_TOL


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
