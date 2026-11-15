"""GPU tests for pynference.quant Q8_0: the selftest as pytest with thresholds.

Real Gemma-4-31B weight shapes. Run only inside a GPU job (2.4); skipped when
the GPUs are hidden. S6-1 observed (report S6): rel err <= 1.6e-03 (M=1),
<= 3.3e-03 (M=8 path), bandwidth 537-590 GB/s.
"""
import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

from pynference.quant import QLinear

DEV = torch.device("xpu", 0)
REL_TOL_M1 = 5e-3       # observed ~1.5e-03
REL_TOL_M8 = 1e-2       # observed ~3e-03 (bf16 F.linear path)
MIN_BANDWIDTH_GBPS = 200.0  # observed 537-590

SHAPES = [(8192, 5376), (5376, 10752), (21504, 5376), (131072, 5376)]


def _q8_linear(n, k, seed):
    torch.manual_seed(seed)
    w = torch.randn(n, k, device=DEV) * 0.02
    d = (w.view(n, k // 32, 32).abs().amax(-1) / 127).clamp(min=1e-8).half()
    qs = torch.round(w.view(n, k // 32, 32) / d.float()[..., None]).clamp(-127, 127) \
        .to(torch.int8).view(n, k)
    return QLinear(qs, d)


@pytest.mark.parametrize("n,k", SHAPES)
def test_gemv_accuracy(n, k):
    lin = _q8_linear(n, k, 0)
    try:
        wref = lin.dequant().float()
        x = torch.randn(1, k, device=DEV)
        ref = x @ wref.T
        err = ((lin(x) - ref).abs().max() / ref.abs().max()).item()
        assert err < REL_TOL_M1, f"{n}x{k}: rel err {err:.2e}"
        x8 = torch.randn(8, k, device=DEV)
        err8 = ((lin(x8) - x8 @ wref.T).abs().max() / (x8 @ wref.T).abs().max()).item()
        assert err8 < REL_TOL_M8, f"{n}x{k}: M=8 rel err {err8:.2e}"
    finally:
        del lin


@pytest.mark.parametrize("n,k", SHAPES)
def test_gemv_bandwidth(n, k):
    import time
    lin = _q8_linear(n, k, 0)
    try:
        x = torch.randn(1, k, device=DEV)
        for _ in range(3):
            lin(x)
        torch.xpu.synchronize()
        t = time.perf_counter()
        it = 50
        for _ in range(it):
            lin(x)
        torch.xpu.synchronize()
        gbps = lin.nbytes / ((time.perf_counter() - t) / it) / 1e9
        print(f"Q8_0 GEMV {n}x{k}: {gbps:.1f} GB/s")
        assert gbps > MIN_BANDWIDTH_GBPS, f"{n}x{k}: {gbps:.1f} GB/s"
    finally:
        del lin
