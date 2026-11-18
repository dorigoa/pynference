"""GPU tests for the S24 hyper-connection mixer: the same torch code on xpu,
plus the GPU-only path — the real blk.0 attn mixer weights Q8_0 through
make_qweight (QLinear Triton GEMM) — checked against the CPU reference
(gguf-py ref_dequant + F32QWeight). hc_combine runs its (small) ops on
device too.

Run only inside a GPU job (plan 2.4); skipped when the GPUs are hidden
(e.g. under scripts/cpu.sh) or when the model file is absent. Acceptance is
the S24 bar: cosine >= 0.999 on the mixer parts. See report S24.
"""
import os
import pathlib

import numpy as np
import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

from pynference.gguf import GGUFFile
from pynference.models.qwen4exp import HCMixer, hc_combine
from pynference.quant.ref import ref_dequant
from pynference.quant.weight import F32QWeight, make_qweight

DEV = torch.device("xpu", 0)
ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

EPS = 1e-6
HC_DIM, N_EMBD, HC_N, RANK = 10240, 2560, 4, 320
TOL_COS = 0.999


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------- the shared torch code on device (no model needed) ----------

def test_hc_synthetic_device_agreement():
    rng = np.random.default_rng(5)
    T, n_embd, hc, rank = 7, 64, 4, 16
    chans = hc * n_embd
    w = {k: rng.standard_normal(s).astype(np.float32) for k, s in {
        "norm": (hc, n_embd), "down": (rank, chans), "up": (chans, rank),
        "inj": (hc, chans)}.items()}
    res = rng.standard_normal((T, chans)).astype(np.float32)

    def build(device):
        t = lambda a: torch.from_numpy(a.copy()).to(device)
        return HCMixer(t(w["norm"]), F32QWeight(t(w["down"]), rank, chans),
                       F32QWeight(t(w["up"]), chans, rank),
                       F32QWeight(t(w["inj"]), hc, chans), eps=EPS)

    m_cpu, i_cpu, parts_cpu = build("cpu").forward(torch.from_numpy(res))
    m_gpu, i_gpu, parts_gpu = build(DEV).forward(torch.from_numpy(res).to(DEV))
    torch.testing.assert_close(m_cpu.to(DEV), m_gpu, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(i_cpu.to(DEV), i_gpu, rtol=1e-5, atol=1e-6)

    # hc_combine's fused-ish torch ops on device
    out = rng.standard_normal((T, n_embd)).astype(np.float32)
    comb_cpu = hc_combine(torch.from_numpy(res), torch.from_numpy(out), i_cpu)
    comb_gpu = hc_combine(torch.from_numpy(res).to(DEV), torch.from_numpy(out).to(DEV), i_gpu)
    torch.testing.assert_close(comb_cpu.to(DEV), comb_gpu, rtol=1e-5, atol=1e-6)


# ---------- the real blk.0 attn mixer on device vs CPU reference ----------

@needs_model
def test_hc_real_mixer_device_vs_cpu():
    g = GGUFFile(MODEL)
    try:
        assert g.tensors["blk.0.hc_attn_down.weight"].type == "Q8_0"
        down_raw = np.ascontiguousarray(g.rows("blk.0.hc_attn_down.weight", 0, RANK))
        up_raw = np.ascontiguousarray(g.rows("blk.0.hc_attn_up.weight", 0, HC_DIM))
        norm = g.raw("blk.0.hc_attn_norm.weight").view(np.float32).reshape(HC_N, N_EMBD).copy()
        inj_w = g.raw("blk.0.hc_attn_inject.weight").view(np.float32).reshape(HC_N, HC_DIM).copy()
    finally:
        g.close()

    torch.manual_seed(0)
    res = torch.randn(5, HC_DIM)  # same input both sides; tests the Q8_0 GEMMs

    cpu = HCMixer(torch.from_numpy(norm.copy()),
                  F32QWeight(torch.from_numpy(ref_dequant(down_raw, "Q8_0")), RANK, HC_DIM),
                  F32QWeight(torch.from_numpy(ref_dequant(up_raw, "Q8_0")), HC_DIM, RANK),
                  F32QWeight(torch.from_numpy(inj_w.copy()), HC_N, HC_DIM), eps=EPS)
    m_cpu, i_cpu, parts_cpu = cpu.forward(res)

    t = lambda a: torch.from_numpy(a.copy()).to(DEV)
    gpu = HCMixer(t(norm),
                  make_qweight(down_raw, "Q8_0", RANK, HC_DIM, DEV),  # QLinear path
                  make_qweight(up_raw, "Q8_0", HC_DIM, RANK, DEV),
                  F32QWeight(t(inj_w), HC_N, HC_DIM), eps=EPS)
    m_gpu, i_gpu, parts_gpu = gpu.forward(res.to(DEV))

    assert _cos(parts_cpu["hc_norm"], parts_gpu["hc_norm"]) >= TOL_COS
    assert _cos(parts_cpu["hc_gate"], parts_gpu["hc_gate"]) >= TOL_COS
    assert _cos(m_cpu, m_gpu) >= TOL_COS
    assert _cos(i_cpu, i_gpu) >= TOL_COS
    assert _rel_err(m_cpu, m_gpu) < 5e-3
    assert _rel_err(i_cpu, i_gpu) < 5e-3

    comb_cpu = hc_combine(res, torch.randn(5, N_EMBD), i_cpu)
    comb_gpu = hc_combine(res.to(DEV), torch.randn(5, N_EMBD).to(DEV), i_gpu)
    assert _cos(comb_cpu, comb_gpu) >= TOL_COS
