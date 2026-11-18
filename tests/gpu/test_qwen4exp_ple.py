"""GPU tests for the S24 PLE module: the same torch code on xpu, plus the two
GPU-only paths — the IQ4_NL table rows dequantized on device
(NativeQWeight.rows_dequant over just the gathered bytes) and the real Q8_0
ple_key/ple_value via make_qweight — checked against the CPU reference
(gguf-py ref_dequant + F32QWeight).

Run only inside a GPU job (plan 2.4); skipped when the GPUs are hidden
(e.g. under scripts/cpu.sh) or when the model file is absent. Acceptance is
the S24 bar: cosine >= 0.999 on the block parts. See report S24.
"""
import json
import os
import pathlib

import numpy as np
import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

from pynference.gguf import GGUFFile
from pynference.models.qwen4exp import PLE, ple_gather_embedding, ple_row_indices
from pynference.quant.ref import ref_dequant
from pynference.quant.weight import F32QWeight, make_qweight

DEV = torch.device("xpu", 0)
ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen4exp"
MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

MD = json.loads((GOLDEN / "facts.json").read_text(encoding="utf-8"))["metadata"]
PLE_C = dict(
    ngram=MD["qwen4exp.ple.ngram_size"],
    heads_per_ngram=MD["qwen4exp.ple.heads_per_ngram"],
    multipliers=MD["qwen4exp.ple.layer_multipliers"],
    head_offsets=MD["qwen4exp.ple.head_offsets"],
    head_vocab_sizes=MD["qwen4exp.ple.head_vocab_sizes"],
    eos_token_id=MD["qwen4exp.ple.eos_token_id"],
)
EPS = 1e-6
TABLE = "per_layer_token_embd.weight"
DUMP_TOKENS = [760, 6511, 314, 9338, 369]  # the S23-1 dump prompt
TOL_COS = 0.999


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------- the device-only gather path ----------

@needs_model
def test_gather_rows_on_device_matches_cpu_reference():
    idx = ple_row_indices(DUMP_TOKENS, [], **PLE_C)
    g = GGUFFile(MODEL)
    try:
        cpu = ple_gather_embedding(g, TABLE, idx, device="cpu")
        gpu = ple_gather_embedding(g, TABLE, idx, device=DEV)
    finally:
        g.close()
    assert gpu.device.type == "xpu" and gpu.shape == cpu.shape
    # IQ4_NL is table[nibble] * fp16_scale: exact in fp32 both ways
    assert torch.allclose(cpu.to(DEV), gpu, rtol=0.0, atol=1e-6)


# ---------- the real blk.1 PLE on device vs CPU reference ----------

@needs_model
def test_ple_real_layer_device_vs_cpu():
    g = GGUFFile(MODEL)
    try:
        key_raw = np.ascontiguousarray(g.rows("blk.1.ple_key.weight", 0, 10240))
        val_raw = np.ascontiguousarray(g.rows("blk.1.ple_value.weight", 0, 2560))
        assert g.tensors["blk.1.ple_key.weight"].type == "Q8_0"
        f32 = lambda name: g.raw(name).view(np.float32)
        nk = f32("blk.1.ple_norm_key.weight").reshape(4, 2560).copy()
        nq = f32("blk.1.ple_norm_query.weight").reshape(4, 2560).copy()
        nc = f32("blk.1.ple_norm_conv.weight").reshape(4, 2560).copy()
        cw = f32("blk.1.ple_conv1d.weight").reshape(10240, 4).copy()  # tap last
        idx = ple_row_indices(DUMP_TOKENS, [], **PLE_C)
        emb_cpu = ple_gather_embedding(g, TABLE, idx, device="cpu")
        emb_gpu = ple_gather_embedding(g, TABLE, idx, device=DEV)
    finally:
        g.close()

    torch.manual_seed(0)
    res = torch.randn(len(DUMP_TOKENS), 10240)
    kw_cpu = F32QWeight(torch.from_numpy(ref_dequant(key_raw, "Q8_0")), 10240, 2560)
    vw_cpu = F32QWeight(torch.from_numpy(ref_dequant(val_raw, "Q8_0")), 2560, 2560)
    t_cpu = lambda a: torch.from_numpy(a.copy())
    ple_cpu = PLE(kw_cpu, vw_cpu, t_cpu(nk), t_cpu(nq), t_cpu(nc), t_cpu(cw),
                  n_embd=2560, ngram=PLE_C["ngram"], eps=EPS)
    out_cpu, state_cpu, parts_cpu = ple_cpu.forward(
        res, emb_cpu, ple_cpu.zero_state())

    kw_gpu = make_qweight(key_raw, "Q8_0", 10240, 2560, DEV)  # the Q8_0 GEMM path
    vw_gpu = make_qweight(val_raw, "Q8_0", 2560, 2560, DEV)
    t_gpu = lambda a: torch.from_numpy(a.copy()).to(DEV)
    ple_gpu = PLE(kw_gpu, vw_gpu, t_gpu(nk), t_gpu(nq), t_gpu(nc), t_gpu(cw),
                  n_embd=2560, ngram=PLE_C["ngram"], eps=EPS)
    out_gpu, state_gpu, parts_gpu = ple_gpu.forward(
        res.to(DEV), emb_gpu, ple_gpu.zero_state(DEV))

    assert _cos(out_cpu, out_gpu) >= TOL_COS
    assert _cos(parts_cpu["ple_gate"], parts_gpu["ple_gate"]) >= TOL_COS
    assert _cos(parts_cpu["ple_gated_value"], parts_gpu["ple_gated_value"]) >= TOL_COS
    assert _cos(parts_cpu["ple_conv_out"], parts_gpu["ple_conv_out"]) >= TOL_COS
    assert _rel_err(out_cpu, out_gpu) < 5e-3
    assert _rel_err(state_cpu, state_gpu) < 5e-3


# ---------- the shared torch code on device (no model needed) ----------

def test_ple_synthetic_device_agreement():
    rng = np.random.default_rng(5)
    T, n_embd, hc, kernel = 7, 64, 4, 4
    chans = hc * n_embd
    w = {k: rng.standard_normal(s).astype(np.float32) for k, s in {
        "key": (chans, n_embd), "value": (n_embd, n_embd), "nk": (hc, n_embd),
        "nq": (hc, n_embd), "nc": (hc, n_embd), "cw": (chans, kernel)}.items()}
    res = rng.standard_normal((T, chans)).astype(np.float32)
    emb = rng.standard_normal((T, n_embd)).astype(np.float32)
    hist = (kernel - 1) * PLE_C["ngram"]
    state = rng.standard_normal((hist, chans)).astype(np.float32)

    def build(device):
        t = lambda a: torch.from_numpy(a.copy()).to(device)
        return PLE(F32QWeight(t(w["key"]), chans, n_embd),
                   F32QWeight(t(w["value"]), n_embd, n_embd),
                   t(w["nk"]), t(w["nq"]), t(w["nc"]), t(w["cw"]),
                   n_embd=n_embd, ngram=PLE_C["ngram"], eps=EPS)

    out_cpu, state_cpu, parts_cpu = build("cpu").forward(
        torch.from_numpy(res), torch.from_numpy(emb), torch.from_numpy(state))
    out_gpu, state_gpu, parts_gpu = build(DEV).forward(
        torch.from_numpy(res).to(DEV), torch.from_numpy(emb).to(DEV),
        torch.from_numpy(state).to(DEV))
    torch.testing.assert_close(out_cpu.to(DEV), out_gpu, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state_cpu.to(DEV), state_gpu, rtol=1e-5, atol=1e-6)
    assert _cos(parts_cpu["ple_conv_out"], parts_gpu["ple_conv_out"]) >= 0.99999
