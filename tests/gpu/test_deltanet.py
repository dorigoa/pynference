"""GPU tests for pynference.deltanet: the Triton kernels vs the CPU references.

Run only inside a GPU job (2.4); skipped when the GPUs are hidden (e.g.
under scripts/cpu.sh). Random data on small shapes and on the real
Qwen3.8-27B DeltaNet shapes, full and per-rank (S19 TP plan); the layer-0
weight check is the S20 job itself. The reference pipeline (conv+SiLU ->
split -> L2 norm -> gated delta) runs on CPU in float64 and reproduces the
state layout and head mapping documented in pynference/deltanet.py.
Acceptance is the S20 bar: state relative error <= 1e-4, output cosine
>= 0.9999.
"""
import os

import numpy as np
import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

from pynference import deltanet

DEV = torch.device("xpu", 0)
EPS = 1e-6
FULL = dict(H_k=16, H_v=48, S=128)
RANK = dict(H_k=8, H_v=24, S=128)
TOL_REL_STATE = 1e-4
TOL_COS_OUT = 0.9999


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _random_inputs(seed, T, H_k, H_v, S, K=4):
    torch.manual_seed(seed)
    C = (2 * H_k + H_v) * S
    x = torch.randn(T, C)
    hist = torch.randn(K - 1, C)
    w = torch.randn(C, K)
    g = -torch.rand(T, H_v) * 2.0
    beta = torch.rand(T, H_v)
    state = torch.randn(H_v, S, S)
    return x, hist, w, g, beta, state


def _ref_pipeline(x, hist, w, g, beta, state, H_k, S):
    """f64 CPU reference: conv+SiLU, q|k|v split, L2 norm, gated delta."""
    x, hist, w, g, beta, state = (t.double() for t in (x, hist, w, g, beta, state))
    y, new_hist = deltanet.conv_prefill(x, hist, w)
    H_v = state.shape[0]
    q = deltanet.l2norm_qk(y[:, :H_k * S].reshape(-1, H_k, S), EPS)
    k = deltanet.l2norm_qk(y[:, H_k * S:2 * H_k * S].reshape(-1, H_k, S), EPS)
    v = y[:, 2 * H_k * S:].reshape(-1, H_v, S)
    out, new_state = deltanet.delta_prefill(q, k, v, g, beta, state)
    return out, new_state, new_hist


def _check(out, state_g, hist_g, ref, tag):
    out_ref, st_ref, hist_ref = ref
    assert _rel_err(state_g.cpu(), st_ref) <= TOL_REL_STATE, f"state {tag}"
    assert _cos(out.cpu(), out_ref) >= TOL_COS_OUT, f"out {tag}"
    torch.testing.assert_close(hist_g.cpu(), hist_ref.float(), rtol=0, atol=0)


# ------------------------------------------------------------------- decode
def _decode_case(seed, H_k, H_v, S, K=4, tag=""):
    x, hist, w, g, beta, state = _random_inputs(seed, 1, H_k, H_v, S, K)
    ref = _ref_pipeline(x, hist, w, g, beta, state, H_k, S)
    hist_g = hist.to(DEV).clone()
    state_g = state.to(DEV).clone()
    out = deltanet.decode(x[0].to(DEV), hist_g, w.to(DEV), g[0].to(DEV),
                          beta[0].to(DEV), state_g, H_k)
    _check(out, state_g, hist_g, ref, tag or f"d1 {H_k}/{H_v}/{S}/{K}")


def test_decode_small():
    _decode_case(1, 2, 4, 16, tag="d small")


def test_decode_small_k2():
    _decode_case(2, 2, 3, 16, K=2, tag="d K=2")


def test_decode_full_shapes():
    _decode_case(3, **FULL, tag="d full")


def test_decode_rank_shapes():
    _decode_case(4, **RANK, tag="d rank")


# ------------------------------------------------------------------ prefill
def _prefill_case(seed, H_k, H_v, S, T, K=4, tag=""):
    x, hist, w, g, beta, state = _random_inputs(seed, T, H_k, H_v, S, K)
    ref = _ref_pipeline(x, hist, w, g, beta, state, H_k, S)
    hist_g = hist.to(DEV).clone()
    state_g = state.to(DEV).clone()
    out = deltanet.prefill(x.to(DEV), hist_g, w.to(DEV), g.to(DEV),
                           beta.to(DEV), state_g, H_k)
    _check(out, state_g, hist_g, ref, tag or f"p {H_k}/{H_v}/{S}/{T}/{K}")


def test_prefill_small():
    _prefill_case(10, 2, 4, 16, 8, tag="p small")


def test_prefill_small_k2():
    _prefill_case(11, 2, 3, 16, 5, K=2, tag="p K=2")


def test_prefill_full_shapes():
    _prefill_case(12, T=16, **FULL, tag="p full")


def test_prefill_rank_shapes():
    _prefill_case(13, T=12, **RANK, tag="p rank")


def test_prefill_then_decode():
    # chunked prefill: kernel prefill of T1 tokens then kernel decode of T2
    # tokens must match the f64 reference over the whole sequence (S21 path)
    H_k, H_v, S, T1, T2 = 2, 4, 32, 5, 3
    x, hist, w, g, beta, state = _random_inputs(14, T1 + T2, H_k, H_v, S)
    ref = _ref_pipeline(x, hist, w, g, beta, state, H_k, S)
    hist_g = hist.to(DEV)
    state_g = state.to(DEV)
    out_p = deltanet.prefill(x[:T1].to(DEV), hist_g, w.to(DEV), g[:T1].to(DEV),
                             beta[:T1].to(DEV), state_g, H_k)
    assert _cos(out_p.cpu(), ref[0][:T1]) >= TOL_COS_OUT, "p1 out"
    for t in range(T1, T1 + T2):
        out = deltanet.decode(x[t].to(DEV), hist_g, w.to(DEV), g[t].to(DEV),
                              beta[t].to(DEV), state_g, H_k)
        assert _cos(out.cpu(), ref[0][t]) >= TOL_COS_OUT, f"d2 out t={t}"
    assert _rel_err(state_g.cpu(), ref[1]) <= TOL_REL_STATE, "chunked state"
    torch.testing.assert_close(hist_g.cpu(), ref[2].float(), rtol=0, atol=0)


def test_decode_matches_ref_step_by_step():
    # consecutive decode calls must equal the prefill reference prefix by prefix
    H_k, H_v, S, T = 2, 5, 32, 6
    x, hist, w, g, beta, state = _random_inputs(5, T, H_k, H_v, S)
    hist_g = hist.to(DEV)
    state_g = state.to(DEV)
    for t in range(T):
        out = deltanet.decode(x[t].to(DEV), hist_g, w.to(DEV), g[t].to(DEV),
                              beta[t].to(DEV), state_g, H_k)
        out_ref, st_ref, hist_ref = _ref_pipeline(x[:t + 1], hist, w,
                                                  g[:t + 1], beta[:t + 1],
                                                  state, H_k, S)
        assert _rel_err(state_g.cpu(), st_ref) <= TOL_REL_STATE, f"state t={t}"
        assert _cos(out.cpu(), out_ref[t]) >= TOL_COS_OUT, f"out t={t}"
        torch.testing.assert_close(hist_g.cpu(), hist_ref.float(), rtol=0, atol=0)


# ------------------------------------------------------- real layer-0 weights
QWEN_MODEL = "/nvme/gguf/Qwen3.8-27B-Q8_0.gguf"
needs_qwen = pytest.mark.skipif(not os.path.exists(QWEN_MODEL),
                                reason="Qwen3.8-27B GGUF not present")
_REAL = None


def _layer0_weights():
    """(wq, conv) from blk.0: wq [10240, 5120] dequantized Q8_0 output rows,
    conv [10240, 4] f32 conv taps (raw GGUF order, deltanet.py docstring).
    Both are copies: the GGUFFile must be closeable (BufferError otherwise).
    """
    global _REAL
    if _REAL is None:
        from pynference.gguf import GGUFFile
        from pynference.quant.ref import ref_dequant
        f = GGUFFile(QWEN_MODEL)
        t = f.tensors["blk.0.attn_qkv.weight"]
        assert t.shape == (5120, 10240) and t.type == "Q8_0"   # S19 note
        wq = ref_dequant(f.rows("blk.0.attn_qkv.weight", 0, t.n_rows), "Q8_0")
        tc = f.tensors["blk.0.ssm_conv1d.weight"]
        assert tc.shape == (4, 10240) and tc.type == "F32"
        conv = np.array(f.rows("blk.0.ssm_conv1d.weight", 0, 10240)
                        .view(np.float32).reshape(10240, 4))
        f.close()
        _REAL = (wq, conv)
    return _REAL


def _chunked_case(x, hist, w, g, beta, state, H_k, T1, tag):
    """prefill of T1 tokens + decode of the rest vs the f64 reference."""
    T = x.shape[0]
    ref = _ref_pipeline(x, hist, w, g, beta, state, H_k, state.shape[2])
    hist_g = hist.to(DEV)
    state_g = state.to(DEV)
    out_p = deltanet.prefill(x[:T1].to(DEV), hist_g, w.to(DEV),
                             g[:T1].to(DEV), beta[:T1].to(DEV), state_g, H_k)
    assert _cos(out_p.cpu(), ref[0][:T1]) >= TOL_COS_OUT, f"{tag} p out"
    for t in range(T1, T):
        out = deltanet.decode(x[t].to(DEV), hist_g, w.to(DEV), g[t].to(DEV),
                              beta[t].to(DEV), state_g, H_k)
        assert _cos(out.cpu(), ref[0][t]) >= TOL_COS_OUT, f"{tag} d out t={t}"
    assert _rel_err(state_g.cpu(), ref[1]) <= TOL_REL_STATE, f"{tag} state"
    torch.testing.assert_close(hist_g.cpu(), ref[2].float(), rtol=0, atol=0)


@needs_qwen
def test_real_weights_full():
    # real qkv projection and conv taps of Qwen3.8-27B layer 0 on random
    # hidden states: x = h @ W_qkv^T, std ~1.1 like the live model
    wq, conv = _layer0_weights()
    T, H_k, H_v, S = 10, 16, 48, 128
    _, hist, _, g, beta, state = _random_inputs(20, T, H_k, H_v, S)
    h = torch.randn(T, 5120)                      # continues seed 20's stream
    x = torch.tensor(h.numpy() @ wq.T)
    _chunked_case(x, hist, torch.tensor(conv), g, beta, state, H_k, T1=8,
                  tag="real full")


@needs_qwen
def test_real_weights_rank():
    # per-rank shape (S19 TP plan): q heads 0-7 | k heads 0-7 | v heads 0-23,
    # 40 heads x 128 = 5120 conv channels
    wq, conv = _layer0_weights()
    T, H_k, H_v, S = 8, 8, 24, 128
    cols = np.r_[0:1024, 2048:3072, 4096:7168]
    _, hist, _, g, beta, state = _random_inputs(21, T, H_k, H_v, S)
    h = torch.randn(T, 5120)
    x = torch.tensor(h.numpy() @ wq[cols].T)
    _chunked_case(x, hist, torch.tensor(conv[cols]), g, beta, state, H_k,
                  T1=6, tag="real rank")
