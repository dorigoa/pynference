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
