"""CPU tests of the Gated DeltaNet references (S20 step 1).

The two independent references in pynference.deltanet — vectorized torch in
float64, and the float32 numpy transcription of the llama.cpp ops.cpp
kernels — must agree within the S20 acceptance criteria (state relative
error <= 1e-4, output cosine >= 0.9999) on both random small shapes and the
real Qwen3.8-27B shapes (full and per-rank, S19 TP plan). The decode entry
points must also match the prefill entry token by token, since the decode
kernel in step 2 is only correct if it reproduces the prefill state.

Random data only; the real-weight checks are the GPU job of step 4.
"""
import os

# pynference.deltanet imports Triton at module level, and this file is
# collected before test_gptoss_attn.py / test_quant_decode_interp.py
# (alphabetical order): the S15 TRITON_INTERPRET contract wants the env var
# set before the first Triton import of the pytest process, or the
# interpreted tests collected after this one lose interpreter mode.
os.environ["TRITON_INTERPRET"] = "1"

import numpy as np
import torch

from pynference import deltanet

# Qwen3.8-27B DeltaNet sizes (S19 map): value heads, key/query heads, head dim
FULL = dict(H_k=16, H_v=48, S=128)
RANK = dict(H_k=8, H_v=24, S=128)   # per-rank shapes after the parity split


def _gen(rng):
    # local generator only: the global torch RNG must stay untouched, the
    # later test files in the suite depend on the state they inherit.
    return torch.Generator().manual_seed(int(rng.integers(1 << 31)))


def _randoms(rng, T, H_k, H_v, S, dtype):
    gen = _gen(rng)
    q = deltanet.l2norm_qk(torch.randn(T, H_k, S, dtype=dtype, generator=gen), 1e-6)
    k = deltanet.l2norm_qk(torch.randn(T, H_k, S, dtype=dtype, generator=gen), 1e-6)
    v = torch.randn(T, H_v, S, dtype=dtype, generator=gen)
    g = -torch.rand(T, H_v, dtype=dtype, generator=gen) * 2.0   # decay exp(g) in (e^-2, 1]
    beta = torch.rand(T, H_v, dtype=dtype, generator=gen)       # sigmoid output range
    state = torch.randn(H_v, S, S, dtype=dtype, generator=gen)
    return q, k, v, g, beta, state


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------------ conv
def test_conv_ops_ref_matches_vectorized():
    rng = np.random.default_rng(1)
    T, C, K = 5, 6, 4
    x = rng.standard_normal((T, C))
    hist = rng.standard_normal((K - 1, C))
    w = rng.standard_normal((C, K))
    y_ref, st_ref = deltanet.conv_prefill(
        torch.tensor(x, dtype=torch.float32), torch.tensor(hist, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32))
    y_ops, st_ops = deltanet.conv_ops_ref(x, hist, w)
    np.testing.assert_allclose(y_ref.numpy(), y_ops, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(st_ref.numpy(), st_ops, rtol=0, atol=0)


def test_conv_decode_matches_prefill():
    rng = np.random.default_rng(2)
    T, C, K = 5, 10240, 4             # real conv width (q 2048 | k 2048 | v 6144)
    x = rng.standard_normal((T, C)).astype(np.float32)
    hist = rng.standard_normal((K - 1, C)).astype(np.float32)
    w = rng.standard_normal((C, K)).astype(np.float32)
    y_pf, st_pf = deltanet.conv_prefill(
        torch.from_numpy(x), torch.from_numpy(hist), torch.from_numpy(w))
    st = torch.from_numpy(hist.copy())
    for t in range(T):
        y_dec, st = deltanet.conv_decode(torch.from_numpy(x[t]), st,
                                         torch.from_numpy(w))
        np.testing.assert_allclose(y_dec.numpy(), y_pf[t].numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(st.numpy(), st_pf.numpy(), rtol=0, atol=0)


# ---------------------------------------------------------------- delta rule
def test_delta_ops_ref_matches_vectorized():
    rng = np.random.default_rng(3)
    torch.manual_seed(rng.integers(1 << 31))
    q, k, v, g, beta, state = _randoms(rng, 4, 2, 4, 8, torch.float64)
    out_ref, st_ref = deltanet.delta_prefill(q, k, v, g, beta, state)
    out_ops, st_ops = deltanet.delta_ops_ref(q, k, v, g, beta, state)
    np.testing.assert_allclose(out_ref.numpy(), out_ops, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(st_ref.numpy(), st_ops, rtol=1e-4, atol=1e-5)


def test_delta_refs_agree_real_shapes():
    for shapes in (FULL, RANK):
        rng = np.random.default_rng(4)
        torch.manual_seed(rng.integers(1 << 31))
        T = 3
        q, k, v, g, beta, state = _randoms(rng, T, **shapes, dtype=torch.float64)
        out_ref, st_ref = deltanet.delta_prefill(q, k, v, g, beta, state)
        out_ops, st_ops = deltanet.delta_ops_ref(q, k, v, g, beta, state)
        assert _rel_err(st_ref, st_ops) <= 1e-4, shapes
        assert _cos(out_ref, out_ops) >= 0.9999, shapes


def test_delta_decode_matches_prefill():
    rng = np.random.default_rng(5)
    torch.manual_seed(rng.integers(1 << 31))
    T = 5
    q, k, v, g, beta, state = _randoms(rng, T, **FULL, dtype=torch.float32)
    out_pf, st_pf = deltanet.delta_prefill(q, k, v, g, beta, state)
    st = state.clone()
    for t in range(T):
        out_d, st = deltanet.delta_decode(q[t], k[t], v[t], g[t], beta[t], st)
        torch.testing.assert_close(out_d, out_pf[t], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(st, st_pf, rtol=2e-6, atol=2e-6)


def test_delta_key_modulo():
    # heads h and h + H_k must share key/query head h % H_k (ops.cpp:11079):
    # clone one head's inputs onto its partner and compare outputs.
    rng = np.random.default_rng(6)
    torch.manual_seed(rng.integers(1 << 31))
    T, H_k, H_v, S = 4, 2, 4, 8
    q, k, v, g, beta, state = _randoms(rng, T, H_k, H_v, S, torch.float64)
    v[:, 2] = v[:, 0]
    g[:, 2] = g[:, 0]
    beta[:, 2] = beta[:, 0]
    state[2] = state[0]
    out_ref, st_ref = deltanet.delta_prefill(q, k, v, g, beta, state)
    out_ops, st_ops = deltanet.delta_ops_ref(q, k, v, g, beta, state)
    np.testing.assert_allclose(out_ref.numpy()[:, 2], out_ref.numpy()[:, 0], rtol=0, atol=0)
    np.testing.assert_allclose(out_ops[:, 2], out_ops[:, 0], rtol=0, atol=0)
    np.testing.assert_allclose(st_ref.numpy()[2], st_ref.numpy()[0], rtol=0, atol=0)
    np.testing.assert_allclose(out_ref.numpy()[:, 2], out_ops[:, 2], rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------- norms
def test_norm_gated_matches_ops_ref():
    rng = np.random.default_rng(7)
    H_v, S, eps = 6, 32, 1e-6
    out = rng.standard_normal((H_v, S)).astype(np.float32)
    z = rng.standard_normal((H_v, S)).astype(np.float32) * 2.0
    w = rng.standard_normal(S).astype(np.float32)
    y_ref = deltanet.norm_gated(torch.from_numpy(out), torch.from_numpy(z),
                                torch.from_numpy(w), eps)
    y_ops = deltanet.norm_gated_ops_ref(out, z, w, eps)
    np.testing.assert_allclose(y_ref.numpy(), y_ops, rtol=1e-6, atol=1e-7)


def test_l2norm_qk():
    rng = np.random.default_rng(8)
    x = torch.randn(3, 7, 128, dtype=torch.float64)
    y = deltanet.l2norm_qk(x, 1e-6)
    ss = (y * y).sum(-1).numpy()
    assert np.allclose(ss, 1.0, atol=1e-4)              # unit rows up to eps
    torch.testing.assert_close(deltanet.l2norm_qk(x * 100, 1e-6), y, rtol=1e-3, atol=0)
