"""CPU tests for the S24 hyper-connection mixer/combine: block math against a
numpy f64 reference (explicit per-stream loops), the no-inject output-head
path, chunked vs single pass.

The GGUF-gated tests run blk.0's real attn mixer on the S23-1 dump prompt
([760, 6511, 314, 9338, 369], no history): hc_init is the token embedding
replicated over the 4 streams, so the mixer inputs are reconstructable
without DeltaNet and every attn-phase node — hc_init, hc_norm-0, hc_gate-0,
hc_mixed-0, hc_inject-0 (1st occurrence = attn pass) — is compared directly
against the samples in tests/golden/qwen4exp/dump_nodes.json. This also
settles whether hc_*_norm stores gamma as-is (map A: (1+w) folded by the
converter). The model lives only on the PC (/nvme); those tests skip
elsewhere. See report S24.
"""
import json
import os
import pathlib

import numpy as np
import pytest
import torch

from pynference.gguf import GGUFFile
from pynference.models.qwen4exp import HCMixer, hc_combine
from pynference.quant.ref import ref_dequant
from pynference.quant.weight import F32QWeight

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen4exp"
MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

EPS = 1e-6  # attention.layer_norm_rms_epsilon rounds to 1e-6 in f32
# The S23-1 dump prompt "The capital of France is" (5 tokens).
DUMP_TOKENS = [760, 6511, 314, 9338, 369]
HC_DIM, N_EMBD, HC_N, RANK = 10240, 2560, 4, 320

NODES = json.loads((GOLDEN / "dump_nodes.json").read_text(encoding="utf-8"))["nodes"]


def node(name, call=0):
    """The call-th printed occurrence of `name` (attn pass first)."""
    return [n for n in NODES if n["name"] == name][call]


def assert_matches_dump(ours, nd, *, gemm=False):
    """Compare an f32 tensor ([T, C] or [T*hc, C]) against a golden node: the
    printed 3+3 samples per row (or the whole innermost dim when the dump
    printed it, as for hc_inject) plus the full-tensor sum.

    `gemm=True` is for nodes whose value passed through a Q8_0 mul_mat
    inside llama.cpp: the dump's lower-precision accumulation caps element
    accuracy there (probe: cos >= 0.99999, sums within 0.4%), so the bar is
    the plan's S24 acceptance: cosine >= 0.999 on the samples + sum within
    1% relative. Exact nodes (F32-only paths) must match print precision.
    """
    o = np.asarray(ours, dtype=np.float64).reshape(len(nd["rows"]), -1)
    samp = np.array(nd["rows"], dtype=np.float64)
    if samp.shape[1] == o.shape[1]:
        ours_s = o
    else:
        cols = list(range(3)) + list(range(o.shape[1] - 3, o.shape[1]))
        ours_s = o[:, cols]
    if gemm:
        a, b = ours_s.ravel(), samp.ravel()
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cos >= 0.999, (nd["name"], cos)
        assert abs(o.sum() - nd["sum"]) <= 1e-2 * abs(nd["sum"]), (o.sum(), nd["sum"])
    else:
        np.testing.assert_allclose(ours_s, samp, atol=2e-4)
        assert abs(o.sum() - nd["sum"]) <= max(1e-2, 1e-4 * abs(nd["sum"])), (o.sum(), nd["sum"])


# ---------- synthetic block math ----------

def _make_mixer(seed, T, *, n_embd=32, hc=4, rank=16, inject=True):
    rng = np.random.default_rng(seed)
    chans = hc * n_embd
    norm = rng.standard_normal((hc, n_embd)).astype(np.float32) * 0.3 + 1.0
    down = rng.standard_normal((rank, chans)).astype(np.float32) * np.float32(1.0 / np.sqrt(chans))
    up = rng.standard_normal((chans, rank)).astype(np.float32) * np.float32(1.0 / np.sqrt(rank))
    inj = rng.standard_normal((hc, chans)).astype(np.float32) * np.float32(1.0 / np.sqrt(chans))
    res = rng.standard_normal((T, chans)).astype(np.float32)
    w = lambda a, n, k: F32QWeight(torch.from_numpy(a), n, k)
    mix = HCMixer(torch.from_numpy(norm), w(down, rank, chans), w(up, chans, rank),
                  w(inj, hc, chans) if inject else None, eps=EPS)
    return mix, res, (norm, down, up, inj)


def _mix_reference(x, weights, *, hc, n_embd, rank):
    """Plain numpy f64 formulation with explicit per-stream loops (the
    module's DSV4_HC_PRE mean and SCALE halves written out)."""
    norm, down, up, inj = (a.astype(np.float64) for a in weights)
    T = x.shape[0]
    chans = hc * n_embd
    x = x.astype(np.float64).reshape(T, hc, n_embd)
    xn = np.empty_like(x)
    for t in range(T):
        for s in range(hc):
            r = x[t, s]
            xn[t, s] = r / np.sqrt((r * r).mean() + EPS) * norm[s]
    xnf = xn.reshape(T, chans)
    lo = (xnf @ down.T) / hc
    lo = lo / (1.0 + np.exp(-lo))  # silu = x * sigmoid(x)
    gate = (lo @ up.T).reshape(T, hc, n_embd)
    sig = 1.0 / (1.0 + np.exp(-gate))
    mixed = np.empty((T, n_embd))
    for t in range(T):
        for e in range(n_embd):
            mixed[t, e] = sum(xn[t, s, e] * sig[t, s, e] for s in range(hc)) / hc
    inject = (xnf @ inj.T) / hc
    return xnf, gate.reshape(T, chans), mixed, inject


def test_hc_mixer_matches_numpy_reference():
    mix, res, weights = _make_mixer(3, 7)
    mixed, inj, parts = mix.forward(torch.from_numpy(res))
    want = _mix_reference(res, weights, hc=4, n_embd=32, rank=16)
    for got, w in zip((parts["hc_norm"], parts["hc_gate"], mixed, inj), want):
        np.testing.assert_allclose(got.numpy(), w, rtol=2e-5, atol=2e-7)


def test_hc_mixer_output_head_has_no_inject():
    mix, res, _ = _make_mixer(5, 3, inject=False)
    mixed, inj, parts = mix.forward(torch.from_numpy(res))
    assert inj is None and "hc_inject" not in parts
    assert mixed.shape == (3, 32)


def test_hc_mixer_chunked_matches_single_pass():
    mix, res, _ = _make_mixer(4, 6)
    m0, i0, _ = mix.forward(torch.from_numpy(res))
    ms, is_ = [], []
    for t in range(res.shape[0]):  # decode one token at a time
        m, i, _ = mix.forward(torch.from_numpy(res[t:t + 1]))
        ms.append(m), is_.append(i)
    # [1,K] vs [T,K] matmul: torch CPU blocks differently, ~1e-4 relative
    np.testing.assert_allclose(torch.cat(ms, 0).numpy(), m0.numpy(), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(torch.cat(is_, 0).numpy(), i0.numpy(), rtol=1e-3, atol=1e-6)


def test_hc_combine_matches_loop():
    rng = np.random.default_rng(9)
    T, hc, C = 5, 4, 64
    res = rng.standard_normal((T, hc * C)).astype(np.float32)
    out = rng.standard_normal((T, C)).astype(np.float32)
    inj = rng.standard_normal((T, hc)).astype(np.float32)
    got = hc_combine(torch.from_numpy(res), torch.from_numpy(out), torch.from_numpy(inj))
    want = np.empty((T, hc, C))
    for t in range(T):
        for s in range(hc):
            w = 2.0 / (1.0 + np.exp(-inj[t, s].astype(np.float64)))
            want[t, s] = res[t, s * C:(s + 1) * C].astype(np.float64) + out[t].astype(np.float64) * w
    np.testing.assert_allclose(got.numpy(), want.reshape(T, hc * C), rtol=2e-5, atol=2e-7)


# ---------- real model: blk.0 attn mixer vs the S23-1 dump ----------

def _embed_rows(g, toks, name="token_embd.weight"):
    """Dequantized embedding rows [len(toks), n_embd] via the CPU reference."""
    rows = np.concatenate([g.rows(name, t, t + 1) for t in toks])
    return torch.from_numpy(ref_dequant(rows, g.tensors[name].type))


def _real_mixer(g, phase):
    f32 = lambda name: g.raw(name).view(np.float32).reshape(HC_N, N_EMBD).copy()
    q8 = lambda name, n, k: F32QWeight(
        torch.from_numpy(ref_dequant(np.ascontiguousarray(g.rows(name, 0, n)), "Q8_0")), n, k)
    p = f"blk.0.hc_{phase}"
    return HCMixer(torch.from_numpy(f32(f"{p}_norm.weight")),
                   q8(f"{p}_down.weight", RANK, HC_DIM),
                   q8(f"{p}_up.weight", HC_DIM, RANK),
                   F32QWeight(torch.from_numpy(
                       g.raw(f"{p}_inject.weight").view(np.float32).reshape(HC_N, HC_DIM).copy()),
                       HC_N, HC_DIM),
                   eps=EPS)


@needs_model
def test_hc_init_is_replicated_embedding():
    g = GGUFFile(MODEL)
    try:
        assert g.tensors["token_embd.weight"].type == "IQ4_XS"
        emb = _embed_rows(g, DUMP_TOKENS)
    finally:
        g.close()
    res = emb.unsqueeze(1).expand(len(DUMP_TOKENS), HC_N, N_EMBD)
    assert_matches_dump(res, node("hc_init"))


@needs_model
def test_hc_attn_mixer_matches_dump():
    g = GGUFFile(MODEL)
    try:
        assert g.tensors["blk.0.hc_attn_down.weight"].type == "Q8_0"
        res = _embed_rows(g, DUMP_TOKENS).unsqueeze(1).expand(
            len(DUMP_TOKENS), HC_N, N_EMBD).reshape(len(DUMP_TOKENS), HC_DIM)
        mixed, inj, parts = _real_mixer(g, "attn").forward(res)
    finally:
        g.close()
    assert parts.keys() >= {"hc_norm", "hc_gate", "hc_mixed", "hc_inject"}
    # hc_norm/hc_inject are exact vs the dump (cos = 1.0): the mixer math is
    # right and hc_*_norm stores gamma with any (1+w) folded (map A).
    assert_matches_dump(parts["hc_norm"], node("hc_norm-0"))
    assert_matches_dump(parts["hc_gate"], node("hc_gate-0"), gemm=True)
    assert_matches_dump(parts["hc_mixed"], node("hc_mixed-0"), gemm=True)
    assert_matches_dump(inj, node("hc_inject-0"))
