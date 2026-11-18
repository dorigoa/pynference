"""CPU tests for the S24 PLE module: n-gram row hash, IQ4_NL table gather,
PLE block math.

Synthetic checks compare the vectorized torch code against plain numpy loops
on torch CPU. The GGUF-gated test recomputes the row indices of the S23-1
dump prompt ([760, 6511, 314, 9338, 369], no history) and compares the
gathered embedding against the sampled `ple_embd` node (first 3 + last 3
values of every token row + the full-tensor sum) in
tests/golden/qwen4exp/dump_nodes.json. The model lives only on the PC (/nvme);
those tests skip elsewhere. See report S24.
"""
import json
import math
import os
import pathlib

import numpy as np
import pytest
import torch

from pynference.gguf import GGUFFile
from pynference.models.qwen4exp import PLE, grouped_norm, ple_gather_embedding, ple_row_indices
from pynference.quant.weight import F32QWeight

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen4exp"
MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

FACTS = json.loads((GOLDEN / "facts.json").read_text(encoding="utf-8"))
MD = FACTS["metadata"]
PLE_C = dict(
    ngram=MD["qwen4exp.ple.ngram_size"],
    heads_per_ngram=MD["qwen4exp.ple.heads_per_ngram"],
    multipliers=MD["qwen4exp.ple.layer_multipliers"],
    head_offsets=MD["qwen4exp.ple.head_offsets"],
    head_vocab_sizes=MD["qwen4exp.ple.head_vocab_sizes"],
    eos_token_id=MD["qwen4exp.ple.eos_token_id"],
)
EPS = 1e-6  # attention.layer_norm_rms_epsilon rounds to 1e-6 in f32
TABLE = "per_layer_token_embd.weight"
# The S23-1 dump prompt "The capital of France is" (5 tokens).
DUMP_TOKENS = [760, 6511, 314, 9338, 369]


# ---------- row hash ----------

# Synthetic constants for the hash tests: 3 sizes, 8 heads each, 64-bit-safe.
SYNTH = dict(
    multipliers=[101, 103, 107],
    head_offsets=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
                  8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000],
    head_vocab_sizes=[997, 1009, 1013, 1019, 977, 983, 991, 1009,
                      971, 983, 991, 997, 1009, 1013, 1019, 977],
)


def _hash_reference(tokens, history, eos):
    """Plain-python reference of set_input 1293-1352: int arithmetic masked
    to uint64, explicit window walk (ngram=3, 8 heads, SYNTH constants)."""
    mask = (1 << 64) - 1
    mul, offs, voc = SYNTH["multipliers"], SYNTH["head_offsets"], SYNTH["head_vocab_sizes"]
    ngram, heads = 3, 8
    # plain python ints: numpy 2 refuses to mask int64 scalars with 2**64-1
    tokens = [int(x) for x in tokens]
    history = [int(x) for x in history]
    stream = history + tokens
    H = len(history)
    out = []
    for i, tok in enumerate(tokens):
        pos = H + i
        ctx = [tok]
        cut = False
        for s in range(1, ngram):
            if cut or pos - s < 0:
                cut, _ = True, ctx.append(eos)
            else:
                t = stream[pos - s]
                if t == eos:
                    cut, _ = True, ctx.append(eos)
                else:
                    ctx.append(t)
        for n in range(2, ngram + 1):
            mixed = 0
            for s in range(n):
                mixed ^= (ctx[s] * mul[s]) & mask
            for h in range((n - 2) * heads, (n - 1) * heads):
                out.append(mixed % voc[h] + offs[h])
    return np.array(out, dtype=np.int64)


def _synth_consts(eos):
    return dict(PLE_C, ngram=3, heads_per_ngram=8, eos_token_id=eos, **SYNTH)


def test_row_indices_random_matches_loop_reference():
    rng = np.random.default_rng(7)
    eos = 99
    for hist_len in (0, 2):
        for _ in range(5):
            tokens = rng.integers(0, 150, size=int(rng.integers(1, 9)))
            tokens[rng.integers(0, len(tokens))] = eos  # some eos in the stream
            history = rng.integers(0, 150, size=hist_len)
            got = ple_row_indices(tokens, history, **_synth_consts(eos))
            want = _hash_reference(tokens, history, eos)
            assert got.shape == want.shape
            assert np.array_equal(got, want), (tokens, history)


def test_row_indices_window_semantics():
    eos = 99
    C = _synth_consts(eos)
    mask = (1 << 64) - 1

    def rows(tokens, history=()):
        return ple_row_indices(np.array(tokens), np.array(history), **C)

    # bigram head 0 of token 57 behind 10 mixes only (57, 10): hand row
    mixed = ((57 * C["multipliers"][0]) ^ (10 * C["multipliers"][1])) & mask
    assert rows([57], [10])[0] == mixed % C["head_vocab_sizes"][0] + C["head_offsets"][0]

    # the token's OWN eos does not cut: its predecessors still participate
    assert not np.array_equal(rows([eos], [10]), rows([eos], [7]))

    # an eos in the window cuts everything further back than it, the near
    # side is kept: 30/42 are hidden, 31/44 are not
    assert np.array_equal(rows([20], [30, eos, 31]), rows([20], [42, eos, 31]))
    assert not np.array_equal(rows([20], [30, eos, 31]), rows([20], [30, eos, 44]))

    # missing predecessors read as eos, like an explicit eos prefix
    assert np.array_equal(rows([20], []), rows([20], [eos, eos]))


# ---------- grouped_norm ----------

def test_grouped_norm_matches_loop():
    rng = np.random.default_rng(11)
    T, hc, C = 5, 4, 64
    x = rng.standard_normal((T, hc, C)).astype(np.float32)
    gamma = rng.standard_normal((hc, C)).astype(np.float32)
    got = grouped_norm(torch.from_numpy(x), torch.from_numpy(gamma), EPS).numpy()
    for t in range(T):
        for s in range(hc):
            r = x[t, s]
            want = r / np.sqrt((r * r).mean() + EPS) * gamma[s]
            np.testing.assert_allclose(got[t, s], want, rtol=1e-6, atol=1e-7)


# ---------- PLE block math ----------

def _make_ple(seed, T, *, n_embd=32, hc=4, kernel=4, ngram=3, device=None):
    rng = np.random.default_rng(seed)
    chans = hc * n_embd
    key_w = rng.standard_normal((chans, n_embd)).astype(np.float32)
    value_w = rng.standard_normal((n_embd, n_embd)).astype(np.float32)
    norms = [rng.standard_normal((hc, n_embd)).astype(np.float32) * 0.3 + 1.0 for _ in range(3)]
    conv_w = rng.standard_normal((chans, kernel)).astype(np.float32)
    kw, vw = F32QWeight(torch.from_numpy(key_w), chans, n_embd), \
             F32QWeight(torch.from_numpy(value_w), n_embd, n_embd)
    t = lambda a: torch.from_numpy(a).to(device) if device is not None else torch.from_numpy(a)
    ple = PLE(kw, vw, t(norms[0]), t(norms[1]), t(norms[2]), t(conv_w),
              n_embd=n_embd, ngram=ngram, eps=EPS)
    res = t(rng.standard_normal((T, chans)).astype(np.float32))
    emb = t(rng.standard_normal((T, n_embd)).astype(np.float32))
    state = t(rng.standard_normal((ple.hist, chans)).astype(np.float32))
    return ple, res, emb, state, (key_w, value_w, norms, conv_w)


def _ple_reference_np(res, emb, state, weights, *, n_embd, hc, kernel, ngram):
    """Independent numpy formulation: per-token loops, padding picked by an
    explicit t-offset instead of the module's shifted slices."""
    key_w, value_w, (nk, nq, nc), conv_w = weights
    T = res.shape[0]
    hist = (kernel - 1) * ngram
    chans = hc * n_embd
    res, emb, state = (a.astype(np.float64) for a in (res, emb, state))

    def rms(a):
        return a / np.sqrt((a * a).mean(-1, keepdims=True) + EPS)

    raw_key = emb @ key_w.T.astype(np.float64)
    key = rms(raw_key.reshape(T, hc, n_embd)) * nk
    query = rms(res.reshape(T, hc, n_embd)) * nq
    s = (key * query).sum(-1) / np.sqrt(n_embd)
    u = np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6))
    gate = 1.0 / (1.0 + np.exp(-u))
    value = emb @ value_w.T.astype(np.float64)
    gated = value[:, None, :] * gate[:, :, None]
    normed = (rms(gated) * nc).reshape(T, chans)
    padded = np.concatenate([state, normed], 0)
    conv = np.zeros((T, chans))
    for t in range(T):
        for k in range(kernel):
            j = t - (kernel - 1 - k) * ngram
            sel = state[hist + j] if j < 0 else normed[j]
            conv[t] += sel * conv_w[:, k]
    conv_out = (conv * (1.0 / (1.0 + np.exp(-conv)))).reshape(T, hc, n_embd)
    out = res + (gated + conv_out).reshape(T, chans)
    return out, padded[-hist:], gate, conv_out


def test_ple_forward_matches_numpy_reference():
    ple, res, emb, state, weights = _make_ple(3, 7)
    got_res, got_state, parts = ple.forward(res, emb, state)
    want = _ple_reference_np(res.numpy(), emb.numpy(), state.numpy(), weights,
                             n_embd=32, hc=4, kernel=4, ngram=3)
    np.testing.assert_allclose(got_res.numpy(), want[0], rtol=2e-5, atol=2e-6)
    # the state rows are f32 norms seen through f64: a few ulps off
    np.testing.assert_allclose(got_state.numpy(), want[1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts["ple_gate"].numpy(), want[2], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(parts["ple_conv_out"].numpy(), want[3], rtol=2e-5, atol=2e-6)


def test_ple_chunked_matches_single_pass():
    ple, res, emb, state, _ = _make_ple(4, 6)
    full_res, full_state, _ = ple.forward(res, emb, state)
    st = state.clone()
    outs = []
    for t in range(res.shape[0]):  # decode one token at a time
        r, st, _ = ple.forward(res[t:t + 1], emb[t:t + 1], st)
        outs.append(r)
    # [1,K] vs [T,K] matmul: torch CPU blocks differently, ~1e-4 relative
    np.testing.assert_allclose(torch.cat(outs, 0).numpy(), full_res.numpy(),
                               rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(st.numpy(), full_state.numpy(), rtol=1e-3, atol=1e-6)


# ---------- real model: hash + gather vs the S23-1 dump ----------

@needs_model
def test_ple_row_indices_on_dump_tokens():
    idx = ple_row_indices(DUMP_TOKENS, [], **PLE_C)
    assert idx.shape == (16 * len(DUMP_TOKENS),)
    end = MD["qwen4exp.ple.head_offsets"][-1] + MD["qwen4exp.ple.head_vocab_sizes"][-1]
    assert idx.min() >= 0 and idx.max() < end  # inside the 320001446 real rows
    # bigram head 0 of the first token has no predecessor (missing -> eos)
    m = (1 << 64) - 1
    mixed = ((DUMP_TOKENS[0] * 23703573157769) ^ (248044 * 20109073645365)) & m
    assert idx[0] == mixed % 20000003 + 0


@needs_model
def test_ple_embd_matches_dump_sample():
    idx = ple_row_indices(DUMP_TOKENS, [], **PLE_C)
    g = GGUFFile(MODEL)
    try:
        info = g.tensors[TABLE]
        assert info.type == "IQ4_NL" and info.shape[0] == 160
        emb = ple_gather_embedding(g, TABLE, idx, device="cpu")
    finally:
        g.close()
    assert emb.shape == (16 * len(DUMP_TOKENS), 160)
    ours = emb.view(len(DUMP_TOKENS), 16, 160).reshape(len(DUMP_TOKENS), 2560)

    nodes = json.loads((GOLDEN / "dump_nodes.json").read_text(encoding="utf-8"))["nodes"]
    node = next(n for n in nodes if n["name"] == "ple_embd")
    assert node["shape"][:2] == [2560, len(DUMP_TOKENS)]
    assert len(node["rows"]) == len(DUMP_TOKENS)
    o = ours.numpy().astype(np.float64)
    for t, row in enumerate(node["rows"]):
        np.testing.assert_allclose(o[t, :3], row[:3], atol=2e-4)
        np.testing.assert_allclose(o[t, -3:], row[3:], atol=2e-4)
    assert abs(o.sum() - node["sum"]) < 5e-3
