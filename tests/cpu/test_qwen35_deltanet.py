"""CPU unit tests for the Qwen35 gated DeltaNet layer (S21 step 2).

Tiny stub model (world=1, all layers recurrent): the real `Qwen35` methods
(`_layer_recurrent[_decode]`, `_ffn`, `forward`, `decode`, `reset`) run on
fake `torch.nn.Linear` weights and the `deltanet` Triton kernels under
TRITON_INTERPRET. Checks:

* the layer math against the vectorized CPU reference of S20 (conv_prefill +
  l2norm_qk + delta_prefill + norm_gated) on the same weights;
* chunked prefill vs per-token decode (states in the cache evolve the same);
* `reset()` re-zeroes conv history and delta state, and the model reproduces
  a fresh one after reset;
* the context limit raises.

TRITON_INTERPRET is set before the imports (S20 lesson: pytest collects
alphabetically, so any test file that can import Triton must enable the
interpreter itself)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import torch.nn.functional as F

import pytest

from pynference import deltanet
from pynference.models.gptoss import rms_norm
from pynference.models.qwen35 import Qwen35, Qwen35Config, _Layer

# Tiny shapes: H_k = 2, H_v = 4, S = 4 -> _qkv_l = (2*2+4)*4 = 32.
N_EMBD, N_KEY_L, N_VALUE_L, S = 16, 2, 4, 4
QKV_L, FFN, N_VOCAB, CONV_K = (2 * N_KEY_L + N_VALUE_L) * S, 32, 9, 4
EPS, MAX_CTX = 1e-6, 32


class TP1:
    world, rank, is_main = 1, 0, True
    device = torch.device("cpu")

    @staticmethod
    def split(n):
        return 0, n

    @staticmethod
    def all_reduce(t):
        return t

    @staticmethod
    def all_gather(t):
        return t


class _Embed:
    def __init__(self, emb):
        self.emb = emb

    def rows_dequant(self, idx):
        return self.emb[idx]


def _lin(n, k, scale=0.05):
    lin = torch.nn.Linear(k, n, bias=False)
    with torch.no_grad():
        lin.weight.normal_(0, scale)
    return lin


def _make_model(seed=0, n_layer=3):
    torch.manual_seed(seed)
    m = Qwen35.__new__(Qwen35)
    m.cfg = Qwen35Config(
        n_layer=n_layer, n_embd=N_EMBD, n_head=2, n_head_kv=1, head_dim=4,
        n_rot=4, ffn=FFN, rope_base=1e7, eps=EPS, n_vocab=N_VOCAB,
        n_ctx_train=64, is_attn=[False] * n_layer, conv_kernel=CONV_K,
        ssm_state=S, n_key_heads=N_KEY_L, n_value_heads=N_VALUE_L)
    m.tp, m.max_ctx, m.kv_dtype = TP1(), MAX_CTX, torch.float16
    m.n_head_l, m.n_kv_l = 2, 1
    m.n_key_l, m.n_value_l, m._qkv_l = N_KEY_L, N_VALUE_L, QKV_L
    m.layers = []
    for _ in range(n_layer):
        L = _Layer()
        L.is_attn = False
        L.attn_norm = torch.ones(N_EMBD)
        L.post_attn_norm = torch.ones(N_EMBD)
        L.ffn_gate, L.ffn_up = _lin(FFN, N_EMBD), _lin(FFN, N_EMBD)
        L.ffn_down = _lin(N_EMBD, FFN)
        L.qkv, L.gate = _lin(QKV_L, N_EMBD), _lin(N_VALUE_L * S, N_EMBD)
        L.alpha, L.beta = _lin(N_VALUE_L, N_EMBD), _lin(N_VALUE_L, N_EMBD)
        L.conv = torch.randn(QKV_L, CONV_K) * 0.3
        L.ssm_a = -torch.rand(N_VALUE_L) * 2 - 0.1   # -exp(A_log) < 0 (Mappa A.4)
        L.dt_bias = torch.randn(N_VALUE_L)
        L.ssm_norm = torch.ones(S)
        L.ssm_out = _lin(N_EMBD, N_VALUE_L * S)
        m.layers.append(L)
    m.v0, m.v1 = 0, N_VOCAB
    m.embed = _Embed(torch.randn(N_VOCAB, N_EMBD) * 0.5)
    m.lm_head = _lin(N_VOCAB, N_EMBD)
    m.output_norm = torch.ones(N_EMBD)
    m.cos = m.sin = None
    m.fast_decode, m.use_graph = True, False
    m._weights_bytes = m._scratch_bytes = 0
    m.reset()
    return m


def _ref_forward(m, tokens, cache):
    """Whole-sequence reference with the vectorized S20 pieces (Mappa A):
    conv_prefill -> l2norm q/k -> delta_prefill -> norm_gated -> ssm_out,
    then the FFN; updates `cache` in place like the kernels do."""
    x = m.embed.emb[torch.tensor(tokens)]
    for il, L in enumerate(m.layers):
        cs, st = cache[il]
        h = rms_norm(x, L.attn_norm, EPS)
        qkv, z = L.qkv(h), L.gate(h)
        alpha, beta = L.alpha(h), torch.sigmoid(L.beta(h))
        g = L.ssm_a * F.softplus(alpha + L.dt_bias)
        y, new_cs = deltanet.conv_prefill(qkv, cs, L.conv)
        T = y.shape[0]
        q = deltanet.l2norm_qk(y[:, :N_KEY_L * S].view(T, N_KEY_L, S), EPS)
        k = deltanet.l2norm_qk(y[:, N_KEY_L * S:2 * N_KEY_L * S].view(T, N_KEY_L, S), EPS)
        v = y[:, 2 * N_KEY_L * S:].view(T, N_VALUE_L, S)
        out, new_st = deltanet.delta_prefill(q, k, v, g, beta, st)
        cs.copy_(new_cs)
        st.copy_(new_st)
        y = deltanet.norm_gated(out, z.view(T, N_VALUE_L, S), L.ssm_norm, EPS)
        x = x + L.ssm_out(y.reshape(T, -1))
        h = rms_norm(x, L.post_attn_norm, EPS)
        x = x + L.ffn_down(F.silu(L.ffn_gate(h)) * L.ffn_up(h))
    return m.lm_head(rms_norm(x, m.output_norm, EPS))


def test_recurrent_layer_matches_reference():
    m = _make_model()
    tokens = list(range(1, 9))
    ref_cache = [(a.clone(), b.clone()) for a, b in m.cache]
    ref = _ref_forward(m, tokens, ref_cache)[-1]
    ours = m.forward(tokens, chunk=3)
    torch.testing.assert_close(ours, ref, rtol=2e-4, atol=2e-4)
    for (rc, rs), (mc, ms) in zip(ref_cache, m.cache):
        torch.testing.assert_close(mc, rc, rtol=1e-4, atol=1e-5)   # conv history
        torch.testing.assert_close(ms, rs, rtol=1e-4, atol=1e-5)   # delta state


def test_chunked_prefill_then_decode_matches_full_prefill():
    m1 = _make_model()
    m2 = _make_model()
    tokens = list(range(1, 13))
    m1.forward(tokens[:4], chunk=4)
    for t in tokens[4:]:
        ours = m1.decode(t)
    full = m2.forward(tokens, chunk=12)
    torch.testing.assert_close(ours, full, rtol=2e-4, atol=2e-4)
    for (c1, s1), (c2, s2) in zip(m1.cache, m2.cache):
        torch.testing.assert_close(c1, c2, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(s1, s2, rtol=1e-4, atol=1e-5)


def test_single_token_forward_uses_decode_path():
    m1 = _make_model()
    m2 = _make_model()
    ours = m1.forward([5])                 # fast_decode -> decode()
    full = m2.forward([5], all_logits=True)[0]
    assert m1.n_past == 1 and m2.n_past == 1
    torch.testing.assert_close(ours, full, rtol=1e-5, atol=1e-5)


def test_reset_zeroes_states_and_reproduces_fresh_run():
    m = _make_model()
    tokens = list(range(1, 7))
    before = m.forward(tokens)
    for c, s in m.cache:
        assert not torch.all(c == 0)
        assert not torch.all(s == 0)
    m.reset()
    assert m.n_past == 0
    for c, s in m.cache:
        assert torch.all(c == 0) and torch.all(s == 0)
    after = m.forward(tokens)
    torch.testing.assert_close(after, before)


def test_kv_bytes_counts_recurrent_states():
    m = _make_model()
    per_layer = (CONV_K - 1) * QKV_L * 4 + N_VALUE_L * S * S * 4
    assert m.kv_bytes() == per_layer * len(m.layers)


def test_context_limit():
    m = _make_model()
    with pytest.raises(ValueError, match="context full"):
        m.forward(list(range(1, MAX_CTX + 2)))
    m.forward(list(range(1, MAX_CTX + 1)))
    with pytest.raises(ValueError, match="context full"):
        m.decode(1)
