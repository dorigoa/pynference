"""CPU unit tests for the Qwen35 full attention layer (S21 step 3).

Same stub-model approach as test_qwen35_deltanet.py (world=1, torch.nn.Linear
"weights", TRITON_INTERPRET for `kernels.attn_decode`). Tiny shapes: 4 query
heads / 2 KV heads (GQA group 2, the local form of kv-head = h/6), head_dim 8
with partial RoPE n_rot=4 (dims 4..7 must pass through), q|gate interleaved
16 rows per head (the 512-row layout of Mappa B.2 at D scale).

Checks:
* the whole-sequence reference (naive causal attention, Mappa B order: q/k
  RMS norms before RoPE, sigmoid gate on the attention output before `o`)
  against the real `_layer_attn` path (two chunks, to exercise pos0), plus
  the fp16 KV cache holding the roped keys;
* chunked prefill + per-token decode (`kernels.attn_decode`) == full
  prefill, on a mixed attn/recurrent model;
* partial RoPE leaves dims n_rot..D untouched;
* the second half of each 2*D head block of `attn_q` is the gate: zeroing
  it makes the layer compute exactly sigmoid(0) = 0.5 gating.

TRITON_INTERPRET is set before the imports (collection-order contract)."""
import math
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import torch.nn.functional as F

from pynference import deltanet
from pynference.models.gptoss import rms_norm, rope_neox
from pynference.models.qwen35 import Qwen35, Qwen35Config, _Layer, rope_table

N_EMBD, HQ, HKV, D, N_ROT = 16, 4, 2, 8, 4
FFN, N_VOCAB, MAX_CTX, EPS, CONV_K, S = 32, 16, 64, 1e-6, 4, 4
QKV_L = (2 * 2 + 4) * S           # recurrent stub: H_k = 2, H_v = 4 value heads


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


def _make_model(seed=0, is_attn=(True, True, True)):
    torch.manual_seed(seed)
    m = Qwen35.__new__(Qwen35)
    m.cfg = Qwen35Config(
        n_layer=len(is_attn), n_embd=N_EMBD, n_head=HQ, n_head_kv=HKV,
        head_dim=D, n_rot=N_ROT, ffn=FFN, rope_base=1e7, eps=EPS,
        n_vocab=N_VOCAB, n_ctx_train=MAX_CTX, is_attn=list(is_attn),
        conv_kernel=CONV_K, ssm_state=S, n_key_heads=2, n_value_heads=4)
    m.tp, m.max_ctx, m.kv_dtype = TP1(), MAX_CTX, torch.float16
    m.n_head_l, m.n_kv_l = HQ, HKV
    m.n_key_l, m.n_value_l, m._qkv_l = 2, 4, QKV_L
    m._scale = 1.0 / math.sqrt(D)
    m._split_full = max(256, ((MAX_CTX + 63) // 64 + 15) // 16 * 16)
    m._nsp_full = (MAX_CTX + m._split_full - 1) // m._split_full
    m.layers = []
    for attn in is_attn:
        L = _Layer()
        L.is_attn = attn
        L.attn_norm = torch.ones(N_EMBD)
        L.post_attn_norm = torch.ones(N_EMBD)
        L.ffn_gate, L.ffn_up = _lin(FFN, N_EMBD), _lin(FFN, N_EMBD)
        L.ffn_down = _lin(N_EMBD, FFN)
        if attn:
            L.q = _lin(2 * HQ * D, N_EMBD)     # q|gate interleaved, Mappa B.2
            L.k, L.v = _lin(HKV * D, N_EMBD), _lin(HKV * D, N_EMBD)
            L.o = _lin(N_EMBD, HQ * D)
            L.q_norm, L.k_norm = torch.ones(D), torch.ones(D)
        else:
            L.qkv, L.gate = _lin(QKV_L, N_EMBD), _lin(4 * S, N_EMBD)
            L.alpha, L.beta = _lin(4, N_EMBD), _lin(4, N_EMBD)
            L.conv = torch.randn(QKV_L, CONV_K) * 0.3
            L.ssm_a = -torch.rand(4) * 2 - 0.1
            L.dt_bias = torch.randn(4)
            L.ssm_norm = torch.ones(S)
            L.ssm_out = _lin(N_EMBD, 4 * S)
        m.layers.append(L)
    m.v0, m.v1 = 0, N_VOCAB
    m.embed = _Embed(torch.randn(N_VOCAB, N_EMBD) * 0.5)
    m.lm_head = _lin(N_VOCAB, N_EMBD)
    m.output_norm = torch.ones(N_EMBD)
    m.cos, m.sin = rope_table(N_ROT, 1e7, MAX_CTX, torch.device("cpu"))
    m.fast_decode, m.use_graph = True, False
    m._weights_bytes = m._scratch_bytes = 0
    m.reset()
    return m


def _causal_attn(q, k, v, G, scale):
    """Naive causal GQA attention: q [T, HQ, D], k, v [T, HKV, D] -> [T, HQ*D]."""
    T, HQn = q.shape[0], q.shape[1]
    tril = torch.ones(T, T).bool().tril()
    out = torch.empty(T, HQn, q.shape[2])
    for hh in range(HQn):
        s = (q[:, hh] @ k[:, hh // G].T) * scale
        out[:, hh] = s.masked_fill(~tril, float("-inf")).softmax(-1) @ v[:, hh // G]
    return out.reshape(T, -1)


def _ref_attn(m, L, x):
    """Mappa B attention on x [T, n_embd]: norms before RoPE, naive causal
    attention, sigmoid gate before `o`. Returns (o(a), roped k [T, HKV, D])."""
    T = x.shape[0]
    h = rms_norm(x, L.attn_norm, EPS)
    qg = L.q(h).view(T, HQ, 2 * D)
    q, gate = qg[..., :D], qg[..., D:]
    q = rms_norm(q, L.q_norm, EPS)
    k = rms_norm(L.k(h).view(T, HKV, D), L.k_norm, EPS)
    v = L.v(h).view(T, HKV, D)
    q = rope_neox(q, m.cos[:T], m.sin[:T])
    k = rope_neox(k, m.cos[:T], m.sin[:T])
    a = _causal_attn(q, k, v, HQ // HKV, m._scale) * torch.sigmoid(gate).reshape(T, -1)
    return L.o(a), k


def _ref_forward(m, tokens):
    """Whole-sequence reference (fresh DeltaNet states: valid for the first
    forward only, no reset in between)."""
    x = m.embed.emb[torch.tensor(tokens)]
    for L in m.layers:
        if L.is_attn:
            blk, _ = _ref_attn(m, L, x)
        else:
            T = x.shape[0]
            h = rms_norm(x, L.attn_norm, EPS)
            qkv, z = L.qkv(h), L.gate(h)
            g = L.ssm_a * F.softplus(L.alpha(h) + L.dt_bias)
            beta = torch.sigmoid(L.beta(h))
            y, _ = deltanet.conv_prefill(qkv, torch.zeros(CONV_K - 1, QKV_L), L.conv)
            q = deltanet.l2norm_qk(y[:, :2 * S].view(T, 2, S), EPS)
            k = deltanet.l2norm_qk(y[:, 2 * S:4 * S].view(T, 2, S), EPS)
            v = y[:, 4 * S:].view(T, 4, S)
            o, _ = deltanet.delta_prefill(q, k, v, g, beta, torch.zeros(4, S, S))
            y = deltanet.norm_gated(o, z.view(T, 4, S), L.ssm_norm, EPS)
            blk = L.ssm_out(y.reshape(T, -1))
        x = x + blk
        h = rms_norm(x, L.post_attn_norm, EPS)
        x = x + L.ffn_down(F.silu(L.ffn_gate(h)) * L.ffn_up(h))
    return m.lm_head(rms_norm(x, m.output_norm, EPS))


def test_attn_layer_matches_reference_and_fills_cache():
    m = _make_model()
    tokens = list(range(1, 11))
    ref = _ref_forward(m, tokens)[-1]
    ours = m.forward(tokens, chunk=5)          # two chunks: exercises pos0
    torch.testing.assert_close(ours, ref, rtol=2e-4, atol=2e-4)
    # layer 0's fp16 KV cache holds the roped keys at the token positions
    x = m.embed.emb[torch.tensor(tokens)]
    _, k = _ref_attn(m, m.layers[0], x)
    kc, vc = m.cache[0]
    torch.testing.assert_close(kc[:, :len(tokens)].float(), k.transpose(0, 1),
                               rtol=1e-3, atol=1e-4)


def test_chunked_prefill_then_decode_matches_full_prefill():
    tokens = list(range(1, 14))
    m1 = _make_model()                          # attn + recurrent + attn
    m2 = _make_model()
    m1.forward(tokens[:5], chunk=5)
    for t in tokens[5:]:
        ours = m1.decode(t)                     # kernels.attn_decode path
    full = m2.forward(tokens, chunk=14)
    torch.testing.assert_close(ours, full, rtol=2e-4, atol=2e-4)
    for L, e1, e2 in zip(m1.layers, m1.cache, m2.cache):
        if L.is_attn:
            for t1, t2 in zip(e1, e2):          # both caches in fp16: exact
                torch.testing.assert_close(t1, t2)


def test_partial_rope_passes_tail_dims():
    """n_rot=4 < D=8: dims 4..7 of roped q/k must equal the pre-RoPE values
    (the MROPE text collapse of Mappa B.4)."""
    x = torch.randn(2, HQ, D)
    r = rope_neox(x, torch.ones(2, N_ROT // 2), torch.zeros(2, N_ROT // 2))
    torch.testing.assert_close(r[..., N_ROT:], x[..., N_ROT:])


def test_gate_is_the_second_half_of_each_head_block():
    """Zeroing the gate rows of `attn_q` (rows D..2D of each head) must make
    the layer compute exactly sigmoid(0) = 0.5 gating - a swapped split would
    zero q instead (uniform attention, a different result)."""
    m = _make_model(is_attn=(True,))
    with torch.no_grad():
        m.layers[0].q.weight.view(HQ, 2 * D, -1)[:, D:, :].zero_()
    tokens = list(range(1, 9))
    ours = m.forward(tokens)
    x = m.embed.emb[torch.tensor(tokens)]
    L = m.layers[0]
    h = rms_norm(x, L.attn_norm, EPS)
    q = L.q(h).view(len(tokens), HQ, 2 * D)[..., :D]
    q = rms_norm(q, L.q_norm, EPS)
    k = rms_norm(L.k(h).view(len(tokens), HKV, D), L.k_norm, EPS)
    v = L.v(h).view(len(tokens), HKV, D)
    q = rope_neox(q, m.cos[:len(tokens)], m.sin[:len(tokens)])
    k = rope_neox(k, m.cos[:len(tokens)], m.sin[:len(tokens)])
    a = 0.5 * _causal_attn(q, k, v, HQ // HKV, m._scale)
    xx = x + L.o(a)
    h2 = rms_norm(xx, L.post_attn_norm, EPS)
    xx = xx + L.ffn_down(F.silu(L.ffn_gate(h2)) * L.ffn_up(h2))
    ref = m.lm_head(rms_norm(xx, m.output_norm, EPS))[-1]
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
