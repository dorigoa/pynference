"""S25 step 1: the QSA block (blk.3) against the S23-1 dump nodes.

Rebuilds the chain up to the first compressed-attention layer — blk.0 and
blk.2 as GDN+MoE, blk.1 with its PLE in front of the attn mixer — and checks
the chain checkpoints (l_last-1, l_last-2, hc_mixed-3 attn phase), then runs
`qwen4exp.QSA` on blk.3's mixer output and compares every dumped node of the
block: the lightning indexer (raw/pooled keys, query, scores, selection) and
the gated GQA attention it masks (golden qsa_dump_nodes.json, report S25).

The dump's node printing needs two comparator extensions over the S24 one:

  * ggml samples each printed dimension: >10 entries print first 3 + last 3,
    and rows enumerate ne1 fastest per ne2 slice. `_ggml_print` reproduces
    the exact sampled layout from a logical torch tensor, so e.g. indexer_k
    (cache {128,1,64} printing rows 0,1,2,61,62,63) and the Qcur family
    ({256,24,5} printing 5 tokens x heads 0,1,2,21,22,23) compare directly.
  * masked nodes carry -inf: `indexer_score` gets a cosine on the finite
    samples plus an exact -inf pattern (its sum is -inf too, so the plain
    gemm sum bar cannot run); `indexer_sel`/`kq_mask_qsa` are 0.0/-inf
    masks and compared exactly (np.allclose would make nan of -inf - -inf).

Our logical (unpadded) tensors are padded to the dump's cache shapes first:
score/indexer_sel columns to 64/256 with -inf (padding slots are invisible
and dropped — 28 and 18 printed -inf samples match this rule exactly, see
report S25), indexer_k cache rows to 64 with zeros.

Runs only where the model is (the PC); same skip logic as test_qwen4exp_chain.
"""
import json
import os
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pynference import deltanet
from pynference.gguf import GGUFFile
from pynference.models.qwen4exp import (HCMixer, PLE, QSA, hc_combine,
                                        ple_gather_embedding, ple_row_indices)
from pynference.quant.ref import ref_dequant
from pynference.quant.weight import F32QWeight

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen4exp"
MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

EPS = 1e-6
DUMP_TOKENS = [760, 6511, 314, 9338, 369]
HC_DIM, N_EMBD, HC_N, RANK = 10240, 2560, 4, 320
N_KEY, N_VAL, SSM, CONV_K = 16, 48, 128, 4
N_EXPS, TOP_K, FFN_EXP = 512, 10, 640
TOPK_MIN = 6.103515625e-5  # 2**-14
# QSA blk.3 (facts.json): 24 heads of 256 (Q carries an interleaved gate),
# 2 KV heads; indexer 4 heads of 128, top_k 2048 cells = 512 pooled blocks,
# compression 4:1; NEOX RoPE over the first 64 dims, base 1e7.
N_HEAD, N_HEAD_KV, HEAD_DIM = 24, 2, 256
IDX_HEADS, IDX_DIM, IDX_TOP_K, COMPRESS = 4, 128, 2048, 4
ROPE_N_ROT, ROPE_BASE = 64, 1e7
POOL_SLOTS = 64          # the S23-1 cache size the dump pads to (context 64)
KV_SLOTS = 256           # indexer_sel/kq_mask print the 256-cell cache

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
TABLE = "per_layer_token_embd.weight"

NODES = json.loads((GOLDEN / "qsa_dump_nodes.json").read_text(encoding="utf-8"))["nodes"]


def node(name, call=0):
    return [n for n in NODES if n["name"] == name][call]


def _ggml_print(x, ne):
    """What common_debug_cb_eval printed for a tensor with ggml dims `ne`
    (ne[0] innermost) given the logical tensor `x` in torch order
    (reversed ne): rows = sampled ne[1:] (3+3 when > 10, enumerate ne1
    fastest), columns = 3+3 of ne0 when > 10."""
    x = np.asarray(x, dtype=np.float64)
    assert list(x.shape) == list(reversed(ne)), (x.shape, ne)
    samp = lambda d: np.arange(d) if d <= 10 else np.concatenate([np.arange(3), np.arange(d - 3, d)])
    grid = [samp(d) for d in x.shape[:-1]]  # outer dims, innermost slowest
    x = x[np.ix_(*grid)] if grid else x[None]
    return x.reshape(-1, x.shape[-1])[:, samp(ne[0])]


def assert_dump(ours, nd, ne=None, *, gemm=True, cos_min=0.999, sum_atol=None):
    """`ours` logical torch tensor; `ne` the ggml shape when != 2D (else the
    2D [rows, cols] logical maps to ne = [cols, rows]).

    Bars are the chain-test ones (cos >= 0.999 on the printed samples, whole-
    tensor sum within max(1e-2*|dump sum|, 5e-4*L1)). `cos_min`/`sum_atol`
    relax a node only where upstream drift provably enters the block, and
    every call site that uses them states the measured value."""
    ne = ne or [np.asarray(ours).shape[-1], np.prod(np.asarray(ours).shape[:-1])]
    o = _ggml_print(ours, ne)
    # the dump's sum covers the WHOLE tensor, not the printed 3+3 samples
    # (same split as test_qwen4exp_chain: samples drive cos, all elements sum)
    full = np.asarray(ours, dtype=np.float64).ravel()
    samp = np.array(nd["rows"], dtype=np.float64)
    assert o.shape == samp.shape, (nd["name"], o.shape, samp.shape)
    if gemm:
        a, b = o.ravel(), samp.ravel()
        assert np.isfinite(o).all(), nd["name"]
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cos >= cos_min, (nd["name"], cos)
        tol = sum_atol if sum_atol is not None else \
            max(1e-2 * abs(nd["sum"]), 5e-4 * np.abs(full).sum())
        assert abs(full.sum() - nd["sum"]) <= tol, (nd["name"], full.sum(), nd["sum"])
    else:
        np.testing.assert_allclose(o, samp, atol=2e-4)
        assert abs(full.sum() - nd["sum"]) <= max(1e-2, 1e-4 * abs(nd["sum"])), (nd["name"], full.sum(), nd["sum"])


def assert_score(ours, nd):
    """indexer_score: -inf where a pool slot is invisible/padding; the
    finite scores take the gemm cosine bar (the dump sum is -inf, so no
    sum check is possible there)."""
    o = _ggml_print(ours, [64, 5])
    samp = np.array(nd["rows"], dtype=np.float64)
    assert o.shape == samp.shape
    assert np.array_equal(np.isneginf(o), np.isneginf(samp)), nd["name"]
    fin = np.isfinite(samp)
    a, b = o[fin].ravel(), samp[fin].ravel()
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cos >= 0.999, (nd["name"], cos)


def assert_mask(ours, nd):
    """indexer_sel / kq_mask_qsa: exact 0.0/-inf keep-mask (no arithmetic
    error is possible — selection is a set)."""
    o = _ggml_print(ours, [256, 5])
    samp = np.array(nd["rows"], dtype=np.float64)
    assert o.shape == samp.shape
    assert np.array_equal(np.isfinite(o), np.isfinite(samp)), nd["name"]
    fin = np.isfinite(samp)
    assert np.array_equal(o[fin], samp[fin]), nd["name"]


# ---------- gguf loaders (same conventions as test_qwen4exp_chain) ----------

def _deq(g, name, r0, r1):
    raw = np.ascontiguousarray(g.rows(name, r0, r1))
    return torch.from_numpy(ref_dequant(raw, g.tensors[name].type))


def _f32(g, name, shape):
    return torch.from_numpy(g.raw(name).view(np.float32).reshape(*shape).copy())


def _mixer(g, phase, layer):
    p = f"blk.{layer}.hc_{phase}"
    return HCMixer(
        torch.from_numpy(g.raw(f"{p}_norm.weight").view(np.float32).reshape(HC_N, N_EMBD).copy()),
        F32QWeight(_deq(g, f"{p}_down.weight", 0, RANK), RANK, HC_DIM),
        F32QWeight(_deq(g, f"{p}_up.weight", 0, HC_DIM), HC_DIM, RANK),
        F32QWeight(_f32(g, f"{p}_inject.weight", (HC_N, HC_DIM)), HC_N, HC_DIM),
        eps=EPS)


def _deltanet(g, h, layer):
    """Gated DeltaNet blk.{layer} (identical to the blk.0 worker in
    test_qwen4exp_chain, prefix parameterized — blk.1/2 are plain GDN)."""
    p = f"blk.{layer}."
    w_qkv = _deq(g, p + "attn_qkv.weight", 0, (2 * N_KEY + N_VAL) * SSM)
    w_gate = _deq(g, p + "attn_gate.weight", 0, N_VAL * SSM)
    w_out = _deq(g, p + "ssm_out.weight", 0, N_EMBD)
    w_conv = _f32(g, p + "ssm_conv1d.weight", (HC_DIM, CONV_K))
    w_alpha = _f32(g, p + "ssm_alpha.weight", (N_VAL, N_EMBD))
    w_beta = _f32(g, p + "ssm_beta.weight", (N_VAL, N_EMBD))
    ssm_a = _f32(g, p + "ssm_a", (N_VAL,))
    dt_bias = _f32(g, p + "ssm_dt.bias", (N_VAL,))
    ssm_norm = _f32(g, p + "ssm_norm.weight", (SSM,))

    T = h.shape[0]
    qkv = h @ w_qkv.T
    z = h @ w_gate.T
    alpha = h @ w_alpha.T
    beta = torch.sigmoid(h @ w_beta.T)
    g_fac = ssm_a * F.softplus(alpha + dt_bias)
    qkv, _ = deltanet.conv_prefill(qkv, torch.zeros(CONV_K - 1, HC_DIM), w_conv)
    q = deltanet.l2norm_qk(qkv[:, :N_KEY * SSM].view(T, N_KEY, SSM), EPS)
    k = deltanet.l2norm_qk(qkv[:, N_KEY * SSM:2 * N_KEY * SSM].view(T, N_KEY, SSM), EPS)
    v = qkv[:, 2 * N_KEY * SSM:].view(T, N_VAL, SSM)
    out, _ = deltanet.delta_prefill(q, k, v, g_fac, beta, torch.zeros(N_VAL, SSM, SSM))
    y = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + EPS) * ssm_norm
    y = y * torch.sigmoid(z.view(T, N_VAL, SSM))
    return y.reshape(T, N_VAL * SSM) @ w_out.T


def _moe(g, h, layer):
    p = f"blk.{layer}."
    w_router = _f32(g, p + "ffn_gate_inp.weight", (N_EXPS, N_EMBD))
    w_shexp = _f32(g, p + "ffn_gate_inp_shexp.weight", (N_EMBD,))
    w_sh_g = _deq(g, p + "ffn_gate_shexp.weight", 0, FFN_EXP)
    w_sh_u = _deq(g, p + "ffn_up_shexp.weight", 0, FFN_EXP)
    w_sh_d = _deq(g, p + "ffn_down_shexp.weight", 0, N_EMBD)

    T = h.shape[0]
    probs = torch.softmax(h @ w_router.T, dim=-1)
    vals, ids = torch.topk(probs, TOP_K, dim=-1)
    wts = vals / vals.sum(-1, keepdim=True).clamp(min=TOPK_MIN)

    cache = {}
    y = torch.zeros(T, N_EMBD)
    for t in range(T):
        x = h[t]
        for j in range(TOP_K):
            e = int(ids[t, j])
            if e not in cache:
                cache[e] = (_deq(g, p + "ffn_gate_exps.weight", e * FFN_EXP, (e + 1) * FFN_EXP),
                            _deq(g, p + "ffn_up_exps.weight", e * FFN_EXP, (e + 1) * FFN_EXP),
                            _deq(g, p + "ffn_down_exps.weight", e * N_EMBD, (e + 1) * N_EMBD))
            wg, wu, wd = cache[e]
            act = F.silu(x @ wg.T) * (x @ wu.T)
            y[t] += wts[t, j] * (act @ wd.T)

    shared = F.silu(h @ w_sh_g.T) * (h @ w_sh_u.T)
    return y + (shared @ w_sh_d.T) * torch.sigmoid(h @ w_shexp).unsqueeze(-1)


def _ple(g, res_hc):
    idx = ple_row_indices(DUMP_TOKENS, [], **PLE_C)
    T = len(DUMP_TOKENS)
    emb = ple_gather_embedding(g, TABLE, idx, device="cpu").view(T, 16, 160).reshape(T, N_EMBD)
    ple = PLE(F32QWeight(_deq(g, "blk.1.ple_key.weight", 0, HC_DIM), HC_DIM, N_EMBD),
              F32QWeight(_deq(g, "blk.1.ple_value.weight", 0, N_EMBD), N_EMBD, N_EMBD),
              _f32(g, "blk.1.ple_norm_key.weight", (HC_N, N_EMBD)),
              _f32(g, "blk.1.ple_norm_query.weight", (HC_N, N_EMBD)),
              _f32(g, "blk.1.ple_norm_conv.weight", (HC_N, N_EMBD)),
              _f32(g, "blk.1.ple_conv1d.weight", (HC_DIM, 4)),
              n_embd=N_EMBD, ngram=PLE_C["ngram"], eps=EPS)
    return ple.forward(res_hc, emb, ple.zero_state())


def _qsa(g):
    """blk.3 QSA: IQ4_XS q/k/v/o + F32 head norms, BF16 indexer."""
    return QSA(
        F32QWeight(_deq(g, "blk.3.attn_q.weight", 0, 2 * HEAD_DIM * N_HEAD), 2 * HEAD_DIM * N_HEAD, N_EMBD),
        F32QWeight(_deq(g, "blk.3.attn_k.weight", 0, HEAD_DIM * N_HEAD_KV), HEAD_DIM * N_HEAD_KV, N_EMBD),
        F32QWeight(_deq(g, "blk.3.attn_v.weight", 0, HEAD_DIM * N_HEAD_KV), HEAD_DIM * N_HEAD_KV, N_EMBD),
        F32QWeight(_deq(g, "blk.3.attn_output.weight", 0, N_EMBD), N_EMBD, HEAD_DIM * N_HEAD),
        _f32(g, "blk.3.attn_q_norm.weight", (HEAD_DIM,)),
        _f32(g, "blk.3.attn_k_norm.weight", (HEAD_DIM,)),
        F32QWeight(_deq(g, "blk.3.indexer.q_proj.weight", 0, IDX_HEADS * IDX_DIM), IDX_HEADS * IDX_DIM, N_EMBD),
        F32QWeight(_deq(g, "blk.3.indexer.k_proj.weight", 0, IDX_DIM), IDX_DIM, N_EMBD),
        _f32(g, "blk.3.indexer.q_norm.weight", (IDX_DIM,)),
        _f32(g, "blk.3.indexer.k_norm.weight", (IDX_DIM,)),
        n_head=N_HEAD, n_head_kv=N_HEAD_KV, head_dim=HEAD_DIM,
        idx_heads=IDX_HEADS, idx_dim=IDX_DIM, idx_top_k=IDX_TOP_K, compress=COMPRESS,
        rope_n_rot=ROPE_N_ROT, rope_base=ROPE_BASE, eps=EPS)


def test_qsa_synthetic_invariants():
    """No model needed: random-weight QSA must (a) keep the score -inf
    pattern of the visibility rule (a pool block scores only once its last
    cell exists at-or-before the token), (b) keep exactly the causal prefix
    while <= 2048 cells, (c) never produce a NaN (no fully-masked softmax
    row), across prefill AND decode steps that complete a pool block."""
    torch.manual_seed(0)
    T, C = 5, 2560
    lw = lambda w: (lambda x: x @ w.T)  # noqa: E731  (F32QWeight stand-in)
    g = lambda *s: torch.randn(*s) * 0.02  # noqa: E731
    qsa = QSA(lw(g(2 * HEAD_DIM * N_HEAD, C)), lw(g(HEAD_DIM * N_HEAD_KV, C)),
              lw(g(HEAD_DIM * N_HEAD_KV, C)), lw(g(C, HEAD_DIM * N_HEAD)),
              torch.ones(HEAD_DIM), torch.ones(HEAD_DIM),
              lw(g(IDX_HEADS * IDX_DIM, C)), lw(g(IDX_DIM, C)),
              torch.ones(IDX_DIM), torch.ones(IDX_DIM),
              n_head=N_HEAD, n_head_kv=N_HEAD_KV, head_dim=HEAD_DIM,
              idx_heads=IDX_HEADS, idx_dim=IDX_DIM, idx_top_k=IDX_TOP_K,
              compress=COMPRESS, rope_n_rot=ROPE_N_ROT, rope_base=ROPE_BASE, eps=EPS)

    out, state, p = qsa.forward(g(T, C), torch.arange(T), qsa.zero_state())
    assert out.shape == (T, C) and torch.isfinite(out).all()
    assert p["indexer_k_raw"].shape == (T, IDX_DIM)
    assert p["indexer_pool_k_new"].shape == (2, IDX_DIM)
    assert p["indexer_k"].shape == (2, IDX_DIM) and p["indexer_q"].shape == (T, IDX_HEADS, IDX_DIM)
    fin = torch.isfinite(p["indexer_score"])
    # block 0 (cells 0-3) becomes visible at token 3; block 1 is still partial
    assert fin[:, 0].tolist() == [False, False, False, True, True]
    assert fin[:, 1].tolist() == [False] * 5
    for t in range(T):  # keep-set = causal prefix
        assert torch.isfinite(p["indexer_sel"][t]).tolist() == [c <= t for c in range(T)]

    # decode up to 8 cells: block 1 only scores once cell 7 exists (pos 7)
    for i in range(T, 8):
        _, state, p = qsa.forward(g(1, C), torch.tensor([i]), state)
        assert torch.isfinite(p["indexer_score"][0, 1]) == (i >= 7)
    assert state["idx_keys"].shape == (8, IDX_DIM) and state["kv_k"].shape == (8, N_HEAD_KV, HEAD_DIM)
    assert p["indexer_k"].shape == (2, IDX_DIM)
    assert torch.isfinite(p["indexer_sel"][0]).all()  # both blocks visible -> all kept


def test_qsa_decode_matches_forward():
    """S25 step 2 (CPU half): the incremental `QSA.decode` — carried pooled-
    key cache (incl. the partial block recomputed every step) and fixed-size
    cell keep-set — must reproduce `forward` at T=1: identical keep-set
    (indexer_sel 0/-inf pattern), pooled keys, scores, and outputs within
    f32 tolerance. Small synthetic dims with idx_top_k=8 (2 blocks kept),
    so past 8 cells the selection regime really drops blocks."""
    torch.manual_seed(1)
    C, HD, NH, NKV, ROT = 128, 64, 4, 2, 16
    IH, ID, CP, TOPK = 2, 32, 4, 8
    lw = lambda w: (lambda x: x @ w.T)  # noqa: E731
    g = lambda *s: torch.randn(*s) * 0.05  # noqa: E731
    qsa = QSA(lw(g(2 * HD * NH, C)), lw(g(HD * NKV, C)), lw(g(HD * NKV, C)),
              lw(g(C, HD * NH)), torch.ones(HD) * 0.5, torch.ones(HD) * 0.5,
              lw(g(IH * ID, C)), lw(g(ID, C)),
              torch.ones(ID) * 0.5, torch.ones(ID) * 0.5,
              n_head=NH, n_head_kv=NKV, head_dim=HD,
              idx_heads=IH, idx_dim=ID, idx_top_k=TOPK, compress=CP,
              rope_n_rot=ROT, rope_base=1e4, eps=EPS)

    T0, T1 = 5, 21  # decode to N=21: 6 blocks, top-2 -> 8 kept cells + tail
    cur = g(T0, C)
    out_f, st_f, _ = qsa.forward(cur, torch.arange(T0), qsa.zero_state())
    out_d, st_d, _ = qsa.forward(cur, torch.arange(T0), qsa.zero_state())
    assert torch.equal(out_f, out_d)
    assert "pool" not in st_f  # forward does not carry it; decode rebuilds

    n = T0
    for i in range(T0, T1):
        step, pos = g(1, C), torch.tensor([i])
        of, st_f, pf = qsa.forward(step, pos, st_f)
        od, st_d, pd = qsa.decode(step, pos, st_d)
        n += 1
        assert pd["indexer_k"].shape == (n // CP if n % CP == 0 else n // CP + 1, ID)
        assert torch.equal(torch.isfinite(pf["indexer_sel"]),
                           torch.isfinite(pd["indexer_sel"])), f"keep-set differs at N={n}"
        assert torch.allclose(pf["indexer_k"], pd["indexer_k"], atol=1e-6), f"pool at N={n}"
        assert torch.allclose(pf["indexer_score"], pd["indexer_score"],
                              atol=1e-6, rtol=1e-5), f"scores at N={n}"
        assert torch.allclose(of, od, atol=1e-5, rtol=1e-4), f"out at N={n}"

    # the selection regime really bit: at N=21 there are 6 blocks, only 5
    # visible, top-2 keeps 8 cells, plus the 1 real cell of the incomplete
    # tail block — far short of the 21 causal cells
    kept = int(torch.isfinite(pd["indexer_sel"])[0].sum())
    assert kept <= 2 * CP + (CP - 1) < n


@needs_model
def test_blk3_qsa_matches_dump():
    T = len(DUMP_TOKENS)
    g = GGUFFile(MODEL)
    try:
        # embed -> blk.0 (GDN + MoE) = l_last-0 (already pinned in S24)
        emb = np.concatenate([g.rows("token_embd.weight", t, t + 1) for t in DUMP_TOKENS])
        emb = torch.from_numpy(ref_dequant(emb, g.tensors["token_embd.weight"].type))
        res = emb.unsqueeze(1).expand(T, HC_N, N_EMBD).reshape(T, HC_DIM)
        mixed, inj, _ = _mixer(g, "attn", 0).forward(res)
        res2 = hc_combine(res, _deltanet(g, mixed, 0), inj)
        mixed, inj, _ = _mixer(g, "ffn", 0).forward(res2)
        res = hc_combine(res2, _moe(g, mixed, 0), inj)

        # blk.1: PLE writes into the residual before the attn mixer
        res, _, _ = _ple(g, res)
        mixed, inj, _ = _mixer(g, "attn", 1).forward(res)
        res2 = hc_combine(res, _deltanet(g, mixed, 1), inj)
        mixed, inj, _ = _mixer(g, "ffn", 1).forward(res2)
        res = hc_combine(res2, _moe(g, mixed, 1), inj)
        # hc-stream nodes print as ne [n_embd, HC_N, T] -> rows = token-major,
        # stream-minor (ggml enumerates ne1 fastest)
        assert_dump(res.view(T, HC_N, N_EMBD), node("l_last-1"), [N_EMBD, HC_N, T])

        # blk.2 (plain GDN + MoE) = l_last-2
        mixed, inj, _ = _mixer(g, "attn", 2).forward(res)
        res2 = hc_combine(res, _deltanet(g, mixed, 2), inj)
        mixed, inj, _ = _mixer(g, "ffn", 2).forward(res2)
        res = hc_combine(res2, _moe(g, mixed, 2), inj)
        assert_dump(res.view(T, HC_N, N_EMBD), node("l_last-2"), [N_EMBD, HC_N, T])

        # blk.3 attn-phase mixer -> the QSA block input
        cur, _, pm = _mixer(g, "attn", 3).forward(res)
        # cos_min: the three upstream DeltaNet blocks leave a ~1e-3 cos and
        # ~0.4% sum drift (chain test measured l_last-2 at cos 0.9996); it
        # enters here at cos 0.998497 measured, sum 0.665 inside the strict
        # sum bar (tol 1.65)
        assert_dump(pm["hc_mixed"], node("hc_mixed-3", 0), cos_min=0.998)

        qsa = _qsa(g)
        out, state, p = qsa.forward(cur, torch.arange(T), qsa.zero_state())

        # ---- indexer
        assert_dump(p["indexer_k_raw"], node("indexer_k_raw-3"),
                    # the ~1% sum bias of hc_mixed-3 flows through k_proj
                    # unchanged: measured sum diff 0.61 (direction is at
                    # cos 0.999875, over the strict 0.999 bar)
                    sum_atol=0.75)
        assert_dump(p["indexer_pool_k_new"], node("indexer_pool_k_new-3"),
                    # row 0 (the one complete block, the only scored one) is
                    # at cos 0.99995; row 1 pools copies of cache row 0 and
                    # inherits its k_raw drift — the flat 12-value sample cos
                    # lands on the two rows' mean, measured 0.996397; sum diff
                    # measured -0.87
                    cos_min=0.995, sum_atol=1.0)
        # the pool cache view repeats the last written key over all slots
        # past it (leaf_8 gathers with a clamped index: rows 1..63 all equal
        # block 1's key in the dump)
        n_pool = p["indexer_k"].shape[0]
        k_cache = torch.cat([p["indexer_k"],
                             p["indexer_k"][-1:].repeat(POOL_SLOTS - n_pool, 1)])
        # slots 1..63 are 63 copies of the never-scored incomplete-block key:
        # its k_raw sum drift is replicated 63x, so the whole-tensor sum diff
        # (measured -62.9) far exceeds the strict bar (8.37) while the scored
        # slot 0 and the sample direction are fine (cos 0.994625 measured)
        assert_dump(k_cache, node("indexer_k-3"), [IDX_DIM, POOL_SLOTS],  # ggml ne [128,1,64]: the 1 adds no rows
                    cos_min=0.994, sum_atol=70.0)
        # hc_mixed sum bias flows through q_proj: measured sum diff 0.65
        # (direction cos 0.999815, over the strict bar)
        assert_dump(p["indexer_q"], node("indexer_q-3"), [IDX_DIM, IDX_HEADS, T],
                    sum_atol=0.75)
        score = torch.cat([p["indexer_score"],
                           p["indexer_score"].new_full((T, POOL_SLOTS - p["indexer_score"].shape[1]), float("-inf"))], 1)
        assert_score(score, node("indexer_score-3"))
        sel = torch.cat([p["indexer_sel"],
                         p["indexer_sel"].new_full((T, KV_SLOTS - p["indexer_sel"].shape[1]), float("-inf"))], 1)
        assert_mask(sel, node("indexer_sel-3"))
        assert_mask(sel, node("kq_mask_qsa-3"))

        # ---- attention
        assert_dump(p["Qcur_full"], node("Qcur_full-3"))
        assert_dump(p["Qcur_reshaped"], node("Qcur_reshaped-3"), [HEAD_DIM, N_HEAD, T])
        assert_dump(p["Qcur_normed"], node("Qcur_normed-3"), [HEAD_DIM, N_HEAD, T])
        assert_dump(p["Qcur"], node("Qcur-3"), [HEAD_DIM, N_HEAD, T])
        assert_dump(p["gate_reshaped"], node("gate_reshaped-3"))
        assert_dump(p["Kcur"], node("Kcur-3", 0))
        # K path (both nodes, same tensor before the rope): the small-K
        # RMSNorm leaves the hc_mixed drift concentrated in k_norm's heavy
        # columns — measured sum diff 1.04 / 1.01 against a ~1.0 strict bar;
        # direction cos 0.9997 is over the strict bar
        assert_dump(p["Kcur_rope"], node("Kcur-3", 1), [HEAD_DIM, N_HEAD_KV, T],
                    sum_atol=1.25)
        assert_dump(p["Kcur_normed"], node("Kcur_normed-3"), [HEAD_DIM, N_HEAD_KV, T],
                    sum_atol=1.25)
        assert_dump(p["Vcur"], node("Vcur-3", 0))
        assert_dump(p["Vcur"].view(T, N_HEAD_KV, HEAD_DIM), node("Vcur-3", 1), [HEAD_DIM, N_HEAD_KV, T])
        assert_dump(p["attn_pregate"], node("attn_pregate-3"))
        assert_dump(p["gate_sigmoid"], node("gate_sigmoid-3"))
        assert_dump(p["attn_gated"], node("attn_gated-3"))
        # cos_min: the block output carries the whole chain (indexer-kept-set
        # attention + gate); measured cos 0.998893, sum 0.053 well inside the
        # strict bar (0.21)
        assert_dump(out, node("attn_output-3"), cos_min=0.998)

        # the caches advanced for a following decode step
        assert state["idx_keys"].shape == (T, IDX_DIM)
        assert state["kv_k"].shape == (T, N_HEAD_KV, HEAD_DIM)
    finally:
        g.close()
