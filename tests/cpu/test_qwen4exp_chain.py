"""S24 step-4 worker: the whole blk.0 chain against the S23-1 dump nodes.

The dump prints hc_init and the attn-phase mixer nodes directly; everything
downstream needs the full `res_hc`, so this worker reconstructs blk.0 with
existing modules on torch CPU: embed (IQ4_XS) -> HC attn mixer -> gated
DeltaNet (`pynference/deltanet.py` functions, world = 1; Flash-Next has no
attn_norm — the mixer's `mixed` is the block input) -> hc_combine -> HC ffn
mixer -> MoE (router softmax top-10, renorm denominator clamped at 2**-14,
plain SwiGLU, scalar-sigmoid-gated shared expert) -> hc_combine = l_last-0
-> the blk.1 PLE block (gathered table rows, zero conv state).

Every dumped node is checked with `assert_matches_dump`: nodes downstream of
llama.cpp's first Q8_0 mul_mat get the plan's bar (cos >= 0.999 on the
samples + sum within 1%, floored at 5e-4 of the L1), F32-only nodes print
precision (report S24,
Valori). Runs only where the model is (`/nvme`, the PC): under
`scripts/cpu.sh` and inside the GPU job alike — no XPU device is needed,
so this lives in tests/cpu (the tests/gpu files skip without a GPU).

See report S24 for the strategy and per-node tolerance choices.
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
from pynference.models.qwen4exp import (HCMixer, PLE, hc_combine,
                                        ple_gather_embedding, ple_row_indices)
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
# blk.0 DeltaNet (Mappa A): 16 key / 48 value heads of 128, conv kernel 4.
N_KEY, N_VAL, SSM, CONV_K = 16, 48, 128, 4
# MoE blk.0: 512 experts, top-10, 640 intermediate, 1 shared expert.
N_EXPS, TOP_K, FFN_EXP = 512, 10, 640
TOPK_MIN = 6.103515625e-5  # 2**-14, llama.cpp moe renorm clamp

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

NODES = json.loads((GOLDEN / "dump_nodes.json").read_text(encoding="utf-8"))["nodes"]


def node(name, call=0):
    """The call-th printed occurrence of `name` (attn pass first)."""
    return [n for n in NODES if n["name"] == name][call]


def assert_matches_dump(ours, nd, *, gemm=False):
    """Compare an f32 tensor ([T, C] or [T*hc, C]) against a golden node: the
    printed 3+3 samples per row (or the whole innermost dim when the dump
    printed it) plus the full-tensor sum. `gemm=True` = the plan's S24 bar
    for nodes that passed a Q8_0 mul_mat inside llama.cpp (same helper as
    test_qwen4exp_hc.py)."""
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
        # Sum bar: 1% relative, floored at 5e-4 of the full-tensor L1. The
        # floor matters when the dump's total nearly cancels or upstream
        # accumulation noise compounds (measured on the chain: linear_attn_out
        # diff 1.4e-4 of L1 at cos 0.9999; the ffn-phase mixer inherits the
        # DeltaNet drift and lands at ~2.5e-4). The plan's acceptance bar is
        # the cosine; the sum only has to catch structural errors, and 5e-4
        # of L1 is well below any of those.
        tol = max(1e-2 * abs(nd["sum"]), 5e-4 * np.abs(o).sum())
        assert abs(o.sum() - nd["sum"]) <= tol, (nd["name"], o.sum(), nd["sum"])
    else:
        np.testing.assert_allclose(ours_s, samp, atol=2e-4)
        assert abs(o.sum() - nd["sum"]) <= max(1e-2, 1e-4 * abs(nd["sum"])), (nd["name"], o.sum(), nd["sum"])


# ---------- gguf loaders ----------

def _deq(g, name, r0, r1):
    """Rows [r0, r1) dequantized through the CPU reference: torch [r1-r0, ne0]."""
    raw = np.ascontiguousarray(g.rows(name, r0, r1))
    return torch.from_numpy(ref_dequant(raw, g.tensors[name].type))


def _f32(g, name, shape):
    return torch.from_numpy(g.raw(name).view(np.float32).reshape(*shape).copy())


def _mixer(g, phase):
    """Real blk.0 `hc_{phase}` mixer, same as test_qwen4exp_hc._real_mixer."""
    p = f"blk.0.hc_{phase}"
    return HCMixer(
        torch.from_numpy(g.raw(f"{p}_norm.weight").view(np.float32).reshape(HC_N, N_EMBD).copy()),
        F32QWeight(_deq(g, f"{p}_down.weight", 0, RANK), RANK, HC_DIM),
        F32QWeight(_deq(g, f"{p}_up.weight", 0, HC_DIM), HC_DIM, RANK),
        F32QWeight(_f32(g, f"{p}_inject.weight", (HC_N, HC_DIM)), HC_N, HC_DIM),
        eps=EPS)


def _deltanet(g, h):
    """Gated DeltaNet blk.0 over the prefill chunk h [T, 2560] (fresh
    sequence: zero conv history and delta state), qwen35._layer_recurrent
    with world = 1 and no attn_norm (HC replaces it)."""
    w_qkv = _deq(g, "blk.0.attn_qkv.weight", 0, (2 * N_KEY + N_VAL) * SSM)
    w_gate = _deq(g, "blk.0.attn_gate.weight", 0, N_VAL * SSM)
    w_out = _deq(g, "blk.0.ssm_out.weight", 0, N_EMBD)
    w_conv = _f32(g, "blk.0.ssm_conv1d.weight", (HC_DIM, CONV_K))  # [channel, tap]
    w_alpha = _f32(g, "blk.0.ssm_alpha.weight", (N_VAL, N_EMBD))
    w_beta = _f32(g, "blk.0.ssm_beta.weight", (N_VAL, N_EMBD))
    ssm_a = _f32(g, "blk.0.ssm_a", (N_VAL,))       # holds -exp(A_log) (A.4)
    dt_bias = _f32(g, "blk.0.ssm_dt.bias", (N_VAL,))
    ssm_norm = _f32(g, "blk.0.ssm_norm.weight", (SSM,))

    T = h.shape[0]
    qkv = h @ w_qkv.T                              # [T, 10240]
    z = h @ w_gate.T                               # [T, 6144]
    alpha = h @ w_alpha.T                          # [T, 48]
    beta = torch.sigmoid(h @ w_beta.T)             # Mappa A.3
    g_fac = ssm_a * F.softplus(alpha + dt_bias)
    qkv, _ = deltanet.conv_prefill(qkv, torch.zeros(CONV_K - 1, HC_DIM), w_conv)
    q = deltanet.l2norm_qk(qkv[:, :N_KEY * SSM].view(T, N_KEY, SSM), EPS)
    k = deltanet.l2norm_qk(qkv[:, N_KEY * SSM:2 * N_KEY * SSM].view(T, N_KEY, SSM), EPS)
    v = qkv[:, 2 * N_KEY * SSM:].view(T, N_VAL, SSM)
    out, _ = deltanet.delta_prefill(q, k, v, g_fac, beta,
                                    torch.zeros(N_VAL, SSM, SSM))
    # Flash-Next gate is sigmoid(z), not qwen35's silu(z) (grid vs the dump:
    # cos 0.99993 vs 0.84168; deltanet.norm_gated is the silu variant).
    y = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + EPS) * ssm_norm
    y = y * torch.sigmoid(z.view(T, N_VAL, SSM))
    return y.reshape(T, N_VAL * SSM) @ w_out.T     # [T, 2560]


def _moe(g, h):
    """MoE blk.0 (Mappa C): 512-expert softmax router top-10 renormalized
    with a 2**-14 floor on the denominator, plain SwiGLU experts, plus the
    640-unit shared expert gated by one scalar sigmoid per token. Only the
    selected experts are dequantized."""
    w_router = _f32(g, "blk.0.ffn_gate_inp.weight", (N_EXPS, N_EMBD))
    w_shexp = _f32(g, "blk.0.ffn_gate_inp_shexp.weight", (N_EMBD,))
    w_sh_g = _deq(g, "blk.0.ffn_gate_shexp.weight", 0, FFN_EXP)
    w_sh_u = _deq(g, "blk.0.ffn_up_shexp.weight", 0, FFN_EXP)
    w_sh_d = _deq(g, "blk.0.ffn_down_shexp.weight", 0, N_EMBD)

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
            if e not in cache:  # rows: gate/up [640, 2560], down [2560, 640]
                cache[e] = (_deq(g, "blk.0.ffn_gate_exps.weight", e * FFN_EXP, (e + 1) * FFN_EXP),
                            _deq(g, "blk.0.ffn_up_exps.weight", e * FFN_EXP, (e + 1) * FFN_EXP),
                            _deq(g, "blk.0.ffn_down_exps.weight", e * N_EMBD, (e + 1) * N_EMBD))
            wg, wu, wd = cache[e]
            act = F.silu(x @ wg.T) * (x @ wu.T)
            y[t] += wts[t, j] * (act @ wd.T)

    shared = F.silu(h @ w_sh_g.T) * (h @ w_sh_u.T)
    return y + (shared @ w_sh_d.T) * torch.sigmoid(h @ w_shexp).unsqueeze(-1)


def _ple(g, res_hc):
    """blk.1 PLE on the fresh sequence (zero conv history), real Q8_0
    key/value + F32 norms/conv (same conventions as tests/gpu)."""
    idx = ple_row_indices(DUMP_TOKENS, [], **PLE_C)
    T = len(DUMP_TOKENS)
    emb = ple_gather_embedding(g, TABLE, idx, device="cpu").view(T, 16, 160).reshape(T, N_EMBD)
    ple = PLE(F32QWeight(_deq(g, "blk.1.ple_key.weight", 0, HC_DIM), HC_DIM, N_EMBD),
              F32QWeight(_deq(g, "blk.1.ple_value.weight", 0, N_EMBD), N_EMBD, N_EMBD),
              _f32(g, "blk.1.ple_norm_key.weight", (HC_N, N_EMBD)),
              _f32(g, "blk.1.ple_norm_query.weight", (HC_N, N_EMBD)),
              _f32(g, "blk.1.ple_norm_conv.weight", (HC_N, N_EMBD)),
              _f32(g, "blk.1.ple_conv1d.weight", (HC_DIM, 4)),  # tap last
              n_embd=N_EMBD, ngram=PLE_C["ngram"], eps=EPS)
    return ple.forward(res_hc, emb, ple.zero_state())


@needs_model
def test_blk0_chain_matches_dump():
    T = len(DUMP_TOKENS)
    g = GGUFFile(MODEL)
    try:
        # embed -> res_hc (the 4 streams all start at the token embedding)
        emb = np.concatenate([g.rows("token_embd.weight", t, t + 1) for t in DUMP_TOKENS])
        emb = torch.from_numpy(ref_dequant(emb, g.tensors["token_embd.weight"].type))
        res = emb.unsqueeze(1).expand(T, HC_N, N_EMBD).reshape(T, HC_DIM)
        assert_matches_dump(res, node("hc_init"))

        # attn-phase mixer (call 0): exact norm/inject, gemm-bar gate/mixed
        mixed_a, inj_a, pa = _mixer(g, "attn").forward(res)
        assert_matches_dump(pa["hc_norm"], node("hc_norm-0", 0))
        assert_matches_dump(pa["hc_gate"], node("hc_gate-0", 0), gemm=True)
        assert_matches_dump(pa["hc_mixed"], node("hc_mixed-0", 0), gemm=True)
        assert_matches_dump(inj_a, node("hc_inject-0", 0))

        # DeltaNet + combine
        attn_out = _deltanet(g, mixed_a)
        assert_matches_dump(attn_out, node("linear_attn_out-0"), gemm=True)
        res2 = hc_combine(res, attn_out, inj_a)
        assert_matches_dump(res2, node("hc_combine-0"), gemm=True)

        # ffn-phase mixer (call 1): input carries the Q8_0 accumulation
        mixed_b, inj_b, pb = _mixer(g, "ffn").forward(res2)
        assert_matches_dump(pb["hc_norm"], node("hc_norm-0", 1), gemm=True)
        assert_matches_dump(pb["hc_gate"], node("hc_gate-0", 1), gemm=True)
        assert_matches_dump(pb["hc_mixed"], node("hc_mixed-0", 1), gemm=True)
        assert_matches_dump(inj_b, node("hc_inject-0", 1), gemm=True)

        # MoE + combine = end of blk.0
        ffn_out = _moe(g, mixed_b)
        assert_matches_dump(ffn_out, node("ffn_out-0"), gemm=True)
        res3 = hc_combine(res2, ffn_out, inj_b)
        assert_matches_dump(res3, node("l_last-0"), gemm=True)

        # blk.1 PLE on that res_hc (also confirms the ple_conv1d view)
        _, _, pp = _ple(g, res3)
        assert_matches_dump(pp["ple_gate"], node("ple_gate-1"), gemm=True)
        assert_matches_dump(pp["ple_gated_value"], node("ple_gated_value-1"), gemm=True)
        assert_matches_dump(pp["ple_conv_out"], node("ple_conv_out-1"), gemm=True)
    finally:
        g.close()
