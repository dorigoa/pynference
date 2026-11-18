"""GPU tests for S25 step 3: QSA against the blk.3 dump, and both context
regimes around the indexer's top_k = 2048 cells (report S23 map B).

Run only inside a GPU job (2.4); skipped when the GPUs are hidden (e.g.
under scripts/cpu.sh). Coverage:

- dump: the REAL blk.3 weights (dequantized with `quant.ref` from the GGUF,
  the blk.3 input `cur` rebuilt on CPU by the same chain the CPU dump test
  uses) run `QSA.forward` on xpu, and every dumped node is compared to the
  golden `qsa_dump_nodes.json` samples. Bars are the CPU test's, including
  the documented per-node relaxations: the xpu-vs-CPU drift (~1e-7 rel on
  scores/outputs, report S25) adds nothing at the scale of those margins,
  and the exact-pattern nodes (keep-set, -inf score mask) hold because at
  N=5 there is no selection and no tie anywhere near the cut.
- short regime: prefill at T=2046 (n_pool 512 = idx_top_k/compress, so the
  indexer keeps EVERY visible block): the keep-set is exactly the causal
  prefix, and xpu matches CPU within the cross-device bars.
- long regime: prefill at T=2058 (515 blocks, 514 visible to the last
  token): top-512 selection prunes the keep-set, and the per-token kept
  cell count is exactly `4 * n_top_pool + tail` (tail = the incomplete
  block's cells at-or-before the token).
- long regime decode: incremental `QSA.decode` from N=2040 to N=2099 at the
  real head shapes (xpu kernel vs CPU sdpa through `_check_cross`), crossing
  the N=2052 step where selection first prunes: kept cells stay causal
  (= N) while at most 512 blocks are visible, then drop to 2048 + tail.

Bars: same as tests/gpu/test_qwen4exp_qsa.py (cross-device cos/rel, keep-set
exact off near-ties, near-tie budget) — see its docstring for why exact
node matches are only asserted against the dump, never across devices.
"""
import os
import pathlib
import sys

import numpy as np
import pytest

import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

torch.backends.cuda.matmul.allow_tf32 = False   # the xpu plugin honors these
torch.backends.cudnn.allow_tf32 = False

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pynference.models.qwen4exp import QSA, hc_combine
from pynference.quant.weight import F32QWeight

from tests.cpu import test_qwen4exp_qsa as cq
from tests.gpu.test_qwen4exp_qsa import DEV, REAL, _check_cross, qsa_pair

needs_model = pytest.mark.skipif(
    not os.path.exists(cq.MODEL), reason=f"model file {cq.MODEL} not present")


def _real_qsa(g, device):
    """blk.3's real QSA (the CPU dump test's cq._qsa) with every tensor on
    `device` — F32QWeight matmuls device-agnostically, so weights alone
    decide where the block runs."""
    H, Hkv, D = cq.N_HEAD, cq.N_HEAD_KV, cq.HEAD_DIM
    deq = lambda name, n, k: F32QWeight(  # noqa: E731
        cq._deq(g, name, 0, n).to(device), n, k)
    f32 = lambda name, shape: cq._f32(g, name, shape).to(device)  # noqa: E731
    return QSA(
        deq("blk.3.attn_q.weight", 2 * D * H, cq.N_EMBD),
        deq("blk.3.attn_k.weight", D * Hkv, cq.N_EMBD),
        deq("blk.3.attn_v.weight", D * Hkv, cq.N_EMBD),
        deq("blk.3.attn_output.weight", cq.N_EMBD, D * H),
        f32("blk.3.attn_q_norm.weight", (D,)),
        f32("blk.3.attn_k_norm.weight", (D,)),
        deq("blk.3.indexer.q_proj.weight", cq.IDX_HEADS * cq.IDX_DIM, cq.N_EMBD),
        deq("blk.3.indexer.k_proj.weight", cq.IDX_DIM, cq.N_EMBD),
        f32("blk.3.indexer.q_norm.weight", (cq.IDX_DIM,)),
        f32("blk.3.indexer.k_norm.weight", (cq.IDX_DIM,)),
        n_head=H, n_head_kv=Hkv, head_dim=D,
        idx_heads=cq.IDX_HEADS, idx_dim=cq.IDX_DIM, idx_top_k=cq.IDX_TOP_K,
        compress=cq.COMPRESS, rope_n_rot=cq.ROPE_N_ROT,
        rope_base=cq.ROPE_BASE, eps=cq.EPS)


def _blk3_input(g):
    """blk.3 attn-phase mixer output (the QSA input `cur`), on CPU: the same
    chain as the CPU dump test (embed -> blk.0/1/2 -> attn mixer of blk.3).
    Dump nodes for blk.3 are conditioned on exactly this input."""
    T = len(cq.DUMP_TOKENS)
    emb = np.concatenate([g.rows("token_embd.weight", t, t + 1)
                          for t in cq.DUMP_TOKENS])
    emb = torch.from_numpy(cq.ref_dequant(emb, g.tensors["token_embd.weight"].type))
    res = emb.unsqueeze(1).expand(T, cq.HC_N, cq.N_EMBD).reshape(T, cq.HC_DIM)
    for layer in (0, 2):  # blk.0/2: plain GDN + MoE
        mixed, inj, _ = cq._mixer(g, "attn", layer).forward(res)
        res2 = hc_combine(res, cq._deltanet(g, mixed, layer), inj)
        mixed, inj, _ = cq._mixer(g, "ffn", layer).forward(res2)
        res = hc_combine(res2, cq._moe(g, mixed, layer), inj)
        if layer == 0:
            res, _, _ = cq._ple(g, res)  # blk.1 PLE precedes its attn mixer
            mixed, inj, _ = cq._mixer(g, "attn", 1).forward(res)
            res2 = hc_combine(res, cq._deltanet(g, mixed, 1), inj)
            mixed, inj, _ = cq._mixer(g, "ffn", 1).forward(res2)
            res = hc_combine(res2, cq._moe(g, mixed, 1), inj)
    cur, _, _ = cq._mixer(g, "attn", 3).forward(res)
    return cur


@needs_model
def test_gpu_blk3_qsa_matches_dump():
    """The xpu prefill of the real blk.3 hits the same dump samples the CPU
    test pins (relaxations and their measured values: report S25)."""
    T = len(cq.DUMP_TOKENS)
    g = cq.GGUFFile(cq.MODEL)
    try:
        cur = _blk3_input(g)
        qsa_g = _real_qsa(g, DEV)
        out, state, p = qsa_g.forward(cur.to(DEV), torch.arange(T).to(DEV),
                                      qsa_g.zero_state(DEV))
        p = {k: v.cpu() for k, v in p.items()}
        out = out.cpu()

        # ---- indexer (same per-node relaxations as the CPU dump test:
        # upstream-chain drift, not device drift — xpu adds ~1e-7 rel here)
        cq.assert_dump(p["indexer_k_raw"], cq.node("indexer_k_raw-3"),
                       sum_atol=0.75)
        cq.assert_dump(p["indexer_pool_k_new"],
                       cq.node("indexer_pool_k_new-3"),
                       cos_min=0.995, sum_atol=1.0)
        n_pool = p["indexer_k"].shape[0]
        k_cache = torch.cat([p["indexer_k"],
                             p["indexer_k"][-1:].repeat(cq.POOL_SLOTS - n_pool, 1)])
        cq.assert_dump(k_cache, cq.node("indexer_k-3"),
                       [cq.IDX_DIM, cq.POOL_SLOTS], cos_min=0.994, sum_atol=70.0)
        cq.assert_dump(p["indexer_q"], cq.node("indexer_q-3"),
                       [cq.IDX_DIM, cq.IDX_HEADS, T], sum_atol=0.75)
        score = torch.cat([p["indexer_score"], p["indexer_score"].new_full(
            (T, cq.POOL_SLOTS - p["indexer_score"].shape[1]), float("-inf"))], 1)
        cq.assert_score(score, cq.node("indexer_score-3"))
        sel = torch.cat([p["indexer_sel"], p["indexer_sel"].new_full(
            (T, cq.KV_SLOTS - p["indexer_sel"].shape[1]), float("-inf"))], 1)
        cq.assert_mask(sel, cq.node("indexer_sel-3"))
        cq.assert_mask(sel, cq.node("kq_mask_qsa-3"))

        # ---- attention
        cq.assert_dump(p["Qcur_full"], cq.node("Qcur_full-3"))
        cq.assert_dump(p["Qcur_reshaped"], cq.node("Qcur_reshaped-3"),
                       [cq.HEAD_DIM, cq.N_HEAD, T])
        cq.assert_dump(p["Qcur_normed"], cq.node("Qcur_normed-3"),
                       [cq.HEAD_DIM, cq.N_HEAD, T])
        cq.assert_dump(p["Qcur"], cq.node("Qcur-3"),
                       [cq.HEAD_DIM, cq.N_HEAD, T])
        cq.assert_dump(p["gate_reshaped"], cq.node("gate_reshaped-3"))
        cq.assert_dump(p["Kcur"], cq.node("Kcur-3", 0))
        cq.assert_dump(p["Kcur_rope"], cq.node("Kcur-3", 1),
                       [cq.HEAD_DIM, cq.N_HEAD_KV, T], sum_atol=1.25)
        cq.assert_dump(p["Kcur_normed"], cq.node("Kcur_normed-3"),
                       [cq.HEAD_DIM, cq.N_HEAD_KV, T], sum_atol=1.25)
        cq.assert_dump(p["Vcur"], cq.node("Vcur-3", 0))
        cq.assert_dump(p["Vcur"].view(T, cq.N_HEAD_KV, cq.HEAD_DIM),
                       cq.node("Vcur-3", 1),
                       [cq.HEAD_DIM, cq.N_HEAD_KV, T])
        cq.assert_dump(p["attn_pregate"], cq.node("attn_pregate-3"))
        cq.assert_dump(p["gate_sigmoid"], cq.node("gate_sigmoid-3"))
        cq.assert_dump(p["attn_gated"], cq.node("attn_gated-3"))
        cq.assert_dump(out, cq.node("attn_output-3"), cos_min=0.998)

        assert state["idx_keys"].shape == (T, cq.IDX_DIM)
        assert state["kv_k"].shape == (T, cq.N_HEAD_KV, cq.HEAD_DIM)
    finally:
        g.close()


def _prefill(qsa_c, qsa_g, T):
    """One prefill on both devices -> the CPU keep-set bools [T, N]; asserts
    the cross-device bars and caps the near-tie tokens. Kept-cell counts are
    printed for the job log."""
    torch.manual_seed(7)
    cur = torch.randn(T, REAL["C"]) * 0.05
    pos = torch.arange(T)
    out_c, _, pc = qsa_c.forward(cur, pos, qsa_c.zero_state())
    out_g, _, pg = qsa_g.forward(cur.to(DEV), pos.to(DEV), qsa_g.zero_state(DEV))
    pg = {k: v.cpu() for k, v in pg.items()}
    kc, n_amb, drift = _check_cross(pc, pg, out_c, out_g.cpu(),
                                    qsa_c.n_top_pool, f"prefill T={T}")
    print(f"drift prefill T={T}: {drift:.2e} near-tie {n_amb}/{T}")
    assert n_amb <= T // 4, "selection near-tie across devices too often"
    return kc


def test_prefill_short_context_keeps_causal():
    # 2046 cells -> 512 blocks = idx_top_k/compress: every visible block
    # fits the indexer's top-k, so the keep-set is exactly the causal prefix
    # (no selection to hide a keep-set bug at this size)
    kc = _prefill(*qsa_pair(seed=11, **REAL), T=2046)
    counts = kc.sum(1)
    assert int(counts.max().item()) == 2046
    assert torch.equal(counts, torch.arange(1, 2047))


def test_prefill_long_context_selects():
    # 2058 cells -> 515 blocks, 514 visible to the last token: selection
    # prunes down to the top-512 blocks (2048 cells) plus the incomplete-
    # tail cells at-or-before the token, exactly `4*512 + t%4+1` (0 when the
    # current block is already complete)
    T = 2058
    kc = _prefill(*qsa_pair(seed=12, **REAL), T=T)
    counts = kc.sum(1)
    # token t keeps t+1 cells while selection cannot prune (t <= 2050):
    # the first 100 tokens must hold the counts 1..100 (job S25-4: a 99-element
    # arange(1, 100) here is a shape bug in the test, not the code)
    assert torch.equal(counts[:100], torch.arange(1, 101))  # still causal early on
    tail = torch.tensor([0 if t % 4 == 3 else t % 4 + 1 for t in range(2051, T)])
    assert torch.equal(counts[2051:], 2048 + tail)


def test_decode_across_top_k_boundary():
    # incremental decode at the real shapes through the selection onset:
    # while at most 512 blocks are visible the keep-set is the causal prefix
    # (= N cells); from N=2052 (513 visible blocks) it is 2048 + tail
    qsa_c, qsa_g = qsa_pair(seed=13, **REAL)
    T0, T1 = 2040, 2099
    torch.manual_seed(14)
    cur = torch.randn(T0, REAL["C"]) * 0.05
    _, st_c, _ = qsa_c.forward(cur, torch.arange(T0), qsa_c.zero_state())
    _, st_g, _ = qsa_g.forward(cur.to(DEV), torch.arange(T0).to(DEV),
                               qsa_g.zero_state(DEV))
    n = n_amb = kept = 0
    for i in range(T0, T1):
        step = torch.randn(1, REAL["C"]) * 0.05
        pos = torch.tensor([i])
        of, st_c, pf = qsa_c.decode(step, pos, st_c)
        og, st_g, pg = qsa_g.decode(step.to(DEV), pos.to(DEV), st_g)
        og, pg = og.cpu(), {k: v.cpu() for k, v in pg.items()}
        n += 1
        N = T0 + n        # context length after this step; kept counts cells
        kc, a, drift = _check_cross(pf, pg, of, og, qsa_c.n_top_pool,
                                    f"decode N={N}")
        if a:
            n_amb += 1
            print(f"drift decode N={N}: {drift:.2e} near-tie "
                  "(keep-set/output skipped, counted)")
        kept = int(kc.sum())
        if N <= 2051:
            assert kept == N, f"N={N}: expected causal prefix ({N} cells), got {kept}"
        else:
            tail = 0 if N % 4 == 0 else (N - 1) % 4 + 1
            assert kept == 2048 + tail, f"N={N}: expected 2048+{tail}, got {kept}"
    assert kept < N, "selection never pruned below the causal prefix"
    print(f"decode boundary: kept {kept} cells at N={N}; near-tie {n_amb}/{n}")
    assert n_amb <= (T1 - T0) // 4, "selection near-tie across devices too often"
