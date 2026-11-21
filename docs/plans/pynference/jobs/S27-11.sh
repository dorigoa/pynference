#!/usr/bin/env bash
# job: S27-11
# timeout: 90
# scopo: ripetizione del programma S27-10 (fallito per infrastruttura: il
#   repo sul PC non e` un repository git e /scratch/ non viene sincronizzato).
#   Tutto il necessario e` incorporato qui: il modulo modello PRE-passo-4
#   (git show 507c831^ sul Mac, 1231 righe) e i due script di diagnostica.
#   (1) A/B dei layer decode qwen4exp: old (path della golden S27-4) vs new
#       (driver segmentato corrente, eager), hook su hc_combine/hc_combine_,
#       PLE, embedding, router logits, uscita QSA, logits; chat-it e code,
#       3 step. Primi record divergenti = componente colpevole (metodo 2.5).
#   (2) Gate C8 + passo-vs-golden sul path FUSED (separa driver vs cattura).
#       Il confronto graph(S27-9)-vs-fused si fa sul Mac dopo il job.
#   (3) gemma4 (C10): comandi EQUIVALENTI di regress.sh gemma4 col modello
#       al nuovo percorso /nvme/gguf/gemma-4/ (deviazione documentata).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
MQ=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
GQ=tests/golden/qwen4exp
MG=/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf
GG=tests/golden/gemma4

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

cat > "$OUT/run_bench.py" <<'PYEOF'
"""Run a bench/ script with load_tokenizer as its tokenizer factory (S17)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from pynference.tokenizer import load_tokenizer  # noqa: E402

module, argv = sys.argv[1], sys.argv[2:]
mod = __import__(module, fromlist=["main"])
mod.Tokenizer = load_tokenizer
sys.argv = [module] + argv
mod.main()
PYEOF

cat > "$OUT/cmp.py" <<'PYEOF'
"""Regress-style comparison (scripts/regress.sh block) between two ours jsons
(+ ppl vs golden ppl): cmp BASE_JSON NOW_JSON PPL_BASE PPL_NOW"""
import json, re, sys

base_path, now_path, ppl_base_path, ppl_now_path = sys.argv[1:5]
problems = []
base = json.load(open(base_path))
now = json.load(open(now_path))
for b, n in zip(base, now):
    if b["name"] != n["name"] or b["prompt"] != n["prompt"]:
        problems.append(f"{b['name']}: prompt mismatch"); continue
    max_dp = 0.0
    for i, (bs, ns) in enumerate(zip(b["steps"], n["steps"])):
        pb = {t["id"]: t["p"] for t in bs["top"]}
        pn = {t["id"]: t["p"] for t in ns["top"]}
        ids_same = pb.keys() == pn.keys()
        if bs["id"] != ns["id"]:
            print(f"{b['name']}: greedy differs at step {i}, ids same: {ids_same}")
        if not ids_same:
            print(f"{b['name']}: top-10 ids differ at step {i}")
            break
        max_dp = max([max_dp] + [abs(pb[k] - pn[k]) for k in pb])
    print(f"{b['name']}: max |dp| {max_dp:.2e} over top-10")
pp = re.compile(r"PPL = ([0-9.]+)")
if ppl_base_path and ppl_now_path:
    b_ppl = float(pp.search(open(ppl_base_path).read()).group(1))
    n_ppl = float(pp.search(open(ppl_now_path).read()).group(1))
    rel = abs(n_ppl - b_ppl) / b_ppl
    print(f"ppl: baseline {b_ppl:.4f}, now {n_ppl:.4f}, rel diff {rel * 100:.3f}%")
    if rel > 1e-3:
        problems.append(f"ppl rel {rel * 100:.3f}% > 0.1%")
print("PROBLEMS" if problems else "ALIGNED")
PYEOF

cat > "$OUT/qwen4exp_v0.py" <<'QWEN4V0_EOF'
"""Qwen3.8-Flash-Next (architecture `qwen4exp`) modules — plan S24.

One device-agnostic FP32 torch implementation: the CPU reference (tests/cpu,
torch CPU) and the GPU run the SAME code on `xpu`, like S20/S22. Layouts are
torch order [T, C]; GGML prints [C, T] with the flat stream index
`e + 2560*s + 10240*t`, so a GGML [hc*n_embd, T] tensor is our [T, hc*n_embd]
(S23 map A). Maps A/C in docs/plans/pynference/reports/ are the reference for
the math; constants live in tests/golden/qwen4exp/facts.json
(metadata.qwen4exp.ple.*).

This file holds the PLE (only blk.1 has it): the n-gram row hash computed
host-side, the IQ4_NL embedding-table gather, and the gated key/value +
dilated depthwise-conv block; the hyper-connection mixer and combine
(`hc_attn_*`/`hc_ffn_*` per layer, `output_hc_*` at the exit head); and the
QSA block (plan S25: every 4th layer, `blk.3` first) — the lightning indexer
with 4:1 key pooling, the top-k cell selection and the gated GQA attention
it masks.

S26 adds the full model class (`Qwen4Exp`, plan S26 step 3). S27 step 2
keeps every dense weight in its native GGUF quant instead of f32
(`_wq` -> QLinear for Q8_0, NativeQWeight/gemv_native for Q6_K,
HalfQWeight for BF16, F32QWeight for F32) and hands the freed VRAM to
resident experts: decode takes the native GEMV, prefill the shared bf16
scratch dequant, the same split the shared-expert trio has always used
(C8 is checked against llama.cpp, not between the two paths). On CPU
`_wq` keeps the FP32 dequant path (F32QWeight), which is what the chain/
reference tests build against. The MoE experts stay natively quantized
in the ExpertStore (S26-1..S26-6);
the DeltaNet driver is the pure-torch chain verified against the dump in
tests/cpu/test_qwen4exp_chain.py (no Triton kernels from deltanet.py).
"""
from __future__ import annotations

import gc
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from pynference.models.gemma4 import rope_neox, rope_table
from pynference.quant.ref import ref_dequant
from pynference.quant.types import TYPES
from pynference.quant.weight import F32QWeight, NativeQWeight, make_qweight
from .gptoss import D8_BUDGET_GB, read_mem_total_bytes
from .qwen35 import qkv_parity_runs, split_run


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def grouped_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last (channel) dim, per residual stream, times gamma.

    x [..., C] f32, gamma [C] or broadcastable [..., C]. This is the
    RMS_NORM+MUL pair llama.cpp builds for `ple_norm_*` (and the HC mixers):
    the stored gamma is multiplied as-is, no (1+w) folding on our side
    (the converter already folded it in the file if the model wants it).
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def ple_row_indices(
    tokens,
    history,
    *,
    ngram: int,
    heads_per_ngram: int,
    multipliers,
    head_offsets,
    head_vocab_sizes,
    eos_token_id: int,
) -> np.ndarray:
    """Table rows for each token: int64 [n_heads * T], token-major, head
    index inner (`rows[t * n_heads + h]`, h = (n-2)*heads_per_ngram + g).

    Host-side replacement for the int64/xor ops ggml lacks (S23 map C,
    qwen4exp.cpp set_input 1293-1352). Window `ctx[s]`: ctx[0] is the token
    itself (its OWN eos never cuts the window); ctx[s] is the predecessor s
    steps back, read as eos when missing; an eos in the window cuts, so every
    position at-or-after it becomes eos. Then, per n-gram size n = 2..ngram,
    `mixed = XOR_s (ctx[s] * multipliers[s])` in uint64 and
    `row = mixed % head_vocab_sizes[h] + head_offsets[h]`.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    history = np.asarray(history, dtype=np.int64)
    n_heads = len(head_offsets)
    assert n_heads == (ngram - 1) * heads_per_ngram
    g = np.concatenate([history, tokens])
    T, H = tokens.shape[0], history.shape[0]

    ctx = np.full((T, ngram), eos_token_id, dtype=np.int64)
    for i in range(T):
        pos = H + i
        ctx[i, 0] = g[pos]
        cut = False
        for s in range(1, ngram):
            if cut or pos - s < 0:
                cut = True  # missing predecessor reads as eos and cuts too
            else:
                t = g[pos - s]
                if t == eos_token_id:
                    cut = True
                else:
                    ctx[i, s] = t

    mul = np.array(multipliers, dtype=np.uint64)
    offs = np.array(head_offsets, dtype=np.uint64)
    voc = np.array(head_vocab_sizes, dtype=np.uint64)
    cu = ctx.astype(np.uint64)
    rows = np.empty((T, n_heads), dtype=np.int64)
    for n in range(2, ngram + 1):
        mixed = cu[:, 0] * mul[0]  # uint64 mul/xor wrap mod 2**64 like C++
        for s in range(1, n):
            mixed = mixed ^ (cu[:, s] * mul[s])
        h0 = (n - 2) * heads_per_ngram
        sel = mixed[:, None] % voc[None, h0:h0 + heads_per_ngram]
        sel = sel + offs[None, h0:h0 + heads_per_ngram]
        rows[:, h0:h0 + heads_per_ngram] = sel
    return rows.reshape(-1)


def ple_gather_embedding(g, name: str, idx: np.ndarray, *, device) -> torch.Tensor:
    """Rows `idx` (int64, from ple_row_indices) of the n-gram table, already
    dequantized float32 [len(idx), head_dim].

    Only the selected bytes move: on CPU they go through `quant.ref`
    (gguf-py, the reference the kernels are checked against); on GPU the
    gathered bytes become a small NativeQWeight and `rows_dequant` expands
    them on device (S24 plan: pinned/mmap gather -> H2D -> dequant).
    """
    info = g.tensors[name]
    head_dim = info.shape[0]
    rows = np.concatenate([g.rows(name, int(r), int(r) + 1) for r in np.asarray(idx)])
    device = torch.device("cpu") if device is None else torch.device(device)
    if device.type == "cpu":
        return torch.from_numpy(ref_dequant(rows, info.type))
    qw = NativeQWeight(info.type, torch.from_numpy(rows).to(device), rows.shape[0], head_dim)
    return qw.rows_dequant(torch.arange(rows.shape[0], device=device))


class HCMixer:
    """The hyper-connection mixer (S23 map A `build_hc_mix`), run on the flat
    `res_hc` [T, hc*n_embd] to produce the block input and its inject weights.

    Weights per phase (`hc_attn_*`/`hc_ffn_*`, and `output_hc_*` at the exit
    head with `inject=None`): `norm` torch [hc, n_embd] F32 — the GGUF
    [hc*n_embd] viewed stream-major (flat index `e + n_embd*s`), multiplied
    as-is (map A: the converter folded any (1+w) into the file); `down`/`up`
    QWeights of the low-rank branch (down: n = rank 320, k = hc*n_embd;
    up: n = hc*n_embd, k = rank); `inject` a QWeight n = hc, k = hc*n_embd.

    Math (dump S23-1 blk.0 node graph): `xn = grouped_norm(res, norm, eps)`,
    `lo = silu(xn_down / hc)`, `gate = lo @ up`, `mixed = mean_s(xn *
    sigmoid(gate))` (the DSV4_HC_PRE op) is the block input, and
    `inject = xn @ inject_w / hc` feeds `hc_combine`. The 1/hc SCALEs are
    explicit ggml nodes (sums in the dump confirm the factor 4).
    """

    def __init__(self, norm, down, up, inject=None, *, eps: float):
        self.norm, self.down, self.up, self.inject = norm, down, up, inject
        self.hc, self.n_embd = norm.shape
        self.channels = self.hc * self.n_embd
        self.eps = eps

    def forward(self, res_hc: torch.Tensor):
        """-> (mixed [T, n_embd], inject [T, hc] | None, parts).

        parts holds the cb()-dumped intermediates in torch layout:
        hc_norm/hc_gate [T, hc*n_embd], hc_mixed [T, n_embd], hc_inject
        [T, hc] (absent on the output-head mixer, which has no inject).
        """
        T = res_hc.shape[0]
        xn = grouped_norm(res_hc.view(T, self.hc, self.n_embd), self.norm, self.eps)
        xn_flat = xn.reshape(T, self.channels)
        lo = silu(self.down(xn_flat) * (1.0 / self.hc))
        gate = self.up(lo)
        mixed = (xn * torch.sigmoid(gate.view(T, self.hc, self.n_embd))).mean(1)
        parts = {"hc_norm": xn_flat, "hc_gate": gate, "hc_mixed": mixed}
        if self.inject is None:
            return mixed, None, parts
        inj = self.inject(xn_flat) * (1.0 / self.hc)
        parts["hc_inject"] = inj
        return mixed, inj, parts


def hc_combine(res_hc: torch.Tensor, block_out: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
    """`res[s] += block_out * 2*sigmoid(inject[s])` (map A `build_hc_combine`,
    DSV4_HC_POST; the 2*SCALE(SIGMOID(inject)) pair is explicit in the dump).

    res_hc [T, hc*n_embd], block_out [T, n_embd], inject [T, hc] -> new
    res_hc, same layout.
    """
    T, channels = res_hc.shape
    hc = inject.shape[1]
    w = 2.0 * torch.sigmoid(inject)
    return (res_hc.view(T, hc, -1)
            + block_out.unsqueeze(1) * w.unsqueeze(-1)).reshape(T, channels)


class PLE:
    """The PLE block (qwen4exp.cpp build_ple 1417-1505), run on `res_hc`
    before blk.1's attention mixer.

    Inputs/outputs are torch [T, ...]: `res_hc` [T, hc*n_embd] (in and out),
    `emb` [T, n_embd] (gathered table rows, head dim slowest: columns
    [160h, 160h+160) = head h), `state` [hist, hc*n_embd] — the causal conv
    history ((kernel-1)*ngram = 9 rows) that the block takes and returns;
    owning that row per sequence is the engine's job (S27).

    `key`/`value` are QWeights (ple_key: n = hc*n_embd, k = n_embd;
    ple_value: n = k = n_embd); `norm_*` are [hc, n_embd] F32; `conv_w` is
    [hc*n_embd, kernel] — the GGUF [4, 10240] F32 viewed with the tap index
    last (flat file index = channel*4 + tap).
    """

    def __init__(self, key, value, norm_key, norm_query, norm_conv, conv_w, *,
                 n_embd: int, ngram: int, eps: float):
        self.key, self.value = key, value
        self.norm_key, self.norm_query, self.norm_conv = norm_key, norm_query, norm_conv
        self.conv_w = conv_w
        self.n_embd, self.ngram, self.eps = n_embd, ngram, eps
        self.hc = norm_key.shape[0]
        self.channels = self.hc * n_embd
        self.kernel = conv_w.shape[1]
        self.hist = (self.kernel - 1) * ngram
        assert conv_w.shape[0] == self.channels

    def zero_state(self, device=None) -> torch.Tensor:
        return torch.zeros(self.hist, self.channels, device=device)

    def forward(self, res_hc: torch.Tensor, emb: torch.Tensor, state: torch.Tensor):
        """-> (res_hc_out, new_state, parts).

        parts keeps the cb()-dumped intermediates for the dump comparison
        (torch layouts: ple_gate [T, hc], *_value/conv [T, hc, n_embd]).
        """
        T = res_hc.shape[0]
        C, hc = self.n_embd, self.hc

        raw_key = self.key(emb).view(T, hc, C)
        key = grouped_norm(raw_key, self.norm_key, self.eps)
        query = grouped_norm(res_hc.view(T, hc, C), self.norm_query, self.eps)
        s = (key * query).sum(-1) * (1.0 / math.sqrt(C))          # [T, hc]
        gate_in = torch.sign(s) * torch.sqrt(s.abs().clamp(min=1e-6))
        gate = torch.sigmoid(gate_in)

        value = self.value(emb)                                    # [T, C]
        gated = value.unsqueeze(1) * gate.unsqueeze(-1)            # [T, hc, C]

        normed = grouped_norm(gated, self.norm_conv, self.eps).reshape(T, self.channels)
        x = torch.cat([state, normed], 0)                          # [hist + T, hc*C]
        conv = torch.zeros(T, self.channels, dtype=res_hc.dtype, device=res_hc.device)
        for k in range(self.kernel):  # causal, dilation = ngram
            start = self.hist - (self.kernel - 1 - k) * self.ngram
            conv += x[start:start + T] * self.conv_w[:, k]
        conv_out = silu(conv).view(T, hc, C)

        res = res_hc + (gated + conv_out).reshape(T, self.channels)
        new_state = x[-self.hist:].clone()  # write-back tail (build_conv_state_at)
        parts = {
            "ple_key_raw": raw_key, "ple_key": key, "ple_value": value,
            "ple_query": query, "ple_s": s, "ple_gate_in": gate_in,
            "ple_gate": gate, "ple_gated_value": gated,
            "ple_conv_normed": normed, "ple_conv_out": conv_out,
        }
        return res, new_state, parts


class QSA:
    """Compressed attention (QSA) with the lightning indexer — plan S25.

    qwen4exp.cpp `build_qsa_sel` 808-943 + `build_attn_qsa` 947-1004 +
    `build_layer_attn` 1006-1081, cross-checked against the S23-1 blk.3 dump
    (report S23 map B). One `forward` runs any step (prefill chunk or one
    decode token); state is the append-only raw-indexer-key cache (with the
    M-RoPE position stored per cell) and the K/V cache. Layout torch [T, C];
    correctness-first like the rest of this file — pooled keys are recomputed
    from the cached raw rows each call (O(N); the incremental kernel is S27).

    Math as the dump shows it:
      * `k_raw = idx_k(cur)` is cached RAW in an f16 cache, gets no norm
        and no RoPE, and pooling reads the cache back through GET_ROWS with
        indices `[4b + j]` whose past-the-end slots are 0, not zeros: an
        incomplete tail block pools `compress` copies of row 0 of the cache
        (dump leaf_108 = [0,1,2,3,0,0,0,0] for N=5, and the gathered sums
        confirm it). Block pool = GET_ROWS rows summed and SCALE'd by a
        fixed 1/compress, then RMSNorm x k_norm gamma — and NOTHING else:
        `indexer_pool_k_new-3` (sum 12.820622) is exactly the normalized
        gamma'd block, no rope (dump chain RESHAPE->SUMS->SCALE->RMS_NORM->
        MUL k_norm.weight). The pool keys the score consumes are these,
        cache values included (leaf_8 gathers the pool slots, repeating the
        last written one over all remaining slots); only `indexer_q` gets
        RoPE, at the token positions (`ROPE(node_513, inp_pos)`).
      * `score[b, t] = (1/sqrt(idx_dim)) * sum_h relu(q_t.h . k_b)`, where
        every indexer head weight is that scale (808:866). A block is
        visible to token t iff its LAST cell exists and is strictly in the
        past or at the token — `pos[4b+3] <= pos[t]`, i.e. complete and
        causal; invisible (and padded) slots are -inf. Verified on the
        dump: blk 0 scores -inf for tokens 0-2, finite from token 3; the
        incomplete block 1 (cell 4) never scores even at token 4.
      * selection (ggml_top_k is unordered, so we compare sets): keep the
        cells of the `idx_top_k/compress` best visible blocks PLUS every
        cached cell at-or-before the token whose block is not yet visible
        (the incomplete tail); everything else dropped. <= 2048 cached cells
        means all visible blocks fit, so the kept set is exactly the causal
        prefix — matches indexer_sel-3 ({256, 5} f16: cell <= token is 0).
      * attention is plain GQA (no softcap/sink) with `sel` as the additive
        kq mask and `kq_scale = 1/sqrt(head_dim)`; q/k/v reach the cache
        already normed+roped, no Hadamard rotations (self_k_rot/self_v_rot
        are a cache option this dump does not use). The Q projection emits
        interleaved per-head [query | gate] (the gate view starts at
        element 256 of each 512 group); `attn_pregate * sigmoid(gate)`
        then `wo`.

    M-RoPE: `rope.dimension_sections = [11, 11, 10, 0]` over 32 pairs with
    one position row per axis — for text all axes agree, so plain NEOX
    rotation of the first `rope_n_rot` dims at the cell position is exact.
    """

    def __init__(self, wq, wk, wv, wo, q_norm, k_norm,
                 idx_q, idx_k, idx_q_norm, idx_k_norm, *,
                 n_head: int, n_head_kv: int, head_dim: int,
                 idx_heads: int, idx_dim: int, idx_top_k: int, compress: int,
                 rope_n_rot: int, rope_base: float, eps: float):
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.q_norm, self.k_norm = q_norm, k_norm
        self.idx_q, self.idx_k = idx_q, idx_k
        self.idx_q_norm, self.idx_k_norm = idx_q_norm, idx_k_norm
        self.n_head, self.n_head_kv, self.head_dim = n_head, n_head_kv, head_dim
        self.idx_heads, self.idx_dim = idx_heads, idx_dim
        self.compress, self.n_top_pool = compress, idx_top_k // compress
        self.rope_n_rot, self.rope_base, self.eps = rope_n_rot, rope_base, eps
        self.kq_scale = 1.0 / math.sqrt(head_dim)
        self.idx_scale = 1.0 / math.sqrt(idx_dim)

    def zero_state(self, device=None) -> dict:
        """Append-only caches: raw indexer keys + their cell positions, the
        K/V cache (cells in sequence order) and, for `decode`, the pooled
        indexer keys per block (`forward` recomputes those from the raw rows
        each call and ignores/does not carry `pool`)."""
        e = lambda *shape: torch.zeros(0, *shape, device=device)  # noqa: E731
        return {"idx_keys": e(self.idx_dim),
                "idx_pos": torch.zeros(0, dtype=torch.int64, device=device),
                "kv_k": e(self.n_head_kv, self.head_dim),
                "kv_v": e(self.n_head_kv, self.head_dim),
                "pool": e(self.idx_dim)}

    def forward(self, cur: torch.Tensor, pos: torch.Tensor, state: dict):
        """cur [T, n_embd], pos [T] int64 absolute M-RoPE positions (all four
        axes agree for text), state from zero_state()/the previous call ->
        (out [T, n_embd], new_state, parts).

        `parts` holds every cb()-dumped node of the block in torch layout,
        logical (unpadded) sizes: the dump pads them to the cache shapes
        (indexer_k to the pool-slot count by repeating the last written key
        — its gather index is clamped to the last slot —, indexer_score and
        indexer_sel/kq_mask to the KV-cell count with -inf) — see the S25
        test. `indexer_pool_k_new` is just the blocks this step (re)wrote.
        """
        T = cur.shape[0]
        dev = cur.device
        pos = pos.to(device=dev, dtype=torch.int64)
        cp = self.compress

        # ---- indexer: pooled keys, query, scores, selection (build_qsa_sel)
        k_raw = self.idx_k(cur)                                       # [T, 128]
        # the cache is an f16 buffer (SET_ROWS f16) that pooling gathers
        # back through, so the pooled keys see f16-rounded raw rows
        idx_keys = torch.cat([state["idx_keys"].to(dev), k_raw], 0).half().float()  # [N, 128]
        idx_pos = torch.cat([state["idx_pos"].to(dev), pos])          # [N]
        N = idx_keys.shape[0]
        n_pool = (N + cp - 1) // cp

        pad = n_pool * cp - N
        # the pooling GET_ROWS indexes past-the-end tail slots with 0, not
        # with zero rows: an incomplete block pools `cp` copies of cache
        # row 0 (dump leaf_108 = [0,1,2,3,0,0,0,0] at N=5). Those blocks
        # never score (visibility below), only the dump nodes see them.
        cell = torch.arange(n_pool * cp, device=dev)
        cell = cell.masked_fill(cell >= N, 0)
        pooled = idx_keys[cell.view(n_pool, cp)].sum(1) * (1.0 / cp)
        pooled = grouped_norm(pooled, self.idx_k_norm, self.eps)
        cos_all, sin_all = rope_table(self.rope_n_rot, self.rope_base, None,
                                      int(idx_pos.max().item()) + 1, dev)

        q_idx = self.idx_q(cur).view(T, self.idx_heads, self.idx_dim)
        q_idx = grouped_norm(q_idx, self.idx_q_norm, self.eps)
        q_idx = rope_neox(q_idx, cos_all[pos], sin_all[pos])

        # block b is visible to token t iff its last (padded) cell position
        # exists and is at-or-before the token: padded slots read +inf.
        pos_pad = torch.cat([idx_pos, idx_pos.new_full((pad,), 1 << 60)])
        vis = pos_pad[torch.arange(n_pool, device=dev) * cp + (cp - 1)].unsqueeze(1) \
            <= pos.unsqueeze(0)                                       # [n_pool, T] -> vis[b, t]
        score = q_idx.new_zeros(n_pool, T)
        for h in range(self.idx_heads):                               # rectified, unweighted sum
            score += torch.relu(pooled @ q_idx[:, h, :].T)            # [n_pool, T]
        score = score * self.idx_scale
        score = torch.where(vis, score, score.new_full((), float("-inf")))

        keep = self._keep_cells(score, vis, idx_pos, pos, n_pool)      # [T, N] bool
        sel = torch.where(keep, torch.zeros((), device=dev),
                          torch.full((), float("-inf"), device=dev))   # additive 0/-inf mask

        # ---- Q / K / V (build_layer_attn 1022-1058)
        q_full = self.wq(cur)                                         # [T, 2*hd*n_head]
        qv = q_full.view(T, self.n_head, 2 * self.head_dim)
        q_pre, gate = qv[..., :self.head_dim], qv[..., self.head_dim:]
        q_normed = grouped_norm(q_pre, self.q_norm, self.eps)
        q_roped = rope_neox(q_normed, cos_all[pos], sin_all[pos])
        k_pre = self.wk(cur).view(T, self.n_head_kv, self.head_dim)
        k_normed = grouped_norm(k_pre, self.k_norm, self.eps)
        k_roped = rope_neox(k_normed, cos_all[pos], sin_all[pos])
        v = self.wv(cur).view(T, self.n_head_kv, self.head_dim)

        kv_k = torch.cat([state["kv_k"].to(dev), k_roped], 0)
        kv_v = torch.cat([state["kv_v"].to(dev), v], 0)

        # ---- masked GQA attention + gate (build_attn_qsa / 1069-1078)
        attn = self._attend(q_roped, kv_k, kv_v, sel)
        attn_flat = attn.reshape(T, self.n_head * self.head_dim)
        gate_sig = torch.sigmoid(gate.reshape(T, self.n_head * self.head_dim))
        out = self.wo(attn_flat * gate_sig)

        parts = {
            "indexer_k_raw": k_raw,
            "indexer_pool_k_new": pooled[state["idx_keys"].shape[0] // cp:],
            "indexer_k": pooled,
            "indexer_q": q_idx,
            "indexer_score": score.T,                                   # [T, n_pool]
            "indexer_sel": sel,
            "kq_mask_qsa": sel,
            "Qcur_full": q_full,
            "Qcur_reshaped": q_pre,
            "Qcur_normed": q_normed,
            "Qcur": q_roped,
            "gate_reshaped": gate.reshape(T, self.n_head * self.head_dim),
            "Kcur": k_pre.reshape(T, self.n_head_kv * self.head_dim),
            "Kcur_normed": k_normed,
            "Kcur_rope": k_roped,          # the dump cb's Kcur a second time post-rope
            "Vcur": v.reshape(T, self.n_head_kv * self.head_dim),
            "attn_pregate": attn_flat,
            "gate_sigmoid": gate_sig,
            "attn_gated": attn_flat * gate_sig,
            "attn_output": out,
        }
        new_state = {"idx_keys": idx_keys, "idx_pos": idx_pos,
                     "kv_k": kv_k, "kv_v": kv_v}
        return out, new_state, parts

    def decode(self, cur: torch.Tensor, pos: torch.Tensor, state: dict):
        """One decode token (T=1) — the incremental twin of `forward` used on
        GPU (plan S25 step 2: prefill with torch, decode with one kernel).
        Same math, cheaper state:

          * pooled keys are carried in `state["pool"]`; only the blocks this
            token (re)touches are recomputed — INCLUDING the last PARTIAL
            block, whose clamped gather (past-the-end members read cache
            row 0) changes every step — never just the newly completed ones;
          * the keep-set is materialized as a fixed-size cell-index list
            (top-k block cells with padding -1, then the incomplete-tail
            slots), no [T, N] bool;
          * attention runs through kernels.qsa_attn_decode (gather by cell
            id, online softmax). On CPU the identical keep-set goes through
            `sel` + `_attend` instead, so the two paths test against each
            other without a GPU.

        Assumes `pos` is the newest position (>= every cached one), as any
        decode step is: then the only block invisible to the token with
        cells at-or-before it is the incomplete last block, and the tail
        rule of `_keep_cells` reduces to exactly that block. Returns
        (out [1, n_embd], new_state with `pool`, parts) — same parts keys as
        `forward`, T=1.
        """
        dev = cur.device
        pos = pos.to(device=dev, dtype=torch.int64).view(1)
        cp = self.compress
        old_n = state["idx_keys"].shape[0]

        k_raw = self.idx_k(cur)                                       # [1, 128]
        idx_keys = torch.cat([state["idx_keys"].to(dev), k_raw], 0).half().float()
        idx_pos = torch.cat([state["idx_pos"].to(dev), pos])
        N = idx_keys.shape[0]
        n_pool = (N + cp - 1) // cp

        pool = state.get("pool")
        if pool is None or pool.device != dev or pool.shape[0] != (old_n + cp - 1) // cp:
            pool, b0 = idx_keys.new_zeros((0, self.idx_dim)), 0
        else:
            b0 = min(old_n // cp, n_pool)   # rows >= floor(old_n/cp) saw their clamp change
        rows = torch.arange(b0, n_pool, device=dev)
        cells = rows[:, None] * cp + torch.arange(cp, device=dev)
        cid = torch.where(cells < N, cells, torch.zeros_like(cells))  # GET_ROWS -> row 0
        new_pool = grouped_norm(idx_keys[cid].sum(1) * (1.0 / cp), self.idx_k_norm, self.eps)
        pool = torch.cat([pool[:b0], new_pool], 0)

        cos_all, sin_all = rope_table(self.rope_n_rot, self.rope_base, None,
                                      int(idx_pos.max().item()) + 1, dev)
        q_idx = self.idx_q(cur).view(1, self.idx_heads, self.idx_dim)
        q_idx = grouped_norm(q_idx, self.idx_q_norm, self.eps)
        q_idx = rope_neox(q_idx, cos_all[pos], sin_all[pos])

        pos_pad = torch.cat([idx_pos, idx_pos.new_full((n_pool * cp - N,), 1 << 60)])
        vis = pos_pad[torch.arange(n_pool, device=dev) * cp + (cp - 1)] <= pos  # [n_pool]
        score = q_idx.new_zeros(n_pool)
        for h in range(self.idx_heads):
            score = score + torch.relu(pool @ q_idx[0, h])
        score = score * self.idx_scale
        score = torch.where(vis, score, score.new_full((), float("-inf")))

        k = min(n_pool, self.n_top_pool)
        top = score.topk(k).indices                                    # [k] block ids
        bcells = top[:, None] * cp + torch.arange(cp, device=dev)      # [k, cp] cells
        ok = vis[top][:, None] & (bcells < N)
        head = torch.where(ok, bcells, torch.full_like(bcells, -1)).reshape(-1)
        tail = (n_pool - 1) * cp + torch.arange(cp, device=dev)
        ok_tail = (tail < N) & (~vis[n_pool - 1]) \
            & (idx_pos[torch.clamp(tail, max=N - 1)] <= pos)
        idx = torch.cat([head, torch.where(ok_tail, tail, torch.full_like(tail, -1))]) \
            .to(torch.int32)

        # ---- Q / K / V (same as forward)
        q_full = self.wq(cur)
        qv = q_full.view(1, self.n_head, 2 * self.head_dim)
        q_pre, gate = qv[..., :self.head_dim], qv[..., self.head_dim:]
        q_normed = grouped_norm(q_pre, self.q_norm, self.eps)
        q_roped = rope_neox(q_normed, cos_all[pos], sin_all[pos])
        k_pre = self.wk(cur).view(1, self.n_head_kv, self.head_dim)
        k_normed = grouped_norm(k_pre, self.k_norm, self.eps)
        k_roped = rope_neox(k_normed, cos_all[pos], sin_all[pos])
        v = self.wv(cur).view(1, self.n_head_kv, self.head_dim)
        kv_k = torch.cat([state["kv_k"].to(dev), k_roped], 0)
        kv_v = torch.cat([state["kv_v"].to(dev), v], 0)

        if dev.type == "cpu":
            sel = torch.full((N,), float("-inf"), device=dev)
            sel[idx[idx >= 0].long()] = 0.0
            attn = self._attend(q_roped, kv_k, kv_v, sel.unsqueeze(0))
        else:
            from pynference import kernels
            attn = kernels.qsa_attn_decode(q_roped[0].contiguous(), kv_k, kv_v,
                                           idx, self.kq_scale).unsqueeze(0)

        attn_flat = attn.reshape(1, self.n_head * self.head_dim)
        gate_sig = torch.sigmoid(gate.reshape(1, self.n_head * self.head_dim))
        out = self.wo(attn_flat * gate_sig)

        sel_parts = torch.full((1, N), float("-inf"), device=dev)
        sel_parts[0, idx[idx >= 0].long()] = 0.0
        parts = {
            "indexer_k_raw": k_raw,
            "indexer_pool_k_new": pool[b0:],
            "indexer_k": pool,
            "indexer_q": q_idx,
            "indexer_score": score.view(1, -1),
            "indexer_sel": sel_parts,
            "kq_mask_qsa": sel_parts,
            "Qcur_full": q_full,
            "Qcur_reshaped": q_pre,
            "Qcur_normed": q_normed,
            "Qcur": q_roped,
            "gate_reshaped": gate.reshape(1, self.n_head * self.head_dim),
            "Kcur": k_pre.reshape(1, self.n_head_kv * self.head_dim),
            "Kcur_normed": k_normed,
            "Kcur_rope": k_roped,
            "Vcur": v.reshape(1, self.n_head_kv * self.head_dim),
            "attn_pregate": attn_flat,
            "gate_sigmoid": gate_sig,
            "attn_gated": attn_flat * gate_sig,
            "attn_output": out,
        }
        new_state = {"idx_keys": idx_keys, "idx_pos": idx_pos,
                     "kv_k": kv_k, "kv_v": kv_v, "pool": pool}
        return out, new_state, parts

    def _keep_cells(self, score: torch.Tensor, vis: torch.Tensor,
                    idx_pos: torch.Tensor, pos: torch.Tensor, n_pool: int) -> torch.Tensor:
        """kept-cell mask [T, N] (bool): the cells of the top-`n_top_pool`
        VISIBLE blocks per token plus the incomplete-tail cells at-or-before
        the token. ggml_top_k's order is undefined, so ties at the cut can
        differ from the dump set-theoretically; with real scores that never
        bites (report S25)."""
        T, N = pos.shape[0], idx_pos.shape[0]
        dev, cp = pos.device, self.compress
        k = min(n_pool, self.n_top_pool)
        top = score.topk(k, dim=0).indices.T                              # [T, k] block ids
        # winners that scored -inf (invisible) keep nothing: pair each with vis
        block_keep = torch.zeros(T, n_pool, dtype=torch.bool, device=dev)
        block_keep.scatter_(1, top, vis.T.gather(1, top))
        block_of_cell = torch.arange(N, device=dev) // cp
        keep = block_keep[:, block_of_cell]                               # cells of kept blocks
        # incomplete-tail cells at-or-before the token: the block's last cell
        # does not exist yet (padded sentinel 1<<60 keeps such blocks open)
        last_pos = torch.full((n_pool,), 1 << 60, device=dev, dtype=idx_pos.dtype)
        complete = torch.arange(n_pool, device=dev) * cp + cp - 1 < N
        last_pos[complete] = idx_pos[torch.arange(n_pool, device=dev)[complete] * cp + cp - 1]
        keep |= (idx_pos.unsqueeze(0) <= pos.unsqueeze(1)) \
            & (last_pos[block_of_cell].unsqueeze(0) > pos.unsqueeze(1))
        return keep

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """GQA attention: q [T, n_head, hd], k/v [N, n_head_kv, hd], additive
        mask [T, N] (0 keep, -inf drop) -> [T, n_head, hd]."""
        # sdpa wants (B, H, L, E): transpose the [seq, heads, hd] layout first
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        try:
            out = torch.nn.functional.scaled_dot_product_attention(
                qh.unsqueeze(0), kh.unsqueeze(0), vh.unsqueeze(0),
                attn_mask=mask.unsqueeze(0), scale=self.kq_scale, enable_gqa=True)
        except TypeError:  # torch < 2.5: expand the KV heads by hand
            rep = self.n_head // self.n_head_kv
            out = torch.nn.functional.scaled_dot_product_attention(
                qh.unsqueeze(0), kh.repeat_interleave(rep, 0).unsqueeze(0),
                vh.repeat_interleave(rep, 0).unsqueeze(0),
                attn_mask=mask.unsqueeze(0), scale=self.kq_scale)
        return out[0].transpose(0, 1)


# --------------------------------------------------------------- MoE FFN (S26)

MOE_N_EXPERTS = 512
MOE_TOP_K = 10
MOE_FFN = 640


def moe_expert_groups(g):
    """One ExpertGroup per (layer, projection) for the Flash-Next experts.

    GGUF layout matches GPT-OSS (models/gptoss.py):
    blk.N.ffn_{gate,up}_exps.weight is [k, n, E] in ggml order (k = 2560
    contiguous inputs, n = 640 outputs, E = 512 experts) and
    ffn_down_exps.weight is [n, k, E] = [640, 2560, E], so global expert e
    occupies rows [e*n, (e+1)*n) of the flat tensor.

    Unlike GPT-OSS (all MXFP4), the "UD" mix keeps a per-(layer, projection)
    type (facts.json per_layer_types: gate/up in IQ3_XXS/IQ4_XS/IQ2_S, down
    in Q5_0/IQ4_NL/Q2_0); all are native quant types the indexed GEMV
    kernels already cover, so each group just carries its tensor's type.
    """
    from ..moe import ExpertGroup  # lazy import: no Triton at module load
    groups = []
    for il in range(int(g.get("block_count"))):
        for proj in ("gate", "up", "down"):
            name = f"blk.{il}.ffn_{proj}_exps.weight"
            t = g.tensors[name]
            k, n, e = t.shape
            groups.append(ExpertGroup(proj, il, e, n, k, t.type,
                                      lambda ex, t=name, n=n: g.rows(t, ex * n, (ex + 1) * n)))
    return groups


def moe_router(g, il: int) -> torch.Tensor:
    """Router f32: ffn_gate_inp.weight [k, E] (ggml order) -> [E, k].

    No bias (S23 map C): Flash-Next routes with plain softmax(logits),
    moe routing "softmax" — no expert_bias, no logits bias, and no
    expert_weights_scale either."""
    t = g.tensors[f"blk.{il}.ffn_gate_inp.weight"]
    assert t.type == "F32", (t.name, t.type)
    k, e = t.shape
    return torch.from_numpy(np.array(g.raw(t.name).view(np.float32))).reshape(e, k)


def moe_shared_expert(g, il: int, device):
    """Dense shared expert per layer: gate/up [2560, 640] ggml -> QWeight
    [640, 2560], down [640, 2560] -> [2560, 640] (types vary per layer:
    mostly Q6_K/Q8_0, read from the tensor), plus the scalar gate vector
    ffn_gate_inp_shexp.weight f32 [2560] applied as sigmoid(x @ gate)."""
    trio = []
    for proj, (n, k) in (("gate", (MOE_FFN, 2560)), ("up", (MOE_FFN, 2560)),
                         ("down", (2560, MOE_FFN))):
        name = f"blk.{il}.ffn_{proj}_shexp.weight"
        t = g.tensors[name]
        assert tuple(t.shape) == (k, n), (name, t.shape)
        trio.append(make_qweight(g.raw(name), t.type, n, k, device))
    gw = moe_gate_shexp(g, il)
    return trio + [gw.to(device)]


def moe_gate_shexp(g, il: int) -> torch.Tensor:
    """Shared-expert scalar gate ffn_gate_inp_shexp.weight, f32 [k]."""
    t = g.tensors[f"blk.{il}.ffn_gate_inp_shexp.weight"]
    assert t.type == "F32", (t.name, t.type)
    return torch.from_numpy(np.array(g.raw(t.name).view(np.float32)))


def moe_layer(g, store, il: int, device):
    """Flash-Next FFN as a MoE (C5): 512 experts, top-10 softmax routing,
    plain swiglu (no clamp/alpha), shared expert 640 with sigmoid scalar
    gate, no biases anywhere (S23 map C)."""
    from ..moe import MoE  # lazy import (module header note)
    shared = moe_shared_expert(g, il, device)
    return MoE(store, il, top_k=MOE_TOP_K,
               router_w=moe_router(g, il).to(device),
               activation="swiglu", routing="softmax",
               shared=shared[:3], shared_gate=shared[3])


# ------------------------------------------------------------- full model (S26)

PLE_TABLE = "per_layer_token_embd.weight"   # gather-only, never loaded whole


class _Q4Layer:
    """Per-block weight namespace: mixers always, then kind-specific block."""


class Qwen4Exp:
    """Qwen3.8-Flash-Next (48 layers): DeltaNet/QSA hybrid attention under
    hyper-connections, PLE on blk.1, 512-expert MoE FFN (C5).

    Layout follows the dump-verified chain tests (S24/S25): blocks take the
    HC mixer's `mixed` directly (no extra pre-norm), residual streams are
    [T, hc*n_embd], and the exit is the `output_hc` mixer (it replaces the
    final norm, so there is no output_norm tensor and no inject at the exit).

    TP (world=2, oneCCL): DeltaNet heads split by PARITY (qkv_parity_runs,
    whose in-rank key map h % n_key_l reproduces the global h % n_key rule —
    dump-verified in TP2, l_last-0). QSA heads split CONTIGUOUSLY: its GQA
    pairing is contiguous grouping (q-head h -> kv-head h // rep, dump +
    tests/cpu/test_s26_pairing.py verified), so rank r owns q-heads
    [r*n_head_l, (r+1)*n_head_l) = kv-head r's whole group + kv-head r;
    parity would scatter a kv group across ranks (S26-11 attn_output-3).
    Embedding and lm_head
    split the vocab by rows; the HC mixers, indexer, PLE and the ssm
    norm/conv metadata are replicated. all_reduce after every output
    projection; all_gather only on the logits.

    Dense weights stay in their native quant (S27-2, `_wq`); experts
    stream from the ExpertStore with the gpt-oss budget ladder (C5/C8).
    Decode GEMV and prefill bf16 matmul differ only in accumulation
    order, as they always did for the shared-expert trio; the margin is
    checked against llama.cpp (C8), not between the paths.
    """

    # C5 floors, same rationale comments as models/gptoss.py (S17/S26):
    # oneCCL + graph scratch live above the store; the host reserve keeps
    # page-cache-free RAM for the OS because the store's pinned ring (and
    # the S27-3 pinned host cache sized by `_host_pin_bytes`) is
    # non-reclaimable.
    EXPERT_BUDGET_MIN_GB = 8.0
    EXPERT_GPU_OVERHEAD_BYTES = 11 << 30
    EXPERT_HOST_RESERVE_BYTES = 16 << 30

    def __init__(self, g, tp, max_ctx: int = 8192):
        self.g, self.tp, self.max_ctx = g, tp, int(max_ctx)
        get, dev, w, r = g.get, tp.device, tp.world, tp.rank
        if not os.environ.get("PYI_NONDETERMINISTIC"):
            torch.backends.mkldnn.deterministic = True   # same C8 rationale as qwen35
        self.n_layer = int(get("block_count"))
        cap = int(os.environ.get("PYI_QWEN4EXP_LAYERS", "0") or 0)
        if cap:
            self.n_layer = min(self.n_layer, cap)
        self.n_embd = int(get("embedding_length"))
        self.n_head = int(get("attention.head_count"))
        self.n_head_kv = int(get("attention.head_count_kv"))
        self.head_dim = int(get("attention.key_length"))
        self.eps = float(get("attention.layer_norm_rms_epsilon", 1e-6))
        self.compress = [int(x) for x in get("attention.compress_ratios")]
        self.idx_heads = int(get("attention.indexer.head_count"))
        self.idx_dim = int(get("attention.indexer.key_length"))
        self.idx_top_k = int(get("attention.indexer.top_k"))
        self.n_key = int(get("ssm.group_count"))
        self.n_value = int(get("ssm.time_step_rank"))
        self.ssm = int(get("ssm.state_size"))
        self.conv_k = int(get("ssm.conv_kernel"))
        assert int(get("ssm.inner_size")) == self.n_value * self.ssm
        self.hc_n = int(get("hyper_connection.count"))
        self.hc_rank = int(get("hyper_connection.low_rank"))
        self.hc_dim = self.hc_n * self.n_embd
        self.ple_dim = int(get("embedding_length_per_layer_input"))
        self.rope_dim = int(get("rope.dimension_count", self.head_dim))
        self.rope_base = float(get("rope.freq_base", 1e7))
        self.n_vocab = int(g.tensors["token_embd.weight"].shape[1])
        ple_layers = [int(x) for x in get("ple.layers", [1])]
        if len(ple_layers) != 1:
            raise NotImplementedError("PLE on several layers not supported")
        self.ple_layer = ple_layers[0]
        self.ple_cfg = dict(
            ngram=int(get("ple.ngram_size")),
            heads_per_ngram=int(get("ple.heads_per_ngram")),
            multipliers=[int(x) for x in get("ple.layer_multipliers")],
            head_offsets=[int(x) for x in get("ple.head_offsets")],
            head_vocab_sizes=[int(x) for x in get("ple.head_vocab_sizes")],
            eos_token_id=int(get("ple.eos_token_id")),
        )
        for name, v in (("attention.head_count", self.n_head),
                        ("attention.head_count_kv", self.n_head_kv),
                        ("ssm.group_count", self.n_key),
                        ("ssm.time_step_rank", self.n_value),
                        ("vocab", self.n_vocab)):
            if v % w:
                raise ValueError(f"{name} ({v}) not divisible by {w} GPUs")
        self.n_head_l, self.n_head_kv_l = self.n_head // w, self.n_head_kv // w
        self.n_key_l, self.n_value_l = self.n_key // w, self.n_value // w
        self._qkv_l = (2 * self.n_key_l + self.n_value_l) * self.ssm
        # The engine may toggle these; this model has a single eager decode
        # path (T == 1 goes through QSA.decode), so they are accepted but do
        # not select different code (correctness first, graphs are S27).
        self.fast_decode, self.use_graph = True, False
        self._dense_bytes = 0
        t0 = time.time()
        self.v0, vn = split_run(self.n_vocab, w, r)
        self.v1 = self.v0 + vn
        self.embed = self._wq("token_embd.weight", runs=[(self.v0, vn)])
        self.lm_head = (self._wq("output.weight", runs=[(self.v0, vn)])
                        if "output.weight" in g.tensors else self.embed)
        self.layers = [self._load_layer(il) for il in range(self.n_layer)]
        self.ple = self._load_ple() if self.ple_layer < self.n_layer else None
        self.exit_mixer = self._mixer("output_hc")
        self.reset()
        self._load_moe()
        if dev.type != "cpu":
            torch.xpu.synchronize(dev)
        self.load_seconds = time.time() - t0
        if tp.is_main:
            print(f"  qwen4exp: load done in {self.load_seconds:.1f}s", flush=True)

    # ------------------------------------------------------------------ loader

    def _dequant(self, raw, qtype, n, k):
        """Raw GGML bytes -> torch f32 [n, k] (whole tensor, row or column
        subset; ref_dequant infers k from the block arithmetic)."""
        flat = np.ascontiguousarray(raw).reshape(-1)
        if qtype == "F32":
            w = flat.view(np.float32)
        elif qtype == "F16":
            w = flat.view(np.uint16).view(np.float16).astype(np.float32)
        elif qtype == "BF16":
            w = (flat.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            w = ref_dequant(flat.reshape(1, -1), qtype)[0] if raw.ndim == 1 \
                else ref_dequant(raw, qtype)
        return torch.from_numpy(np.array(w).reshape(n, k)).to(self.tp.device)

    def _wq(self, name, *, runs=None, cols=None):
        """Dense weight -> QWeight [n, k] kept in its native quant (S27-2):
        make_qweight gives QLinear for Q8_0, NativeQWeight (gemv_native)
        for Q6_K and friends, HalfQWeight for F16/BF16, F32QWeight for F32
        — the shared-expert machinery, so decode is a native GEMV and
        prefill dequantizes through the shared bf16 scratch. On CPU the
        FP32 dequant path stays (F32QWeight): the chain/reference tests
        build on CPU and bind `_dequant` to a stub expecting `.w` there.
        `runs` takes output-side rows — quant blocks are packed along
        the input axis INSIDE each GGUF row, so output-row slicing needs no
        block alignment (verified: row-slice dequant == slice of full
        dequant, S26-7 debug), and the concatenated raw rows are a valid
        native matrix. `cols` byte-slices input-side blocks, so it does
        (valid because every col run is type-block aligned: Q6_K 256,
        Q8_0 32, and the parity runs are multiples of 128/256), which again
        leaves [n, row_bytes(qtype, k)] raw bytes."""
        t = self.g.tensors[name]
        K, N = t.shape                     # ggml order: (inputs, outputs)
        blk = TYPES[t.type].block
        if runs is not None:
            assert cols is None
            raw = np.concatenate([self.g.rows(name, a, a + l) for a, l in runs], 0)
            n, k = sum(l for _, l in runs), K
        elif cols is not None:
            assert all(a % blk == 0 and l % blk == 0 for a, l in cols), name
            bb = TYPES[t.type].block_bytes
            raw = self.g.rows(name, 0, N)
            raw = np.concatenate([raw[:, (a // blk) * bb:((a + l) // blk) * bb]
                                  for a, l in cols], 1)
            n, k = N, sum(l for _, l in cols)
        else:
            raw = np.array(self.g.raw(name))
            n, k = N, K
        if self.tp.device.type == "cpu":
            w = self._dequant(raw, t.type, n, k)
            self._dense_bytes += w.numel() * 4
            return F32QWeight(w, n, k)
        w = make_qweight(raw, t.type, n, k, self.tp.device)
        self._dense_bytes += w.nbytes
        return w

    def _f32(self, name, shape=None):
        t = self.g.tensors[name]
        assert t.type == "F32", (name, t.type)
        w = torch.from_numpy(np.array(self.g.raw(name).view(np.float32)))
        if shape is not None:
            w = w.reshape(shape)
        self._dense_bytes += w.numel() * 4
        return w.to(self.tp.device)

    def _f32_gather(self, name, shape, ids):
        w = self._f32(name, shape)[torch.from_numpy(np.asarray(ids, np.int64))]
        return w.contiguous()

    def _mixer(self, p: str) -> HCMixer:
        """HC mixer for the `{p}_norm/down/up[/inject]` family; the exit head
        (`output_hc`) has no inject tensor."""
        inj = f"{p}_inject.weight"
        return HCMixer(self._f32(f"{p}_norm.weight", (self.hc_n, self.n_embd)),
                       self._wq(f"{p}_down.weight"),
                       self._wq(f"{p}_up.weight"),
                       self._wq(inj) if inj in self.g.tensors else None,
                       eps=self.eps)

    def _load_layer(self, il: int):
        w, r = self.tp.world, self.tp.rank
        p = f"blk.{il}."
        L = _Q4Layer()
        L.kind = "delta" if self.compress[il] == 0 else "qsa"
        L.mix_attn = self._mixer(p + "hc_attn")
        L.mix_ffn = self._mixer(p + "hc_ffn")
        if L.kind == "delta":
            qkv_runs, value_runs, ids, _ = qkv_parity_runs(
                self.n_key, self.n_value, self.ssm, w, r)
            L.qkv = self._wq(p + "attn_qkv.weight", runs=qkv_runs)
            L.gate = self._wq(p + "attn_gate.weight", runs=value_runs)
            chans = np.concatenate([np.arange(a, a + l) for a, l in qkv_runs])
            L.conv = self._f32_gather(p + "ssm_conv1d.weight",
                                      ((2 * self.n_key + self.n_value) * self.ssm,
                                       self.conv_k), chans)     # [channel, tap]
            # alpha/beta share one [2*n_value_l, n_embd] product (driver
            # splits it again); the rank keeps value heads `ids`.
            L.ab = F32QWeight(
                torch.cat([self._f32_gather(p + "ssm_alpha.weight",
                                            (self.n_value, self.n_embd), ids),
                           self._f32_gather(p + "ssm_beta.weight",
                                            (self.n_value, self.n_embd), ids)], 0),
                2 * self.n_value_l, self.n_embd)
            L.ssm_a = self._f32_gather(p + "ssm_a", (self.n_value,), ids)
            L.dt_bias = self._f32_gather(p + "ssm_dt.bias", (self.n_value,), ids)
            L.out = self._wq(p + "ssm_out.weight", cols=value_runs)
            L.ssm_norm = self._f32(p + "ssm_norm.weight")
        else:
            # GQA pairing is CONTIGUOUS GROUPING (q-head h -> kv-head
            # h // (n_head/n_head_kv)): torch's SDPA enable_gqa does
            # repeat_interleave and the S23-1 dump matches it — verified by
            # tests/cpu/test_s26_pairing.py + the S25 TP1 attn comparisons
            # (a modular mapping would mis-feed half the heads, no cos
            # 0.9989). So heads split CONTIGUOUSLY, not by parity: rank r
            # owns q-heads [r*n_head_l, (r+1)*n_head_l) — exactly the q
            # block of kv-head r's group — and kv-head r. A parity split
            # scatters each kv group over the ranks while each rank keeps
            # only kv-head r, which is S26-11's attn_output-3 divergence
            # (cos 0.43-0.49 on the rank-local attn_pregate, cpu2-repro).
            def _head_rows(n_heads, dim):
                a, l = split_run(n_heads * dim, w, r)
                return [[a, l]]
            L.qsa = QSA(
                self._wq(p + "attn_q.weight",
                         runs=_head_rows(self.n_head, 2 * self.head_dim)),
                self._wq(p + "attn_k.weight",
                         runs=_head_rows(self.n_head_kv, self.head_dim)),
                self._wq(p + "attn_v.weight",
                         runs=_head_rows(self.n_head_kv, self.head_dim)),
                self._wq(p + "attn_output.weight",
                         cols=_head_rows(self.n_head, self.head_dim)),
                self._f32(p + "attn_q_norm.weight"),
                self._f32(p + "attn_k_norm.weight"),
                self._wq(p + "indexer.q_proj.weight"),    # BF16 -> HalfQWeight
                self._wq(p + "indexer.k_proj.weight"),
                self._f32(p + "indexer.q_norm.weight"),
                self._f32(p + "indexer.k_norm.weight"),
                n_head=self.n_head_l, n_head_kv=self.n_head_kv_l,
                head_dim=self.head_dim, idx_heads=self.idx_heads,
                idx_dim=self.idx_dim, idx_top_k=self.idx_top_k,
                compress=int(self.compress[il]), rope_n_rot=self.rope_dim,
                rope_base=self.rope_base, eps=self.eps)
        return L

    def _load_ple(self):
        p = f"blk.{self.ple_layer}."
        return PLE(self._wq(p + "ple_key.weight"),
                   self._wq(p + "ple_value.weight"),
                   self._f32(p + "ple_norm_key.weight", (self.hc_n, self.n_embd)),
                   self._f32(p + "ple_norm_query.weight", (self.hc_n, self.n_embd)),
                   self._f32(p + "ple_norm_conv.weight", (self.hc_n, self.n_embd)),
                   self._f32(p + "ple_conv1d.weight", (self.hc_dim, 4)),  # tap last
                   n_embd=self.n_embd, ngram=self.ple_cfg["ngram"], eps=self.eps)

    # ------------------------------------------------------------- state/kv/ram

    def reset(self):
        """Fresh sequence: the delta/conv states are replacement-style (never
        mutated in place), the QSA state dict is empty-and-append, and the PLE
        conv history plus the host-side token history restart."""
        self.n_past = 0
        dev = self.tp.device
        self.delta_cache = {
            il: (torch.zeros(self.conv_k - 1, self._qkv_l, device=dev),
                 torch.zeros(self.n_value_l, self.ssm, self.ssm, device=dev))
            for il, L in enumerate(self.layers) if L.kind == "delta"}
        self.qsa_cache = {il: L.qsa.zero_state(dev)
                          for il, L in enumerate(self.layers) if L.kind == "qsa"}
        self.ple_state = self.ple.zero_state(dev) if self.ple is not None else None
        self._ple_hist = np.zeros(0, dtype=np.int64)

    def kv_bytes(self) -> int:
        """Cache footprint for the memory report. Delta/PLE states are exact;
        the QSA caches grow by torch.cat with the context, so this reserves
        the worst case per QSA layer: max_ctx cells of (indexer key f32 +
        pos i64 + 2 KV rows f32) plus the pooled keys, which accumulate at
        1/compress of the cells but are bounded by idx_dim per cell."""
        b = sum((self.conv_k - 1) * self._qkv_l
                + self.n_value_l * self.ssm * self.ssm
                for _ in self.delta_cache) * 4
        if self.ple_state is not None:
            b += self.ple_state.numel() * 4
        per_cell = (2 * self.idx_dim + 8 + 2 * self.n_head_kv_l * self.head_dim) * 4
        return b + len(self.qsa_cache) * self.max_ctx * per_cell

    def memory_report(self) -> dict:
        """C3 report; `experts`/`pinned_host` cover the store (0 on CPU)."""
        experts = self.store.memory_bytes() if self.store is not None else 0
        pinned = self.store.pinned_bytes() if self.store is not None else 0
        cache = self.kv_bytes()
        return {"weights": self._weights_bytes, "experts": experts,
                "cache": cache, "scratch": self._scratch_bytes,
                "pinned_host": pinned,
                "device_total": self._weights_bytes + experts + cache
                + self._scratch_bytes}

    # --------------------------------------------------------------------- MoE

    def _expert_budget(self, dense, kv, scratch, host_total=None):
        """Expert-store budget: the model must fit with a floor above it
        (EXPERT_BUDGET_MIN_GB); caps: per-rank D8 device budget, host RAM
        reserve (world-shared), and the live device total minus overhead.

        S27-3 (measured): with native dense weights the live-device cap
        binds — 34.3e9 total - EXPERT_GPU_OVERHEAD_BYTES - 2.68e9 dense -
        0.34e9 kv - 1.07e9 scratch = 18.4e9 (store allocated 18.38 GB, no
        OOM through decode). The host cap is (121.5 GiB - 16 GiB) / 2
        - dense - kv - scratch ~ 52e9 and no longer binds since the PC
        grew its RAM on 2026-11-19; the overhead's real cost measured
        GPUActive <= 13.7 GiB and one OOM-free graph-less run, so 11 GiB
        stays for now (step 4 graphs will want it)."""
        plan = max(int(D8_BUDGET_GB[self.tp.rank] * 1e9) - dense - kv - scratch, 0)
        if host_total is None and torch.xpu.device_count() > 0:
            host_total = read_mem_total_bytes()
        if host_total and host_total > 0:
            host_cap = max((host_total - self.EXPERT_HOST_RESERVE_BYTES)
                           // self.tp.world - dense - kv - scratch, 0)
            plan = min(plan, host_cap)
        if torch.xpu.device_count() == 0:
            return plan
        gc.collect()
        torch.xpu.empty_cache()
        try:
            total = torch.xpu.mem_get_info(self.tp.device)[1]
        except (AttributeError, RuntimeError):
            total = int(D8_BUDGET_GB[self.tp.rank] * 1e9)
        cap = max(total - self.EXPERT_GPU_OVERHEAD_BYTES - dense - kv - scratch, 0)
        return min(plan, cap)

    def _host_pin_bytes(self, dense, kv, scratch, budget, host_total=None):
        """S27-3: RAM safe to page-lock for the non-resident experts (the
        store's host cache, so a swap miss never reads the disk). The xe
        driver holds host RAM for device memory too (S26: GPUActive tracked
        the reservations), so the store's full device budget is counted
        against the host side, once per rank, plus dense/kv/scratch and the
        EXPERT_HOST_RESERVE_BYTES floor for the OS. ExpertStore still only
        pins when the whole non-resident set fits, and moe.PINNED_LIMIT_BYTES
        caps it globally."""
        if host_total is None:
            host_total = read_mem_total_bytes() if torch.xpu.device_count() else -1
        if host_total <= 0:
            return 0
        return max(host_total - self.EXPERT_HOST_RESERVE_BYTES
                   - self.tp.world * (dense + kv + scratch + budget), 0)

    def _expert_store(self, groups, tp, budget, host_pin_bytes=0):
        from ..moe import ExpertStore  # lazy import (module header note)
        return ExpertStore(groups, tp, budget, verbose=True,
                           host_pin_bytes=host_pin_bytes)

    def _expert_store_ladder(self, groups, tp, budget, host_pin_bytes=0):
        """Retry the store at 90% budget on OOM down to the floor (gpt-oss
        S17 recipe, unchanged; the host pin allowance rides along — it is a
        RAM budget, not a device one, so shrinking the device budget does
        not change it)."""
        b, floor = budget, int(self.EXPERT_BUDGET_MIN_GB * 1e9)
        while b >= floor:
            try:
                return self._expert_store(groups, tp, b, host_pin_bytes)
            except torch.OutOfMemoryError:
                gc.collect()
                if torch.device(tp.device).type == "xpu":
                    torch.xpu.empty_cache()
                nb = int(b * 0.9)
                if nb < floor:
                    raise
                if tp.is_main:
                    print(f"  MoE: OOM at {b / 1e9:.2f} GB, retrying {nb / 1e9:.2f} GB",
                          flush=True)
                b = nb
        raise RuntimeError(f"expert budget {b / 1e9:.2f} GB below floor "
                           f"{floor / 1e9:.2f} GB")

    def _load_moe(self):
        dev = self.tp.device
        if dev.type == "cpu" or os.environ.get("PYI_QWEN4EXP_SKIP_MOE"):
            # CPU scaffold: the ExpertStore page-locks real expert RAM, so
            # the model test runs the FFN block's zero path instead.
            self.store, self.moe = None, [None] * self.n_layer
            self._weights_bytes, self._scratch_bytes = self._dense_bytes, 0
            if self.tp.is_main:
                print("  qwen4exp: MoE experts skipped (CPU)", flush=True)
            return
        groups = moe_expert_groups(self.g)
        tiny = 0   # routers, shared gates, shared-expert trio (loaded natively
        for il in range(self.n_layer):    # by moe_layer, above the store)
            for t in ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                      "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                      "ffn_down_shexp.weight"):
                tiny += self.g.tensors[f"blk.{il}.{t}"].nbytes
        dense = self._dense_bytes + tiny
        kv, scratch = self.kv_bytes(), 1 << 30
        budget = self._expert_budget(dense, kv, scratch)
        host_pin = self._host_pin_bytes(dense, kv, scratch, budget)
        self._weights_bytes, self._scratch_bytes = dense, scratch
        if self.tp.is_main:
            print(f"  MoE: {int(self.g.get('expert_count'))} experts x "
                  f"{self.n_layer} layers, budget {budget / 1e9:.2f} GB "
                  f"(dense {dense / 1e9:.2f}, kv {kv / 1e9:.2f}, "
                  f"scratch {scratch / 1e9:.2f}, "
                  f"host pin {host_pin / 1e9:.2f} GB)", flush=True)
        self.store = self._expert_store_ladder(groups, self.tp, budget,
                                               host_pin)
        self.moe = [moe_layer(self.g, self.store, il, dev)
                    for il in range(self.n_layer)]
        if self.tp.is_main:
            tot = self.n_layer * len(self.store.local_ids[0])
            res = sum(len(m.resident) for m in self.store.maps.values())
            swp = sum(m.S - m.R for m in self.store.maps.values())
            print(f"    {res} of {tot} local experts resident, {swp} swap slots, "
                  f"store {self.store.memory_bytes() / 1e9:.2f} GB, "
                  f"pinned {self.store.pinned_bytes() / 1e9:.2f} GB", flush=True)

    # ---------------------------------------------------------------- forward

    @torch.inference_mode()
    def forward(self, tokens, chunk: int = 512, all_logits: bool = False):
        if self.n_past + len(tokens) > self.max_ctx:
            raise ValueError(f"context full: {self.n_past} + {len(tokens)} "
                             f"> {self.max_ctx}")
        dev = self.tp.device
        got, last = [], None
        for i0 in range(0, len(tokens), chunk):
            part = list(tokens[i0:i0 + chunk])
            T = len(part)
            ids = torch.tensor(part, dtype=torch.long, device=dev)
            pos = torch.arange(self.n_past, self.n_past + T, dtype=torch.long,
                               device=dev)
            x = self.embed_tokens(ids)
            # hc_init (Mappa A): the embedding broadcast over the streams.
            res = x.unsqueeze(1).expand(T, self.hc_n, self.n_embd) \
                   .reshape(T, self.hc_dim)
            for il, L in enumerate(self.layers):
                if self.ple is not None and il == self.ple_layer:
                    res = self._ple_step(res, part, dev)
                mixed, inj, _ = L.mix_attn.forward(res)
                if L.kind == "delta":
                    blk = self._delta_block(il, L, mixed)
                elif T == 1:
                    out, state = L.qsa.decode(mixed, pos[:1], self.qsa_cache[il])[:2]
                    self.qsa_cache[il] = state
                    blk = self.tp.all_reduce(out)
                else:
                    out, state = L.qsa.forward(mixed, pos, self.qsa_cache[il])[:2]
                    self.qsa_cache[il] = state
                    blk = self.tp.all_reduce(out)
                res = hc_combine(res, blk, inj)
                mixed2, inj2, _ = L.mix_ffn.forward(res)
                res = hc_combine(res, self._ffn_block(il, mixed2), inj2)
            mixed, _, _ = self.exit_mixer.forward(res)
            lg = self.tp.all_gather(self.lm_head(mixed))
            if all_logits:
                got.append(lg)
            last = lg[-1]
            self.n_past += T
        return torch.cat(got, 0) if all_logits else last

    def embed_tokens(self, ids):
        local = (ids >= self.v0) & (ids < self.v1)
        rows = self.embed.rows_dequant(
            (ids - self.v0).clamp(0, self.v1 - self.v0 - 1))
        return self.tp.all_reduce(rows * local.unsqueeze(-1))

    def _ple_step(self, res, part, dev):
        idx = ple_row_indices(part, self._ple_hist, **self.ple_cfg)
        emb = ple_gather_embedding(self.g, PLE_TABLE, idx, device=dev)
        nheads = (self.ple_cfg["ngram"] - 1) * self.ple_cfg["heads_per_ngram"]
        emb = emb.view(len(part), nheads, self.ple_dim).reshape(len(part),
                                                                self.n_embd)
        res, self.ple_state, _ = self.ple.forward(res, emb, self.ple_state)
        self._ple_hist = np.concatenate([self._ple_hist,
                                         np.asarray(part, dtype=np.int64)])
        return res

    def _delta_block(self, il, L, mixed):
        """The chain-verified pure-torch DeltaNet (test_qwen4exp_chain
        ._deltanet, TP-split); states are replacement-style, so conv_prefill/
        delta_prefill run read-only on the cached ones."""
        from .. import deltanet  # lazy: Triton kernels live in that module
        T, S = mixed.shape[0], self.ssm
        nk, nv = self.n_key_l, self.n_value_l
        qkv = L.qkv(mixed)                       # [T, qkv_l]
        z = L.gate(mixed)                        # [T, nv*S]
        ab = L.ab(mixed)                         # [T, 2*nv]
        g = L.ssm_a * F.softplus(ab[:, :nv] + L.dt_bias)
        beta = torch.sigmoid(ab[:, nv:])         # Mappa A.3
        conv, state = self.delta_cache[il]
        qkv, conv_new = deltanet.conv_prefill(qkv, conv, L.conv)
        q = deltanet.l2norm_qk(qkv[:, :nk * S].view(T, nk, S), self.eps)
        k = deltanet.l2norm_qk(qkv[:, nk * S:2 * nk * S].view(T, nk, S), self.eps)
        v = qkv[:, 2 * nk * S:].view(T, nv, S)
        out, st_new = deltanet.delta_prefill(q, k, v, g, beta, state)
        self.delta_cache[il] = (conv_new, st_new)
        # Flash-Next gate is sigmoid(z), not qwen35's silu(z) (S24 chain).
        y = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps) \
            * L.ssm_norm
        y = y * torch.sigmoid(z.view(T, nv, S))
        return self.tp.all_reduce(L.out(y.reshape(T, -1)))

    def _ffn_block(self, il, x):
        M = self.moe[il]
        if M is None:
            return torch.zeros_like(x)
        if x.shape[0] == 1:
            return M.decode(x[0]).view(1, -1)   # [k] flat, all-reduced, +shared
        return M.prefill(x)                     # [T, k], all-reduced, +shared
QWEN4V0_EOF
wc -l "$OUT/qwen4exp_v0.py"

cat > "$OUT/s2711_ab.py" <<'ABPY_EOF'
"""S27-11 diagnostic (2.5: find the first component that diverges).

Decodes the chat-it and code prompts through the PRE-step-4 model module
(mode `old`, file embedded in the job -> $OUT/qwen4exp_v0.py — the path that
produced the S27-4 golden) or through the CURRENT segmented eager driver
(mode `new`, fast_decode/use_graph exactly as `compare_probs ours --decode
fused`), and dumps the tensors the two share a 1:1 alignment of:

  * every hc_combine/`hc_combine_` (sequence c0=attn L0, c1=ffn/moe L0, ...)
    — block output and inject;
  * PLE.forward output; Qwen4Exp.embed_tokens output;
  * MoE._logits router logits per layer (routing ids follow deterministically);
  * QSA.decode pre-reduce attention output per QSA layer;
  * the step logits (top of the vocab distribution, full tensor).

Only the decode steps are recorded (DUMP gate; prefill is not the suspect:
ppl vs llama is green). Run twice under torchrun, then diff the dumps with
$OUT/s2711_diff.py.

usage (inside a GPU job):
  S2710_OLD_MOD=$OUT/qwen4exp_v0.py torchrun --nproc_per_node=2 \
      $OUT/s2711_ab.py old MODEL OUT        # or: new
"""
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
MODE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
assert MODE in ("old", "new")

if MODE == "old":
    import pynference.models  # noqa: F401  package first: the old file's
    # relative imports resolve against it; the module is never imported by
    # __init__ (load_model imports qwen4exp lazily), so the injection below
    # wins for `from .qwen4exp import Qwen4Exp` inside load_model.
    spec = importlib.util.spec_from_file_location(
        "pynference.models.qwen4exp", os.environ["S2710_OLD_MOD"])
    qmod = importlib.util.module_from_spec(spec)
    sys.modules["pynference.models.qwen4exp"] = qmod
    spec.loader.exec_module(qmod)
else:
    from pynference.models import qwen4exp as qmod

from bench.compare_probs import PROMPTS              # noqa: E402
from pynference import moe as moemod                 # noqa: E402
from pynference.gguf import GGUFFile                 # noqa: E402
from pynference.models import load_model             # noqa: E402
from pynference.parallel import TP                   # noqa: E402
from pynference.tokenizer import load_tokenizer      # noqa: E402

DUMP = {"on": False}
NS = {}
SEQ = {"c": 0, "q": 0, "l": 0}


def seqreset():
    SEQ["c"] = SEQ["q"] = SEQ["l"] = 0


def _rec(name, t):
    NS[name] = t.detach().float().cpu().numpy().copy()


def _hooks():
    seq = SEQ

    orig = qmod.hc_combine if MODE == "old" else qmod.hc_combine_

    def comb(res_hc, block_out, inject):
        out = orig(res_hc, block_out, inject)
        if DUMP["on"]:
            _rec(f"c{seq['c']}_blk", block_out)
            _rec(f"c{seq['c']}_inj", inject)
            seq["c"] += 1
        return out

    if MODE == "old":
        qmod.hc_combine = comb
    else:
        qmod.hc_combine_ = comb

    orig_ple = qmod.PLE.forward

    def ple(self, res, emb, state, **kw):
        out = orig_ple(self, res, emb, state, **kw)
        if DUMP["on"]:
            _rec("ple", out[0])
        return out

    qmod.PLE.forward = ple

    orig_emb = qmod.Qwen4Exp.embed_tokens

    def emb(self, ids):
        r = orig_emb(self, ids)
        if DUMP["on"]:
            _rec("emb", r)
        return r

    qmod.Qwen4Exp.embed_tokens = emb

    orig_qd = qmod.QSA.decode

    def qsa_dec(self, cur, pos, state):
        r = orig_qd(self, cur, pos, state)
        if DUMP["on"]:
            _rec(f"qsa{seq['q']}_out", r[0])
            seq["q"] += 1
        return r

    qmod.QSA.decode = qsa_dec

    orig_lg = moemod.MoE._logits

    def logits(self, x):
        r = orig_lg(self, x)
        if DUMP["on"]:
            _rec(f"lg{seq['l']}", r)
            seq["l"] += 1
        return r

    moemod.MoE._logits = logits


def main():
    tp = TP()
    g = GGUFFile(MODEL)
    tok = load_tokenizer(g)
    model = load_model(g, tp, max_ctx=1024)
    model.fast_decode = True
    model.use_graph = False
    _hooks()
    for name, text in PROMPTS:
        if name not in ("chat-it", "code"):
            continue
        ids = tok.encode(text)
        model.reset()
        logits = model.forward(ids)
        for t in range(3):
            NS.clear()
            seqreset()
            DUMP["on"] = True
            logits = model.forward([int(torch.softmax(logits.double(), -1).argmax())])
            DUMP["on"] = False
            _rec("logits", logits)
            if tp.is_main:
                d = f"{OUT}/ab/{MODE}"
                os.makedirs(d, exist_ok=True)
                np.savez_compressed(f"{d}/{name}_step{t}.npz", **NS)
                print(f"{MODE} {name} step {t}: {len(NS)} records", flush=True)
    tp.close()


if __name__ == "__main__":
    main()
ABPY_EOF

cat > "$OUT/s2711_diff.py" <<'DIFFPY_EOF'
"""S27-11: compare the old/new decode dumps of $OUT/s2711_ab.py.

Prints, per (prompt, step), the first dumped tensor (in sequence order) whose
max abs diff exceeds TOL, plus the top-10 ids/probs of both logits at that
step. usage: python $OUT/s2711_diff.py OUT [TOL]
"""
import sys

import numpy as np

OUT = sys.argv[1]
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
KEYORD = ("emb", "ple", "logits")


def order(k):
    if k.startswith("c"):
        return (0, int(k[1:].split("_")[0]))
    if k.startswith("lg"):
        return (1, int(k[2:]))
    if k.startswith("qsa"):
        return (2, int(k[3:].split("_")[0]))
    return (3, KEYORD.index(k) if k in KEYORD else 9)


def top10(a):
    a = np.asarray(a, dtype=np.float64)
    p = np.exp(a - a.max())
    p /= p.sum()
    idx = np.argsort(-p)[:10]
    return [(int(i), round(float(p[i]), 4)) for i in idx]


for f in sorted(__import__("glob").glob(OUT + "/ab/old/*.npz")):
    name = f.split("/")[-1]
    o = np.load(f)
    n = np.load(OUT + "/ab/new/" + name)
    keys = sorted(set(o.files) & set(n.files), key=order)
    first = None
    bad = 0
    for k in keys:
        d = float(np.max(np.abs(o[k].astype(np.float64) - n[k].astype(np.float64))))
        if d > TOL:
            bad += 1
            if first is None:
                first = (k, d)
    tag = "OK" if first is None else f"first diverging: {first[0]} max|d| {first[1]:.3e}"
    print(f"{name}: {bad}/{len(keys)} records diverge; {tag}")
    if first is not None and "logits" in o.files:
        print("   old logits top10:", top10(o["logits"]))
        print("   new logits top10:", top10(n["logits"]))
DIFFPY_EOF

# ---- 1) qwen4exp A/B dei layer (pre-passo-4 vs corrente, eager) -------------
ra=0; rb=0
for MODE in old new; do
  S2710_OLD_MOD="$OUT/qwen4exp_v0.py" timeout 1500 torchrun --nproc_per_node=2 \
    "$OUT/s2711_ab.py" "$MODE" "$MQ" "$OUT" > "$OUT/ab_$MODE.log" 2>&1
  rc=$?
  if [ "$MODE" = old ]; then ra=$rc; else rb=$rc; fi
  echo "== ab $MODE exit $rc ($(date +%T)) (atteso 0)"
  clean "$OUT/ab_$MODE.log" | tail -n 4
done
python "$OUT/s2711_diff.py" "$OUT" > "$OUT/ab_diff.log" 2>&1; rd=$?
echo "== ab diff exit $rd ($(date +%T))"
cat "$OUT/ab_diff.log"

# ---- 2) qwen4exp: gate C8 + golden-vs-passi sul path FUSED (eager) ----------
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$GQ/ref_probs.json" -m "$MQ" --decode fused -o "$OUT/ours_q4_fused.json" \
  > "$OUT/ours_q4_fused.log" 2>&1
rc_f=$?
echo "== q4 ours(fused) exit $rc_f ($(date +%T)) (atteso 0)"
clean "$OUT/ours_q4_fused.log" | tail -n 4
python "$OUT/run_bench.py" bench.compare_probs diff --strict "$GQ/ref_probs.json" \
  "$OUT/ours_q4_fused.json" > "$OUT/diff_q4_fused.log" 2>&1
rc_df=$?
echo "== q4 diff strict (fused) exit $rc_df ($(date +%T)) (atteso 0 = gate)"
cat "$OUT/diff_q4_fused.log"
python "$OUT/cmp.py" "$GQ/ours_baseline.json" "$OUT/ours_q4_fused.json" "" "" \
  > "$OUT/cmp_q4_fused_vs_golden.log" 2>&1
echo "== q4 fused vs golden (json S27-4):"
cat "$OUT/cmp_q4_fused_vs_golden.log"

# ---- 3) gemma4 C10: comandi equivalenti a regress.sh, path nuovo ------------
timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
  --ref "$GG/ref_probs.json" -m "$MG" --decode graph -o "$OUT/ours_gemma.json" \
  > "$OUT/ours_gemma.log" 2>&1
rc_g=$?
echo "== gemma ours(graph) exit $rc_g ($(date +%T)) (atteso 0)"
clean "$OUT/ours_gemma.log" | tail -n 4
timeout 900 torchrun --nproc_per_node=2 bench/ppl.py -m "$MG" -c 512 --chunks 8 \
  > "$OUT/ppl_gemma.log" 2>&1
rc_pg=$?
echo "== gemma ppl exit $rc_pg ($(date +%T)) (atteso 0)"
grep -a "Final estimate" "$OUT/ppl_gemma.log"
python "$OUT/cmp.py" "$GG/ours_baseline.json" "$OUT/ours_gemma.json" \
  "$GG/ours_ppl.txt" "$OUT/ppl_gemma.log" > "$OUT/cmp_gemma.log" 2>&1
rc_cg=$?
echo "== gemma cmp exit $rc_cg ($(date +%T))"
cat "$OUT/cmp_gemma.log"

echo "== summary: ab=$((ra + rb)) diff=$rd q4ours=$rc_f q4fused_gate=$rc_df gemma=$((rc_g + rc_pg + rc_cg))"
echo "== end $(date +%T)"
