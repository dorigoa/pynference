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


def hc_combine_(res_hc: torch.Tensor, block_out: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
    """`hc_combine` writing into `res_hc` in place (same numbers) — the
    decode/graph path needs `res_hc` to keep one address across steps."""
    T = res_hc.shape[0]
    hc = inject.shape[1]
    w = 2.0 * torch.sigmoid(inject)
    res_hc.view(T, hc, -1).add_(block_out.unsqueeze(1) * w.unsqueeze(-1))
    return res_hc


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

    def forward(self, res_hc: torch.Tensor, emb: torch.Tensor, state: torch.Tensor,
                *, inplace_state: bool = False):
        """-> (res_hc_out, new_state, parts).

        parts keeps the cb()-dumped intermediates for the dump comparison
        (torch layouts: ple_gate [T, hc], *_value/conv [T, hc, n_embd]).
        With `inplace_state` the write-back tail lands in `state` itself and
        is returned (same values; the S27 graph decode needs a fixed buffer).
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
        tail = x[-self.hist:]  # write-back tail (build_conv_state_at)
        if inplace_state:
            state.copy_(tail)
            new_state = state
        else:
            new_state = tail.clone()
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

    def zero_state(self, device=None, capacity: int = 0, tables=None) -> dict:
        """Append-only caches: raw indexer keys + their cell positions, the
        K/V cache (cells in sequence order) and, for `decode`, the pooled
        indexer keys per block (`forward` recomputes those from the raw rows
        each call and ignores/does not carry `pool`).

        `capacity > 0` instead returns the S27 STATIC state: every cache is a
        fixed buffer sized for `capacity` cells, a device scalar `n` counts
        the valid ones, and `decode` updates the buffers in place (no cat, no
        rope-table rebuild, no host sync — the shape the XPUGraph replay
        needs). `tables` = the shared (cos, sin) rope tables for the
        positions. `absorb` folds an append-style step (prefill chunk) into a
        static state. Static states are recognized by their `n` entry."""
        if capacity <= 0:
            e = lambda *shape: torch.zeros(0, *shape, device=device)  # noqa: E731
            return {"idx_keys": e(self.idx_dim),
                    "idx_pos": torch.zeros(0, dtype=torch.int64, device=device),
                    "kv_k": e(self.n_head_kv, self.head_dim),
                    "kv_v": e(self.n_head_kv, self.head_dim),
                    "pool": e(self.idx_dim)}
        cp = self.compress
        pool_cap = (capacity + cp - 1) // cp
        cells = pool_cap * cp           # whole blocks, so `vis` can read any slot
        cos, sin = tables if tables is not None else \
            rope_table(self.rope_n_rot, self.rope_base, None, capacity, device)
        return {"n": torch.zeros((), dtype=torch.int64, device=device),
                "idx_keys": torch.zeros(cells, self.idx_dim, device=device),
                # padded cells read "far in the future": never visible, never kept
                "idx_pos": torch.full((cells,), 1 << 60, dtype=torch.int64, device=device),
                "kv_k": torch.zeros(capacity, self.n_head_kv, self.head_dim, device=device),
                "kv_v": torch.zeros(capacity, self.n_head_kv, self.head_dim, device=device),
                "pool": torch.zeros(pool_cap, self.idx_dim, device=device),
                "cos": cos, "sin": sin}

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
        if "n" in state:
            # static state (S27): run the step on views of its first `n0`
            # cells and hand back an append-style state; the caller folds it
            # in place with `absorb`. Host sync is fine — prefill is not
            # graph-captured.
            n0 = int(state["n"])
            prev = {k: state[k][:n0]
                    for k in ("idx_keys", "idx_pos", "kv_k", "kv_v")}
        else:
            prev = state

        # ---- indexer: pooled keys, query, scores, selection (build_qsa_sel)
        k_raw = self.idx_k(cur)                                       # [T, 128]
        # the cache is an f16 buffer (SET_ROWS f16) that pooling gathers
        # back through, so the pooled keys see f16-rounded raw rows
        idx_keys = torch.cat([prev["idx_keys"].to(dev), k_raw], 0).half().float()  # [N, 128]
        idx_pos = torch.cat([prev["idx_pos"].to(dev), pos])          # [N]
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

        kv_k = torch.cat([prev["kv_k"].to(dev), k_roped], 0)
        kv_v = torch.cat([prev["kv_v"].to(dev), v], 0)

        # ---- masked GQA attention + gate (build_attn_qsa / 1069-1078)
        attn = self._attend(q_roped, kv_k, kv_v, sel)
        attn_flat = attn.reshape(T, self.n_head * self.head_dim)
        gate_sig = torch.sigmoid(gate.reshape(T, self.n_head * self.head_dim))
        out = self.wo(attn_flat * gate_sig)

        parts = {
            "indexer_k_raw": k_raw,
            "indexer_pool_k_new": pooled[prev["idx_keys"].shape[0] // cp:],
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

        A static state (`zero_state(capacity=...)`) takes `_decode_static`
        instead: same numbers, in-place fixed-shape buffers, parts=None.
        """
        if "n" in state:
            return self._decode_static(cur, pos, state)
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

    def _decode_static(self, cur: torch.Tensor, pos: torch.Tensor, state: dict):
        """One decode token against a static state (S27 graph decode): every
        cache is a fixed buffer updated in place, `state["n"]` (device
        int64) counts the valid cells, and the whole step is device-only and
        fixed-shape — no cat, no rope-table rebuild (shared tables in the
        state), no `.item()`, so `kernels.qsa_attn_decode` and the segment
        XPUGraph can bake it. Same numbers as `decode`:

          * the only pool row whose clamp can change is the one the new cell
            lands in — exactly one row per step, `nb = n // cp`;
          * scoring runs over the pool rows that exist (`n_new // cp`), like
            `decode`, not over the whole capacity: the top-k cut is a
            relative ranking, so scoring the sentinel-hidden padding rows
            too would let a winner be a -inf row and drop a real block whose
            score is lower; rows past the cut could never be picked anyway;
          * attention gathers from the full-capacity K/V cache; the kernel
            (and the CPU `sel`) only touch cells in the fixed idx list.
        parts is None: the dump-parity nodes belong to the append path.
        """
        dev = cur.device
        pos = pos.to(device=dev, dtype=torch.int64).view(1)
        p0 = pos.view(())
        cp = self.compress
        n = state["n"]
        n_new = n + 1

        k_raw = self.idx_k(cur)                                       # [1, 128]
        state["idx_keys"].index_copy_(0, n.view(1), k_raw.half().float())
        state["idx_pos"].index_copy_(0, n.view(1), pos)

        nb = torch.div(n, cp, rounding_mode="floor")                  # the new cell's block
        cells = nb * cp + torch.arange(cp, device=dev)
        cid = torch.where(cells <= n, cells, torch.zeros_like(cells))  # GET_ROWS -> row 0
        row = grouped_norm(state["idx_keys"][cid].sum(0) * (1.0 / cp),
                           self.idx_k_norm, self.eps)
        state["pool"].index_copy_(0, nb.view(1), row.unsqueeze(0))
        pool = state["pool"]
        pool_cap = pool.shape[0]

        cos_all, sin_all = state["cos"], state["sin"]
        q_idx = self.idx_q(cur).view(1, self.idx_heads, self.idx_dim)
        q_idx = grouped_norm(q_idx, self.idx_q_norm, self.eps)
        q_idx = rope_neox(q_idx, cos_all[pos], sin_all[pos])

        n_pool = torch.div(n_new, cp, rounding_mode="floor").clamp(max=pool_cap)
        rows = torch.arange(pool_cap, device=dev) < n_pool
        last = state["idx_pos"][torch.arange(pool_cap, device=dev) * cp + (cp - 1)]
        vis = (last <= p0) & rows
        score = q_idx.new_zeros(pool_cap)
        for h in range(self.idx_heads):
            score = score + torch.relu(pool @ q_idx[0, h])
        score = score * self.idx_scale
        score = torch.where(vis, score, score.new_full((), float("-inf")))
        score = torch.where(rows, score, score.new_full((), float("-inf")))

        k = min(pool_cap, self.n_top_pool)
        top = score.topk(k).indices
        bcells = top[:, None] * cp + torch.arange(cp, device=dev)
        # `vis` is never indexed with a tensor here: torch's bool advanced
        # indexing (`vis[top]`, `vis[nb]`) enqueues an XPU kernel that
        # Level Zero refuses to record into a graph (UR error 44,
        # UNSUPPORTED_FEATURE — S27-6 crashed at `vis[nb]`, right after
        # `topk` and `vis[top]` had captured fine; `topk` itself is
        # capturable). `vis` is (last <= p0) & (arange < n_pool), so both
        # lookups are rebuilt from int64 gathers and comparisons — same
        # values, only capture-friendly ops.
        ok = ((last.gather(0, top) <= p0) & (top < n_pool))[:, None] \
            & (bcells < n_new)
        head = torch.where(ok, bcells, torch.full_like(bcells, -1)).reshape(-1)
        tail = nb * cp + torch.arange(cp, device=dev)                 # the incomplete last block
        nb_vis = (last.gather(0, nb.view(1)) <= p0) & (nb < n_pool)
        ok_tail = (tail < n_new) & (~nb_vis) \
            & (state["idx_pos"][torch.minimum(tail, n)] <= p0)
        idx = torch.cat([head, torch.where(ok_tail, tail, torch.full_like(tail, -1))]) \
            .to(torch.int32)

        q_full = self.wq(cur)
        qv = q_full.view(1, self.n_head, 2 * self.head_dim)
        q_pre, gate = qv[..., :self.head_dim], qv[..., self.head_dim:]
        q_normed = grouped_norm(q_pre, self.q_norm, self.eps)
        q_roped = rope_neox(q_normed, cos_all[pos], sin_all[pos])
        k_pre = self.wk(cur).view(1, self.n_head_kv, self.head_dim)
        k_normed = grouped_norm(k_pre, self.k_norm, self.eps)
        k_roped = rope_neox(k_normed, cos_all[pos], sin_all[pos])
        v = self.wv(cur).view(1, self.n_head_kv, self.head_dim)
        state["kv_k"].index_copy_(0, n.view(1), k_roped)
        state["kv_v"].index_copy_(0, n.view(1), v)

        if dev.type == "cpu":
            sel = torch.full((state["kv_k"].shape[0],), float("-inf"), device=dev)
            sel[idx[idx >= 0].long()] = 0.0
            attn = self._attend(q_roped, state["kv_k"], state["kv_v"], sel.unsqueeze(0))
        else:
            from pynference import kernels
            attn = kernels.qsa_attn_decode(q_roped[0].contiguous(), state["kv_k"],
                                           state["kv_v"], idx, self.kq_scale).unsqueeze(0)

        attn_flat = attn.reshape(1, self.n_head * self.head_dim)
        gate_sig = torch.sigmoid(gate.reshape(1, self.n_head * self.head_dim))
        out = self.wo(attn_flat * gate_sig)
        state["n"].add_(1)
        return out, state, None

    def absorb(self, state: dict, new: dict) -> None:
        """Fold an append-style step state (`forward`'s `new_state`) into a
        static `state` in place: copy the added rows, rebuild the pool rows
        from floor(n/cp) up — the previously-partial block's clamped gather
        changed too — and set `n`. Prefill-only (host values fine); the pool
        formula is `decode`'s, so the folded state scores identically."""
        n = int(state["n"])
        nn = new["idx_keys"].shape[0]
        assert nn <= state["kv_k"].shape[0], "static QSA state over capacity"
        if nn > n:
            # `new` is forward's full concatenated state, not just the delta
            for key in ("idx_keys", "idx_pos", "kv_k", "kv_v"):
                state[key][n:nn].copy_(new[key][n:])
        cp = self.compress
        n_pool = (nn + cp - 1) // cp
        b0 = n // cp
        if n_pool > b0:
            dev = state["pool"].device
            rows = torch.arange(b0, n_pool, device=dev)
            cells = rows[:, None] * cp + torch.arange(cp, device=dev)
            cid = torch.where(cells < nn, cells, torch.zeros_like(cells))
            state["pool"][b0:n_pool] = grouped_norm(
                state["idx_keys"][cid].sum(1) * (1.0 / cp), self.idx_k_norm, self.eps)
        state["n"].fill_(nn)

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
        mutated in place), the PLE conv history plus the host-side token
        history restart.

        On XPU the QSA states are the S27 STATIC buffers (fixed at max_ctx,
        shared rope tables) — the shape the segmented graph decode replays;
        the CPU keeps the append-style state so the reference tests run the
        original code path. ~254 MB/rank at max_ctx, inside the kv_bytes()
        reservation. A new reset invalidates any captured decode graphs (they
        pin the freed buffers), so `_capture` runs again on the next decode."""
        self.n_past = 0
        dev = self.tp.device
        self.delta_cache = {
            il: (torch.zeros(self.conv_k - 1, self._qkv_l, device=dev),
                 torch.zeros(self.n_value_l, self.ssm, self.ssm, device=dev))
            for il, L in enumerate(self.layers) if L.kind == "delta"}
        cap = self.max_ctx if torch.xpu.device_count() > 0 else 0
        if cap and not hasattr(self, "_rope_cs"):
            self._rope_cs = rope_table(self.rope_dim, self.rope_base, None,
                                       self.max_ctx, dev)
        tables = getattr(self, "_rope_cs", None)
        self.qsa_cache = {il: L.qsa.zero_state(dev, cap, tables)
                          for il, L in enumerate(self.layers) if L.kind == "qsa"}
        self.ple_state = self.ple.zero_state(dev) if self.ple is not None else None
        self._ple_hist = np.zeros(0, dtype=np.int64)
        self._g0 = None
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
        if (len(tokens) == 1 and not all_logits and self.fast_decode
                and self.store is not None):
            return self.decode(tokens[0])
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
                else:
                    call, pos_arg = (L.qsa.decode, pos[:1]) if T == 1 \
                        else (L.qsa.forward, pos)
                    out, new = call(mixed, pos_arg, self.qsa_cache[il])[:2]
                    if new is not self.qsa_cache[il]:
                        if "n" in self.qsa_cache[il]:
                            L.qsa.absorb(self.qsa_cache[il], new)  # keep the buffers
                        else:
                            self.qsa_cache[il] = new
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

    def _delta_block(self, il, L, mixed, inplace: bool = False):
        """The chain-verified pure-torch DeltaNet (test_qwen4exp_chain
        ._deltanet, TP-split). States are replacement-style by default;
        `inplace=True` (S27 graph decode, T=1) writes the new conv/ssm state
        back into the cached buffers via the `state_out` kwargs — the graph
        segments replay against fixed addresses — and the T=1 gate tail runs
        the fused `norm_sigmoid_decode` kernel (CPU twin in tests)."""
        from .. import deltanet  # lazy: Triton kernels live in that module
        T, S = mixed.shape[0], self.ssm
        nk, nv = self.n_key_l, self.n_value_l
        qkv = L.qkv(mixed)                       # [T, qkv_l]
        z = L.gate(mixed)                        # [T, nv*S]
        ab = L.ab(mixed)                         # [T, 2*nv]
        g = L.ssm_a * F.softplus(ab[:, :nv] + L.dt_bias)
        beta = torch.sigmoid(ab[:, nv:])         # Mappa A.3
        conv, state = self.delta_cache[il]
        if inplace:
            qkv, _ = deltanet.conv_prefill(qkv, conv, L.conv, state_out=conv)
        else:
            qkv, conv_new = deltanet.conv_prefill(qkv, conv, L.conv)
        q = deltanet.l2norm_qk(qkv[:, :nk * S].view(T, nk, S), self.eps)
        k = deltanet.l2norm_qk(qkv[:, nk * S:2 * nk * S].view(T, nk, S), self.eps)
        v = qkv[:, 2 * nk * S:].view(T, nv, S)
        if inplace:
            out, _ = deltanet.delta_prefill(q, k, v, g, beta, state,
                                            state_out=state)
        else:
            out, st_new = deltanet.delta_prefill(q, k, v, g, beta, state)
            self.delta_cache[il] = (conv_new, st_new)
        # Flash-Next gate is sigmoid(z), not qwen35's silu(z) (S24 chain).
        if T == 1:
            y = deltanet.norm_sigmoid_decode(out[0], z.view(nv, S),
                                             L.ssm_norm, self.eps) \
                  .reshape(1, nv * S)
        else:
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

    # ---- segmented decode (S27, the gptoss S18 pattern) ----
    # One token becomes 1 + 2*n_layer + 1 small graphs: G0 (embedding -> hc
    # residual), A[il] (previous layer's MoE reduce+shared + PLE step + attn
    # mixer/delta-or-static-QSA + ffn mixer + router logits), B[il] (MoE
    # expert body), GF (last reduce + exit mixer + head). The one host sync
    # per layer is `decode_route` between A[il] and B[il]. Every buffer the
    # graphs touch is static: _res/_ffn_in/_ple_emb/_inj slots, the MoE
    # decode buffers, the static QSA states, the delta/PLE state buffers
    # (`_delta_block(inplace=True)`, PLE `inplace_state=True`), and the
    # device twins of token/pos. Cross-graph tensors (_lg, _inj) are pool
    # tensors held by Python references, so their addresses survive replays
    # (gptoss precedent). Requires the store (GPU): on CPU `store` is None
    # and `forward` keeps running the reference path.

    def _init_decode(self) -> None:
        if getattr(self, "_res", None) is not None:
            return
        assert self.store is not None and all(m is not None for m in self.moe), \
            "segmented decode needs the expert store (GPU)"
        dev = self.tp.device
        host = torch.zeros(2, dtype=torch.long)
        self._host = host.pin_memory()
        self._dbuf = torch.zeros(2, dtype=torch.long, device=dev)
        self._ids, self._pos = self._dbuf[0:1], self._dbuf[1:2]
        self._res = torch.zeros(1, self.hc_dim, dtype=torch.float32, device=dev)
        self._ffn_in = torch.zeros(1, self.n_embd, dtype=torch.float32, device=dev)
        self._ple_emb = torch.zeros(1, self.n_embd, dtype=torch.float32, device=dev)
        self._inj = [None] * self.n_layer
        for m in self.moe:
            m._init_decode_bufs(dev)

    def _embed_body(self) -> None:
        """Segment G0: token lookup -> hc_init (embedding broadcast over the
        streams) written into the static residual."""
        x = self.embed_tokens(self._ids)
        self._res.copy_(x.unsqueeze(1).expand(1, self.hc_n, self.n_embd)
                        .reshape(1, self.hc_dim))

    def _moe_finish(self, il: int) -> torch.Tensor:
        """Layer il's MoE: all-reduce the [k] expert accumulator (B[il]
        replayed) and add the gated shared expert — computed from _ffn_in,
        which still holds layer il's ffn input at this point (same order as
        MoE.decode: reduce first, replicated shared added after)."""
        y = self.tp.all_reduce(self.moe[il]._d["y"])
        if self.moe[il].shared is not None:
            y = y + self.moe[il]._shared_out(self._ffn_in)[0]
        return y.view(1, -1)

    def _layer_body(self, il: int):
        """Segment A[il]; returns the router logits for the host routing step
        (None is never produced on GPU — every layer has its MoE)."""
        L = self.layers[il]
        if il > 0:
            hc_combine_(self._res, self._moe_finish(il - 1), self._inj[il - 1])
        if self.ple is not None and il == self.ple_layer:
            # the emb rows are gathered host-side in decode(); only the PLE
            # math runs here, with the conv state written in place
            out, self.ple_state, _ = self.ple.forward(
                self._res, self._ple_emb, self.ple_state, inplace_state=True)
            self._res.copy_(out)   # PLE.forward returns a new tensor: keep the address
        mixed, inj, _ = L.mix_attn.forward(self._res)
        if L.kind == "delta":
            blk = self._delta_block(il, L, mixed, inplace=True)
        else:
            out, _, _ = L.qsa.decode(mixed, self._pos, self.qsa_cache[il])
            blk = self.tp.all_reduce(out)
        hc_combine_(self._res, blk, inj)
        mixed2, inj2, _ = L.mix_ffn.forward(self._res)
        self._ffn_in.copy_(mixed2)
        self._inj[il] = inj2       # pool tensor: held here, read by A[il+1]
        lg = self.moe[il]._logits(self._ffn_in)
        self.moe[il]._d["y"].zero_()
        return lg

    def _final_body(self) -> torch.Tensor:
        """Segment GF: last MoE reduce, exit mixer, vocab-split head: [1, n_vocab]."""
        hc_combine_(self._res, self._moe_finish(self.n_layer - 1),
                    self._inj[self.n_layer - 1])
        mixed, _, _ = self.exit_mixer.forward(self._res)
        return self.tp.all_gather(self.lm_head(mixed))

    def _run_layers_eager(self) -> None:
        """The driver of the graph, eager: A[il], the host routing step, then
        one B[il] replay per resolve batch (identical bodies, no capture)."""
        for il in range(self.n_layer):
            lg = self._layer_body(il)
            m = self.moe[il]
            nb = m.decode_fill(*m.decode_route(lg))
            for bi in range(nb):
                m.decode_fill_batch(bi)
                m.decode_body(self._ffn_in)

    def _decode_eager(self) -> torch.Tensor:
        self._embed_body()
        self._run_layers_eager()
        return self._final_body()[0]

    def _decode_graph(self) -> torch.Tensor:
        self._g0.replay()
        for il in range(self.n_layer):
            self._ga[il].replay()
            m = self.moe[il]
            nb = m.decode_fill(*m.decode_route(self._lg[il]))
            for bi in range(nb):
                m.decode_fill_batch(bi)
                self._gb[il].replay()
        self._gf.replay()
        return self._graph_out[0]

    def _capture(self) -> None:
        """Warm the whole path twice eagerly on a side stream, then capture
        G0 / A[il] / B[il] / GF. The warmups run real steps, so the states
        they mutate (delta buffers in place, PLE conv state, the QSA static
        `n`/rows with garbage tokens) are snapshotted before and restored
        after — capture itself records without executing, and the garbage
        rows sit behind the `idx_pos` sentinel / below `n` and get
        recomputed before they could ever be scored."""
        dev = self.tp.device
        snap = {il: (c.clone(), s.clone())
                for il, (c, s) in self.delta_cache.items()}
        ple_snap = self.ple_state.clone() if self.ple_state is not None else None
        n0 = (int(next(iter(self.qsa_cache.values()))["n"])
              if self.qsa_cache else 0)
        s = torch.xpu.Stream(dev)
        s.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(s):
            for _ in range(2):
                self._decode_eager()
            torch.xpu.current_stream().wait_stream(s)
        torch.xpu.synchronize(dev)
        self._g0 = torch.xpu.XPUGraph()
        with torch.xpu.graph(self._g0):
            self._embed_body()
        self._ga = [torch.xpu.XPUGraph() for _ in range(self.n_layer)]
        self._gb = [torch.xpu.XPUGraph() for _ in range(self.n_layer)]
        self._lg = [None] * self.n_layer
        for il in range(self.n_layer):
            with torch.xpu.graph(self._ga[il]):
                self._lg[il] = self._layer_body(il)
            with torch.xpu.graph(self._gb[il]):
                self.moe[il].decode_body(self._ffn_in)
        self._gf = torch.xpu.XPUGraph()
        with torch.xpu.graph(self._gf):
            self._graph_out = self._final_body()
        torch.xpu.synchronize(dev)
        for il, (c, s_) in snap.items():
            self.delta_cache[il][0].copy_(c)
            self.delta_cache[il][1].copy_(s_)
        if ple_snap is not None:
            self.ple_state.copy_(ple_snap)
        for state in self.qsa_cache.values():
            state["n"].fill_(n0)
            state["idx_pos"][n0:].fill_(1 << 60)

    @torch.inference_mode()
    def decode(self, token: int) -> torch.Tensor:
        """One decode token -> logits [n_vocab]. Host-side work only: the PLE
        n-gram rows (numpy hash + table gather into _ple_emb), then the
        pinned token/pos pair, then graph replay or its eager twin."""
        if self.n_past >= self.max_ctx:
            raise ValueError(f"context full: {self.n_past} >= {self.max_ctx}")
        self._init_decode()
        if self.ple is not None:
            idx = ple_row_indices([token], self._ple_hist, **self.ple_cfg)
            emb = ple_gather_embedding(self.g, PLE_TABLE, idx, device=self.tp.device)
            nheads = (self.ple_cfg["ngram"] - 1) * self.ple_cfg["heads_per_ngram"]
            emb = emb.view(1, nheads, self.ple_dim).reshape(1, self.n_embd)
            self._ple_emb.copy_(emb)
            self._ple_hist = np.concatenate([self._ple_hist,
                                             np.asarray([token], dtype=np.int64)])
        self._host[0] = token
        self._host[1] = self.n_past
        self._dbuf.copy_(self._host, non_blocking=True)
        self.n_past += 1
        if self.use_graph:
            if getattr(self, "_g0", None) is None:
                self._capture()
            return self._decode_graph()
        return self._decode_eager()
