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
"""
from __future__ import annotations

import math

import numpy as np
import torch

from pynference.models.gemma4 import rope_neox, rope_table
from pynference.quant.ref import ref_dequant
from pynference.quant.weight import NativeQWeight, make_qweight


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
