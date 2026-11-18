# S23 map B — compressed attention and indexer (qwen4exp.cpp)

Reference: `scratch/ref/llama.cpp/src/models/qwen4exp.cpp`. Line numbers
refer to it. Facts in `tests/golden/qwen4exp/facts.json`.
Layers: blk.3,7,...,47 (`compress_ratios=4` on all; no plain attention).

## Tensor set per QSA layer (builder 261-273)

| tensor | shape | note |
| --- | --- | --- |
| `attn_q` (wq) | (2560, 12288) | `[q|gate]` interleaved per head: 24 heads x 2 x 256 |
| `attn_k`, `attn_v` | (2560, 512) | 2 KV heads x 256 (GQA 24/2) |
| `attn_output` (wo) | (6144, 2560) | |
| `attn_q_norm`, `attn_k_norm` | (256) | per-head RMSNorm on head dim 256 |
| `indexer_q_proj` | (2560, 512) BF16 | 4 heads x 128 |
| `indexer_k_proj` | (2560, 128) BF16 | 1 key head |
| `indexer_q_norm`, `indexer_k_norm` | (128) BF16 | |

head dim k = v = 256; RoPE on the first 64 dims only (`rope.dimension_count`
64, base 1e7, `dimension_sections` [11,11,10,0] = 32 pairs split over 4
M-RoPE sections — "IMRoPE", comment 1052). Same 64-dim RoPE for the indexer
q/pooled keys (128-dim rows, rope touches first 64). `kq_scale` = 1/sqrt(256)
(1060). QSA condition: kpool input present and `compress_ratios[il] > 0`
(1017) — always true here; dense fallback path unused.

## Layer flow — `build_layer_attn` (1006-1081)

1. `sel = build_qsa_sel(...)` on the same block input `cur` as q/k/v (1019).
2. `Qcur_full = wq·cur` [12288,T] → `Qcur_full`. Per-head 512-wide block:
   first 256 = Q (`Qcur_reshaped`, strided view), last 256 = `gate`
   (`gate_reshaped`). Q RMS-norm `attn_q_norm` → `Qcur_normed`.
3. `Kcur = wk·cur` reshape (256,2,T), RMS-norm `attn_k_norm` → `Kcur_normed`;
   `Vcur = wv·cur` reshape (256,2,T).
4. `build_rope(Qcur)`, `build_rope(Kcur)` with `inp_pos` → cb `Qcur/Kcur/Vcur`.
5. `cur = build_attn_qsa(...)` with mask `sel`, `n_sel` (map below).
6. Gate: `cur *= sigmoid(gate)` (`gate_sigmoid`, `attn_gated`), then `wo` →
   `attn_output`. (Qwen3-Next style sigmoid gate, no tanh.)

## Indexer selection — `build_qsa_sel` (808-943)

Constants: `kpool = compress_ratio = 4`, `indexer_top_k = 2048` →
`n_top_pool = min(n_pool, 2048/4) = 512` blocks; kept cells
`n_sel = kpool*min(n_pool,512) + kpool-1` = up to 4*512+3 = 2051.

Scoring keys: blocks of 4 consecutive cache cells pooled into ONE key.
The indexer cache keeps per-cell rows `raw key | pooled key`
(`indexer_kpool_row=2`, hparams 81-83): pooling happens BEFORE norm and RoPE,
so cached raw keys get neither. Blocks are groups of visible tokens in cache
order; the incomplete tail is always kept (`by_order`, `select_tail`).

Per ubatch:
1. `k_raw = index_k_proj·cur` [128,T] → `indexer_k_raw`; stored (zero-
   padded to 2*128) into the indexer cache rows via `cpy_k` (826-830).
2. Pool only blocks this ubatch completes/regroups (`new_pool_flat` member
   cells): gather member raw keys, mean over the 4 (slice sum, 840-846),
   RMS-norm `index_k_norm`, `build_rope` with `new_pool_pos` = M-RoPE
   position of each block's FIRST member → `indexer_pool_k_new`; scatter
   into rep rows (`new_pool_rep`), gather all `pool_cells` → `indexer_k`
   [128, n_pool] (848-861). Older blocks reuse rows written earlier.
3. `q = index_q_proj·cur` → (128,4,T), RMS-norm `index_q_norm`, RoPE
   `inp_pos` → `indexer_q`.
4. `score = lightning_indexer(q, pooled, weights=1/sqrt(128) all heads,
   pool_mask)` [n_pool,T] = `(1/sqrt(128)) * sum_h relu(q_h . k_block)`
   (867-870) → `indexer_score`.
5. `top_k = ggml_top_k(score, n_top_pool)` — UNORDERED → `indexer_top_k`.
6. Build cell mask `sel` [n_kv,T]: 0 for cells of the selected live pools
   plus the token's tail cells (`tail_idxs`, up to 3), -inf elsewhere
   (880-941). Fused path `top_k_mask(keep, top_k, score, pool_idxs,
   tail_idxs)`; fallback path scatters zeros into an -inf buffer using a
   "dump row" trick for dead slots (padded pools / missing tail share the
   `n_kv` sentinel; a pool is live iff its score >= 0, invisible = -inf).
   For us: just build the boolean keep-set — top-512 block ids ∪ tail.

K-pool graph inputs (shared by all QSA layers, 713-805): `k_idxs` (cell of
each new token), `pool_cells` (rep cell per block), `pool_idxs` [4,n_pool]
members (n_kv sentinel = padding), `pool_mask` F16 [n_pool,T] (block not yet
visible to token), `tail_idxs` [3,T], `new_pool_*`. These come from the
hybrid memory context — we must reproduce the same bookkeeping per sequence:
per-cell raw keys, per-block pooled keys + rep, block membership in cache
order.

## QSA attention — `build_attn_qsa` (946-1002)

1. Hadamard self-rotation `self_k_rot` on q and k, `self_v_rot` on v —
   present ONLY when the KV cache was created with rotation matrices
   (`mctx->get_k_rot()`, llama-graph.cpp 3322): a cache option, not a GGUF
   tensor. Rotation is self-inverse, undone on the output (999-1001).
   If our KV cache is plain F16/F32 we can skip it — decide in S25.
2. K,V → cache; q current, k/v gathered from cache.
3. `sel` replaces the causal kq_mask: reshaped across the K-view streams →
   `kq_mask_qsa`; dense GQA MHA `build_attn_mha(q,k,v, mask=sel, n_sel)` →
   `kqv_out`. No softcap, no sink; the mask already encodes causality
   (all kept cells are in the token's past, comment 878).

## cb() names for dumps (layer il=3)

`indexer_k_raw`, `indexer_pool_k_new` (partial, per ubatch), `indexer_k`,
`indexer_q`, `indexer_score`, `indexer_top_k`, `indexer_sel`,
`Qcur_full`, `Qcur_reshaped`, `Qcur_normed`, `Kcur`, `Kcur_normed`, `Vcur`,
`gate_reshaped`, `Qcur`, `Kcur`, `Vcur`, `kq_mask_qsa`, `kqv_out`,
`attn_pregate`, `gate_sigmoid`, `attn_gated`, `attn_output`.

## Notes for S25

- Dump comparison: pooled-key pooling and top-k are cache-stateful; easiest
  dump target is a single short prompt in one ubatch (prefill covers
  `indexer_k_raw`, `indexer_q`, `indexer_score`, mask, `kqv_out`).
- Selection is by score with ties unordered: compare the keep-SET, not
  ordering. `ggml_top_k` pads with -inf rows when n_pool < 512 (dead slots →
  dump rows; live test: score >= 0).
- indexer matmuls are BF16; q/k/v/attn are the usual quantized types.
- `pool_mask` marks blocks a token cannot see yet (built by memory, not by
  the graph) — with per-sequence state we can score all existing blocks.
- The `n_embd_head_k == n_embd_head_v` assert (1014) holds: 256 both.
