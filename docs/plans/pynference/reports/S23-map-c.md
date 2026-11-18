# S23 map C — PLE, MoE + shared expert, output head (qwen4exp.cpp)

Reference: `scratch/ref/llama.cpp/src/models/qwen4exp.cpp` (PLE/FFN) and
`src/llama-graph.cpp` (`build_moe_ffn`, 2252-2650). Line numbers refer to
those files. Facts in `tests/golden/qwen4exp/facts.json`.

## PLE — n-gram hash embedding (blk.1 only; builder 87-160, input 1269-1415, build 1417-1505)

Global table `per_layer_token_embd.weight`: IQ4_NL, GGUF dims (160,
320001536) = (head_dim, rows), 26.82 GiB. `ple_n_heads = (ngram-1)*8 = 16`
heads, `ple_head_dim = n_embd/16 = 160`, head layout: heads of bigram n=2
first (indices 0-7), trigram heads 8-15.

### Hash + row selection — host-side (`set_input`, 1293-1352)

ggml has no int64/xor: we compute I32 row indices ourselves per token:

1. Window `ctx[0..n-1]`: `ctx[0]` = the token itself; predecessors come from
   the attention KV cells (`get_prev_tokens`, oldest-first; a missing
   predecessor reads as EOS). An EOS anywhere in the window cuts: every
   position at-or-after it becomes EOS. The token's OWN EOS does not cut its
   window. Embd (image) batches hash `image_token_id` 248056. EOS = 248044.
2. For each n-gram size n = 2, 3:
   `mixed = (t0*m[0]) ^ (t1*m[1]) ^ ... (t[n-1]*m[n-1])` (uint64, multipliers
   `ple_layer_multipliers` = 23703573157769, 20109073645365, 8052911324071);
   for each of the 8 heads g of that n:
   `row = mixed % ple_head_vocab_sizes[h] + ple_head_offsets[h]`,
   `h = (n-2)*8 + g` (offsets ~20M apart, vocab ~20M each; table end 320001446
   + 90 pad rows).
3. Rows tensor `[16*T]` laid out token-major, head index inside
   (`idx[i*n_heads + h]`); `get_rows` + reshape to `[16*160=2560, T]` puts
   the head dim slowest (comment 1409) — match that layout.
   Restriction: no token shared across sequences (assert 1320).

### PLE block math (blk.1, runs on `res_hc` BEFORE the attn mixer)

- `key = ple_key·emb` (2560→10240, Q8_0); `value = ple_value·emb`
  (2560→2560, Q8_0).
- grouped_norm(x, w): reshape (2560, 4, T), per-stream RMSNorm
  (`f_norm_rms_eps`), multiply by w `[2560,4]` (GGUF flat [10240] F32,
  reshape allowed). Applied to key (`ple_norm_key`), to `hidden` as query
  (`ple_norm_query`), and later to gated (`ple_norm_conv`). Note: plain
  rms_norm*weight — unlike the HC mixers, no evidence the converter folded
  (1+w) here; verify against the blk.1 dump.
- gate: `s = sum(key*query, over 2560)` `[4,T]` × 1/sqrt(2560);
  `gate = sigmoid(sign(s) * sqrt(clamp(|s|, 1e-6)))` (cb `ple_gate`).
- `gated` = value (2560,1,T) repeated over the 4 streams × gate
  `[2560,4,T]` (cb `ple_gated_value`).
- conv: grouped_norm(gated)→[10240,T]; depthwise causal conv, kernel 4,
  dilation = ngram 3, history (4-1)*3 = 9 tokens × 10240 channels stored
  per-sequence in its OWN recurrent row (`get_p_l(il)`; blk.1 has two
  conv-state rows — this and the DeltaNet one; `build_conv_state_at`
  1368-1395 prepends history, writes the new tail back).
  `out[c,t] = Σ_k w[k,c]·x[c, t-(3-k)*3]`; weights = row k of `ple_conv1d`
  `[4, 10240]` F32 (cast if not F32), then `silu` → `[2560,4,T]`
  (cb `ple_conv_out`).
- Output: `res_hc = hidden + (gated + conv_out)`.

PLE conv-state size/seq: `(kernel-1)*ngram*hc*n_embd = 9*10240` floats
(llama-hparams.cpp 274).

## MoE FFN — every layer (`build_layer_ffn` 1215-1263; `build_moe_ffn`
2252-2650). 512 experts, top-10, FFN 640, shared expert 640.

- Router: `logits = ffn_gate_inp·cur` (2560,512) F32 (cb
  `ffn_moe_logits`); `probs = softmax(logits)` — arch qwen4exp: NO bias, NO
  expert groups (n_expert_groups=1), gating SOFTMAX (2304).
- `selected = argsort_top_k(probs, 10)` (cb `ffn_moe_topk`);
  `weights = probs[selected]`; norm_w: `weights /= clamp(sum, 6.1035e-5,
  inf)` (F16-min clamp; the 1e-20 variant is DEEPSEEK41 only, 2403-2409);
  `expert_weights_scale` absent in the GGUF (default 0) → NO extra scale
  (2417).
- Experts via gather-GEMM (`mul_mat_id`), separate gate/up — the GGUF has NO
  fused `ffn_gate_up_exps` tensor despite the llama.cpp support (2449):
  `h = silu(gate_exps[e]·x) * (up_exps[e]·x)` (swiglu_split, no clamp:
  `swiglu_clamp_exp` unset), `y = down_exps[e]·h` (cb `ffn_moe_gate/up/
  swiglu/down`), `moe_out = Σ_e weight_e * y_e` (cb `ffn_moe_weighted`,
  summed → `ffn_moe_out`).
- Mixed per-layer quant types ("UD", see report Deviazioni): gate/up ∈
  {IQ3_XXS, IQ4_XS, IQ2_S}, down ∈ {Q5_0, IQ4_NL, Q2_0} — the expert loader
  must read the type per tensor.
- Shared expert: `shexp = down_shexp·(silu(gate_shexp·cur) * (up_shexp·cur))`
  (Q6_K, blk.2 Q8_0; cb `ffn_shexp`); `gate = sigmoid(ffn_gate_inp_shexp·cur)`
  one scalar/token `[1,T]` F32 (cb `shared_expert_gate(_sigmoid)`);
  `ffn_out = moe_out + shexp*gate` (cb `ffn_shexp_gated`, `ffn_out`).

## Output head (graph tail 556-573; mixer math in map A)

`build_hc_mix(res_hc, output_hc_norm, output_hc_down, output_hc_up,
nullptr, nullptr, -1)` — same low-rank mixer, NO inject branch; GGUF names
`output_hc_{norm,down,up}` (llama.cpp calls them `hc_head_*`). Then
`result_norm` (plain RMSNorm, final eps) → `output` mm (2560,248320) →
`result_output`. No logits softmax/quant anywhere; golden compares pre-argmax.

## cb() names for dumps

- PLE (il=1 only): `ple_embd`(-1), `ple_gate`, `ple_gated_value`,
  `ple_conv_out`, plus `conv_state_at` (1) (the state AFTER write-back).
- MoE (any il): `ffn_moe_logits`, `ffn_moe_probs`, `ffn_moe_argsort`,
  `ffn_moe_topk`, `ffn_moe_weights`, `ffn_moe_weights_sum(_clamped)`,
  `ffn_moe_weights_norm`, `ffn_moe_gate`, `ffn_moe_up`, `ffn_moe_swiglu`,
  `ffn_moe_down`, `ffn_moe_weighted`, `ffn_moe_out`, `ffn_shexp`,
  `shared_expert_gate`, `shared_expert_gate_sigmoid`, `ffn_shexp_gated`,
  `ffn_out`.
- Output: `hc_norm`, `hc_gate`, `hc_mixed` (il=-1), `result_norm`,
  `result_output`.

## Notes for S24/S26

- PLE dumps are NOT in the S23-1 targets (layer 0 + first QSA layer per the
  card); if S24 needs `ple_*` tensors, ask for a blk.1 dump — add
  `--dump-layer 1` style coverage when defining the job.
- `ffn_moe_weights` order follows `argsort_top_k` (sorted by score desc);
  weights are a bag over experts — compare the (expert, weight) SET, and
  beware score ties (F32 router, deterministic softmax should match bitwise).
- Shared-expert gate tensor is `[1, T]` — broadcast along n_embd in the mul
  (ggml broadcasts over ne[0]).
