# S23 map A — layer structure and hyper-connections (qwen4exp.cpp)

Reference: `scratch/ref/llama.cpp/src/models/qwen4exp.cpp` (1505 lines).
Line numbers below refer to it. Facts in `tests/golden/qwen4exp/facts.json`.

## Layer structure

48 layers, `n_embd=2560`. The builder is `llama_model_qwen4exp::graph`
(graph ctor, 447-575). Three kinds, per `facts.json → layer_kinds`:

| kind | layers | block functions |
| --- | --- | --- |
| deltanet | 35 (0,2,4,...; all but the 12 below) | `build_layer_attn_linear` (1083) — same DeltaNet as Qwen3.8-27B/S20, hidden 2560 |
| deltanet+ple | blk.1 only | PLE via `build_ple` (1417) runs on the HC residual *before* the attn mixer; then as deltanet |
| compressed_attention | blk.3,7,...,47 (every 4) | `build_layer_attn` (1006) → QSA branch (all have `compress_ratios=4`; no plain attention exists) |

Dispatch in the loop (509-513): `hparams.is_recr(il)` → linear, else QSA
attention. PLE hook at 501-503. Layer kinds come from
`LLM_KV_ATTENTION_RECURRENT_LAYERS` / `LLM_KV_PLE_LAYERS` (152-159).
MoE FFN on every layer: `build_layer_ffn` (1215) — router `ffn_gate_inp`
(2560,512) F32, 512 experts top-10 FFN 640, shared expert 640 gated by
`ffn_gate_inp_shexp` (1215-1257; details in map C). No `attn_norm` /
`post_attention_norm`: the norms are folded into the HC mixers. No MTP
tensors in this GGUF (`graph_mtp`, 577, is dead code for us).

Per-kind tensor sets = `facts.json → block_tensor_signatures` (deltanet 22
tensors incl. 8 `hc_*` + 8 ssm/attn; compressed_attention 26 incl. 6
indexer; deltanet+ple adds 6 `ple_*`).

## Hyper-connections flow

HC keeps `hc=4` parallel residual streams `res_hc [n_embd, hc, T]` instead
of plain residual adds + norms (332-445). `hc_dim = hc*n_embd = 10240`,
low rank `hc_lr = 320`. Stream layout = concatenated streams (reshape of
[n_embd, hc, T] to [hc_dim, T]).

Per layer, in order (loop 487-548):

1. (blk.1 only) `res_hc = build_ple(recr_cache, ple_emb, res_hc, il)`.
2. `hc_mix attn_mix = build_hc_mix(res_hc, hc_attn_norm, hc_attn_down,
   hc_attn_up, hc_attn_inject, nullptr, il)` → `cur [n_embd, T]` +
   `inject [hc, T]`.
3. `cur =` DeltaNet or QSA attention on `cur`.
4. `res_hc = build_hc_combine(res_hc, cur, attn_mix.inject, il)`.
5. Same mixer pair with `hc_ffn_*` around `build_layer_ffn` (`ffn_out`).

`build_hc_mix` (347-415):
- grouped RMSNorm over one stream (`eps = f_norm_rms_eps`), scaled by gamma
  `w_norm [n_embd, hc]` — GGUF stores it flat [10240] F32 (reshape allowed);
  **the converter already folded gamma to (1 + w)**; then reshape [hc_dim,T].
- `lo = silu( (1/hc) * down·xn )` with `down [hc_dim, 320]` (Q8_0);
  `gate = up·lo` with `up [320, hc_dim]` (Q8_0) → `[hc_dim, T]`.
- `mixed = mean_streams( xn * sigmoid(gate) )` → `[n_embd, T]` = block input.
- `inject = (1/hc) * inject_w·xn` with `inject_w [hc_dim, hc]` (F32).
  GGUF has separate `hc_*_down` + `hc_*_inject`; llama.cpp also accepts a
  merged `hc_*_down_inject [hc_dim, hc_lr+hc]` — not used by this file.

`build_hc_combine` (417-445): `w = 2*sigmoid(inject) [hc,T]` (centred on 1,
so zero inject = plain residual); each stream `res[s] += block_out * w[s]`.

Final head (556-573): the output norm *is* a mixer — `build_hc_mix(res_hc,
output_hc_norm, output_hc_down, output_hc_up, nullptr, nullptr, -1)` — then
`result_norm → output (2560,248320) → result_output`. No inject branch here.
Global tensor names in GGUF are `output_hc_{norm,down,up}` (llama.cpp calls
them `hc_head_*`).

## cb() names for dumps

Top-level (any layer): `model.input_embed`(-1), `hc_init`(-1), `layer_inp`(il,
only when requested), `hc_norm`, `hc_gate`, `hc_mixed`, `hc_inject`,
`hc_combine` (twice per layer: after attn mix/combine and after ffn mix —
dumps keep the last call), `ffn_out`, `l_last` (the HC-wide layer output),
`h_nextn`(-1), `result_norm`, `result_output`.

DeltaNet layer internals (1083-1213): `beta_alpha`, `beta`, `alpha`,
`beta_sigmoid`, `a_softplus`, `gate`, `state_predelta`, `conv_output_raw`,
`conv_output_silu`, `q_conv`, `k_conv`, `v_conv`, `q/k/v_conv_predelta`,
`final_output`, `linear_attn_out`.

QSA layer internals: see map B. MoE internals: `ffn_moe_out`, `ffn_shexp`,
`shared_expert_gate`, `shared_expert_gate_sigmoid`, `ffn_shexp_gated`.
PLE: `ple_embd`(-1), `ple_gate`, `ple_gated_value`, `ple_conv_out` (map C).

Note: `hc_combine`/`hc_*` cb names repeat per phase; a dump of layer 0 gives
the first `hc_mixed` = attn-mixer output and the second = ffn-mixer output,
distinguished by order within the layer.
