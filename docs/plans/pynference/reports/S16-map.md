# S16 — Mappa di riferimento: attention GPT-OSS (sinks, finestra, YaRN)

Riferimento: `scratch/ref/llama.cpp` a commit 2d8a325, arch `gpt-oss` (`openai-moe`).
Fatti GGUF (S13, `tests/golden/gpt-oss-120b/facts.json`): hidden 2880, 36 layer,
64 teste Q / 8 KV, dim testa 64, RoPE NEOX base 150000, YaRN factor 32 su ctx
originale 4096 (beta_fast 32, beta_slow 1, attn_factor assente), sliding window
128, eps RMSNorm 1e-5, attn_sinks [64] F32 per layer, pesi attn Q8_0 + bias F32.

## 1. Grafo del layer — `src/models/openai-moe.cpp:62-161`
- RMSNorm(attn_norm) → `build_qkv` → `build_rope` su Q e K → `build_attn` con
  sinks e scala `1/sqrt(64)` (riga 101) → `cb "attn_out"` (103) → somma residuo
  → RMSNorm(attn_post_norm) (115-117) → MoE → residuo. In cima: output_norm → `output`.
- Tensor: tok_embd (27), output_norm e output (30-31), attn_norm/attn_post_norm
  (36-37), QKV via `create_tensor_qkv` (39), wo (40), attn_sinks (42), wo_b (49).
- hparams: window `n_swa` dal GGUF (6), `swa_type` STANDARD (8), pattern 2 (9);
  righe 11-12: `rope_freq_scale_train_swa = rope_freq_scale_train` e base uguale
  (13, chiave `ROPE_FREQ_BASE_SWA` assente nel GGUF) → YaRN con gli STESSI
  parametri su tutti i layer, windowed compresi (differenza rispetto a HF).

## 2. `build_qkv` e bias — `src/llama-graph.cpp:1864-1976`
- Strada con wq/wk/wv separati (no fusione in graph): Q = x·Wq, poi bias DOPO
  la proiezione: Q += wq_b (1924-1925), K += wk_b (1938-1939), V += wv_b (1952-1953).
- Nessun clamp; reshape in teste: Q [64×64], K e V [8×64].

## 3. Attention — `build_attn` iswa e `build_attn_mha`
- `llama-graph.cpp:3666-3762` ( overload iswa usato da openai-moe, input da riga 71
  di openai-moe.cpp): layer windowed usano cache e maschera SWA (3697, 3734);
  K,V copiati in cache (3722-3732); attenzione in 3741; poi O: `wo` (3749-3751)
  e bias `+ wo_b` (3757-3759). Nessun Hadamard/scale-KV (tensori assenti).
- `llama-graph.cpp:3072-3201` (`build_attn_mha`, via non-flash): prodotto
  kq = ⟨q, k⟩ per (query, chiave) con accumulatore BF16 (3154); kq non ha bias
  (kq_b nullo); `soft_max_ext(kq, maschera, scala)` (3183) con
  wp[i,j] = kq[i,j]·(1/√64) + maschera[i,j], maschera additiva 0 oppure −inf;
  poi `soft_max_add_sinks` (3184).
- Softmax con sink — `ggml/src/ggml-cpu/ops.cpp:5709-5819`: per ogni riga i di
  testa h: m = max( max_j wp[i,j] , sink[h] ); p[i,j] = exp(wp[i,j] − m);
  den = Σ_j p[i,j] + exp(sink[h] − m); out[i,j] = p[i,j]/den. Sink indicizzato
  per testa Q (64 valori), condiviso da tutte le righe: la massa mancante
  (Σ p < 1) è "assegnata al sink", i pesi non sommano a 1.

## 4. Finestra: quali layer, quali chiavi
- Pattern: `src/llama-hparams.cpp:8-22`, `set_swa_pattern(2, dense_first=false)`:
  is_swa[il] = il%2 < 1 → layer PARI (0,2,…,34) con finestra 128; dispari dense.
- Maschera: `src/llama-hparams.h:544-556` (STANDARD): la chiave in posizione p0
  è mascherata se p1 − p0 ≥ 128; chiamata (src/llama-kv-cache.cpp:1109) con
  p1 = pos(query)+1 → query alla posizione q vede solo le chiavi con
  q+1−p0 < 128, cioè le ultime 128 incluse se stessa; le altre −inf.
  Layer dispari: nessuna finestra, solo causalità/padding.

## 5. RoPE: tipo e parametri YaRN
- Tipo NEOX: `src/llama-model.cpp:3793,3822` (OPENAI_MOE → NEOX); cache degli
  angoli con rotazione a coppie (i, i+32): `ops.cpp:6188-6204` + chiamata 6337.
- Chiavi GGUF → hparams: `llama-model.cpp:1581-1613`, in particolare 1607-1611:
  scaling.factor=32 → `rope_freq_scale_train = 1/32`; scaling.type=yarn;
  ctx originale 4096; beta_fast 32, beta_slow 1. Parametri per layer:
  `llama-model.cpp:2883-2899`; freq_scale per layer windowed: 2917-2919 → con
  openai-moe.cpp:12 il valore è lo stesso dei layer dense (vedi §1).
- `src/llama-context.cpp:265-306`: ext_factor −1 → 1.0 per YARN (265-267);
  yarn_attn_factor = (1+0.1·ln 32) (295) · 1/(1+0.1·ln 32) (303) · 1.0 (306) = 1.0;
  yarn_beta_fast/slow 32/1; n_ctx_orig 4096.

## 6. `rope_yarn` e `ggml_rope_yarn_corr_dims`
- `ggml/src/ggml-cpu/ops.cpp:6076-6098` (rope_yarn): per la coppia k:
  theta_interp = theta/32; ramp(k) = clamp01(1 − (k−8)/(18−8));
  theta = theta_interp·(1−ramp) + theta·ramp; mscale effettiva su cos/sin =
  yarn_attn_factor·(1+0.1·ln 32) = 1.34657.
- Cache angoli: `ops.cpp:6100-6114`, theta_scale = 150000^(−2/64) (6268):
  θ_k(pos) = pos·150000^(−2k/64), k = 0..31; cos/sin moltiplicati per mscale.
- `ggml/src/ggml.c:4373-4389` corr_dims: d(β) = 64·ln(4096/(β·2π))/(2·ln 150000);
  truncate è TRUE di default (chiamata 6271 + default `ggml.c:4447-4458`) →
  floor/ceil e clamp [0,63]: corr_dims = [8, 18] (valori calcolati: 8.09, 17.40).
- Effetto per coppia: k ≤ 8 angolo non scalato; k ≥ 18 angolo /32; 9..17 interpolazione lineare.

## 7. GQA
- Mappa testa Q h → KV h//8: broadcast del mul_mat (`ggml/src/ggml.c:3352`,
  ggml_can_mul_mat) e indice `ggml/src/ggml-cpu/ggml-cpu.c:1230` (i02 = i12/r2).
- I sink sono per testa Q (64), quindi indicizzati con h, non con h//8.

## 8. Impatto sulla nostra implementazione (passi 2-3)
- TP (piano 3.4): righe di Q,K,V divise per teste [q_r; k_r; v_r]; O con colonne
  di input divise + bias solo su rank0 prima dell'all_reduce; embd e output su
  righe di vocabolario; norme replicate; sink tagliati per le teste Q del rank
  (sono per testa Q: il rank tiene sinks[r·n_head_l : (r+1)·n_head_l]).
- Una sola tabella YaRN cos/sin [max_ctx, 64] (mscale inclusa), coppie NEOX
  (i, i+32), identica per tutti i layer.
- `attn_decode` (pynference/kernels.py:200): aggiungere `sinks=None`; la finestra
  nei layer pari si ottiene tenendo in cache al più le ultime 128 chiavi valide.
- Prefill: torch, con maschera §4 per layer pari e softmax con sink §3.

## 9. Da confermare nel passo 4
- Dump `attn_out-0` (cb openai-moe.cpp:103, layer 0 che è windowed): se il
  confronto tornasse male, l'ipotesi in gioco è §1/§5 (YaRN anche sui layer
  windowed) e la prova alternativa è freq_scale = 1 su layer pari.
- Il nostro riferimento CPU usa F32 dove la reference ha BF16 (accumulo kq):
  soglia cos ≥ 0.9999 tollerante.
