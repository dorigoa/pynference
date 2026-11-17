# S19 — Mappa del riferimento Qwen3.8-27B (llama.cpp in `scratch/ref/llama.cpp`)

Dimensioni DeltaNet: d=5120, H_k=`ssm_n_group`=16, H_v=`ssm_dt_rank`=48,
S=128 (stato 128×128/testa), interno 6144=48×128, conv k=4. File:riga =
`src/models/qwen35.cpp` (q35), `src/models/delta-net-base.cpp` (dnb),
`src/models/models.h` (mh), `ggml/src/ggml-cpu/ops.cpp` (ops),
`src/llama-graph.cpp` (lg).

## Mappa A — layer DeltaNet (48 layer, `is_recr(il)`)

Layer: `llm_build_delta_net_base::operator()` q35:145; ramo lineare q35:176
→ `build_layer_attn_linear` q35:357.

1. `x = RMSNorm(h, eps=1e-6)` q35:167.
2. Proiezioni (q35:376 → `build_qkvz` q35:243; qui `wqkv_z` assente → ramo
   separato q35:277-283): `qkv = attn_qkv·x` → 10240 (canali q 2048 | k 2048 |
   v 6144); `z = attn_gate·x` → 6144. Vista 3D `qkv` [10240, n_t, n_seq].
3. `beta = sigmoid(ssm_beta·x)` → [1,48,n_t,n_seq] q35:391,400.
4. `alpha = ssm_alpha·x` [396]; `g = ssm_a · softplus(alpha + ssm_dt.bias)`
   [402-407], reshape [1,48,...] [412]: `ssm_a`=[48] contiene già
   −exp(A_log) (commento q35:407). Per-testa scalare → percorso non-KDA.
5. Conv (dnb:53 `build_conv_state`): `conv_input = concat([hist(3×10240),
   qkvᵀ], asse 0)` → [3+n_t, 10240, n_seq]; lo stato salvato = le ultime
   k−1=3 colonne [dnb:74 → lg:4148 `build_conv_state_store`].
6. `ssm_conv` (ops:9811): `y[t,c] = Σ_{i<4} in[t+i,c]·w[i,c]`, kernel
   `ssm_conv1d` [4,10240] F32, poi `silu` ops/q35:425,428. Niente bias.
7. Split di y (q35:438-455): `q` [128,16], `k` [128,16], `v` [128,48] per token.
8. Norm q,k: `build_gdn_l2_norm` mh:14 = `rms_norm(x, eps/n)·1/√n` =
   `x/√(Σx²+eps)`, eps=1e-6 su n=128 (L2, non RMS) q35:463-464.
9. Regola delta (dnb:8 → `ggml_gated_delta_net` K=1; kernel ops:11003). Per
   (testa v h, seq), testa k/q = **h mod 16** (ops:11079-11080); stato
   M[i][j] con layout trasposto `s_out[j*S+i]` (commento ops:11104). Per token
   t:
   - `M ← M · exp(g_h[t])` (scalare per testa) ops:11120;
   - `δ = β_h[t] · (v_h[t] − Mᵀ k_h[t])` ops:11122-11127;
   - `M ← M + k_h[t] ⊗ δ` ops:11130-11132;
   - `out_h[t] = Mᵀ q_h[t] / √128` ops:11073,11134-11139.
   Variante KDA `neg0==S_v` (decay per-canale, ops:11041): non usata qui.
   Uscita op = [punteggi S·H·n_t·n_seq | stati], vista output+new_state
   dnb:36-48; nuovo stato scritto nel cache `n_embd_s`=128·128·48/testa·seq
   dnb:107-110. Chunking: righe = H·n_seq, 4 chunk/thread ops:11160-11196.
10. Norma con gate (q35:481 → `build_norm_gated` q35:275):
    `y = RMSNorm(out, 1e-6 per testa 128) ⊙ silu(z)`.
11. Uscita: `h += ssm_out·y`, `ssm_out` [6144,5120] q35:488.
12. Condivisione testa v→k: `H_v % H_k == 0` assert dnb:24; llama.cpp usa
    j mod 16 (round-robin), **non** j/3 come ipotizzato in 3.9; nessun
    permute dei pesi al load (grep su q35). I golden vengono da
    llama-server → la mappatura da rispettare in S20/S21 è quella di
    llama.cpp (annotato in Deviazioni nel report S19).

## Mappa B — layer attention (16+MTP, `!is_recr(il)`) e resto

Loop layer: ctor graph q35:144; `build_inp_mem_hybrid` (cache+mask causale)
q35:156; RMSNorm `attn_norm` eps 1e-6 q35:168; ramo `build_layer_attn`
q35:179; residuo "attn_residual" q35:188; `attn_post_norm` q35:195; FFN
q35:199; residuo FFN q35:203.

1. Proiezioni (q35:297-301): `attn_q` [5120,12288] con n_q=2·24·256
   (`create_tensor_qkv` q35:72), `attn_k` [5120,1024], `attn_v` [5120,1024]
   (4 KV × 256), niente bias.
2. **QG interlacciato per testa**: la vista `Qcur` ha nb1=2·256 per testa
   (q35:306-308; `ggml_view_3d` ha nb0=element-size di default, ggml.h:1509),
   la vista `gate` ha offset +256 (q35:320-324). Testa h: q = righe
   512h..512h+255 di `attn_q`, gate = righe 512h+256..512h+511. NON le
   "seconde 6144 colonne" (nota del passo 1, corretta).
3. Norme pre-RoPE su q e k per testa 256: RMS `attn_q_norm` [256] q35:312,
   `attn_k_norm` [256] q35:317. V non normato.
4. RoPE solo su Q,K (q35:330-331): `build_rope` → GGUF senza `rope.type`,
   tipo derivato `use_mrope() ? MROPE : NEOX` con sezioni [11,11,10,0] ≠ 0
   (llama-model.cpp:3838-3840) → `ggml_rope_multi` (llama-graph.cpp:1659-
   1662). Kernel: MROPE ruota coppie stile NEOX (x_j, x_{j+32}) j=0..31 sui
   primi 64 di 256 dim, canali 64..255 pass-through (ops.cpp:6337-6338,
   6351-6361); base 1e7, sezioni M-RoPE 11+11+10=32 coppie. In testo le 3
   posizioni coincidono → per pynference basta RoPE parziale 64/256.
5. Attention (q35:338-342): scala 1/√256, mask causale; GQA 24Q/4KV con
   testa kv = **h/6** contigua (`rk2=neq2/nek2=6`, `ik2=iq2/rk2`,
   ops.cpp:8776,8834) — diversa da DeltaNet (j mod 16).
6. Gate su uscita: `attn_out *= sigmoid(gate)` con gate 2D [6144,n_tokens]
   (q35:324, 345-348); proj `attn_output` [6144,5120] (q35:72, mm 351).
7. FFN dense (q35:494 → `build_ffn(LLM_FFN_SILU, LLM_FFN_PAR)`
   llama-graph.cpp:1979): `down( silu(gate·x) ⊙ (up·x) )`; `ffn_gate`/
   `ffn_up` [5120,17408], `ffn_down` [17408,5120], niente bias (q35:99-101).
8. Testi residuali: `attn_post_norm` RMS [5120] prima della FFN (q35:195),
   il residuo FFN riaggiunge il valore pre-norm (q35:192,203).
9. Uscita modello: RMS `output_norm` (q35:214) → logits = `output` legato a
   `token_embd` [5120,248320] DUPLICATED (q35:44,51-52; mm 237). No cls.
10. MTP blk.64: caricato come blocco attention denso (q35:107-126, comment
    15-16) con norm `nextn.*` annidate, mai eseguito nel loop principale;
    graph separato `graph_mtp` (q35:510+). Ignorato (D4).
11. Tipo layer: `is_recr(il) = (il+1) % 4 != 0` (q35:17-21) → attention nei
    blk 3,7,...,63 (16) + blk.64.

## Piano TP e memoria

(aggiunta nel passo 4)
