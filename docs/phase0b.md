# Phase 0b — GGUF loader and first model: Gemma-4-31B (2026-11-14)

Status: Gemma-4-31B runs on both GPUs and matches llama.cpp (perplexity within
0.26%, identical greedy tokens). Decode speed: see "Speed" below.

## Model

`ggml-org/gemma-4-31B-it-GGUF`, file `gemma-4-31B-it-Q8_0.gguf` (30.39 GiB,
sha256 `fcd52ceb…5c9c` verified), in `/nvme/gguf/gemma-4-31B/`.

| Parameter | Value |
|---|---|
| layers | 60: 5 sliding-window + 1 full attention, repeated |
| hidden / FFN | 5376 / 21504 (GELU-tanh, gated) |
| heads | 32 query; KV 16 (sliding, head dim 256) or 4 (full, head dim 512) |
| full-attention layers | no V projection (K reused as V), RoPE base 1M on 128 of 512 dims |
| sliding layers | window 1024, RoPE base 10k on all 256 dims |
| norms | RMSNorm on Q, K (weighted) and V (plain); sandwich norms; per-layer output scale |
| attention scale | 1.0 |
| vocabulary | 262144, output head tied to the embedding, final soft-cap 30 |
| tensor types | 411 Q8_0, 422 F32 |

Reference implementation followed: llama.cpp `src/models/gemma4.cpp` at the
commit of the binary on the PC (`2d8a32502`).

## Tensor split on 2 GPUs

| Weight | Split | Per GPU |
|---|---|---|
| Q, K, V (fused into one matrix) | by heads: 16 query + 8 or 2 KV heads | |
| attention output | by input columns, then all-reduce | |
| FFN gate + up (fused) | by columns (10752 each) | |
| FFN down | by input columns, then all-reduce | |
| embedding / output head | by vocabulary (131072 rows each); embedding rows summed with all-reduce, logits gathered | |
| norms, RoPE tables | replicated | |

Two all-reduces per layer (120 per token) plus one for the embedding and one
all-gather of the logits. Weights take ~15.2 GiB per GPU; the KV cache
(fp16) takes ~0.4 GiB for the sliding layers plus 40 KiB per token of context
for the full-attention layers.

## Components and test results

| Component | Test | Result |
|---|---|---|
| GGUF reader (`pynference/gguf.py`) | metadata and 833 tensors of the Q8_0 file | OK |
| Tokenizer (`pynference/tokenizer.py`) | mixed text (special tokens, code, Unicode, emoji, tabs) vs `llama-tokenize` | identical ids (84/84), loads in 0.8 s |
| Q8_0 GEMV, Triton (`pynference/quant.py`) | decode shapes of Gemma-4-31B, xpu:0 | 554–592 GB/s (91–97% of the card's bandwidth) |
| Q8_0 dequant + bf16 matmul (prompt path) | vs fp32 reference | rel. error ~3e-3 (bf16 activations) |

Q8_0 weights are repacked on load into int8 values + fp16 block scales, so the
GEMV kernel reads plain 2D tiles; the int8 data is the same size as in GGUF.

## Comparison with llama.cpp

`scripts/verify_gemma4.sh` (llama.cpp build `2d8a32502`, both GPUs, `-sm layer`).

Perplexity, 8 chunks × 512 tokens of `/usr/share/common-licenses/GPL-3`:

| Engine | PPL |
|---|---|
| llama.cpp, batch 512 (one sequence per batch) | 10039.0 |
| llama.cpp, batch 2048 (four sequences per batch) | 10201.7 |
| pynference | 10064.8 (+0.26% vs llama.cpp b512) |

The absolute value is high because the instruction-tuned model predicts raw
licence text poorly; what matters is the agreement, chunk by chunk too. The
llama.cpp result moves by 1.6% with the batch size alone, more than the gap
between the two engines.

Greedy generation with top-10 probabilities at every step (4 prompts: Italian
and English chat, code, raw text): **81/81 identical tokens**. Mean partial KL
over the top-10 is 1e-7 to 1e-4 on three prompts, 3e-3 on the raw-text one,
due to a single step where the model hesitates between " The" (0.43 llama.cpp /
0.60 ours), " Augustus" and " This". Our two numeric paths (bf16 prompt
matmuls vs everything in fp32, prompt fed token by token) agree with each other
there (0.596 / 0.594), so the gap comes from llama.cpp, whose Q8_0 mat-vec
quantizes the activations to 8 bits.

### llama.cpp on this PC uses OpenCL, not Level Zero

The native build in `/opt/llama.cpp` only sees the GPUs as `opencl:gpu`: the
system oneAPI runtime cannot load its Level Zero adapter because `libumf.so.1`
is not on the library path. With OpenCL, llama.cpp aborts on GPU-to-GPU copies
("Experimental P2P feature is not implemented for OpenCL adapter"). Fix used in
`scripts/verify_gemma4.sh`, without system changes:
`LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.2/lib`.

### oneCCL and large all-reduces

All-reduces of the 512-token prompt chunks (11 MB) on ever-changing torch
buffers failed with `UR_RESULT_ERROR_OUT_OF_RESOURCES` inside oneCCL. Every
collective now goes through one persistent buffer per dtype
(`pynference/parallel.py`), which also keeps addresses fixed for graph capture.

## Speed (both GPUs, Q8_0)

Decode step, `bench/decode_profile.py` (32 greedy tokens, ~60 tokens of context):

| Decode path | ms/token | tok/s |
|---|---|---|
| plain torch ops (~90 launches per layer) | 92.9 | 10.8 |
| fused Triton kernels (~14 launches per layer) | 46.3 | 21.6 |
| fused kernels captured as one XPUGraph (default) | 33.0 | **30.3** |

With plain torch ops the GEMVs take 29.8 ms per token, as expected from the
bandwidth, and the rest is launch overhead: the two processes drift apart and
each all-reduce waits for the slower one (44 ms per token of all-reduce
"device time", mostly waiting). The XPUGraph capture includes the xccl
all-reduces (the open question of phase 0a: yes, they can be captured).

End to end with `pynference.generate` (sampling on the CPU included):

| Run | Prompt | Generation |
|---|---|---|
| 1964-token prompt (GPL text summary), greedy | 786 tok/s | 28.4 tok/s (300 tokens) |
| 46-token prompt, 1800 tokens of story, temp 1 | – | 24.6 tok/s (incl. graph capture) |

Load: 16 s with the file in the page cache (≈110 s from disk), plus 13 s of
warm-up (Triton compile/autotune, graph capture). Memory: 16.2 GiB on GPU 0
with `--ctx 8192`.

The 1800-token run goes well past the 1024-token sliding window and stays
coherent.

### Decode step

`pynference/kernels.py`, all with positions taken from a device buffer so the
step can be replayed as a graph:

| Kernel | Replaces |
|---|---|
| `rms_norm`, `add_norm` | RMSNorm, residual add + post-norm + layer scale + next RMSNorm |
| `norm_rope` | per-head RMSNorm + NEOX RoPE, writing K/V straight into the cache slot |
| `attn_decode` | GQA attention over the cache (split over keys + combine) |
| `gelu_mul` | GELU-tanh(gate) × up |

## Next

1. Prompt path: dequantize-then-matmul in bf16 is fine for now (786 tok/s);
   a Triton int8 GEMM would avoid the bf16 copy.
2. Long contexts: the full-attention decode kernel uses a fixed number of
   splits over `--ctx`; graphs per context bucket would avoid scanning empty
   splits.
3. Other quant types (Q4_K, Q5_K, Q6_K, IQ4_XS, MXFP4) for the next models.
