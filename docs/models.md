# Supported models

Validated end-to-end on the reference PC (2× Intel Arc Pro B70 31.9 GiB,
63.6 GiB host RAM, tensor parallel over 2 GPUs). All numbers below come from
the final validation (`docs/plans/pynference/reports/S28.md`, jobs S28-2/3/6/7,
2026-11-22), measured with `bench/decode_profile.py --mode graph --no-profile`
(2 ranks) and `scripts/regress.sh`.

Start any model the same way, one process per GPU:

```bash
torchrun --nproc_per_node=2 -m pynference.cli chat -m <FILE> [options]
torchrun --nproc_per_node=2 -m pynference.server -m <FILE> [options]
```

The architecture and tokenizer are read from the GGUF; no per-model flags are
needed. Paths are the ones on the reference PC.

| Model | File (on the PC) | Size |
|---|---|---|
| Gemma-4-31B-it | `/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf` | 30.4 GiB |
| Qwen3.8-27B | `/nvme/gguf/qwen3.8/Qwen3.8-27B-Q8_0.gguf` | 27.1 GiB |
| gpt-oss-120b | `/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf` | 59.0 GiB |
| Qwen3.8-Flash-Next (MoE) | `/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` (4 shards, pass the first) | 87.2 GiB |

## Speed and memory (C9 benchmark, `--decode graph`, `--ctx 8192`)

Decode = wall ms/token at the given prompt length; prefill = tokens/s on a
2000-token prompt. Memory = `torch` reserved per rank after load → after
decode (every GPU byte also pins ~1 GiB host RAM 1:1 on this PC, so host RAM
is the binding constraint).

| Model | decode @ctx~512 | prefill 2000 | decode @ctx~8192 | GPU memory per rank |
|---|---|---|---|---|
| Gemma-4-31B (Q8_0) | **29.0 tok/s** (34.4 ms) | 887 tok/s | 27.6 tok/s (36.2 ms) | 16.5 → 20.7 GiB |
| Qwen3.8-27B (Q8_0) | **33.1 tok/s** (30.2 ms) | 404 tok/s | 31.9 tok/s (31.4 ms) | 14.4 → 16.5 GiB (weights 13.3 G + cache ~0.4 G) |
| gpt-oss-120b (MXFP4) | **5.8 tok/s** (171.3 ms, expert store 16.7 GiB copied) | 364 tok/s | n/d (see below) | 22.4 → 23.2 GiB |
| Qwen3.8-Flash-Next (IQ4_XS) | **9.3 tok/s** (107.4 ms) | 68 tok/s | 5.7 tok/s (174.2 ms, store 165 GiB copied) | 19.9 → 23.5 GiB (weights 2.7 G + experts 16.9 G + 11.5 G pinned host) |

gpt-oss decode @ctx~8192 is **not obtainable** on this PC: a single ~8192-token
prompt stalls the first prefill (reproduced with both decode paths, graph 2/2
and fused 1/1 — a rank desync inside the dense attention prefill that
materialises ~17 GB of fp32 scores). Keep `--ctx 8192` and normal prompt
lengths; 2000-token prompts run fine.

## Recommended options

- **Gemma-4-31B**: defaults (`--decode graph`, `--ctx 8192`). Reference
  model of the project: bit-exact regression canary (`scripts/regress.sh
  gemma4`).
- **Qwen3.8-27B**: defaults. It is a reasoning model: the answer arrives in
  `reasoning_content` first (in the server response) — the CLI hides it unless
  you pass `--show-reasoning`, and `--no-reasoning` renders a non-thinking
  prompt.
- **gpt-oss-120b**: defaults; keep prompts well under `--ctx` (see above).
  ~54% of the experts stay resident per rank, the rest streams from a pinned
  host cache — the decode rate above already includes the store misses
  (report the miss-inclusive rate per C9). Server load ≈ 40–45 s + warm-up.
- **Qwen3.8-Flash-Next**: defaults. Biggest model: experts stream from a
  ~12 GiB pinned host cache and the store copies a lot (hit rate drives the
  rate). Load 60–85 s + warm-up ~70 s on a server; right after another large
  model has evicted the page cache, its load can take minutes — allow up to
  ~12 min for `/health` on a cold server.
- Sampling defaults (temp/top-k/top-p) come from each GGUF when not given;
  `--temp 0` is greedy.

## Verification status

All four pass the final validation: the two dense models pass the strict
regression (`scripts/regress.sh`, top-10 probabilities vs llama.cpp golden,
`max |dp| ≤ 3.3e-4`, ppl relative diff 0.000%); the two MoE pass the C8 gate
(`bench/compare_probs.py diff --strict`: 24/24 greedy tokens identical on all
4 prompts; gpt-oss ppl 1.6083 vs llama 1.5995, Flash-Next 1.0005 vs 1.0002).
Their rigid `regress.sh` snapshots deviate only by the documented ~1-ULP bf16
cross-process noise. `pytest tests/gpu`: 142 passed, 7 skipped.
