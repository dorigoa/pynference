# pynference

LLM inference engine in Python for 2× Intel Arc Pro B70 (tensor split), with
RAM offload and on-demand MoE experts.

Developed on the Mac, run and tested on the Ubuntu PC (`ssh ubuntu`).

## Workflow

```bash
scripts/remote.sh                         # rsync the repo to ubuntu:~/pynference
scripts/remote.sh scripts/setup_env.sh    # one-time: create .venv with PyTorch XPU
scripts/remote.sh python bench/env_check.py
```

## Running a model

One process per GPU, started by torchrun:

```bash
scripts/remote.sh 'torchrun --nproc_per_node=2 -m pynference.generate \
    -m /nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf --chat -p "Ciao, chi sei?"'
```

Options: `-n` tokens to generate, `--temp/--top-k/--top-p` (defaults from the
GGUF), `--ctx` max context, `--decode graph|fused|torch` decode path,
`--n-probs N` prints the top-N probabilities.

Supported (all verified against llama.cpp, `--decode graph` on 2× B70):
Gemma-4-31B Q8_0 (~29 tok/s decode, ~890 tok/s prefill), Qwen3.8-27B Q8_0
(~33/~404), gpt-oss-120b MXFP4 (~5.8/~364, experts streamed from a pinned
host cache), Qwen3.8-Flash-Next MoE IQ4_XS (~9.3/68, 4 shards). Per-model
files, memory and options: [docs/models.md](docs/models.md).

### Chat

```bash
scripts/remote.sh 'torchrun --nproc_per_node=2 -m pynference.cli chat \
    -m /nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf --system "Sei breve."'
```

Rank 0 reads your lines; the others follow. Commands: `/reset` (new
conversation), `/system TEXT`, `/exit`. `--script FILE` feeds one user turn per
line (for tests), `-n` max answer tokens, `--temp 0` for greedy. After each
answer a line reports the new prompt tokens (conversation history is reused,
only the new tail is prefilled) and tok/s. `python -m pynference.cli generate`
takes the same options as `python -m pynference.generate`.

### OpenAI-compatible server

```bash
scripts/remote.sh 'torchrun --nproc_per_node=2 -m pynference.server \
    -m /nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf --host 127.0.0.1 --port 8090'
```

Serves `GET /health`, `GET /v1/models` and `POST /v1/chat/completions` with the
usual OpenAI fields (`messages`, `max_tokens` / `max_completion_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `stream`), including
SSE streaming (`delta` chunks, `data: [DONE]`) and `usage`. One request at a
time; the others queue. Defaults: `--ctx 8192`, `-n 512` max tokens (overridden
per request), `--decode graph`. Stop with Ctrl-C / SIGTERM on rank 0 (waits for
the running request, then all ranks exit with "server stopped cleanly").

From the Mac, keep the PC-local bind and tunnel the port:

```bash
ssh -N -L 8090:127.0.0.1:8090 ubuntu   # in another terminal
curl -s http://127.0.0.1:8090/health
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma","messages":[{"role":"user","content":"Ciao!"}],"stream":false}'
```

(`--host 0.0.0.0` exposes the server on the LAN without any auth: at your own
risk.)

## Layout

| Path | Content |
|---|---|
| `pynference/gguf.py` | GGUF reader (mmap, split files, row slices) |
| `pynference/tokenizer.py` | tokenizer from GGUF metadata, chat format |
| `pynference/quant.py` | Q8_0 repacking and Triton kernels (GEMV, dequant) |
| `pynference/kernels.py` | fused Triton kernels for the decode step (norms, RoPE, attention) |
| `pynference/parallel.py` | one process per GPU, xccl all-reduce/all-gather |
| `pynference/models/` | model architectures (`gemma4.py`, `gptoss.py`, `qwen35.py`, `qwen4exp.py`) |
| `pynference/generate.py` | command-line generation |
| `pynference/server.py` | OpenAI-compatible HTTP server (rank 0 serves, others follow) |
| `bench/ppl.py` | perplexity, same method as llama-perplexity |
| `bench/compare_probs.py` | greedy tokens and top probabilities vs llama-server |
| `bench/decode_profile.py` | decode time per token and per kernel |
| `scripts/regress.sh <model>` | regression vs llama.cpp golden (probabilities + ppl), 4 models (needs free GPUs) |
| `scripts/verify_gemma4.sh` | full comparison with llama.cpp for Gemma (needs free GPUs) |

## Phase 0a benchmarks (`bench/`)

| Script | Measures |
|---|---|
| `env_check.py` | torch/XPU/Triton/xccl versions, device properties, sanity matmul |
| `h2d.py` | RAM→GPU, GPU→RAM and GPU→GPU bandwidth vs transfer size, one and both GPUs |
| `devbw.py` | GPU memory bandwidth: torch read/copy, bf16 GEMV, Triton bf16/int8 GEMV |
| `allreduce.py` | all-reduce latency: xccl (torchrun, 2 procs) and host round-trip (1 proc) |
| `launch.py` | per-kernel launch overhead from Python, eager vs graph capture |
| `triton_smoke.py` | smallest Triton kernel; prints the full compile error if it fails |

Results and design consequences: [docs/phase0a.md](docs/phase0a.md).
Phase 0b (GGUF, Gemma 4): [docs/phase0b.md](docs/phase0b.md).

## Plan

The development plan (S1–S28, all completed) is in
[docs/plans/pynference-plan.md](docs/plans/pynference-plan.md), run in
sections by a local agent with
`docs/plans/pynference/tools/run-sections.sh` (GPU jobs run while the LLM
server on the PC is paused). Section reports:
`docs/plans/pynference/reports/`.
