#!/usr/bin/env bash
# Check pynference against llama.cpp on a GGUF model (runs on the PC, GPUs free).
#   scripts/remote.sh scripts/verify_gemma4.sh [MODEL.gguf]
# 1. llama.cpp reference: perplexity on 8 chunks of 512 tokens, then greedy
#    top-10 probabilities from llama-server for the prompts in bench/compare_probs.py.
# 2. The same with pynference on 2 GPUs.
# 3. Comparison. Results and logs go to results/verify/.
set -euo pipefail

M="${1:-/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf}"
OUT=results/verify
TEXT=/usr/share/common-licenses/GPL-3
LL=/opt/llama.cpp
# The system oneAPI runtime only exposes the GPUs through OpenCL unless the
# Level Zero adapter can find libumf (otherwise llama.cpp aborts on GPU-to-GPU copies).
LLENV=(env "LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
mkdir -p "$OUT"

echo "== llama.cpp perplexity"
"${LLENV[@]}" "$LL/llama-perplexity" -m "$M" -f "$TEXT" -c 512 --chunks 8 --device SYCL0,SYCL1 -sm layer -ngl 999 \
    > "$OUT/ref_ppl.log" 2>&1
grep -E "Final estimate" "$OUT/ref_ppl.log"

echo "== llama.cpp top probabilities (llama-server)"
"${LLENV[@]}" "$LL/llama-server" -m "$M" --device SYCL0,SYCL1 -sm layer -ngl 999 -c 4096 --port 8091 \
    > "$OUT/ref_server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for _ in $(seq 300); do curl -sf localhost:8091/health >/dev/null && break; sleep 1; done
python bench/compare_probs.py fetch --url http://localhost:8091 -m "$M" -o "$OUT/ref_probs.json"
kill $SRV; wait $SRV 2>/dev/null || true
trap - EXIT

echo "== pynference perplexity"
torchrun --nproc_per_node=2 bench/ppl.py -m "$M" -f "$TEXT" -c 512 --chunks 8 2>&1 \
    | grep -v CCL_WARN | tee "$OUT/our_ppl.log" | grep -E "Final estimate"

echo "== pynference top probabilities"
torchrun --nproc_per_node=2 bench/compare_probs.py ours -m "$M" -o "$OUT/our_probs.json" 2>&1 | grep -v CCL_WARN

echo "== comparison"
grep -h "Final estimate" "$OUT/ref_ppl.log" "$OUT/our_ppl.log"
python bench/compare_probs.py diff "$OUT/ref_probs.json" "$OUT/our_probs.json"
