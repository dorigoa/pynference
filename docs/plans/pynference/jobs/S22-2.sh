#!/usr/bin/env bash
# job: S22-2
# timeout: 120
# scopo: S22 passo 3 (job finale) — misura della via graph su Qwen3.8-27B e
#   regressioni complete (§2.9). Ordine:
#   (1) suite tests/gpu sul GPU;
#   (2) graph speed TP2 ctx 8192, prompt ~60, con breakdown kernel
#       (confronto direto con fused60 di S22-1: baseline 14.23 tok/s);
#   (3) graph speed con prompt ~2000 --no-profile: decode contesto lungo +
#       prefill token/s (C9: >=25 decode, >=500 prefill);
#   (4) regress.sh qwen35-27b (golden fused + ppl, C10) e regress.sh gemma4
#       (graph, immutata);
#   (5) numerica graph: compare_probs --decode graph contro ref_probs.json
#       (llama.cpp). Serve anche da candidato golden se il passo (4)
#       fallisse solo sul token greedy: in quel caso il golden si
#       rigenera dalla nuova via, protetto da |delta p| <= 1e-3 su llama.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/Qwen3.8-27B-Q8_0.gguf
G=tests/golden/qwen35-27b

# bench/ hardcodes the Gemma-only Tokenizer; rebind load_tokenizer at runtime (S17)
cat > "$OUT/run_bench.py" <<'PYEOF'
"""Run a bench/ script with load_tokenizer as its tokenizer factory (S17)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from pynference.tokenizer import load_tokenizer  # noqa: E402

module, argv = sys.argv[1], sys.argv[2:]
mod = __import__(module, fromlist=["main"])
mod.Tokenizer = load_tokenizer
sys.argv = [module] + argv
mod.main()
PYEOF

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

# ---- 1) GPU test suite ----
python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
echo "== pytest_gpu exit $? ($(date +%T))"
clean "$OUT/pytest_gpu.log" | tail -n 4

# ---- 2) graph decode, kernel breakdown (vs fused60 di S22-1) ----
timeout 2400 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
  -m "$M" --mode graph --ctx 8192 --steps 32 > "$OUT/graph60.log" 2>&1
echo "== graph60 exit $? ($(date +%T))"
clean "$OUT/graph60.log" | grep -a -E "^\[|token/s|ms/token|kernel time" | head -n 8
clean "$OUT/graph60.log" | grep -a -A 26 "kernel .*ms/token" | head -n 30 > "$OUT/graph60_kernels.txt"

# ---- 3) graph decode at ~2000-token prompt: long-context decode + prefill ----
timeout 2400 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
  -m "$M" --mode graph --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/graph2000.log" 2>&1
echo "== graph2000 exit $? ($(date +%T))"
clean "$OUT/graph2000.log" | grep -a -E "^\[" | head -n 4

# ---- 4a) regressione qwen35-27b (golden fused + ppl, C10) ----
timeout 3600 scripts/regress.sh qwen35-27b > "$OUT/regress_qwen35.log" 2>&1
echo "== regress_qwen35 exit $? ($(date +%T))"
clean "$OUT/regress_qwen35.log" | grep -a -E "PASS|FAIL|ppl|delta" | tail -n 12

# ---- 4b) regressione gemma4 (graph) ----
timeout 3600 scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
echo "== regress_gemma4 exit $? ($(date +%T))"
clean "$OUT/regress_gemma4.log" | grep -a -E "PASS|FAIL|ppl|delta" | tail -n 12

# ---- 5) numerica graph vs llama.cpp (C8; e candidato golden) ----
timeout 1800 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$G/ref_probs.json" -m "$M" --decode graph -o "$OUT/graph_now.json" \
  > "$OUT/graph_compare.log" 2>&1
echo "== graph_compare exit $? ($(date +%T))"
clean "$OUT/graph_compare.log" | tail -n 6
timeout 1800 python bench/compare_probs.py diff "$G/ref_probs.json" \
  "$OUT/graph_now.json" --strict > "$OUT/graph_compare_strict.log" 2>&1
echo "== graph_compare_strict exit $? ($(date +%T))"
clean "$OUT/graph_compare_strict.log" | tail -n 6

echo "== end $(date +%T)"
