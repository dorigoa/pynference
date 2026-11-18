#!/usr/bin/env bash
# job: S22-1
# timeout: 120
# scopo: S22 passo 1 — profilo di partenza del decode di Qwen3.8-27B (S21 ha
#   lasciato la sola via eager fused, use_graph non implementata). Tre run di
#   bench/decode_profile.py su TP2, --ctx 8192 (C9):
#   (1) mode torch, prompt ~60, con breakdown kernel (riferimento: quanto
#       valgono già i kernel fusi);
#   (2) mode fused, prompt ~60, con breakdown kernel: baseline vera del passo;
#   (3) mode fused, prompt ~2000 (--prompt-tokens), --no-profile: decode a
#       contesto lungo + prefill token/s (entrambi in C9).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/Qwen3.8-27B-Q8_0.gguf

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

# ---- 1) torch mode: kernel breakdown without fused decode ----
timeout 2400 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
  -m "$M" --mode torch --ctx 8192 --steps 32 > "$OUT/torch60.log" 2>&1
echo "== torch60 exit $? ($(date +%T))"
clean "$OUT/torch60.log" | grep -a -E "^\[|token/s|ms/token|kernel time" | head -n 8
clean "$OUT/torch60.log" | grep -a -A 16 "kernel .*ms/token" | head -n 20 > "$OUT/torch60_kernels.txt"

# ---- 2) fused mode: the S21 baseline, kernel breakdown ----
timeout 2400 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
  -m "$M" --mode fused --ctx 8192 --steps 32 > "$OUT/fused60.log" 2>&1
echo "== fused60 exit $? ($(date +%T))"
clean "$OUT/fused60.log" | grep -a -E "^\[|token/s|ms/token|kernel time" | head -n 8
clean "$OUT/fused60.log" | grep -a -A 26 "kernel .*ms/token" | head -n 30 > "$OUT/fused60_kernels.txt"

# ---- 3) fused mode at ~2000-token prompt: long-context decode + prefill ----
timeout 2400 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
  -m "$M" --mode fused --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/fused2000.log" 2>&1
echo "== fused2000 exit $? ($(date +%T))"
clean "$OUT/fused2000.log" | grep -a -E "^\[" | head -n 4

echo "== end $(date +%T)"
