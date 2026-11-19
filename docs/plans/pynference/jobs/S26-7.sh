#!/usr/bin/env bash
# job: S26-7
# timeout: 120
# scopo: (S26 passo 4) verifica d'oro Flash-Next su GPU (TP world=2), modello
#   completo Qwen4Exp (48 layer, MoE in ExpertStore, PLE in RAM):
#   1) torchrun bench/compare_probs.py ours --ref (prompt ids da
#      tests/golden/qwen4exp/ref_probs.json, S23-1) --decode graph
#      (accettato e ignorato: percorso unico eager, grafi a S27);
#      run_bench.py ribinda la factory tokenizer come in regress.sh (S17);
#      stdout INTERO in $OUT (trappola tail S26-2/4) + poller meminfo/3 s;
#   2) compare_probs.py diff ref_probs.json ours_probs.json --strict → C8
#      (near-tie 0,15, KL <= 1e-2, greedy-identicali >= 3/4);
#   3) torchrun bench.ppl -c 512 --chunks 8 vs ref_ppl.txt (llama.cpp
#      --n-cpu-moe 48: PPL = 1,0002 +/- 0,00008).
# Nessun test Gemma qui: la regressione gemma4 (C10) e regress.sh qwen4exp
# stanno nel job del passo 5.
set -uo pipefail
OUT=$PYI_JOB_OUT
echo "== start $(date +%T)"
grep -E "GPUActive|MemAvailable|MemTotal" /proc/meminfo

MODEL=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
G=tests/golden/qwen4exp

poll() { # poll PID LOG: campiona meminfo ogni 3 s in LOG finché PID vivo
  # (morto o zombie: il figlio resta zombie finché non lo si aspetta)
  local pid=$1 log=$2 st
  while st=$(awk -F') ' 'NR==1 {print substr($2, 1, 1)}' "/proc/$pid/stat" 2>/dev/null) \
         && [ -n "$st" ] && [ "$st" != "Z" ]; do
    printf '%s ' "$(date +%T)" >> "$log"
    grep -E "MemAvailable|GPUActive" /proc/meminfo | tr '\n' ' ' >> "$log"
    echo >> "$log"
    sleep 3
  done
}

summarize() { # summarize LOG: floor MemAvailable e max GPUActive (GiB)
  if [ ! -s "$1" ]; then echo "(no samples: $1)"; return; fi
  awk '/MemAvailable/ {if (m == "" || $3 < m) m = $3}
       /GPUActive/    {if ($6 > g) g = $6}
       END {printf "floor MemAvailable %.2f GiB, max GPUActive %.2f GiB\n",
            m / 2**20, g / 2**20}' "$1"
}

# tokenizer factory ribindato per qwen4exp (identico a regress.sh, S17)
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

# ---- 1: ours --ref (prompt ids dal golden llama.cpp)
echo "== compare_probs ours --ref ($(date +%T))"
timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
    --ref "$G/ref_probs.json" -m "$MODEL" --decode graph -o "$OUT/ours_probs.json" \
    > "$OUT/ours.log" 2>&1 &
OPID=$!
poll "$OPID" "$OUT/meminfo_ours.log"
wait $OPID; RO=$?
echo "== ours exit $RO ($(date +%T)) (atteso 0)"
grep -E "store: expandable_segments on|Traceback|Error|error" "$OUT/ours.log" | head -n 10
grep -E "^(chat-it|chat-en|code|raw): " "$OUT/ours.log" | head -n 8
summarize "$OUT/meminfo_ours.log"

# ---- 2: C8 strict vs llama.cpp
echo "== diff --strict ($(date +%T))"
timeout 300 python bench/compare_probs.py diff "$G/ref_probs.json" "$OUT/ours_probs.json" \
    --strict > "$OUT/diff_strict.log" 2>&1
RD=$?
echo "== diff exit $RD ($(date +%T)) (atteso 0 = C8)"
cat "$OUT/diff_strict.log"

# ---- 3: ppl vs llama.cpp (ref 1,0002 +/- 0,00008)
echo "== ppl ($(date +%T))"
timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$MODEL" \
    -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1 &
PPID_=$!
poll "$PPID_" "$OUT/meminfo_ppl.log"
wait $PPID_; RP=$?
echo "== ppl exit $RP ($(date +%T)) (atteso 0)"
grep -E "Final estimate|tokens,|Traceback|Error" "$OUT/ppl.log" | head -n 6
echo "ref ppl:"
cat "$G/ref_ppl.txt"
summarize "$OUT/meminfo_ppl.log"

echo "== summary ours=$RO diff=$RD ppl=$RP $(date +%T)"
grep -E "GPUActive|MemAvailable" /proc/meminfo
echo "== end $(date +%T)"
