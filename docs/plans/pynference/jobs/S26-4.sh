#!/usr/bin/env bash
# job: S26-4
# timeout: 90
# scopo: (S26 passo 1, via chirurgica) verificare su GPT-OSS il toggle
#   expandable_segments attorno ai soli buffer di ExpertStore (moe.py), la
#   via verde di S26-3 (setter ok, global/setterkeep SIGABRT con oneCCL).
#   Nessun env var: il toggle e' nel codice, la conf globale resta off.
#   1) regress.sh gpt-oss-120b (C8) con poller MemAvailable/GPUActive ogni
#      3 s: lo store ~20 GiB/rank ora allocato in segmenti espandibili non
#      deve specchiare in RAM (floor MemAvailable molto sopra l'OOM S17-6);
#   2) se regress verde: decode_profile.py --mode graph --no-profile --ctx
#      8192 (in S18 4,2-4,6 token/s) sotto lo stesso poller.
#   Log interi in $OUT (S26-2: mai tubare il regress in tail corto).
set -uo pipefail
OUT=$PYI_JOB_OUT
MODEL=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
echo "== start $(date +%T)"
grep -E "GPUActive|MemAvailable|MemTotal" /proc/meminfo

poll() { # poll PID LOG: campiona meminfo ogni 3 s finche' il PID non e'
  # morto o zombie (il figlio di questo shell resta zombie finche' non lo
  # si aspetta con wait, quindi /proc/$pid esiste ancora)
  local pid=$1 log=$2 st
  while st=$(awk -F') ' 'NR==1 {print substr($2, 1, 1)}' "/proc/$pid/stat" 2>/dev/null) \
         && [ -n "$st" ] && [ "$st" != "Z" ]; do
    printf '%s ' "$(date +%T)"
    grep -E "MemAvailable|GPUActive" /proc/meminfo | tr '\n' ' '
    echo
    sleep 3
  done
}

summarize() { # summarize LOG: floor MemAvailable e max GPUActive (MiB)
  awk '/MemAvailable/ {if (m == "" || $3 < m) m = $3}
       /GPUActive/    {if ($6 > g) g = $6}
       END {printf "floor MemAvailable %.2f GiB, max GPUActive %.2f GiB\n",
            m / 2**20, g / 2**20}' "$1"
}

echo "== regress gpt-oss-120b ($(date +%T))"
timeout 3600 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1 &
RPID=$!
poll "$RPID" "$OUT/meminfo_regress.log"
wait $RPID; RC=$?
echo "== regress exit $RC ($(date +%T))"
grep -c "expandable_segments on" "$OUT/regress_gptoss.log" || echo "TOGGLE NOT SEEN"
grep -E "store: layer" "$OUT/regress_gptoss.log" | head -2
grep -E "store: layer" "$OUT/regress_gptoss.log" | tail -1
summarize "$OUT/meminfo_regress.log"
grep -v -E "CCL_WARN|^W1" "$OUT/regress_gptoss.log" | tail -n 12

# bench/ rebinder for non-Gemma models, same as scripts/regress.sh (S17).
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

if [ "$RC" -eq 0 ]; then
  echo "== decode_profile graph --no-profile ($(date +%T))"
  timeout 1800 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
      -m "$MODEL" --mode graph --no-profile --ctx 8192 > "$OUT/decode.log" 2>&1 &
  DPID=$!
  poll "$DPID" "$OUT/meminfo_decode.log" >/dev/null 2>&1
  wait $DPID; DC=$?
  echo "== decode exit $DC ($(date +%T))"
  summarize "$OUT/meminfo_decode.log"
  grep -v -E "CCL_WARN|^W1" "$OUT/decode.log" | tail -n 20
else
  echo "== regress red: skipping decode_profile ($(date +%T))"
fi

grep -E "GPUActive|MemAvailable" /proc/meminfo
echo "== end $(date +%T)"
