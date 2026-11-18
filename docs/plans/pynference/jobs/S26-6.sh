#!/usr/bin/env bash
# job: S26-6
# timeout: 60
# scopo: (S26 passo 1) conferma dopo il re-baseline di
#   tests/golden/gpt-oss-120b/ours_baseline.json al profilo S26 (decisione
#   S26-5: A deterministico + B regime S18 PASS + C8 strict verde → lo
#   spostamento dp vs il vecchio golden S18 (max 1,37e-04) è accettato).
#   1) regress.sh gpt-oss-120b: atteso PASS (probs vs nuovo golden + ppl vs
#      ours_ppl.txt, invariato); stdout intero in $OUT (mai tail corto:
#      trappola S26-2/4) con poller meminfo/3 s (atteso floor MemAvailable
#      come S26-5 ~41 GiB, marker toggle >=1);
#   2) pytest -q tests/gpu completa (S25 l'ultima verde, 116 passed in
#      ~4 min in S18-4; ora ci sono anche i test qwen4exp di S24/S25);
#   3) regress.sh gemma4 (C10 / 2.9.2).
set -uo pipefail
OUT=$PYI_JOB_OUT
echo "== start $(date +%T)"
grep -E "GPUActive|MemAvailable|MemTotal" /proc/meminfo

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

# ---- 1: regress gpt-oss-120b contro il nuovo golden
echo "== regress gpt-oss-120b ($(date +%T))"
timeout 1800 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1 &
RPID=$!
poll "$RPID" "$OUT/meminfo_regress.log"
wait $RPID; RC=$?
echo "== regress gpt-oss exit $RC ($(date +%T)) (atteso 0)"
echo "toggle markers: $(grep -c 'expandable_segments on' "$OUT/regress_gptoss.log") (atteso >=1)"
grep -E "max \|dp\||ppl: baseline|PASS|FAIL" "$OUT/regress_gptoss.log" | tail -n 12
summarize "$OUT/meminfo_regress.log"

# ---- 2: suite GPU completa
echo "== pytest tests/gpu ($(date +%T))"
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
RGP=$?
echo "== pytest gpu exit $RGP ($(date +%T)) (atteso 0)"
tail -n 3 "$OUT/gpu_all.log"

# ---- 3: regressione Gemma (C10; 2.9.2 la vuole verde)
echo "== regress gemma4 ($(date +%T))"
timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
RG4=$?
echo "== gemma4 exit $RG4 ($(date +%T)) (atteso 0)"
grep -E "max \|dp\||ppl: baseline|PASS|FAIL" "$OUT/regress_gemma4.log" | tail -n 10

echo "== summary regress=$RC pytestgpu=$RGP gemma4=$RG4 $(date +%T)"
grep -E "GPUActive|MemAvailable" /proc/meminfo
echo "== end $(date +%T)"
