#!/usr/bin/env bash
# job: S27-8
# timeout: 120
# scopo: (S27 passo 5, job finale) verdezza completa della sezione.
#   (1) pytest -q tests/gpu suite COMPLETA (2.9.2): include
#       tests/gpu/test_graph_capture.py — il fixture `_select_inputs` è
#       stato corretto (score generato su CPU + .to(DEV), come `keys`;
#       in S27-7 falliva per `Expected a 'xpu' device type for generator`).
#   (2) bash scripts/regress.sh qwen4exp: ATTESO PASS — nota: da S27 il
#       flag graph del golden non è più inerte, la regressione gira ora
#       sul grafo reale (corpi funzione condivisi con l'eager → stessi
#       valori della golden S27-4).
#   (3) bash scripts/regress.sh gemma4 (C10, richiesta 2.9.2).
#   (4) bash scripts/regress.sh gpt-oss-120b: moe.py toccato al passo 3
#       (host cache pinned, opt-in) → richiesta dalla scheda.
#   (5,6) velocità finale: graph @60/@2000 identici a S27-7 (--mode graph
#       --no-profile), per la misura conclusiva nel report.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf

poll() { # poll PID LOG: campiona meminfo ogni 3 s in LOG finché PID vivo
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

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

# ---- 1) suite GPU completa -------------------------------------------------
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1; rg=$?
echo "== pytest gpu exit $rg ($(date +%T))"
tail -n 3 "$OUT/gpu_all.log"

# ---- 2) regress qwen4exp (ora sul grafo reale) -------------------------------
timeout 1800 bash scripts/regress.sh qwen4exp > "$OUT/regress_qwen4exp.log" 2>&1; rq=$?
echo "== regress qwen4exp exit $rq ($(date +%T)) (atteso 0)"
clean "$OUT/regress_qwen4exp.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL|greedy|top-10" | tail -n 12

# ---- 3) regress gemma4 (C10) ----------------------------------------------
timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1; r4=$?
echo "== regress gemma4 exit $r4 ($(date +%T)) (atteso 0)"
clean "$OUT/regress_gemma4.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10

# ---- 4) regress gpt-oss-120b (moe.py toccato nel passo 3) -------------------
timeout 1800 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1 &
P0=$!
poll "$P0" "$OUT/meminfo_gptoss.log"
wait $P0; ro=$?
echo "== regress gpt-oss-120b exit $ro ($(date +%T)) (atteso 0)"
clean "$OUT/regress_gptoss.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10
summarize "$OUT/meminfo_gptoss.log"

# ---- 5) velocità finale graph @ ~60 token -----------------------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --no-profile \
  > "$OUT/graph60.log" 2>&1 &
P1=$!
poll "$P1" "$OUT/meminfo_graph60.log"
wait $P1; r1=$?
echo "== graph60 exit $r1 ($(date +%T))"
clean "$OUT/graph60.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|host cache"
clean "$OUT/graph60.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph60.log"

# ---- 6) velocità finale graph @ ~2000 token ----------------------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/graph2000.log" 2>&1 &
P2=$!
poll "$P2" "$OUT/meminfo_graph2000.log"
wait $P2; r2=$?
echo "== graph2000 exit $r2 ($(date +%T))"
clean "$OUT/graph2000.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|host cache"
clean "$OUT/graph2000.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph2000.log"

echo "== summary: gpu=$rg q4exp=$rq gemma4=$r4 gptoss=$ro graph60=$r1 graph2000=$r2"
echo "== end $(date +%T)"
