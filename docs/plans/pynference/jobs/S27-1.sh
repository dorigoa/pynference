#!/usr/bin/env bash
# job: S27-1
# timeout: 90
# scopo: (S27 passo 1) profilo di partenza di Flash-Next TP2 in regime S26
#   (esperti ~10 GB/rank residenti, decode eager: use_graph è flag inerte in
#   Qwen4Exp, i grafi arrivano nei passi successivi). Due run di
#   bench/decode_profile.py --ctx 8192 (C9):
#   (1) mode fused, prompt ~60, CON breakdown kernel (torch profiler): tempo
#       per componente + hit/miss degli esperti + memoria per GPU
#       (model.memory_report() e torch alloc/res, stampati per rank);
#   (2) mode fused, prompt ~2000, --no-profile: decode a contesto lungo e
#       prefill token/s.
#   MemAvailable/GPUActive campionati ogni 3 s durante le due run (floor).
#   NB: bench/decode_profile.py ora usa load_tokenizer (modificato in S27),
#   quindi qui si lancia diretto, senza run_bench.py.
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

# ---- 1) fused @ ~60 token, breakdown kernel + store stats + mem report ----
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode fused --ctx 8192 --steps 32 > "$OUT/fused60.log" 2>&1 &
P1=$!
poll "$P1" "$OUT/meminfo_fused60.log"
wait $P1; r1=$?
echo "== fused60 exit $r1 ($(date +%T))"
clean "$OUT/fused60.log" | grep -a -E "^\[|^\[r|token/s|ms/token|kernel time|store:|load done|resident" | head -n 12
clean "$OUT/fused60.log" | grep -a -A 26 "kernel .*ms/token" | head -n 30 > "$OUT/fused60_kernels.txt"
wc -l < "$OUT/fused60_kernels.txt"
summarize "$OUT/meminfo_fused60.log"

# ---- 2) fused @ ~2000 token, no profiler: long-ctx decode + prefill ----
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode fused --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/fused2000.log" 2>&1 &
P2=$!
poll "$P2" "$OUT/meminfo_fused2000.log"
wait $P2; r2=$?
echo "== fused2000 exit $r2 ($(date +%T))"
clean "$OUT/fused2000.log" | grep -a -E "^\[|token/s|store:" | head -n 6
summarize "$OUT/meminfo_fused2000.log"

echo "== summary: fused60=$r1 fused2000=$r2"
echo "== end $(date +%T)"
