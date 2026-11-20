#!/usr/bin/env bash
# job: S27-3
# timeout: 90
# scopo: (S27 passo 2) pesi densi nel loro quant nativo: `_wq` ora da'
#   QLinear (Q8_0), NativeQWeight/gemv_native (Q6_K), HalfQWeight (BF16),
#   F32QWeight (F32) su GPU; su CPU resta il path f32. La VRAM liberata
#   (~7 GB/rank attesi) va agli esperti residenti perche' `_dense_bytes`
#   ora conta i byte nativi e `_expert_budget` ne sottrae di meno.
#   (1) scripts/regress.sh qwen4exp: C8 contro llama.cpp (compare_probs
#       ours --ref ref_probs.json) + baseline nostra (ours_baseline.json,
#       dp <= 1e-3) + ppl vs ours_ppl.txt. Atteso: C8 verde; sulla baseline
#       possibile deriva (i numerici cambiano: GEMV nativo in decode e
#       bf16 in prefill invece del GEMM f32) -> si ri-baselinea al passo 5.
#   (2,3) le due run di bench/decode_profile.py identiche a S27-2 (fused
#       @60 con breakdown kernel; fused @2000 --no-profile) per confrontare
#       decode/prefill tok/s, hit store e memoria per GPU (weights giu',
#       experts su') con la baseline del passo 1.
#   MemAvailable/GPUActive campionati ogni 3 s durante le due run (floor).
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

# ---- 1) C8 contro llama.cpp + regressione sulla baseline (deriva attesa) --
timeout 3500 scripts/regress.sh qwen4exp > "$OUT/regress.log" 2>&1; r0=$?
echo "== regress exit $r0 ($(date +%T))"
clean "$OUT/regress.log" | head -n 40

# ---- 2) fused @ ~60 token, breakdown kernel + store stats + mem report ----
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode fused --ctx 8192 --steps 32 > "$OUT/fused60.log" 2>&1 &
P1=$!
poll "$P1" "$OUT/meminfo_fused60.log"
wait $P1; r1=$?
echo "== fused60 exit $r1 ($(date +%T))"
clean "$OUT/fused60.log" | grep -a -E "^\[r[0-9]\] mem|^\[fused\]|GPU kernel time|store: hits|resident .*swap slots|MoE: .*budget"
clean "$OUT/fused60.log" | grep -a -A 26 "kernel .*ms/token" | head -n 30 > "$OUT/fused60_kernels.txt"
wc -l < "$OUT/fused60_kernels.txt"
summarize "$OUT/meminfo_fused60.log"

# ---- 3) fused @ ~2000 token, no profiler: long-ctx decode + prefill ----
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode fused --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/fused2000.log" 2>&1 &
P2=$!
poll "$P2" "$OUT/meminfo_fused2000.log"
wait $P2; r2=$?
echo "== fused2000 exit $r2 ($(date +%T))"
clean "$OUT/fused2000.log" | grep -a -E "^\[r[0-9]\] mem|^\[fused\]|store: hits"
summarize "$OUT/meminfo_fused2000.log"

echo "== summary: regress=$r0 fused60=$r1 fused2000=$r2"
echo "== end $(date +%T)"
