#!/usr/bin/env bash
# job: S27-6
# timeout: 120
# scopo: (S27 passo 4) verifica GPU del decode a grafo di segmenti e dei
#   kernel fusi nuovi.
#   (1) tests/gpu/{test_deltanet,test_qwen4exp_qsa,test_qwen4exp_qsa_ctx,
#       test_qwen4exp_hc,test_qwen4exp_ple}.py: i rami Triton dei kernel
#       fusi (norm_sigmoid_decode, PLE inplace, QSA/HC) girano solo su
#       GPU; i gemelli CPU sono già verdi in S27 passo 4.
#   (2,3) fused @60/@2000 IDENTICHE a S27-5: il driver eager è stato
#       ricostruito (G0/A/B/GF senza grafo) → regressione di velocità
#       inattesa, atteso ~5,53 / ~4,93 tok/s (±pochi %).
#   (4,5) graph @60/@2000 con --mode graph (XPUGraph a segmenti + routing
#       host): atteso guadagno grande (lancio per kernel e collettivi
#       assorbiti dal grafo; in S27-2/3 il breakdown mostrava allreduce
#       81,5 ms/tok e migliaia di call). Con --no-profile: niente rottura
#       profilatore+cattura, e il tok/s è la misura che serve.
#   MemAvailable/GPUActive campionati ogni 3 s durante le quattro run
#   (floor).
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

# ---- 1) unit test GPU dei kernel fusi (rami Triton) -----------------------
timeout 1800 python -m pytest -q tests/gpu/test_deltanet.py \
  tests/gpu/test_qwen4exp_qsa.py tests/gpu/test_qwen4exp_qsa_ctx.py \
  tests/gpu/test_qwen4exp_hc.py tests/gpu/test_qwen4exp_ple.py \
  > "$OUT/test_fused_gpu.log" 2>&1; rt=$?
echo "== test_fused_gpu exit $rt ($(date +%T))"
tail -n 5 "$OUT/test_fused_gpu.log"

# ---- 2) fused @ ~60 token, breakdown kernel (guardia sul driver nuovo) -----
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode fused --ctx 8192 --steps 32 > "$OUT/fused60.log" 2>&1 &
P1=$!
poll "$P1" "$OUT/meminfo_fused60.log"
wait $P1; r1=$?
echo "== fused60 exit $r1 ($(date +%T))"
clean "$OUT/fused60.log" | grep -a -E "^\[r[0-9]\] mem|^\[fused\]|GPU kernel time|store: hits|resident .*swap slots|MoE: .*budget|host cache|host pin"
clean "$OUT/fused60.log" | grep -a -A 26 "kernel .*ms/token" | head -n 30 > "$OUT/fused60_kernels.txt"
wc -l < "$OUT/fused60_kernels.txt"
summarize "$OUT/meminfo_fused60.log"

# ---- 3) fused @ ~2000 token, no profile ------------------------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode fused --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/fused2000.log" 2>&1 &
P2=$!
poll "$P2" "$OUT/meminfo_fused2000.log"
wait $P2; r2=$?
echo "== fused2000 exit $r2 ($(date +%T))"
clean "$OUT/fused2000.log" | grep -a -E "^\[r[0-9]\] mem|^\[fused\]|store: hits|host cache|host pin"
summarize "$OUT/meminfo_fused2000.log"

# ---- 4) graph @ ~60 token: XPUGraph a segmenti -----------------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --no-profile \
  > "$OUT/graph60.log" 2>&1 &
P3=$!
poll "$P3" "$OUT/meminfo_graph60.log"
wait $P3; r3=$?
echo "== graph60 exit $r3 ($(date +%T))"
clean "$OUT/graph60.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|resident .*swap slots|MoE: .*budget|host cache|host pin"
clean "$OUT/graph60.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph60.log"

# ---- 5) graph @ ~2000 token -------------------------------------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/graph2000.log" 2>&1 &
P4=$!
poll "$P4" "$OUT/meminfo_graph2000.log"
wait $P4; r4=$?
echo "== graph2000 exit $r4 ($(date +%T))"
clean "$OUT/graph2000.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|host cache|host pin"
clean "$OUT/graph2000.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph2000.log"

echo "== summary: testfused=$rt fused60=$r1 fused2000=$r2 graph60=$r3 graph2000=$r4"
echo "== end $(date +%T)"
