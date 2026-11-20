#!/usr/bin/env bash
# job: S27-7
# timeout: 40
# scopo: (S27 passo 4, ripetizione del grafo) in S27-6 i rami Triton dei
#   kernel fusi e le due run fused erano verdi (5,61 / 5,53 tok/s), ma la
#   cattura di A[il] sui layer QSA è caduta a qwen4exp.py:696 `vis[nb]`:
#   Level Zero rifiuta l'advanced indexing bool (UR error 44); topk,
#   gather int64, index_copy_ e where nello stesso corpo catturavano ok.
#   Fix: `vis` ricostruita con gather int64 e confronti (valori identici).
#   (1) tests/gpu/test_graph_capture.py: pattern esatto del blocco di
#       selezione + kernel Triton qsa_attn_decode dentro XPUGraph, replay
#       bit-identico all'eager (regressione permanente della capture).
#   (2,3) graph @60/@2000 come in S27-6 (--mode graph --no-profile):
#       atteso exit 0 e tok/s >= fused (5,61 / 5,53) — il grafo assorbe
#       launch e collettivi (S27-3 breakdown: allreduce 81,5 ms/tok).
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

# ---- 1) test GPU di capture (blocco selezione + kernel QSA) ---------------
timeout 1800 python -m pytest -q tests/gpu/test_graph_capture.py \
  > "$OUT/test_capture.log" 2>&1; rt=$?
echo "== test_capture exit $rt ($(date +%T))"
tail -n 5 "$OUT/test_capture.log"

# ---- 2) graph @ ~60 token ----------------------------------------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --no-profile \
  > "$OUT/graph60.log" 2>&1 &
P1=$!
poll "$P1" "$OUT/meminfo_graph60.log"
wait $P1; r1=$?
echo "== graph60 exit $r1 ($(date +%T))"
clean "$OUT/graph60.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|resident .*swap slots|MoE: .*budget|host cache|host pin"
clean "$OUT/graph60.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph60.log"

# ---- 3) graph @ ~2000 token ---------------------------------------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/graph2000.log" 2>&1 &
P2=$!
poll "$P2" "$OUT/meminfo_graph2000.log"
wait $P2; r2=$?
echo "== graph2000 exit $r2 ($(date +%T))"
clean "$OUT/graph2000.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|host cache|host pin"
clean "$OUT/graph2000.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph2000.log"

echo "== summary: captest=$rt graph60=$r1 graph2000=$r2"
echo "== end $(date +%T)"
