#!/usr/bin/env bash
# job: S27-5
# timeout: 90
# scopo: (S27 passo 3) host cache degli esperti non residenti (RAM 121,5
#   GiB): ExpertStore(host_pin_bytes=...) con l'allowance di
#   `_host_pin_bytes` = MemTotal − 16 GiB − world×(dense+kv+scratch+budget).
#   Se copre i byte non residentesi, il fill pinned parte al caricamento e
#   i miss di swap copiano diritti dalla riga pinned (niente anello,
#   niente lettura GGUF in decode). Atteso: "host cache ON ~12,5 GB",
#   pinned_host ≈ 12,5 G per rank nei mem report, floor MemAvailable ~25
#   GiB più basso di S27-3 ma lontano da 0 (S17-6: mai affamare l'host).
#   (1) tests/gpu/test_moe_store.py: il percorso host cache (copy diritti
#       dalle righe, anello mai allocato) e il caso OFF verficati su GPU.
#   (2,3) le due run di bench/decode_profile.py IDENTICHE a S27-2/S27-3
#       (fused @60 con breakdown; fused @2000 --no-profile) per confrontare
#       tok/s, hit store, copie e pinned_host con la baseline S27-3
#       (4,86 / 4,41 tok/s; hit 64% / 58%; copie 11,07 / 45,70 GiB).
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

# ---- 1) unit test GPU di ExpertStore (host cache ON/OFF, ring) ----------
timeout 900 python -m pytest -q tests/gpu/test_moe_store.py \
  > "$OUT/test_moe_store.log" 2>&1; rt=$?
echo "== test_moe_store exit $rt ($(date +%T))"
tail -n 5 "$OUT/test_moe_store.log"

# ---- 2) fused @ ~60 token, breakdown kernel + store stats + mem report ----
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

# ---- 3) fused @ ~2000 token, no profiler: long-ctx decode + prefill ----
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode fused --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/fused2000.log" 2>&1 &
P2=$!
poll "$P2" "$OUT/meminfo_fused2000.log"
wait $P2; r2=$?
echo "== fused2000 exit $r2 ($(date +%T))"
clean "$OUT/fused2000.log" | grep -a -E "^\[r[0-9]\] mem|^\[fused\]|store: hits|host cache|host pin"
summarize "$OUT/meminfo_fused2000.log"

echo "== summary: teststore=$rt fused60=$r1 fused2000=$r2"
echo "== end $(date +%T)"
