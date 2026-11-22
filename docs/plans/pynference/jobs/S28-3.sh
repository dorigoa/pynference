#!/usr/bin/env bash
# job: S28-3
# timeout: 120
# scopo: S28 passo 2 — benchmark per modello (bench/decode_profile.py
#   --mode graph --no-profile, C9): decode a contesto ~512 token
#   (prompt 512, 32 step), prefill di 2000 token (16 step di decode a
#   contesto ~2000), decode a contesto ~8192 token (prompt 8192, ctx
#   statico 9728, 8 step); memoria per GPU = righe "mem" per rank +
#   poll /proc/meminfo (GPUActive/MemAvailable, verbatim S27-2). Poi una
#   prova del server per modello (avvio torchrun, /health, /v1/models,
#   chat non-streaming + streaming SSE, SIGTERM rank 0, "server stopped
#   cleanly", verbatim pattern S8-2 su porta 8092). Canaro gemma4
#   (regress C10) per primo e suite tests/gpu per ultima (2.9.2).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

MG=/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf
MQW=/nvme/gguf/qwen3.8/Qwen3.8-27B-Q8_0.gguf
MO=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
MQ=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf

ENVOK=0 CANARY=0 rc_g=255
rc_gA=255 rc_gB=255 rc_gC=255 rc_wA=255 rc_wB=255 rc_wC=255
rc_oA=255 rc_oB=255 rc_oC=255 rc_qA=255 rc_qB=255 rc_qC=255
rc_sg=255 rc_sw=255 rc_so=255 rc_sq=255 rc_gpu=255

# ---- guardia (A): fatti di filesystem NVIDIA (verbatim S27-24) --------------
echo "== nvidia probe $(date +%T)"
NV_FILES=$(ls /dev/nvidia* 2>/dev/null | head -n 6)
NV_VER=$(head -n 1 /proc/driver/nvidia/version 2>/dev/null)
NV_MODS=$(lsmod 2>/dev/null | grep -ci nvidia || true)
echo "nvidia device files: ${NV_FILES:-(none)}"
echo "nvidia version: ${NV_VER:-(none)}"
echo "nvidia kernel modules: ${NV_MODS:-0}"
if [ -n "$NV_FILES" ] || [ -n "$NV_VER" ] || [ "${NV_MODS:-0}" != "0" ]; then
  echo "== ENV-BROKEN: tracce NVIDIA su filesystem — matrice saltata $(date +%T)"
  echo "== summary: env=$ENVOK canary=$CANARY gemma=$rc_g gA=$rc_gA gB=$rc_gB gC=$rc_gC wA=$rc_wA wB=$rc_wB wC=$rc_wC oA=$rc_oA oB=$rc_oB oC=$rc_oC qA=$rc_qA qB=$rc_qB qC=$rc_qC srv_g=$rc_sg srv_w=$rc_sw srv_o=$rc_so srv_q=$rc_sq gpu=$rc_gpu"
  echo "== end $(date +%T)"
  exit 0
fi
echo "== ENV OK: nessun segno NVIDIA su filesystem $(date +%T)"
ENVOK=1

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

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

# ---- guardia (B): canaro = regress.sh gemma4 (C10, atteso 0) ----------------
timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_g=$?
echo "== regress gemma4 exit $rc_g ($(date +%T)) (atteso 0 = canaro + C10)"
clean "$OUT/regress_gemma4.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 8
if [ $rc_g -ne 0 ]; then
  echo "== CANARY-RED: gemma regress rossa — resto saltato $(date +%T)"
  echo "== summary: env=$ENVOK canary=$CANARY gemma=$rc_g gA=$rc_gA gB=$rc_gB gC=$rc_gC wA=$rc_wA wB=$rc_wB wC=$rc_wC oA=$rc_oA oB=$rc_oB oC=$rc_oC qA=$rc_qA qB=$rc_qB qC=$rc_qC srv_g=$rc_sg srv_w=$rc_sw srv_o=$rc_so srv_q=$rc_sq gpu=$rc_gpu"
  echo "== end $(date +%T)"
  exit 0
fi
CANARY=1

# ---- matrice benchmark (C9: --mode graph --no-profile) ----------------------
bench_run() { # bench_run TAG VAR TMO MODEL CTX PROMPT_TOKENS STEPS
  local tag=$1 var=$2 tmo=$3 model=$4 ctx=$5 pt=$6 steps=$7
  local lg="$OUT/bench_$tag.log" pl="$OUT/poll_$tag.log" rc
  timeout "$tmo" torchrun --nproc_per_node=2 bench/decode_profile.py \
    -m "$model" --mode graph --ctx "$ctx" --prompt-tokens "$pt" \
    --steps "$steps" --no-profile > "$lg" 2>&1 &
  local p=$!
  poll "$p" "$pl"
  wait "$p"; rc=$?
  printf -v "$var" %s "$rc"
  echo "== bench $tag exit $rc ($(date +%T)) ctx=$ctx pt=$pt steps=$steps (atteso 0)"
  clean "$lg" | grep -aE "prefill [0-9]+ tokens|wall time|\] mem |store:" | tail -n 12
  summarize "$pl"
}

# a) contesto ~512 (ctx statico 8192, C9)  b) prefill 2000  c) contesto ~8192
echo "== MODEL gemma4 $(date +%T)"
bench_run gemma512  rc_gA 600  "$MG"  8192 512  32
bench_run gemma2k   rc_gB 600  "$MG"  8192 2000 16
bench_run gemma8k   rc_gC 900  "$MG"  9728 8192 8

echo "== MODEL qwen35-27b $(date +%T)"
bench_run qwen512   rc_wA 600  "$MQW" 8192 512  32
bench_run qwen2k    rc_wB 600  "$MQW" 8192 2000 16
bench_run qwen8k    rc_wC 900  "$MQW" 9728 8192 8

echo "== MODEL gpt-oss-120b $(date +%T)"
bench_run gos512    rc_oA 900  "$MO"  8192 512  32
bench_run gos2k     rc_oB 900  "$MO"  8192 2000 16
bench_run gos8k     rc_oC 1200 "$MO"  9728 8192 8

echo "== MODEL qwen4exp $(date +%T)"
bench_run q4e512    rc_qA 900  "$MQ"  8192 512  32
bench_run q4e2k     rc_qB 1200 "$MQ"  8192 2000 16
bench_run q4e8k     rc_qC 1800 "$MQ"  9728 8192 8

# ---- prova del server per modello (pattern S8-2, porta 8092) ----------------
srv_test() { # srv_test TAG VAR MODEL : health, models, chat sync+SSE, stop
  local tag=$1 var=$2 model=$3
  local slog="$OUT/server_$tag.log" pl="$OUT/server_${tag}_poll.log" rc=1 ok=""
  torchrun --nproc_per_node=2 -m pynference.server -m "$model" \
    --host 127.0.0.1 --port 8092 > "$slog" 2>&1 &
  local p=$!
  poll "$p" "$pl" &
  local poll_pid=$!
  stop_server() {
    local r
    r=$(pgrep -o -P "$p" -f "pynference.server" || true)
    [ -n "$r" ] && kill -TERM "$r"
    wait "$p" 2>/dev/null
  }
  for i in $(seq 1 120); do
    ok=$(curl -sf -m 3 http://127.0.0.1:8092/health || true)
    [ -n "$ok" ] && break
    kill -0 "$p" 2>/dev/null || break
    sleep 3
  done
  if [ -z "$ok" ]; then
    echo "== server $tag FAIL: /health non pronto $(date +%T)"
    tail -n 25 "$slog" | grep -v -E "CCL_WARN|^W[0-9]"
    stop_server; echo "== server fermato (FAIL-branche) $(date +%T)"
    rc=1
  else
    echo "== server $tag health: $ok ($(date +%T))"
    curl -sf -m 10 http://127.0.0.1:8092/v1/models > "$OUT/server_${tag}_models.json"
    echo "models: $(head -c 160 "$OUT/server_${tag}_models.json")"
    curl -sf -m 600 http://127.0.0.1:8092/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Ciao! Presentati in due frasi."}],"max_tokens":24,"temperature":0}' \
      > "$OUT/server_${tag}_sync.json"
    local e_sync=$?
    python - "$OUT/server_${tag}_sync.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d["choices"][0]["message"]
print("sync finish=%s usage=%s content=%r" %
      (d["choices"][0]["finish_reason"], d.get("usage"), (m.get("content") or "")[:80]))
PY
    curl -sN -m 600 http://127.0.0.1:8092/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"One two three"}],"max_tokens":16,"temperature":0,"stream":true}' \
      > "$OUT/server_${tag}_sse.txt"
    local e_sse=$?
    echo "stream: righe data=$(grep -c '^data:' "$OUT/server_${tag}_sse.txt") done=$(grep -c '\[DONE\]' "$OUT/server_${tag}_sse.txt") curl_sync=$e_sync curl_sse=$e_sse"
    echo "== arresto server $tag $(date +%T)"
    stop_server
    rc=$?
    echo "== server $tag exit $rc ($(date +%T)) clean=$(grep -c 'server stopped cleanly' "$slog")"
    clean "$slog" | grep -aE "POST /v1/chat|loaded in|server stopped" | tail -n 6
  fi
  wait "$poll_pid" 2>/dev/null
  summarize "$pl"
  printf -v "$var" %s "$rc"
}

echo "== SERVER gemma4 $(date +%T)"
srv_test gemma rc_sg "$MG"
echo "== SERVER qwen35-27b $(date +%T)"
srv_test qwen rc_sw "$MQW"
echo "== SERVER gpt-oss-120b $(date +%T)"
srv_test goss rc_so "$MO"
echo "== SERVER qwen4exp $(date +%T)"
srv_test q4e rc_sq "$MQ"

# ---- suite tests/gpu per ultima (2.9.2) --------------------------------------
python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
rc_gpu=$?
echo "== gpu tests exit $rc_gpu ($(date +%T)) (atteso 0)"
tail -n 2 "$OUT/pytest_gpu.log"

echo "== summary: env=$ENVOK canary=$CANARY gemma=$rc_g gA=$rc_gA gB=$rc_gB gC=$rc_gC wA=$rc_wA wB=$rc_wB wC=$rc_wC oA=$rc_oA oB=$rc_oB oC=$rc_oC qA=$rc_qA qB=$rc_qB qC=$rc_qC srv_g=$rc_sg srv_w=$rc_sw srv_o=$rc_so srv_q=$rc_sq gpu=$rc_gpu"
echo "== end $(date +%T)"
