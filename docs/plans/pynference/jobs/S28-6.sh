#!/usr/bin/env bash
# job: S28-6
# timeout: 90
# scopo: S28 passo 2 (chiusura dopo S28-5: canaro VERDE byte-identico =
#   transiente chiuso; gos8k graph hang 2/2, job ucciso dall'esterno a ~13 min
#   prima del watchdog → fused/server/suite saltati).
#   Ordine: risultato-costoso-prima-dopo-canaro così i parziali restano utili
#   se la finestra si chiude di nuovo.
#   (0) guardia NVIDIA (verbatim S28-5).
#   (1) canaro gemma4 try1; se rosso: dump proprietà XPU + try2 REGRESS_KEEP;
#       due rossi → stop (2.6). (S28-5 try1 era verde: ci si aspetta verde.)
#   (2) gos8k SOLO fused ctx 9728 pt 8192 steps 8, tmo 1200 + watchdog SIGABRT
#       a 1140 + PYTHONFAULTHANDLER (graph rosso 2/2 S28-3/S28-5: retry graph
#       non ripetuto, fallback già approvato nel piano S28-5 — deviazione C9).
#       rc!=0 → documenta e prosegui (i server valgono comunque).
#   (3) server q4e PER PRIMO (hwt 240 = 12 min, echo ogni 40), poi
#       gemma/qwen/goss. Blocchi verbatim S28-5.
#   (4) pytest tests/gpu ultima (2.9.2).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

MO=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
MG=/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf
MQW=/nvme/gguf/qwen3.8/Qwen3.8-27B-Q8_0.gguf
MQ=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf

ENVOK=0 CANARY=0
rc_sk=skip rc_oCf=255 rc_sq=255 rc_sg=255 rc_sw=255 rc_so=255 rc_gpu=255

# ---- guardia (A): fatti di filesystem NVIDIA (verbatim S27-24/S28-5) --------
echo "== nvidia probe $(date +%T)"
NV_FILES=$(ls /dev/nvidia* 2>/dev/null | head -n 6)
NV_VER=$(head -n 1 /proc/driver/nvidia/version 2>/dev/null)
NV_MODS=$(lsmod 2>/dev/null | grep -ci nvidia || true)
echo "nvidia device files: ${NV_FILES:-(none)}"
echo "nvidia version: ${NV_VER:-(none)}"
echo "nvidia kernel modules: ${NV_MODS:-0}"
if [ -n "$NV_FILES" ] || [ -n "$NV_VER" ] || [ "${NV_MODS:-0}" != "0" ]; then
  echo "== ENV-BROKEN: tracce NVIDIA su filesystem — resto saltato $(date +%T)"
  echo "== summary: env=$ENVOK canary=$CANARY oC=$rc_sk oCf=$rc_oCf srv_q=$rc_sq srv_g=$rc_sg srv_w=$rc_sw srv_o=$rc_so gpu=$rc_gpu"
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
            m / 2^20, g / 2^20}' "$1"
}

wdog() { # wdog SECONDS PATTERN: dopo SECONDS invia SIGABRT ai processi che
         # matchano PATTERN (PYTHONFAULTHANDLER stampa gli stack dove sono)
  sleep "$1"
  local c
  for c in $(pgrep -f "$2" 2>/dev/null); do kill -ABRT "$c" 2>/dev/null; done
}

# ---- guardia (B): canaro = regress.sh gemma4, max 2 tentativi ---------------
canary() { # canary N: regress gemma4 (REGRESS_KEEP per gli ours_now), echo esito
  local n=$1 rc
  REGRESS_KEEP="$OUT" timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4_try$n.log" 2>&1
  rc=$?
  echo "== regress gemma4 try$n exit $rc ($(date +%T)) (atteso 0)"
  clean "$OUT/regress_gemma4_try$n.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10
  return $rc
}

canary 1
if [ $? -ne 0 ]; then
  echo "== CANARY-RED try1: dump proprietà XPU $(date +%T)"
  python - <<'PY'
import torch
print("torch", torch.__version__, "devices", torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    p = torch.xpu.get_device_properties(i)
    print(i, p.name, "uuid", getattr(p, "uuid", "?"), "mem %.1f GiB" % (p.total_memory / 2**30))
PY
  canary 2
  if [ $? -ne 0 ]; then
    echo "== CANARY-RED x2: stop diagnostica (S28-4 + S28-6 try1/try2) $(date +%T)"
    echo "== summary: env=$ENVOK canary=$CANARY oC=$rc_sk oCf=$rc_oCf srv_q=$rc_sq srv_g=$rc_sg srv_w=$rc_sw srv_o=$rc_so gpu=$rc_gpu"
    echo "== end $(date +%T)"
    exit 0
  fi
  echo "== canaro verde al try2: S28-4 era transiente $(date +%T)"
fi
CANARY=1

# ---- (2) gos8k: SOLO fused (graph rosso 2/2, fallback approvato) -------------
gos_run() { # gos_run TAG VAR MODE TMO : decode @ctx~8192, watchdog a TMO-60
  local tag=$1 var=$2 mode=$3 tmo=$4
  local lg="$OUT/bench_$tag.log" pl="$OUT/poll_$tag.log" rc wd
  PYTHONFAULTHANDLER=1 timeout "$tmo" torchrun --nproc_per_node=2 \
    bench/decode_profile.py -m "$MO" --mode "$mode" --ctx 9728 \
    --prompt-tokens 8192 --steps 8 --no-profile > "$lg" 2>&1 &
  local p=$!
  wdog $((tmo - 60)) 'bench/decode_profile.py -m /nvme/gguf/gpt-oss-120b' &
  wd=$!
  poll "$p" "$pl"
  wait "$p"; rc=$?
  kill "$wd" 2>/dev/null
  printf -v "$var" %s "$rc"
  echo "== bench $tag exit $rc ($(date +%T)) mode=$mode ctx=9728 pt=8192 steps=8 (atteso 0)"
  clean "$lg" | grep -aE "prefill [0-9]+ tokens|wall time|\] mem |store:|Thread 0x|File \"|line [0-9]+ in" | tail -n 20
  summarize "$pl"
}

echo "== GOS8K fused $(date +%T) (graph rosso 2/2: S28-3, S28-5)"
gos_run gos8kf rc_oCf fused 1200
if [ "$rc_oCf" -ne 0 ]; then
  echo "== GOS8K FUSED ROSSO (rc=$rc_oCf): @8192 gpt-oss non ottenibile, proseguo con i server $(date +%T)"
fi

# ---- (3) servers: q4e per primo (cache GGUF calda), attesa /health 12 min ---
srv_test() { # srv_test TAG VAR MODEL [HWT] : health, models, chat sync+SSE, stop
  local tag=$1 var=$2 model=$3 hwt=${4:-120}
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
  for i in $(seq 1 "$hwt"); do
    ok=$(curl -sf -m 3 http://127.0.0.1:8092/health || true)
    [ -n "$ok" ] && break
    kill -0 "$p" 2>/dev/null || break
    [ $((i % 40)) -eq 0 ] && echo "  wait $tag $i/$(date +%T): $(wc -l < "$slog") righe log, $(grep -c 'store:' "$slog") store"
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

echo "== SERVER qwen4exp $(date +%T) (hwt=240)"
srv_test q4e rc_sq "$MQ" 240
echo "== SERVER gemma4 $(date +%T)"
srv_test gemma rc_sg "$MG"
echo "== SERVER qwen35-27b $(date +%T)"
srv_test qwen rc_sw "$MQW"
echo "== SERVER gpt-oss-120b $(date +%T)"
srv_test goss rc_so "$MO"

# ---- (4) suite tests/gpu per ultima (2.9.2) ---------------------------------
python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
rc_gpu=$?
echo "== gpu tests exit $rc_gpu ($(date +%T)) (atteso 0)"
tail -n 2 "$OUT/pytest_gpu.log"

echo "== summary: env=$ENVOK canary=$CANARY oC=$rc_sk oCf=$rc_oCf srv_q=$rc_sq srv_g=$rc_sg srv_w=$rc_sw srv_o=$rc_so gpu=$rc_gpu"
echo "== end $(date +%T)"
