#!/usr/bin/env bash
# job: S28-7
# timeout: 40
# scopo: S28 passo 2, chiusura formale (dopo S28-6: canaro verde 3/3; gos8k
#   graph 2/2 + fused 1/1 hang -> @8192 gpt-oss GIA' documentato non
#   ottenibile, nessun retry bench; server q4e/gemma/qwen verdi; server gpt-oss
#   health/models/sync verdi, SSE stall dopo 12 chunk = flake intermittente,
#   in S28-3 stesso test pienamente verde; job ucciso dall'esterno a ~37 min
#   PRIMA della suite -> manca 2.9.2 nell'ultimo job).
#   Ordine scelto per sopravvivere a una nuova chiusura: (0) guardia NVIDIA
#   (verbatim S28-6); (1) canaro gemma4 try1, se rosso dump proprietà XPU +
#   try2 REGRESS_KEEP, due rossi -> stop (2.6); (2) pytest tests/gpu SUBITO
#   dopo il canaro (obbligatorio per 2.9.2, ~4 min); (3) per ULTIMO,
#   skippabile, riprova del server gpt-oss come conferma opzionale: curl
#   limitato a -m 300 per limitare il costo di uno stall; rc registrato ma
#   non bloccante (la prova server gpt-oss verde esiste gia' da S28-3).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

MO=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf

ENVOK=0 CANARY=0
rc_gpu=255 rc_so=255

# ---- guardia (A): fatti di filesystem NVIDIA (verbatim S27-24/S28-6) --------
echo "== nvidia probe $(date +%T)"
NV_FILES=$(ls /dev/nvidia* 2>/dev/null | head -n 6)
NV_VER=$(head -n 1 /proc/driver/nvidia/version 2>/dev/null)
NV_MODS=$(lsmod 2>/dev/null | grep -ci nvidia || true)
echo "nvidia device files: ${NV_FILES:-(none)}"
echo "nvidia version: ${NV_VER:-(none)}"
echo "nvidia kernel modules: ${NV_MODS:-0}"
if [ -n "$NV_FILES" ] || [ -n "$NV_VER" ] || [ "${NV_MODS:-0}" != "0" ]; then
  echo "== ENV-BROKEN: tracce NVIDIA su filesystem — resto saltato $(date +%T)"
  echo "== summary: env=$ENVOK canary=$CANARY gpu=$rc_gpu srv_o=$rc_so"
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
    echo "== CANARY-RED x2: stop diagnostica $(date +%T)"
    echo "== summary: env=$ENVOK canary=$CANARY gpu=$rc_gpu srv_o=$rc_so"
    echo "== end $(date +%T)"
    exit 0
  fi
  echo "== canaro verde al try2: rosso precedente transiente $(date +%T)"
fi
CANARY=1

# ---- suite tests/gpu SUBITO (obbligatorio 2.9.2, ~4 min) --------------------
python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
rc_gpu=$?
echo "== gpu tests exit $rc_gpu ($(date +%T)) (atteso 0)"
tail -n 2 "$OUT/pytest_gpu.log"

# ---- ultimo pezzo skippabile: riprova server gpt-oss -------------------------
srv_test() { # srv_test TAG VAR MODEL [HWT] [MX] : health, models, chat sync+SSE,
             # stop; MX = timeout curl (default 600)
  local tag=$1 var=$2 model=$3 hwt=${4:-120} mx=${5:-600}
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
    curl -sf -m "$mx" http://127.0.0.1:8092/v1/chat/completions \
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
    curl -sN -m "$mx" http://127.0.0.1:8092/v1/chat/completions \
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

echo "== SERVER gpt-oss-120b retry (ultimo, opzionale; curl -m 300) $(date +%T)"
srv_test goss2 rc_so "$MO" 120 300
if [ "$rc_so" -ne 0 ]; then
  echo "== retry goss non verde: resta la prova verde S28-3 + sync S28-6 (flake ri-documentato) $(date +%T)"
fi

echo "== summary: env=$ENVOK canary=$CANARY gpu=$rc_gpu srv_o=$rc_so"
echo "== end $(date +%T)"
