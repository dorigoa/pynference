#!/usr/bin/env bash
# job: S8-2
# timeout: 60
# scopo: come S8-1 (riferimento in S8-1.sh), ma ref.txt ripulita dalle righe
#        di progresso del caricatore (`models/gemma4.py` stampa su stdout) e
#        dai warning CCL: in S8-1 il test live ha confrontato una REF sporca.
#        Il grezzo resta in ref_raw.txt. Più righe di output pytest al fallimento.
set -uo pipefail
echo "== start $(date +%T)"
M=/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf
OUT=$PYI_JOB_OUT
PROMPT="Ciao! Presentati in due frasi."
mkdir -p "$OUT"

torchrun --nproc_per_node=2 -m pynference.cli generate -m "$M" --chat \
    -p "$PROMPT" -n 64 --temp 0 --ctx 4096 > "$OUT/ref_raw.txt" 2> "$OUT/ref.err"
echo "== ref exit $? ($(date +%T))"
[ -s "$OUT/ref_raw.txt" ] || { echo "!! ref_raw.txt vuoto"; tail -n 20 "$OUT/ref.err"; exit 1; }
grep -v -E 'CCL_WARN|loaded [0-9]+/[0-9]+ layers' "$OUT/ref_raw.txt" > "$OUT/ref.txt"
echo "== ref righe $(grep -c . "$OUT/ref.txt"), $(wc -c < "$OUT/ref.txt") byte"

torchrun --nproc_per_node=2 -m pynference.server -m "$M" \
    --host 127.0.0.1 --port 8090 --ctx 4096 > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "== server launcher pid $SERVER_PID ($(date +%T))"

stop_server() {   # SIGTERM sul solo rank 0 (figlio più vecchio del launcher)
    local r
    r=$(pgrep -o -P "$SERVER_PID" -f "pynference.server" || true)
    [ -n "$r" ] && kill -TERM "$r"
    wait "$SERVER_PID" 2>/dev/null
}

HEALTH=""
for i in $(seq 1 72); do
    HEALTH=$(curl -sf http://127.0.0.1:8090/health || true)
    [ -n "$HEALTH" ] && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "!! server morto"; tail -n 30 "$OUT/server.log"; exit 1; }
    sleep 5
done
if [ -z "$HEALTH" ]; then echo "!! /health non pronto"; stop_server; tail -n 30 "$OUT/server.log"; exit 1; fi
echo "== health: $HEALTH ($(date +%T))"

PYI_SERVER_URL=http://127.0.0.1:8090 PYI_REF_FILE="$OUT/ref.txt" \
PYI_REF_PROMPT="$PROMPT" python -m pytest -q tests/gpu/test_server_live.py 2>&1 | tail -n 60
echo "== pytest live exit ${PIPESTATUS[0]} ($(date +%T))"

echo "== arresto server ($(date +%T))"
stop_server
echo "== server launcher exit $? ($(date +%T))"
echo "== 'server stopped cleanly' righe: $(grep -c "server stopped cleanly" "$OUT/server.log")"
tail -n 8 "$OUT/server.log" | grep -v -E "CCL_WARN|^W[0-9]"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
