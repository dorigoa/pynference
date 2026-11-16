#!/usr/bin/env bash
# job: S16-1
# timeout: 90
# scopo: S16 passo 4 — attenzione del layer 0 di gpt-oss-120b sulle 2 GPU (TP,
#        worker torchrun dentro tests/gpu/test_gptoss_attn0.py) contro:
#        (a) riferimento CPU f32 con pesi attn Q8_0 dequantizzati via gguf-py e
#            trascrizione indipendente delle formule (YaRN, NEOX, sink, window,
#            wo+wo_b) — soglia cos >= 0.9999;
#        (b) dump attn_out-0 di llama-eval-callback (llama_ref.sh dump con
#            --n-cpu-moe 36, prompt corto = pochi token, quindi tutte le righe;
#            colonne troncata a prime/ultime 3) — soglia cos >= 0.999 (il dump
#            accumula kq in bf16 e stampa 4 decimali).
#        Poi suite tests/gpu completa e regressione Gemma (C10).
#        Log completi in $PYI_JOB_OUT; righe RES in evidenza.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
T=docs/plans/pynference/tools/llama_ref.sh
export GPTOSS_GGUF=$M

bash $T dump $M "The capital of France is" "$OUT/dump.txt" --n-cpu-moe 36
echo "== dump exit $? ($(date +%T))"
grep -a -m 3 'attn_out-0 = ' "$OUT/dump.txt" || echo "ATTN_OUT_0_HEADER_NOT_FOUND"
grep -a -m 1 'number of input tokens' "$OUT/dump.txt" || true

GPTOSS_DUMP=$OUT/dump.txt python -m pytest -q tests/gpu/test_gptoss_attn0.py \
  > "$OUT/attn0.log" 2>&1
rc=$?
echo "== attn0 pytest exit $rc ($(date +%T))"
tail -n 12 "$OUT/attn0.log"
grep -a 'RES ' "$OUT/attn0.log" || true
if [ "$rc" -ne 0 ]; then
  grep -aE '^(FAILED|E +|.*Error)' "$OUT/attn0.log" | head -n 40
fi

python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc=$?
echo "== pytest gpu exit $rc ($(date +%T))"
tail -n 8 "$OUT/pytest.log"
if [ "$rc" -ne 0 ]; then
  grep -a '^FAILED' "$OUT/pytest.log"
  grep -aE '^(tests/.*:[0-9]+:|E +)' "$OUT/pytest.log" | head -n 60
fi

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
