#!/usr/bin/env bash
# job: S10-1
# timeout: 90
# scopo: kernel dei K-quant (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) nel layout GGUF
#        nativo, come S9-3 per i tipi a blocchi da 32: dequant contro gguf-py
#        su super-blocchi casuali e su righe di tensori reali (Q3_K/Q4_K/Q5_K
#        da Qwen3.8-27B-UD-Q4_K_XL, Q6_K da Flash-Next IQ4_XS; Q2_K solo
#        casuale perche' non esiste in questi file), GEMV contro fp32 con error
#        relativo <= 1e-4, GEMV indicizzata contro un ciclo python, banda su
#        8192x5376; poi suite tests/gpu e regressione Gemma (C10).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

python -m pytest -q -s tests/gpu/test_quant_kquant.py 2>&1 | tail -n 120
echo "== pytest kquant exit ${PIPESTATUS[0]} ($(date +%T))"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
