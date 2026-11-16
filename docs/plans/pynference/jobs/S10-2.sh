#!/usr/bin/env bash
# job: S10-2
# timeout: 90
# scopo: riesecuzione di S10-1 dopo l'ampliamento di _GEMV_CONFIGS (13 configs,
#        con 16 warps e BLOCK_K 1024) per portare la banda GEMV di Q2_K sopra
#        i 100 GB/s (job S10-1: 98.5/99.1, tutto il resto verde). Stessi passi
#        di S10-1: suite K-quant, suite tests/gpu, regressione Gemma (C10).
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
