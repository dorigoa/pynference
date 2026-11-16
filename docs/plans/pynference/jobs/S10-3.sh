#!/usr/bin/env bash
# job: S10-3
# timeout: 90
# scopo: riesecuzione di S10-1 dopo la seconda correzione mirata alla banda
#        GEMV di Q2_K (job S10-1: 98.5/99.1; job S10-2, sole configs amplify:
#        98.8/98.0). Cambiati ora i kernel: la coda k viene clampata dentro la
#        riga al posto del tl.where per-elemento (meno istruzioni per elemento
#        e nessun caricamento oltre l'ultima riga) in _gemv_kernel e
#        _gemv_indexed_kernel (store anche mascherato sulle righe oltre N);
#        aggiunte 3 configurazioni autotune con num_stages=2. Stessi passi di
#        S10-1: suite K-quant, suite tests/gpu, regressione Gemma (C10).
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
