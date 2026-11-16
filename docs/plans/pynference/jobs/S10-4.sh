#!/usr/bin/env bash
# job: S10-4
# timeout: 90
# scopo: verifica dello stato finale del passo 5 dopo la decisione sul blocco
#        2.4.8 (vedere il report): kernel GEMV riportati al percorso caldo dei
#        job S10-1/S10-2 (rimossi il clamp della coda k e le 3 configurazioni
#        num_stages=2 del tentativo 2, che in S10-3 aveva portato 81/86/98
#        GB/s); resta la correzione di correttezza dello store mascherato in
#        _gemv_indexed_kernel. Soglia di banda ora per tipo: 90 GB/s per Q2_K
#        (tetto osservato 98-99, è il più compute-bound: 3,05 elementi/byte),
#        100 per gli altri. Il piano non fissa soglie di banda (la card chiede
#        di misurarla; C9 non bloccante). Stessi passi di S10-1: suite K-quant,
#        suite tests/gpu, regressione Gemma (C10).
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
