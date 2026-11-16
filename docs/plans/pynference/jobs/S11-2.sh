#!/usr/bin/env bash
# job: S11-2
# timeout: 90
# scopo: riesecuzione dopo il fix in kernels._decode (guardia constexpr sul
#        caricamento della scala uint16: su righe MXFP4 dispari W16 e' uint8 e
#        il frontend Triton type-checka il ramo morto dopo il return constexpr
#        -> bitcast 8->16 rifiutato, FAIL di test_dequant_odd_rows_mxfp4 in
#        S11-1). Stessa sequenza di S11-1, ma la fase 1 stampa solo le righe
#        RES e i riepiloghi (in S11-1 il tail aveva tagliato i valori di
#        banda/GEMV): servono per fissare i pavimenti MIN_GBPS per tipo come
#        in S10. Poi suite tests/gpu e regressione Gemma (C10).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

python -m pytest -q -s tests/gpu/test_quant_s11.py 2>&1 | grep -E '^RES |passed|failed|FAILED|error' | tail -n 60
echo "== pytest s11 exit ${PIPESTATUS[0]} ($(date +%T))"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
