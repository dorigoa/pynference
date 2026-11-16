#!/usr/bin/env bash
# job: S9-1
# timeout: 90
# scopo: verifica GPU dei kernel Triton nel layout nativo dei blocchi da 32
#        (S9 passo 5): dequant vs gguf-py su blocchi casuali e tensori reali,
#        GEMV vs fp32 <= 1e-4, GEMV indicizzata vs ciclo, banda su 8192x5376,
#        GEMV F16/BF16. Poi la suite tests/gpu completa e la regressione
#        Gemma (C10), obbligatoria perché il codice quant condiviso è cambiato
#        (pynference/quant.py -> pynference/quant/).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

python -m pytest -q -s tests/gpu/test_quant_block32.py 2>&1 | tail -n 120
echo "== pytest block32 exit ${PIPESTATUS[0]} ($(date +%T))"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
