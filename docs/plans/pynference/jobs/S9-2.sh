#!/usr/bin/env bash
# job: S9-2
# timeout: 90
# scopo: ripetizione di S9-1 dopo il fix dell'indice byte dei nibble in
#        _decode ((e % 16) invece di (e % 16) // 2) e la chiave autotune per
#        tipo; real Q5_0 ora preso dal tensore 3D degli esperti (in Flash-Next
#        Q5_0 non esiste su tensori 2D). Stesso programma di S9-1: dequant vs
#        gguf-py, GEMV vs fp32 <= 1e-4, GEMV indicizzata vs ciclo, banda
#        8192x5376, GEMV F16/BF16; suite tests/gpu; regressione Gemma (C10).
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
