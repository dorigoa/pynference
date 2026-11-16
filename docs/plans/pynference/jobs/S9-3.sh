#!/usr/bin/env bash
# job: S9-3
# timeout: 90
# scopo: ripetizione di S9-2 dopo il byte-load di qh in _decode (bit e&7 del
#        byte qh e>>3, offset QHB) al posto dei 2 caricamenti uint16 + shift
#        uint32, che causavano il FAIL banda di Q5_0 (24.5 GB/s < 100 mentre
#        Q5_1 con lo stesso codice stava a 193). Stesso programma di S9-2:
#        dequant vs gguf-py, GEMV vs fp32 <= 1e-4, GEMV indicizzata vs ciclo,
#        banda 8192x5376, GEMV F16/BF16; suite tests/gpu; regressione Gemma (C10).
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
