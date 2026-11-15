#!/usr/bin/env bash
# job: S6-3
# timeout: 40
# scopo: riprova S6-2 (seconda finestra GPU): pytest tests/gpu (kernel, quant
#        Q8_0 con soglie) e regressione Gemma C10 (scripts/regress.sh gemma4),
#        entrambi verdi. Fix: ppl.py ora con torchrun (in S6-2 python singolo
#        -> 33 GB su una sola GPU da 31,89 GiB: hang fino al timeout); output
#        di regress su log visibile per diagnostica.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
