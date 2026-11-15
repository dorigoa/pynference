#!/usr/bin/env bash
# job: S6-2
# timeout: 30
# scopo: seconda finestra GPU: pytest tests/gpu (kernel, quant Q8_0 con soglie)
#        e regressione Gemma C10 (scripts/regress.sh gemma4), entrambi verdi
set -uo pipefail
echo "== start $(date +%T)"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]}"

bash scripts/regress.sh gemma4 2>&1 | tail -n 30
echo "== regress exit ${PIPESTATUS[0]}"

echo "== end $(date +%T)"
