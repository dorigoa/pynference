#!/usr/bin/env bash
# job: S28-1
# timeout: 120
# scopo: validazione finale: regress.sh su tutti e quattro i modelli; pytest tests/gpu (S28 passo 1)
set -uo pipefail
echo "== start $(date +%T)"
python -m pytest -q tests/gpu 2>&1 | tail -n 5
echo "== gpu tests exit ${PIPESTATUS[0]} ($(date +%T))"
for m in gemma4 gpt-oss-120b qwen35-27b qwen4exp; do
  scripts/regress.sh "$m" > "$PYI_JOB_OUT/regress-$m.log" 2>&1
  echo "== regress $m exit $? ($(date +%T))"
  tail -n 8 "$PYI_JOB_OUT/regress-$m.log"
done
echo "== end $(date +%T)"
