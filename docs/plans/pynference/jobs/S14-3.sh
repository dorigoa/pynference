#!/usr/bin/env bash
# job: S14-3
# timeout: 90
# scopo: chiusura sezione S14 (2.9.2): la sezione ha toccato codice condiviso
#        (`tokenizer.py`: BPETokenizer + load_tokenizer, passo 2; nessuna
#        modifica a engine/kernels/models in questo job). Esegue la suite
#        `python -m pytest -q tests/gpu` e la regressione di Gemma (C10,
#        `scripts/regress.sh gemma4`: greedy contro ours_baseline.json,
#        scarto max di p sulle top-10 <= 1e-3, ppl entro 0,1%), entrambe verdi.
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
