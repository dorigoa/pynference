#!/usr/bin/env bash
# job: S12-1
# timeout: 90
# scopo: kernel di IQ3_XXS e IQ2_S nel layout GGUF nativo, come S9-1/S11-1:
#        dequant contro gguf-py su blocchi casuali e su righe di tensori
#        reali (IQ3_XXS blk.0.ffn_up_exps.weight e IQ2_S blk.8.ffn_up_exps.weight,
#        stack MoE 3-D da Flash-Next IQ4_XS letti come righe piatte, k=2560),
#        GEMV contro fp32 con errore relativo <= 1e-4, GEMV indicizzata contro
#        un ciclo python, banda su 8192x5376 (pavimento basso 40 GB/s alla
#        prima passata: i valori si fissano dopo); poi suite tests/gpu e
#        regressione Gemma (C10, codice quant comune).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

python -m pytest -q -s tests/gpu/test_quant_s12.py 2>&1 | tail -n 120
echo "== pytest s12 exit ${PIPESTATUS[0]} ($(date +%T))"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
