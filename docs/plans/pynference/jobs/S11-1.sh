#!/usr/bin/env bash
# job: S11-1
# timeout: 90
# scopo: kernel di MXFP4, IQ4_XS e Q2_0 nel layout GGUF nativo, come S10-1
#        per i K-quant: dequant contro gguf-py su blocchi casuali e su righe
#        di tensori reali (IQ4_XS token_embd.weight e Q2_0
#        blk.8.ffn_down_exps.weight da Flash-Next IQ4_XS; MXFP4 non esiste in
#        nessun file locale, quindi solo casi casuali, come permette la
#        scheda), GEMV contro fp32 con errore relativo <= 1e-4, GEMV
#        indicizzata contro un ciclo python, banda su 8192x5376 (pavimento
#        basso 40 GB/s alla prima passata: i valori si fissano dopo);
#        poi suite tests/gpu e regressione Gemma (C10, codice quant comune).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

python -m pytest -q -s tests/gpu/test_quant_s11.py 2>&1 | tail -n 120
echo "== pytest s11 exit ${PIPESTATUS[0]} ($(date +%T))"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
