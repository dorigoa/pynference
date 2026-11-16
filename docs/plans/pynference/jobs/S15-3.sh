#!/usr/bin/env bash
# job: S15-3
# timeout: 60
# scopo: rerun di S15-2 dopo il fix dell'esperto condiviso: in MoE.decode/
#        prefill la tripla shared (replicata, make_qweight non sharda) viene
#        sommata dopo all_reduce, non prima — con world=2 prima era contata
#        due volte e moe_check perdeva decode/prefill vs riferimento CPU
#        (rel 0,68-0,86) su tutti i tipi. Stesso programma: bench/
#        moe_check.py con torchrun --nproc_per_node=2 (32 esperti, top-4,
#        K = FFN = 2880 per Q8_0/MXFP4, 2816 per IQ4_XS; budget pieno e
#        zero; decode/prefill vs riferimento numpy+gguf-py, accordo tra
#        rank, banda dei miss H2D), suite tests/gpu completa, regressione
#        Gemma (C10). Log pytest completo in $PYI_JOB_OUT; se fallisce,
#        FAILED + righe E in chiaro.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

torchrun --nproc_per_node=2 bench/moe_check.py > "$OUT/moe_check.log" 2>&1
echo "== moe_check exit $? ($(date +%T))"
grep -a '^RES \|ALL OK\|^FAIL\|^Traceback\|Error' "$OUT/moe_check.log" | tail -n 60

python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc=$?
echo "== pytest gpu exit $rc ($(date +%T))"
tail -n 8 "$OUT/pytest.log"
if [ "$rc" -ne 0 ]; then
  grep -a '^FAILED' "$OUT/pytest.log"
  grep -aE '^(tests/.*:[0-9]+:|E +)' "$OUT/pytest.log" | head -n 60
fi

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
