#!/usr/bin/env bash
# job: S15-4
# timeout: 60
# scopo: rerun di S15-3 dopo il fix del bench: make_data in bench/moe_check.py
#        scala i token per tipo (gate/up pre-clip ~ std 5) e compensa il
#        router (logit e margine top-k invariati); prima i pesi sintetici
#        MXFP4 (E8M0 a 2^±10, rms_w ~ 1.5e3) portavano gate/up a ~±2e4,
#        swiglu_oai saturava ovunque e il bf16 del prefill divergeva ~0,1 dal
#        riferimento f32. Stesso programma: bench/moe_check.py con torchrun
#        --nproc_per_node=2 (32 esperti, top-4, K = FFN = 2880 per
#        Q8_0/MXFP4, 2816 per IQ4_XS; budget pieno e zero; decode/prefill vs
#        riferimento numpy+gguf-py, accordo tra rank, banda dei miss H2D),
#        suite tests/gpu completa, regressione Gemma (C10). Log pytest
#        completo in $PYI_JOB_OUT; se fallisce, FAILED + righe E in chiaro.
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
