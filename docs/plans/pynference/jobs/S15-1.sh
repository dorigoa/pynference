#!/usr/bin/env bash
# job: S15-1
# timeout: 60
# scopo: MoE generico + ExpertStore (C5) su 2 GPU: bench/moe_check.py con
#        torchrun --nproc_per_node=2 (32 esperti, top-4, K = FFN = 2880 per
#        Q8_0/MXFP4, K = FFN = 2816 per IQ4_XS; budget pieno e zero = tutto
#        in RAM pinned; decode/prefill vs riferimento numpy+gguf-py, accordo
#        tra rank, banda dei miss H2D). Poi suite tests/gpu completa (i nuovi
#        test_moe_store/test_moe_layer girano in world 1) e regressione
#        Gemma (C10): quant/** e' cambiato (import pigri di q8_0/Triton).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

torchrun --nproc_per_node=2 bench/moe_check.py > "$OUT/moe_check.log" 2>&1
echo "== moe_check exit $? ($(date +%T))"
grep -a '^RES \|ALL OK\|FAIL' "$OUT/moe_check.log" | tail -n 40
tail -n 15 "$OUT/moe_check.log"

python -m pytest -q tests/gpu 2>&1 | tail -n 30
echo "== pytest gpu exit ${PIPESTATUS[0]} ($(date +%T))"

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 30 "$OUT/regress.log"

echo "== end $(date +%T)"
