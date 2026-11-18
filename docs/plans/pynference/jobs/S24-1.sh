#!/usr/bin/env bash
# job: S24-1
# timeout: 120
# scopo: accettazione S24 — catena blk.0+PLE blk.1 contro i dump S23-1
#        (worker CPU tests/cpu/test_qwen4exp_chain.py: cos >= 0.999 sui
#        campioni + sum entro l'1% per i nodi post-GEMM), poi i worker GPU
#        di S24 (HC e PLE su XPU), la suite tests/gpu completa e il
#        regress Gemma C10 (richiesti da 2.9.2 per chiudere la sezione).
set -uo pipefail
echo "== start $(date +%T)"
OUT="$PYI_JOB_OUT"

# 1) confronto con i dump (barra del piano: "Fatto quando")
timeout 2400 python -m pytest -q tests/cpu/test_qwen4exp_chain.py > "$OUT/chain.log" 2>&1
rc_chain=$?
echo "== chain exit $rc_chain ($(date +%T))"
grep -E "assert|Error|passed|failed" "$OUT/chain.log" | tail -n 8

# 2) worker GPU S24: HC e PLE su XPU (CPU vs device, Q8_0 vs F32)
timeout 1200 python -m pytest -q tests/gpu/test_qwen4exp_hc.py tests/gpu/test_qwen4exp_ple.py > "$OUT/gpu_s24.log" 2>&1
rc_gpu24=$?
echo "== gpu_s24 exit $rc_gpu24 ($(date +%T))"
tail -n 3 "$OUT/gpu_s24.log"

# 3) suite GPU completa (2.9.2)
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
rc_gpuall=$?
echo "== gpu_all exit $rc_gpuall ($(date +%T))"
grep -E "FAILED|ERROR|passed|failed" "$OUT/gpu_all.log" | tail -n 5

# 4) regress Gemma C10 (2.9.2; exec bit mancante -> bash)
timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
rc_reg=$?
echo "== regress exit $rc_reg ($(date +%T))"
tail -n 4 "$OUT/regress.log"

rc=0
for r in $rc_chain $rc_gpu24 $rc_gpuall $rc_reg; do [ "$r" -ne 0 ] && rc=1; done
echo "== summary: chain $rc_chain gpu_s24 $rc_gpu24 gpu_all $rc_gpuall regress $rc_reg"
exit $rc
