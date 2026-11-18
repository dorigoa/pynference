#!/usr/bin/env bash
# job: S24-3
# timeout: 120
# scopo: accettazione GPU finale di S24 (2.9.2) dopo la seconda correzione
#        dei test tests/gpu/test_qwen4exp_{hc,ple}.py emersa da S24-2:
#        block_out condiviso in hc_combine (due randn indipendenti davano
#        cos 0.48) e emb ripiegato [16T,160]->[T,2560] prima di PLE.forward
#        (QLinear device non appiattisce come F32QWeight). Sorgente
#        pynference/ invariato da S24-1; il chain test dump e il regress
#        erano già verdi.
set -uo pipefail
echo "== start $(date +%T)"
OUT="$PYI_JOB_OUT"

# 1) worker GPU S24 corretti: HC e PLE su XPU (CPU vs device, Q8_0 vs F32)
timeout 1200 python -m pytest -q tests/gpu/test_qwen4exp_hc.py tests/gpu/test_qwen4exp_ple.py > "$OUT/gpu_s24.log" 2>&1
rc_gpu24=$?
echo "== gpu_s24 exit $rc_gpu24 ($(date +%T))"
tail -n 3 "$OUT/gpu_s24.log"

# 2) suite GPU completa (2.9.2)
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
rc_gpuall=$?
echo "== gpu_all exit $rc_gpuall ($(date +%T))"
grep -E "FAILED|ERROR|passed|failed" "$OUT/gpu_all.log" | tail -n 5

# 3) regress Gemma C10 (2.9.2; exec bit mancante -> bash)
timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
rc_reg=$?
echo "== regress exit $rc_reg ($(date +%T))"
tail -n 4 "$OUT/regress.log"

rc=0
for r in $rc_gpu24 $rc_gpuall $rc_reg; do [ "$r" -ne 0 ] && rc=1; done
echo "== summary: gpu_s24 $rc_gpu24 gpu_all $rc_gpuall regress $rc_reg"
exit $rc
