#!/usr/bin/env bash
# job: S25-2
# timeout: 60
# scopo: ri-accettazione GPU del passo 2 di S25 dopo i fix del test
#        (job S25-1: kernel selftest verde ma reference del test kernel con
#        * invece di @, e barre atol 1e-5 impossibili cross-device per i
#        pool keys passati dalla cache f16). tests/gpu/test_qwen4exp_qsa.py
#        con -s (stampa drift/near-tie misurati), selftest kernel, suite GPU
#        completa. Il regress Gemma C10 (2.9.2) gira nel job finale della
#        sezione, quello del passo 3.
set -uo pipefail
echo "== start $(date +%T)"
OUT="$PYI_JOB_OUT"

# 1) selftest kernel su XPU (già verde in S25-1; ripassata a ogni job)
timeout 900 python -m pynference.kernels > "$OUT/kernels_selftest.log" 2>&1
rc_k=$?
echo "== kernels_selftest exit $rc_k ($(date +%T))"
grep -E "qsa_attn_decode|rror" "$OUT/kernels_selftest.log" | tail -n 4

# 2) test GPU S25 passo 2 (con -s: righe "drift ... near-tie ...")
timeout 1200 python -m pytest -q -s tests/gpu/test_qwen4exp_qsa.py > "$OUT/gpu_qsa.log" 2>&1
rc_qsa=$?
echo "== gpu_qsa exit $rc_qsa ($(date +%T))"
grep -E "drift|AMBIG|passed|failed|FAILED" "$OUT/gpu_qsa.log" | tail -n 12

# 3) suite GPU completa
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
rc_all=$?
echo "== gpu_all exit $rc_all ($(date +%T))"
grep -E "FAILED|ERROR|passed|failed" "$OUT/gpu_all.log" | tail -n 5

rc=0
for r in $rc_k $rc_qsa $rc_all; do [ "$r" -ne 0 ] && rc=1; done
echo "== summary: kernels_selftest $rc_k gpu_qsa $rc_qsa gpu_all $rc_all"
exit $rc
