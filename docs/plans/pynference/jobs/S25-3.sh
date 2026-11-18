#!/usr/bin/env bash
# job: S25-3
# timeout: 60
# scopo: ri-accettazione GPU del passo 2 di S25 dopo il fix del guard sul
#        near-tie singolo (job S25-2: tutto verde tranne
#        test_decode_matches_cpu, che su un passo decode T=1 tutto near-tie
#        asseriva su una slice vuota -> NaN). Stesso programma di S25-2.
#        Il regress Gemma C10 (2.9.2) gira nel job finale della sezione,
#        quello del passo 3.
set -uo pipefail
echo "== start $(date +%T)"
OUT="$PYI_JOB_OUT"

# 1) selftest kernel su XPU (verde in S25-1/S25-2; ripassata a ogni job)
timeout 900 python -m pynference.kernels > "$OUT/kernels_selftest.log" 2>&1
rc_k=$?
echo "== kernels_selftest exit $rc_k ($(date +%T))"
grep -E "qsa_attn_decode|rror" "$OUT/kernels_selftest.log" | tail -n 4

# 2) test GPU S25 passo 2 (con -s: righe "drift ... near-tie ...")
timeout 1200 python -m pytest -q -s tests/gpu/test_qwen4exp_qsa.py > "$OUT/gpu_qsa.log" 2>&1
rc_qsa=$?
echo "== gpu_qsa exit $rc_qsa ($(date +%T))"
grep -E "drift|passed|failed|FAILED" "$OUT/gpu_qsa.log" | tail -n 14

# 3) suite GPU completa
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
rc_all=$?
echo "== gpu_all exit $rc_all ($(date +%T))"
grep -E "FAILED|ERROR|passed|failed" "$OUT/gpu_all.log" | tail -n 5

rc=0
for r in $rc_k $rc_qsa $rc_all; do [ "$r" -ne 0 ] && rc=1; done
echo "== summary: kernels_selftest $rc_k gpu_qsa $rc_qsa gpu_all $rc_all"
exit $rc
