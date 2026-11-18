#!/usr/bin/env bash
# job: S25-5
# timeout: 90
# scopo: ri-accettazione GPU del passo 3 di S25 (job finale della sezione):
#        stesso programma di S25-4 con i due bug di test corretti
#        (arange off-by-one nel prefill 2058; lunghezza contesto N=T0+n al
#        posto del contatore di passi nel decode). Dump del blk.3 su xpu con
#        i pesi reali + i due regimi di lunghezza intorno ai 2048 token.
#        Include la suite GPU completa e il regress Gemma C10 (2.9.2).
set -uo pipefail
echo "== start $(date +%T)"
OUT="$PYI_JOB_OUT"

# 1) selftest kernel su XPU (verde in S25-1..4; ripassata a ogni job)
timeout 900 python -m pynference.kernels > "$OUT/kernels_selftest.log" 2>&1
rc_k=$?
echo "== kernels_selftest exit $rc_k ($(date +%T))"
grep -E "qsa_attn_decode|rror" "$OUT/kernels_selftest.log" | tail -n 4

# 2) test GPU S25 passo 3 (con -s: righe "drift ... near-tie ..." e i
#    conteggi delle celle tenute ai confini dei due regimi)
timeout 1500 python -m pytest -q -s tests/gpu/test_qwen4exp_qsa_ctx.py > "$OUT/gpu_qsa_ctx.log" 2>&1
rc_ctx=$?
echo "== gpu_qsa_ctx exit $rc_ctx ($(date +%T))"
grep -E "drift|kept|passed|failed|FAILED" "$OUT/gpu_qsa_ctx.log" | tail -n 20

# 3) suite GPU completa
timeout 3000 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
rc_all=$?
echo "== gpu_all exit $rc_all ($(date +%T))"
grep -E "FAILED|ERROR|passed|failed" "$OUT/gpu_all.log" | tail -n 5

# 4) regress Gemma C10 (2.9.2): exec bit mancante -> bash (lezione S24-3)
timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
rc_reg=$?
echo "== regress exit $rc_reg ($(date +%T))"
grep -E "PASS|FAIL|ok|error" "$OUT/regress.log" | tail -n 6

rc=0
for r in $rc_k $rc_ctx $rc_all $rc_reg; do [ "$r" -ne 0 ] && rc=1; done
echo "== summary: kernels_selftest $rc_k gpu_qsa_ctx $rc_ctx gpu_all $rc_all regress $rc_reg"
exit $rc
