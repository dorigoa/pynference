#!/usr/bin/env bash
# job: S25-1
# timeout: 60
# scopo: accettazione GPU del passo 2 di S25 (versione GPU della QSA):
#        selftest dei kernel su XPU (incl. il nuovo qsa_attn_decode),
#        tests/gpu/test_qwen4exp_qsa.py (prefill torch su xpu vs CPU —
#        regime di selezione e forme reali 24/2 GQA head_dim 256 —,
#        kernel decode vs softmax torch con slot di padding -1, cache f32
#        e f16, decode e2e xpu vs CPU attraverso il confine dei blocchi),
#        suite GPU completa. Il regress Gemma C10 (2.9.2) gira nel job
#        finale della sezione, quello del passo 3.
set -uo pipefail
echo "== start $(date +%T)"
OUT="$PYI_JOB_OUT"

# 1) selftest kernel su XPU (compila ed esegue qsa_attn_decode con confronto torch)
timeout 900 python -m pynference.kernels > "$OUT/kernels_selftest.log" 2>&1
rc_k=$?
echo "== kernels_selftest exit $rc_k ($(date +%T))"
grep -E "qsa_attn_decode|rror" "$OUT/kernels_selftest.log" | tail -n 4

# 2) test GPU S25 passo 2
timeout 1200 python -m pytest -q tests/gpu/test_qwen4exp_qsa.py > "$OUT/gpu_qsa.log" 2>&1
rc_qsa=$?
echo "== gpu_qsa exit $rc_qsa ($(date +%T))"
tail -n 3 "$OUT/gpu_qsa.log"

# 3) suite GPU completa (kernels.py è toccato: regress sui kernel condivisi)
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
rc_all=$?
echo "== gpu_all exit $rc_all ($(date +%T))"
grep -E "FAILED|ERROR|passed|failed" "$OUT/gpu_all.log" | tail -n 5

rc=0
for r in $rc_k $rc_qsa $rc_all; do [ "$r" -ne 0 ] && rc=1; done
echo "== summary: kernels_selftest $rc_k gpu_qsa $rc_qsa gpu_all $rc_all"
exit $rc
