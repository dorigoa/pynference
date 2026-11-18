#!/usr/bin/env bash
# job: S22-3
# timeout: 90
# scopo: S22 passo 3 (chiusura) — in S22-2 le due regressioni sono fallite
#   con exit 126 ("Permission denied"): lo script le invocava come
#   `scripts/regress.sh ...` ma il file è committato senza bit di esecuzione
#   (git mode 100644). S21-3 e S18-4 invocavano lo stesso script con
#   `bash scripts/regress.sh ...`. Qui si riesegue solo il necessario;
#   essendo l'ultimo job della sezione (§2.9.2) si include anche la suite
#   tests/gpu:
#   (1) suite tests/gpu;
#   (2) bash scripts/regress.sh qwen35-27b (golden fused + ppl, C10);
#   (3) bash scripts/regress.sh gemma4 (C10, §2.9.2).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

# ---- 1) GPU test suite (2.9.2) ----
timeout 1200 python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
rc_pytest=$?
echo "== pytest_gpu exit $rc_pytest ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1|UserWarning|warn" "$OUT/pytest_gpu.log" | tail -n 3

# ---- 2) regressione qwen35-27b (golden fused + ppl) ----
timeout 3000 bash scripts/regress.sh qwen35-27b > "$OUT/regress_qwen35.log" 2>&1
rc_rq=$?
echo "== regress_qwen35 exit $rc_rq ($(date +%T))"
grep -a -E "PASS|FAIL|ppl:|max \|dp\|" "$OUT/regress_qwen35.log" | tail -n 12

# ---- 3) regressione gemma4 (C10, 2.9.2) ----
timeout 3000 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_rg=$?
echo "== regress_gemma4 exit $rc_rg ($(date +%T))"
grep -a -E "PASS|FAIL|ppl:|max \|dp\|" "$OUT/regress_gemma4.log" | tail -n 12

echo "== summary: pytest=$rc_pytest regressqwen=$rc_rq regressgemma=$rc_rg"
echo "== end $(date +%T)"
