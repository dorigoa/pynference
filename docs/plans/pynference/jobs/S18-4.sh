#!/usr/bin/env bash
# job: S18-4
# timeout: 45
# scopo: chiudere S18 dopo il fix della correzione di moe.decode_body (usava
#   K.swiglu_oai fisso al posto di self.act: 3 rossi in tests/gpu su synthetic
#   swiglu, job S18-3). Il percorso gpt-oss (act=swiglu_oai) non cambia
#   valori (alpha/limit passati esplicitamente = stessi default 1.702/7.0).
#   (1) pytest tests/gpu completo (2.9.2, sub-timeout tarato su S18-3: 685 s);
#   (2) regress.sh gpt-oss-120b (decode_body e` il percorso decode del modello);
#   (3) regress.sh gemma4 (C10, 2.9.2). Microbench GEMV e tempi graph gia`
#   misurati in S18-3: non si ripetono.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

echo "== baseline meminfo:"
grep -aE 'MemAvailable|GPUActive' /proc/meminfo || echo "GPUACTIVE_FIELD_MISSING"

# ---- full GPU test suite (2.9.2)
echo "== pytest tests/gpu ($(date +%T))"
timeout 2400 python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest_gpu.log"
[ "$rc_pytest" -eq 0 ] || grep -a '^FAILED' "$OUT/pytest_gpu.log" | head -n 12

# ---- regressions: gpt-oss golden (decode_body) + Gemma (C10)
echo "== regress gpt-oss-120b ($(date +%T))"
timeout 900 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1
rc_rg=$?
echo "== regress gpt-oss exit $rc_rg ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1|UserWarning" "$OUT/regress_gptoss.log" | tail -n 10

echo "== regress gemma4 ($(date +%T))"
timeout 600 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_rgm=$?
echo "== regress gemma4 exit $rc_rgm ($(date +%T))"
tail -n 6 "$OUT/regress_gemma4.log"

echo "-- oom lines (job safety net):"
journalctl -k --no-pager --since "45 min ago" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 5 || true

echo "== summary: pytest=$rc_pytest regress_gptoss=$rc_rg regress_gemma4=$rc_rgm"
echo "== end $(date +%T)"
