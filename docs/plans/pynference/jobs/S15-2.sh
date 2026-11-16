#!/usr/bin/env bash
# job: S15-2
# timeout: 60
# scopo: rerun di S15-1 dopo i fix: ExpertGroup.expert_bytes appiattisce
#        raw_fn 2D (make_test_blocks) — era il crash di moe_check e dei due
#        test zero-budget (HostPool.fill); test_moe_store porta le want su
#        device (torch.equal cpu vs xpu). Stesso programma: bench/moe_check.py
#        con torchrun --nproc_per_node=2 (32 esperti, top-4, K = FFN = 2880
#        per Q8_0/MXFP4, 2816 per IQ4_XS; budget pieno e zero; decode/prefill
#        vs riferimento numpy+gguf-py, accordo tra rank, banda dei miss H2D),
#        suite tests/gpu completa, regressione Gemma (C10). Novita': il log
#        pytest completo va in $PYI_JOB_OUT e, se fallisce, il job stampa le
#        righe FAILED e le righe di errore (S15-1 tagliava via i traceback).
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
