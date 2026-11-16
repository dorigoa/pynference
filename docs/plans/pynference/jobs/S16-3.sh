#!/usr/bin/env bash
# job: S16-3
# timeout: 90
# scopo: S16 passo 4 — terzo giro. Da S16-2 (nessun problema numerico emerso):
#   - parse_dump: nel dump reale le righe degli id di token portano il log stamp
#     di llama.cpp ("0.28.208.203 I   976") -> parser corretto nel worker
#     (tests/gpu/_gptoss_attn0_worker.py) + test CPU con stamp;
#   - guardia maxrel del confronto col riferimento: il piano chiede cos >= 0,9999
#     (misurato 0,9999955), la guardia aggiuntiva era 2e-4, troppo stretta per
#     il motore f16 (maxrel_sum_ranks ~4,5e-03 su pesi reali): ora 2e-2, rete
#     contro errori grossolani;
#   - banda test_quant_s11 flaky in S16-2 (MXFP4 104,5<120, Q2_0 81,7<100; verdi
#     in S16-1): qui la suite tests/gpu gira per prima, a GPU fredda (in S16-2
#     girava dopo ~5 min tra dump del 120B e attn0); i RES di banda sono
#     catturati con pytest -s senza àncore (lec. S11).
# Ordine: suite tests/gpu -> dump (llama_ref.sh, --n-cpu-moe 36) -> attn0 col
# dump (cos ref >= 0,9999, cos dump >= 0,999) -> regress gemma4 (C10).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
T=docs/plans/pynference/tools/llama_ref.sh
export GPTOSS_GGUF=$M

python -m pytest -q -s tests/gpu > "$OUT/pytest.log" 2>&1
rc=$?
echo "== pytest gpu exit $rc ($(date +%T))"
tail -n 8 "$OUT/pytest.log"
grep -a 'RES.*gbps' "$OUT/pytest.log" | head -n 6 || true
if [ "$rc" -ne 0 ]; then
  grep -a '^FAILED' "$OUT/pytest.log"
  grep -aE '^(tests/.*:[0-9]+:|E +)' "$OUT/pytest.log" | head -n 40
fi

bash $T dump $M "The capital of France is" "$OUT/dump.txt" --n-cpu-moe 36
echo "== dump exit $? ($(date +%T))"
grep -a -m 1 'attn_out-0 = ' "$OUT/dump.txt" || echo "ATTN_OUT_0_HEADER_NOT_FOUND"

GPTOSS_DUMP=$OUT/dump.txt python -m pytest -q tests/gpu/test_gptoss_attn0.py \
  > "$OUT/attn0.log" 2>&1
rc=$?
echo "== attn0 pytest exit $rc ($(date +%T))"
tail -n 12 "$OUT/attn0.log"
grep -a 'RES ' "$OUT/attn0.log" || true
if [ "$rc" -ne 0 ]; then
  grep -aE '^(FAILED|E +|.*Error)' "$OUT/attn0.log" | head -n 40
fi

bash scripts/regress.sh gemma4 > "$OUT/regress.log" 2>&1
echo "== regress exit $? ($(date +%T))"
tail -n 12 "$OUT/regress.log"

echo "== end $(date +%T)"
