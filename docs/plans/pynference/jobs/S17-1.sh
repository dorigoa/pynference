#!/usr/bin/env bash
# job: S17-1
# timeout: 90
# scopo: S17 passo 3 — verifica C8 del modello GPT-OSS completo:
#   1) suite tests/gpu per prima, a GPU fredda (lezione S16-3; rigira anche
#      test_moe_layer.py aggiornato al nuovo swiglu_oai del passo 1);
#   2) compare_probs ours --ref senza grafo (--decode fused; il grafo arriva
#      in S18) + diff --strict (C8: greedy quasi-pareggio, KL top-10 <= 1e-2,
#      >= 3 prompt su 4 identici);
#   3) bench/ppl.py 8 blocchi da 512 contro ref_ppl.txt (C8: scarto <= 1%);
#   4) se un confronto o la suite non torna: dump dei layer (2.5) nello
#      stesso job, stesso prompt di S16 ("The capital of France is"),
#      per il passo 4.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
T=docs/plans/pynference/tools/llama_ref.sh
export GPTOSS_GGUF=$M

python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 6 "$OUT/pytest.log"
if [ "$rc_pytest" -ne 0 ]; then
  grep -a '^FAILED' "$OUT/pytest.log" | head -n 12
fi

timeout 3000 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
  --ref tests/golden/gpt-oss-120b/ref_probs.json -m "$M" --decode fused \
  -o "$OUT/ours.json" > "$OUT/ours.log" 2>&1
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
grep -v -E "CCL_WARN|^W1" "$OUT/ours.log" | tail -n 10

python bench/compare_probs.py diff --strict \
  tests/golden/gpt-oss-120b/ref_probs.json "$OUT/ours.json" > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T))"
tail -n 15 "$OUT/diff.log"

timeout 3000 torchrun --nproc_per_node=2 bench/ppl.py -m "$M" -c 512 --chunks 8 \
  > "$OUT/ppl.log" 2>&1
rc_ppl=$?
echo "== ppl exit $rc_ppl ($(date +%T))"
grep -v -E "CCL_WARN|^W1" "$OUT/ppl.log" | tail -n 4

python - "$OUT/ppl.log" tests/golden/gpt-oss-120b/ref_ppl.txt <<'PYEOF'
import re, sys
now = open(sys.argv[1]).read()
ref = open(sys.argv[2]).read()
pn = re.search(r"PPL = ([0-9.]+)", now)
pr = re.search(r"PPL = ([0-9.]+)", ref)
if not pn or not pr:
    print("PPL_EXTRACT_FAIL")
    sys.exit(3)
n, r = float(pn.group(1)), float(pr.group(1))
rel = abs(n - r) / r
print(f"RES ppl ours {n:.4f} ref {r:.4f} rel {rel * 100:.3f}% (limit 1%)")
sys.exit(0 if rel <= 0.01 else 1)
PYEOF
rc_pplcmp=$?
echo "== ppl compare exit $rc_pplcmp ($(date +%T))"

if [ "$rc_pytest" -ne 0 ] || [ "$rc_ours" -ne 0 ] || [ "$rc_diff" -ne 0 ] \
   || [ "$rc_ppl" -ne 0 ] || [ "$rc_pplcmp" -ne 0 ]; then
  echo "== something failed: layer dump for step 4 ($(date +%T))"
  bash $T dump $M "The capital of France is" "$OUT/dump.txt" --n-cpu-moe 36
  echo "== dump exit $? ($(date +%T))"
  for h in attn_out-0 attn_norm-1 ffn_out-0 ffn_out-1; do
    grep -a -m 1 "$h = " "$OUT/dump.txt" || echo "MISSING_HEADER $h"
  done
fi

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp"
echo "== end $(date +%T)"
