#!/usr/bin/env bash
# job: S17-2
# timeout: 110
# scopo: S17 passo 3 (ripetizione di S17-1, bloccato dal tokenizer gpt2):
#   1) suite tests/gpu per prima, a GPU fredda (lezione S16-3) — ora anche
#      test_gptoss_attn0: il worker usa PYI_GPTOSS_SKIP_MOE (niente
#      ExpertStore, fix della regressione OOM di S17-1);
#   2) compare_probs ours --ref senza grafo (--decode fused) + diff --strict
#      (C8), e bench/ppl.py 8x512 contro ref_ppl.txt (C8 <= 1%), entrambi via
#      run_bench.py che ricarica Tokenizer -> load_tokenizer (bench/ non e'
#      modificabile in S17: il simbolo del modulo viene sostituito a runtime,
#      nessun file di bench/ toccato);
#   3) se un confronto o la suite non torna: dump dei layer (2.5) con
#      --n-cpu-moe 34 (in S17-1 con 36 gli header ffn_out non comparivano),
#      e lista degli header ffn_out effettivamente presenti.
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

# bench/ scripts hardcode the Gemma-only Tokenizer class; load_tokenizer is the
# GGUF-aware factory (pynference/tokenizer.py, used by cli/server). bench/ is
# out of S17's file scope, so rebind the module symbol at runtime instead.
cat > "$OUT/run_bench.py" <<'PYEOF'
"""Run a bench/ script with load_tokenizer as its tokenizer factory (S17)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from pynference.tokenizer import load_tokenizer  # noqa: E402

module, argv = sys.argv[1], sys.argv[2:]
mod = __import__(module, fromlist=["main"])
mod.Tokenizer = load_tokenizer
sys.argv = [module] + argv
mod.main()
PYEOF

timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref tests/golden/gpt-oss-120b/ref_probs.json -m "$M" --decode fused \
  -o "$OUT/ours.json" > "$OUT/ours.log" 2>&1
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
grep -v -E "CCL_WARN|^W1" "$OUT/ours.log" | tail -n 10

python "$OUT/run_bench.py" bench.compare_probs diff --strict \
  tests/golden/gpt-oss-120b/ref_probs.json "$OUT/ours.json" > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T))"
tail -n 15 "$OUT/diff.log"

timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$M" \
  -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1
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
  echo "== something failed: layer dump for step 4, n-cpu-moe 34 ($(date +%T))"
  bash $T dump $M "The capital of France is" "$OUT/dump.txt" --n-cpu-moe 34
  rc_dump=$?
  echo "== dump exit $rc_dump ($(date +%T))"
  for h in attn_out-0 attn_norm-1; do
    grep -a -m 1 "$h = " "$OUT/dump.txt" || echo "MISSING_HEADER $h"
  done
  echo "-- ffn headers present:"
  grep -a -o -E 'ffn_out-[0-9]+' "$OUT/dump.txt" | sort -u | tr '\n' ' '
  echo ""
  if ! grep -q . <<<"$(grep -a -o -E 'ffn_out-[0-9]+' "$OUT/dump.txt" | sort -u)"; then
    echo "== no ffn header: retry dump with n-cpu-moe 30 ($(date +%T))"
    bash $T dump $M "The capital of France is" "$OUT/dump30.txt" --n-cpu-moe 30
    echo "== dump30 exit $? ($(date +%T))"
    grep -a -o -E 'ffn_out-[0-9]+' "$OUT/dump30.txt" | sort -u | tr '\n' ' '
    echo ""
  fi
fi

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp"
echo "== end $(date +%T)"
