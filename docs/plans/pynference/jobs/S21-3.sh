#!/usr/bin/env bash
# job: S21-3
# timeout: 150
# scopo: S21 passo 5 — verifica C8 dopo il fix lm_head (output.weight NON tied,
#   S21TIE). S21-2: C8 FAIL con PPL 759165 ≈ uniforme; causa radice trovata a
#   costo zero (lm_head = embed alimentava l'embedding di ingresso con l'hidden
#   finale) e corretta in qwen35.py; golden TP1 TRITON_INTERPRET 8/8 OK
#   (scratch/logs/S21GOLD1FIX2.log). Job di riesecuzione GPU TP2:
#   (1) suite tests/gpu (regressione dopo la modifica a qwen35.py);
#   (2) compare_probs ours --ref + diff --strict (C8, greedy 4 prompt golden);
#   (3) bench/ppl.py 8x512 GPL-3 vs ref_ppl.txt (C8 <=1%);
#   (4) se (2)+(3) verdi: prima baseline qwen35 (ours_baseline.json +
#       ours_ppl.txt in $G, copia in $OUT per persistenza nel repo) e poi
#       scripts/regress.sh qwen35-27b, che ora esiste (DECODE=fused);
#   (5) cli chat --script, ragionamento attivo (default, --show-reasoning) e
#       spento (--no-reasoning), --decode fused (use_graph=False, C3), temp 0
#       per output deterministico;
#   (6) scripts/regress.sh gemma4 (C10, l'altro modello non deve regredire).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/Qwen3.8-27B-Q8_0.gguf
G=tests/golden/qwen35-27b

# bench/ hardcodes the Gemma-only Tokenizer; rebind load_tokenizer at runtime (S17)
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

# ---- 1) GPU suite (qwen35.py changed since S21-2) ----
python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest.log"
[ "$rc_pytest" -eq 0 ] || grep -a '^FAILED' "$OUT/pytest.log" | head -n 12

# ---- 2) greedy probs vs llama-server golden (C8) ----
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$G/ref_probs.json" -m "$M" --decode fused -o "$OUT/ours.json" > "$OUT/ours.log" 2>&1
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
grep -a "load done" "$OUT/ours.log" | head -n 1
grep -a -v -E "CCL_WARN|^W1|UserWarning|stage.copy_" "$OUT/ours.log" | tail -n 5

python "$OUT/run_bench.py" bench.compare_probs diff --strict "$G/ref_probs.json" "$OUT/ours.json" \
  > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T))"
cat "$OUT/diff.log"

# ---- 3) perplexity vs golden (C8 <= 1%) ----
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$M" \
  -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1
rc_ppl=$?
echo "== ppl exit $rc_ppl ($(date +%T))"
grep -a "Final estimate" "$OUT/ppl.log"
python - "$OUT/ppl.log" "$G/ref_ppl.txt" <<'PYEOF'
import re, sys
pn = re.search(r"PPL = ([0-9.]+)", open(sys.argv[1]).read())
pr = re.search(r"PPL = ([0-9.]+)", open(sys.argv[2]).read())
if not pn or not pr:
    print("PPL_EXTRACT_FAIL")
    sys.exit(3)
n, r = float(pn.group(1)), float(pr.group(1))
print(f"RES ppl ours {n:.4f} ref {r:.4f} rel {abs(n - r) / r * 100:.3f}% (limit 1%)")
sys.exit(0 if abs(n - r) / r <= 0.01 else 1)
PYEOF
rc_pplcmp=$?
echo "== ppl compare exit $rc_pplcmp ($(date +%T))"

# ---- 4) baseline + regress qwen35 (solo con C8 verde: altrimenti la baseline
#      sarebbe spazzatura; è la prima per questo modello, S21) ----
rc_boot=0 rc_rq=3
if [ "$rc_diff" -eq 0 ] && [ "$rc_pplcmp" -eq 0 ]; then
  cp "$OUT/ours.json" "$G/ours_baseline.json"
  python - "$OUT/ppl.log" > "$G/ours_ppl.txt" <<'PYEOF'
import re, sys
m = re.search(r"PPL = ([0-9.]+)", open(sys.argv[1]).read())
if not m:
    sys.exit(3)
print(f"PPL = {m.group(1)}")
PYEOF
  rc_boot=$?
  [ "$rc_boot" -eq 0 ] || echo "== baseline bootstrap FAILED rc=$rc_boot"
  # copia in $OUT: torna sul Mac col job e va committata in tests/golden (S21)
  cp "$G/ours_baseline.json" "$G/ours_ppl.txt" "$OUT/" 2>/dev/null
  if [ "$rc_boot" -eq 0 ]; then
    bash scripts/regress.sh qwen35-27b > "$OUT/regress_qwen35.log" 2>&1
    rc_rq=$?
  fi
else
  echo "== regress qwen35 SKIP (C8 non verde: diff=$rc_diff pplcmp=$rc_pplcmp)"
fi
echo "== regress qwen35 exit $rc_rq ($(date +%T))"
tail -n 8 "$OUT/regress_qwen35.log" 2>/dev/null

# ---- 5) chat con e senza ragionamento (2 turni: testa anche la history) ----
cat > "$OUT/chat.txt" <<'TXTEOF'
Quanto fa 2 + 2? Rispondi in una frase.
What is the capital of France? Answer in one sentence.
TXTEOF
timeout 1500 torchrun --nproc_per_node=2 -m pynference.cli chat -m "$M" \
  --script "$OUT/chat.txt" -n 192 --temp 0 --decode fused --show-reasoning \
  > "$OUT/chat_reasoning.log" 2>&1
rc_cr=$?
echo "== chat reasoning exit $rc_cr ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1|UserWarning|stage.copy_" "$OUT/chat_reasoning.log" | tail -n 25

timeout 1500 torchrun --nproc_per_node=2 -m pynference.cli chat -m "$M" \
  --script "$OUT/chat.txt" -n 96 --temp 0 --decode fused --no-reasoning \
  > "$OUT/chat_noreasoning.log" 2>&1
rc_cn=$?
echo "== chat no-reasoning exit $rc_cn ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1|UserWarning|stage.copy_" "$OUT/chat_noreasoning.log" | tail -n 20

# ---- 6) regress gemma4 (C10) ----
bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_rg=$?
echo "== regress gemma4 exit $rc_rg ($(date +%T))"
tail -n 8 "$OUT/regress_gemma4.log"

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp boot=$rc_boot regressqwen=$rc_rq chatreason=$rc_cr chatnoreason=$rc_cn regressgemma=$rc_rg"
echo "== end $(date +%T)"
