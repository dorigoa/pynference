#!/usr/bin/env bash
# job: S17-8
# timeout: 60
# scopo: S17 passo 4 (verifica C8) + passo 5 (regress gpt-oss + chat 2 turni).
#   Unica correzione dall'ultimo job: in `GPTOss._attention_decode` mancava la
#   scala 1/sqrt(head_dim) sul q — il kernel `attn_decode` calcola i punteggi
#   a scala 1.0 per convenzione (gemma4), il prefill passava self._scale ma
#   il decode no -> punteggio 8x troppo appuntito dal primo token di decode
#   (KL 0,21-0,78 su tutti i prompt, greedy 0/4, mentre il ppl tutto-prefill
#   era a 0,825%). Nuovo test CPU di guardia (rosso senza fix, verde con).
#   Qui: suite GPU completa, ours + `diff --strict` (C8), ppl 8x512 vs ref,
#   `regress.sh gpt-oss-120b` (al primo giro la baseline golden viene
#   materializzata da questa esecuzione e committata dopo il job), `cli chat
#   --script` di 2 turni (--decode fused, il grafo e' S18), `regress.sh
#   gemma4` (C10). Dump condizionale solo se un confronto numerico manca.
set -uo pipefail
echo "== start $(date +%T)"
T0=$(date +%s)
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
T=docs/plans/pynference/tools/llama_ref.sh
G=tests/golden/gpt-oss-120b
export GPTOSS_GGUF=$M

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

python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest.log"
if [ "$rc_pytest" -ne 0 ]; then
  grep -a '^FAILED' "$OUT/pytest.log" | head -n 12
fi

timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$G/ref_probs.json" -m "$M" --decode fused -o "$OUT/ours.json" > "$OUT/ours.log" 2>&1
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
grep -a -E 'MoE: .*resident|load done' "$OUT/ours.log" | head -n 3
grep -a -v -E "CCL_WARN|^W1|store: " "$OUT/ours.log" | tail -n 8

python "$OUT/run_bench.py" bench.compare_probs diff --strict \
  "$G/ref_probs.json" "$OUT/ours.json" > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T))"
tail -n 15 "$OUT/diff.log"

timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$M" \
  -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1
rc_ppl=$?
echo "== ppl exit $rc_ppl ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1" "$OUT/ppl.log" | tail -n 4

python - "$OUT/ppl.log" "$G/ref_ppl.txt" <<'PYEOF'
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

# passo 5: the golden baseline for gpt-oss does not exist yet; materialize it
# from this very run (first execution) so the regress branch runs end-to-end.
# The copies come back in $OUT too: after the job they get committed to $G.
if [ "$rc_ours" -eq 0 ] && [ ! -f "$G/ours_baseline.json" ]; then
  cp "$OUT/ours.json" "$G/ours_baseline.json"
  echo "== baseline ours copied into $G ($(date +%T))"
fi
if [ "$rc_ppl" -eq 0 ] && [ ! -f "$G/ours_ppl.txt" ]; then
  cp "$OUT/ppl.log" "$G/ours_ppl.txt"
  echo "== baseline ppl copied into $G ($(date +%T))"
fi

bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1
rc_gptoss=$?
echo "== regress gpt-oss exit $rc_gptoss ($(date +%T))"
tail -n 8 "$OUT/regress_gptoss.log"

# passo 5: chat with a 2-turn script (fused: the graph decode is S18)
cat > "$OUT/script.txt" <<'TXT'
Qual è la capitale d'Italia? Rispondi in una sola frase.
E la capitale della Francia? Rispondi in inglese, in una sola frase.
TXT
timeout 1200 torchrun --nproc_per_node=2 -m pynference.cli chat -m "$M" \
  --script "$OUT/script.txt" --decode fused --temp 0 -n 160 --ctx 4096 \
  > "$OUT/chat.out" 2> "$OUT/chat.err"
rc_chat=$?
echo "== chat exit $rc_chat ($(date +%T))"
grep -a "\[stats\]" "$OUT/chat.out" || true
tail -n 25 "$OUT/chat.out"
if [ "$rc_chat" -ne 0 ]; then
  grep -a -v -E "CCL_WARN|^W1" "$OUT/chat.err" | tail -n 12
fi

# C10 guard, as on the card: the full matrix reruns the Gemma path too
bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_gemma4=$?
echo "== regress gemma4 exit $rc_gemma4 ($(date +%T))"
tail -n 5 "$OUT/regress_gemma4.log"

# dump only when the model loaded but a numeric comparison misses (step 4)
if [ "$rc_ours" -eq 0 ] && { [ "$rc_diff" -ne 0 ] || [ "$rc_pplcmp" -ne 0 ]; }; then
  echo "== comparison off: layer dump, n-cpu-moe 34 ($(date +%T))"
  bash $T dump $M "The capital of France is" "$OUT/dump.txt" --n-cpu-moe 34
  echo "== dump exit $? ($(date +%T))"
fi

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp regress_gptoss=$rc_gptoss chat=$rc_chat gemma4=$rc_gemma4"
echo "== end $(date +%T)"
