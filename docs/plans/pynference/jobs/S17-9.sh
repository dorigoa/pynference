#!/usr/bin/env bash
# job: S17-9
# timeout: 60
# scopo: S17 passo 4/5, terzo job sul C8 (S17-7 scala del decode, S17-8 2/4
#   greedy su quasi-pareggi): ultimo prima della regola dei 3.
#   Correzioni dopo S17-8, trovate offline:
#   (1) MoE.decode rimetteva gli esperti residenti in ogni blocco di miss:
#       con piu' miss degli slot di swap (2 per layer) un residente veniva
#       sommato 2-3 volte, circa una volta per token. Corretto + test CPU e GPU.
#   (2) torch.backends.mkldnn.deterministic in GPTOss: in S17-8 due esecuzioni
#       dello stesso codice differivano (dp fino a 6e-3 gia' nel prefill), e
#       la regressione gpt-oss confrontava quindi due risultati diversi.
#   (3) q/k/v/o del prefill in fp32 (gptoss.dense_prefill) invece del
#       percorso bf16 di QLinear.
#   Riferimento su CPU (teacher forcing sui token di llama.cpp,
#   scratch/s17_cpuref.py): fp32 esatto 4/4 prompt identici a llama.cpp;
#   la nostra aritmetica con prefill bf16 3/4 (chat-en gira al passo 3);
#   con q/k/v/o del prefill in fp32 4/4 (chat-en We 0,42 contro User 0,25).
#   Qui: suite GPU; ours due volte (determinismo, confronto bit a bit);
#   diff --strict (C8); ppl vs ref; regress gpt-oss (baseline da questa
#   esecuzione); chat 2 turni; regress gemma4 (C10); per diagnosi un terzo
#   ours con PYI_NONDETERMINISTIC=1.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
G=tests/golden/gpt-oss-120b
export GPTOSS_GGUF=$M

# bench/ hardcodes the Gemma-only Tokenizer; rebind load_tokenizer at runtime
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

ours() {   # $1 = output json, rest = env assignments
  local out=$1; shift
  env "$@" timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
    --ref "$G/ref_probs.json" -m "$M" --decode fused -o "$out" > "${out%.json}.log" 2>&1
}

python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest.log"
[ "$rc_pytest" -eq 0 ] || grep -a '^FAILED' "$OUT/pytest.log" | head -n 12

ours "$OUT/ours.json"
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
grep -a -E 'local experts resident|load done' "$OUT/ours.log" | head -n 2
grep -a -v -E "CCL_WARN|^W1|store: |UserWarning|stage.copy_" "$OUT/ours.log" | tail -n 5

python "$OUT/run_bench.py" bench.compare_probs diff --strict \
  "$G/ref_probs.json" "$OUT/ours.json" > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T))"
cat "$OUT/diff.log"

ours "$OUT/ours2.json"
echo "== ours (2nd run) exit $? ($(date +%T))"
ours "$OUT/ours_nd.json" PYI_NONDETERMINISTIC=1
echo "== ours (PYI_NONDETERMINISTIC=1) exit $? ($(date +%T))"
python - "$OUT/ours.json" "$OUT/ours2.json" "$OUT/ours_nd.json" <<'PYEOF' > "$OUT/determinism.log" 2>&1
import json, sys
def load(p):
    try:
        return {e["name"]: e for e in json.load(open(p))}
    except Exception as ex:  # a run that failed
        return ex
a = load(sys.argv[1])
for label, path in (("2nd run", sys.argv[2]), ("PYI_NONDETERMINISTIC=1", sys.argv[3])):
    b = load(path)
    if isinstance(a, Exception) or isinstance(b, Exception):
        print(f"{label}: cannot compare ({a if isinstance(a, Exception) else b})")
        continue
    same_all, worst = True, 0.0
    for n in a:
        sa, sb = a[n]["steps"], b[n]["steps"]
        same = [x["id"] for x in sa] == [y["id"] for y in sb]
        dp = max((abs(x["p"] - y["p"]) for s, t in zip(sa, sb) for x, y in zip(s["top"], t["top"])
                  if x["id"] == y["id"]), default=0.0)
        bit = sa == sb
        same_all &= bit
        worst = max(worst, dp)
        print(f"{label}: {n:8s} greedy {'same' if same else 'DIFFERENT'}, bitwise {'yes' if bit else 'no'}, max |dp| {dp:.2e}")
    print(f"{label}: {'IDENTICAL' if same_all else 'not identical'} (worst |dp| {worst:.2e})")
PYEOF
cat "$OUT/determinism.log"

timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$M" \
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

# step 5: the gpt-oss golden baseline comes from this run (committed after the job)
if [ "$rc_ours" -eq 0 ] && [ "$rc_ppl" -eq 0 ]; then
  cp "$OUT/ours.json" "$G/ours_baseline.json"
  cp "$OUT/ppl.log" "$G/ours_ppl.txt"
  echo "== baseline ours/ppl copied into $G"
fi
bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1
rc_gptoss=$?
echo "== regress gpt-oss exit $rc_gptoss ($(date +%T))"
grep -a -v -E "UserWarning|stage.copy_" "$OUT/regress_gptoss.log" | tail -n 12

cat > "$OUT/script.txt" <<'TXT'
Qual è la capitale d'Italia? Rispondi in una sola frase.
E la capitale della Francia? Rispondi in inglese, in una sola frase.
TXT
timeout 1200 torchrun --nproc_per_node=2 -m pynference.cli chat -m "$M" \
  --script "$OUT/script.txt" --decode fused --temp 0 -n 160 --ctx 4096 \
  > "$OUT/chat.out" 2> "$OUT/chat.err"
rc_chat=$?
echo "== chat exit $rc_chat ($(date +%T))"
grep -a -v -E "store: |CCL_WARN|^W1" "$OUT/chat.out" | tail -n 8
[ "$rc_chat" -eq 0 ] || grep -a -v -E "CCL_WARN|^W1" "$OUT/chat.err" | tail -n 12

bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_gemma4=$?
echo "== regress gemma4 exit $rc_gemma4 ($(date +%T))"
tail -n 5 "$OUT/regress_gemma4.log"

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp regress_gptoss=$rc_gptoss chat=$rc_chat gemma4=$rc_gemma4"
echo "== end $(date +%T)"
