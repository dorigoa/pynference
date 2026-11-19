#!/usr/bin/env bash
# job: S26-12
# timeout: 150
# scopo: (S26 passo 5) verifica della correzione della divisione TP delle
#   teste QSA: pairing GQA CONTIGUO (q-head h ↔ kv-head h//12, misurato:
#   test_s26_pairing.py + confronto TP1 di S25), non modulare → la split a
#   parità di S26-11 alimentava metà delle teste con la kv sbagliata
#   (attn_output-3 cos 0,243). Ora rank r possiede le teste contigue
#   [r*12,(r+1)*12) + la kv-head r. Verifica cpu2 (emulazione TP2 su CPU,
#   s26_nodes.py cpu2) verde: tutte le parti rank-local bit-identiche alle
#   slice TP1, SUM attn_output-3 cos 0,998893.
#   Sequenza (S21-3 + S26-11):
#   1) replay xpu (s26_nodes.py xpu) → atteso FIRST_DIVERGING: none (o solo
#      i 3 FAIL noti di layout, che a world=2 sono SKIP) + SDPA_PROBE con
#      cos ~1 (convenzione torch = grouping su GPU);
#   2) pytest -q tests/gpu (regressione dopo qwen4exp.py);
#   3) C8: compare_probs ours --ref + diff --strict + bench/ppl.py vs
#      ref_ppl.txt (limite 1%);
#   4) se C8 verde: prima baseline qwen4exp (ours_baseline.json +
#      ours_ppl.txt in $G, copia in $OUT da committare in tests/golden) e
#      scripts/regress.sh qwen4exp (supporto appena aggiunto, DECODE=graph
#      inerte: percorso eager unico);
#   5) cli chat --script con ragionamento attivo (--show-reasoning),
#      --temp 0 (carta: "chat con ragionamento");
#   6) scripts/regress.sh gemma4 (C10, l'altro modello non deve regredire).
set -uo pipefail
OUT=$PYI_JOB_OUT
echo "== start $(date +%T)"
grep -E "GPUActive|MemAvailable|MemTotal" /proc/meminfo

M=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
G=tests/golden/qwen4exp

poll() { # poll PID LOG: campiona meminfo ogni 3 s in LOG finché PID vivo
  # (morto o zombie: il figlio resta zombie finché non lo si aspetta)
  local pid=$1 log=$2 st
  while st=$(awk -F') ' 'NR==1 {print substr($2, 1, 1)}' "/proc/$pid/stat" 2>/dev/null) \
         && [ -n "$st" ] && [ "$st" != "Z" ]; do
    printf '%s ' "$(date +%T)" >> "$log"
    grep -E "MemAvailable|GPUActive" /proc/meminfo | tr '\n' ' ' >> "$log"
    echo >> "$log"
    sleep 3
  done
}

summarize() { # summarize LOG: floor MemAvailable e max GPUActive (GiB)
  if [ ! -s "$1" ]; then echo "(no samples: $1)"; return; fi
  awk '/MemAvailable/ {if (m == "" || $3 < m) m = $3}
       /GPUActive/    {if ($6 > g) g = $6}
       END {printf "floor MemAvailable %.2f GiB, max GPUActive %.2f GiB\n",
            m / 2**20, g / 2**20}' "$1"
}

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

# ---- 1) replay xpu: attesa FIRST_DIVERGING: none ----
echo "== node replay xpu ($(date +%T))"
timeout 3000 torchrun --nproc_per_node=2 tests/cpu/s26_nodes.py xpu \
    > "$OUT/nodes.log" 2>&1 &
NPID=$!
poll "$NPID" "$OUT/meminfo_nodes.log"
wait $NPID; RN=$?
echo "== nodes exit $RN ($(date +%T)) (atteso 0)"
grep -E "PROBE|FAIL|compared|FIRST_DIVERGING|SDPA_PROBE|Traceback|Error" "$OUT/nodes.log" | head -n 40
echo "OK=$(grep -cE 'OK$' "$OUT/nodes.log") \
SKIP=$(grep -cE 'SKIP' "$OUT/nodes.log")"
summarize "$OUT/meminfo_nodes.log"

# ---- 2) GPU suite (qwen4exp.py cambiato dopo S26-11) ----
python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest.log"
[ "$rc_pytest" -eq 0 ] || grep -a '^FAILED' "$OUT/pytest.log" | head -n 12

# ---- 3) C8: greedy probs vs llama.cpp + ppl ----
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$G/ref_probs.json" -m "$M" --decode graph -o "$OUT/ours.json" > "$OUT/ours.log" 2>&1
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
grep -a "load done" "$OUT/ours.log" | head -n 1
grep -a -v -E "CCL_WARN|^W1|UserWarning|stage.copy_" "$OUT/ours.log" | tail -n 5

python "$OUT/run_bench.py" bench.compare_probs diff --strict "$G/ref_probs.json" "$OUT/ours.json" \
  > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T)) (atteso 0)"
cat "$OUT/diff.log"

timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$M" \
  -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1
rc_ppl=$?
echo "== ppl exit $rc_ppl ($(date +%T)) (atteso 0)"
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

# ---- 4) baseline + regress qwen4exp (solo con C8 verde: altrimenti la baseline
#      sarebbe spazzatura; è la prima per questo modello, S26) ----
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
  # copia in $OUT: torna sul Mac col job e va committata in tests/golden (S26)
  cp "$G/ours_baseline.json" "$G/ours_ppl.txt" "$OUT/" 2>/dev/null
  if [ "$rc_boot" -eq 0 ]; then
    bash scripts/regress.sh qwen4exp > "$OUT/regress_qwen4exp.log" 2>&1
    rc_rq=$?
  fi
else
  echo "== regress qwen4exp SKIP (C8 non verde: diff=$rc_diff pplcmp=$rc_pplcmp)"
fi
echo "== regress qwen4exp exit $rc_rq ($(date +%T))"
tail -n 8 "$OUT/regress_qwen4exp.log" 2>/dev/null

# ---- 5) chat con ragionamento (2 turni: testa anche la history) ----
cat > "$OUT/chat.txt" <<'TXTEOF'
Quanto fa 2 + 2? Rispondi in una frase.
What is the capital of France? Answer in one sentence.
TXTEOF
timeout 1500 torchrun --nproc_per_node=2 -m pynference.cli chat -m "$M" \
  --script "$OUT/chat.txt" -n 192 --temp 0 --decode graph --show-reasoning \
  > "$OUT/chat_reasoning.log" 2>&1
rc_cr=$?
echo "== chat reasoning exit $rc_cr ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1|UserWarning|stage.copy_" "$OUT/chat_reasoning.log" | tail -n 25

# ---- 6) regress gemma4 (C10) ----
bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_rg=$?
echo "== regress gemma4 exit $rc_rg ($(date +%T))"
tail -n 8 "$OUT/regress_gemma4.log"

echo "== summary: nodes=$RN pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp boot=$rc_boot regressqwen4exp=$rc_rq chatreason=$rc_cr regressgemma=$rc_rg"
echo "== end $(date +%T)"
