#!/usr/bin/env bash
# job: S27-4
# timeout: 45
# scopo: (S27 passo 2, dopo S27-3) gate C8 reale per i pesi densi nativi.
#   S27-3 ha visto regress.sh qwen4exp FAIL solo sul confronto con la
#   baseline nostra di S26 (top-10 id diversi ai passi 0-3, max |dp| 2e-2,
#   ppl rel 0,03%): deriva attesa perche' il decode GEMV nativo non e'
#   bit-identico al GEMM f32 dequant. regress.sh NON confronta llama.cpp
#   (usa ref_probs.json solo per gli id dei prompt): il gate C8 e'
#   compare_probs diff --strict. Regola S17/S18 (gia' usata in S26-12):
#   prima diff --strict + ppl vs ref_ppl.txt (<=1%), POI rigenerare la
#   baseline (ours_baseline.json + ours_ppl.txt) e ri-eseguire
#   scripts/regress.sh qwen4exp, che a quel punto deve essere PASS.
#   Copia della nuova baseline in $OUT per committarla in tests/golden.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
G=tests/golden/qwen4exp

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

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

# ---- 1) C8 greedy: ours vs llama.cpp (ref_probs.json) + diff --strict ----
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$G/ref_probs.json" -m "$M" --decode graph -o "$OUT/ours.json" > "$OUT/ours.log" 2>&1
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
clean "$OUT/ours.log" | tail -n 5

python "$OUT/run_bench.py" bench.compare_probs diff --strict "$G/ref_probs.json" "$OUT/ours.json" \
  > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T)) (atteso 0)"
cat "$OUT/diff.log"

# ---- 2) ppl vs llama.cpp (C8: scarto <= 1%) ----
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

# ---- 3) se C8 verde: rigenera la baseline (regola S17) e regress ----
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
  [ "$rc_boot" -eq 0 ] || echo "== baseline regen FAILED rc=$rc_boot"
  # copia in $OUT: torna sul Mac col job e va committata in tests/golden
  cp "$G/ours_baseline.json" "$G/ours_ppl.txt" "$OUT/" 2>/dev/null
  if [ "$rc_boot" -eq 0 ]; then
    bash scripts/regress.sh qwen4exp > "$OUT/regress_qwen4exp.log" 2>&1
    rc_rq=$?
  fi
else
  echo "== regress qwen4exp SKIP (C8 non verde: diff=$rc_diff pplcmp=$rc_pplcmp)"
fi
echo "== regress qwen4exp exit $rc_rq ($(date +%T))"
clean "$OUT/regress_qwen4exp.log" | tail -n 8

echo "== summary: ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp boot=$rc_boot regressqwen4exp=$rc_rq"
echo "== end $(date +%T)"
