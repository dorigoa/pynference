#!/usr/bin/env bash
# job: S17-5
# timeout: 110
# scopo: S17 passo 3/4, quarta esecuzione del confronto. S17-4 ha
#   identificato il killer: OOM globale dell'HOST (kernel log + mem.log:
#   avail 20->0 GB, free <1 GB al kill), non la GPU (probe2: 31,50 GiB
#   retesi col pattern reale). Cause: ~14,3 GB di HostPool pinned a load
#   (eager, non-reclaimable) + page cache del GGUF + heap contro 62 GB di
#   RAM; e il tetto GPU effettivo e' ~24,3 GB di store (OOM torch in ppl),
#   non 31. Fix nel codice (moe.py, gptoss.py, test): HostPool lazy (niente
#   pinned a load, righe pinned solo al miss reale) e
#   EXPERT_GPU_OVERHEAD_BYTES 7->11 GiB (store ~20 GB, ~2 GiB sotto la zona
#   di morte GPU; la ladder resta sotto il cappetto). Attesa: "MoE: ~76
#   experts resident, pinned 0.00 GB" a fine load; poi ours/diff/ppl veri.
#   Resta: sampler host ogni 10 s, kernel log dopo ours, dump condizionale.
set -uo pipefail
echo "== start $(date +%T)"
T0=$(date +%s)
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
T=docs/plans/pynference/tools/llama_ref.sh
export GPTOSS_GGUF=$M

# memory sampler (our own child, killed at the end)
( for i in $(seq 1 660); do
    echo "T $(date +%T)"
    free -b | awk '/^Mem:/{print "host free "$4" avail "$7}'
    ps -eo pid,comm,rss --sort=-rss | awk 'NR>1 && /python|torchrun/ {print "rss", $1, $2, $3; if (NR>5) exit}'
    sleep 10
  done ) > "$OUT/mem.log" 2>&1 &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest.log"
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
grep -a -E 'MoE: .*budget|MoE: .*resident|OOM at|load done' "$OUT/ours.log" | head -n 6
grep -a -v -E "CCL_WARN|^W1|store: " "$OUT/ours.log" | tail -n 8

echo "-- kernel oom lines:"
journalctl -k --no-pager --since "@$T0" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 15 || true
dmesg 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 5 || echo "JOURNAL_UNAVAILABLE"

python "$OUT/run_bench.py" bench.compare_probs diff --strict \
  tests/golden/gpt-oss-120b/ref_probs.json "$OUT/ours.json" > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T))"
tail -n 15 "$OUT/diff.log"

timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$M" \
  -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1
rc_ppl=$?
echo "== ppl exit $rc_ppl ($(date +%T))"
grep -a -E 'MoE: .*budget|MoE: .*resident|OOM at|load done' "$OUT/ppl.log" | head -n 6
grep -a -v -E "CCL_WARN|^W1" "$OUT/ppl.log" | tail -n 4

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

# dump only when the model loaded but a numeric comparison misses (step 4)
if [ "$rc_ours" -eq 0 ] && { [ "$rc_diff" -ne 0 ] || [ "$rc_pplcmp" -ne 0 ]; }; then
  echo "== comparison off: layer dump, n-cpu-moe 34 ($(date +%T))"
  bash $T dump $M "The capital of France is" "$OUT/dump.txt" --n-cpu-moe 34
  echo "== dump exit $? ($(date +%T))"
fi

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp"
echo "== end $(date +%T)"
