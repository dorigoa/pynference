#!/usr/bin/env bash
# job: S17-7
# timeout: 120
# scopo: S17 passo 3/4, job di correzione autorizzato dopo la diagnosi
#   S17-6 (candidato 2: il driver xe trattiene RAM host ~1:1 con le
#   riserve torch di ogni rank; GPUActive 40.24 GiB a fine load gia' con
#   store_pinned 0, +12.7 GiB al prefill -> global_oom). Due fix nel
#   codice (moe.py, gptoss.py + test, commit aa70adf):
#   (a) anello pinned da 4 buffer slot-sized nel path di swap
#       (ExpertStore._copy_misses): mai pin di pool, ~18 MB/rank bloccati;
#   (b) cappetto d'RAM host sul budget expert: (MemTotal -
#       EXPERT_HOST_RESERVE_BYTES 24 GiB) // world - dense - kv - scratch
#       -> ~16,3 GB/rank di store (~55% expert residenti).
#   Qui: suite GPU completa (moe.py e' codice condiviso), ours pieno col
#   PYI_MEMDIAG (attesa: riga "MoE: ... budget" ~16 GB/rank, store_pinned
#   ~0.02 GiB, GPUActive ben sotto i ~58 GiB del kill e nessun global_oom
#   nel kernel log), diff --strict (C8), ppl 8x512 vs ref (<=1%), e
#   regress.sh gemma4 (C10, perche' moe.py tocca anche Gemma). Dump
#   condizionale solo se il modello carica ma un confronto numerico manca.
set -uo pipefail
echo "== start $(date +%T)"
T0=$(date +%s)
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
T=docs/plans/pynference/tools/llama_ref.sh
export GPTOSS_GGUF=$M

echo "== baseline meminfo (Strata fermo, GPU fredde):"
grep -aE 'MemTotal|MemAvailable|GPUActive' /proc/meminfo || echo "GPUACTIVE_FIELD_MISSING"

# memory sampler (our own child, killed at the end); GPUActive = RAM held by
# the xe driver for GPU processes (kernel 7.3), the counter S17-5 lacked
( for i in $(seq 1 660); do
    echo "T $(date +%T)"
    free -b | awk '/^Mem:/{print "host free "$4" avail "$7}'
    awk '/^MemAvailable/{a=$2} /^GPUActive/{g=$2} END{print "meminfo avail_kb "a" gpu_active_kb "g}' /proc/meminfo
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

# full reference run with diagnostics on: this is both the C8 candidate and
# the verification that the ring + host cap keep the host alive
PYI_MEMDIAG="$OUT/memdiag.log" timeout 3000 torchrun --nproc_per_node=2 \
  "$OUT/run_bench.py" bench.compare_probs ours \
  --ref tests/golden/gpt-oss-120b/ref_probs.json -m "$M" --decode fused \
  -o "$OUT/ours.json" > "$OUT/ours.log" 2>&1
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
grep -a -E 'MoE: .*budget|MoE: .*resident|OOM at|load done' "$OUT/ours.log" | head -n 6
grep -a -v -E "CCL_WARN|^W1|store: " "$OUT/ours.log" | tail -n 8

echo "-- kernel oom lines:"
journalctl -k --no-pager --since "@$T0" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 15 || true
dmesg 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 5 || echo "JOURNAL_UNAVAILABLE"

echo "-- memdiag lines (rank, phase, host avail, driver-held, torch, pinned):"
cat "$OUT/memdiag.log" 2>/dev/null || echo "MEMDIAG_MISSING"
echo "-- mem.log gpu_active trend:"
grep -a 'gpu_active_kb' "$OUT/mem.log" | awk 'NR==1 || NR%6==0'

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

# C10: moe.py is shared with the Gemma path, so guard it in this job too
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

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp gemma4=$rc_gemma4"
echo "== end $(date +%T)"
