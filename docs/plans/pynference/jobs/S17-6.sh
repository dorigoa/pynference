#!/usr/bin/env bash
# job: S17-6
# timeout: 60
# scopo: diagnosi host-OOM (decisione utente 2026-11-17, dopo S17-5). Il
#   kernel log di S17-5 dice gpu_active ~58 GiB: RAM tenuta dal driver xe,
#   non pagine mmap. Candidati: (1) pin HostPool a miss (tutto il pool al
#   primo miss), (2) VRAM oltre il limite scaricata in RAM dal driver.
#   Strumenti: righe PYI_MEMDIAG (gptoss.py) su un solo ours con prompt
#   breve (1 prompt, quello piu corto di ref_probs.json): load, prefill e
#   ogni 8 token di decode, con MemAvailable, GPUActive, torch
#   allocated/reserved, store_pinned; sampler mem.log con GPUActive ogni
#   10 s; kernel log dopo ours. Attesa discriminante: GPUActive che cresce
#   con store_pinned -> candidato 1; gia alto a fine load con store_pinned
#   0, o cresce con torch_resv -> candidato 2. In piu: rigiro i test GPU
#   dello store aggiornati dopo S17-5 (test_moe_store, test_moe_layer).
#   Niente suite completa, niente ppl: e' un job corto di sola diagnosi.
set -uo pipefail
echo "== start $(date +%T)"
T0=$(date +%s)
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
export GPTOSS_GGUF=$M

echo "== baseline meminfo (Strata fermo, GPU fredde):"
grep -aE 'MemAvailable|GPUActive' /proc/meminfo || echo "GPUACTIVE_FIELD_MISSING"

# memory sampler (our own child, killed at the end); GPUActive = RAM held by
# the xe driver for GPU processes (kernel 7.3), the counter S17-5 lacked
( for i in $(seq 1 330); do
    echo "T $(date +%T)"
    free -b | awk '/^Mem:/{print "host free "$4" avail "$7}'
    awk '/^MemAvailable/{a=$2} /^GPUActive/{g=$2} END{print "meminfo avail_kb "a" gpu_active_kb "g}' /proc/meminfo
    ps -eo pid,comm,rss --sort=-rss | awk 'NR>1 && /python|torchrun/ {print "rss", $1, $2, $3; if (NR>5) exit}'
    sleep 10
  done ) > "$OUT/mem.log" 2>&1 &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

python -m pytest -q tests/gpu/test_moe_store.py tests/gpu/test_moe_layer.py \
  > "$OUT/pytest.log" 2>&1
rc_pytest=$?
echo "== pytest store/moe exit $rc_pytest ($(date +%T))"
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

# one short prompt: the entry with the fewest tokens in the golden ref file
python - tests/golden/gpt-oss-120b/ref_probs.json "$OUT/one.json" <<'PYEOF'
import json, sys
ref = json.load(open(sys.argv[1]))
one = min(ref, key=lambda e: len(e["prompt"]))
json.dump([one], open(sys.argv[2], "w"))
print(f"prompt {one['name']}: {len(one['prompt'])} tokens, {len(one['steps'])} steps in ref")
PYEOF

PYI_MEMDIAG="$OUT/memdiag.log" timeout 900 torchrun --nproc_per_node=2 \
  "$OUT/run_bench.py" bench.compare_probs ours --ref "$OUT/one.json" \
  -m "$M" --decode fused -o "$OUT/ours.json" > "$OUT/ours.log" 2>&1
rc_ours=$?
echo "== ours exit $rc_ours ($(date +%T))"
grep -a -E 'MoE: .*budget|MoE: .*resident|OOM at|load done|prompt' "$OUT/ours.log" | head -n 6
grep -a -v -E "CCL_WARN|^W1|store: " "$OUT/ours.log" | tail -n 8

echo "-- kernel oom lines:"
journalctl -k --no-pager --since "@$T0" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 15 || true
dmesg 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 5 || echo "JOURNAL_UNAVAILABLE"

echo "-- memdiag lines (rank, phase, host avail, driver-held, torch, pinned):"
cat "$OUT/memdiag.log" 2>/dev/null || echo "MEMDIAG_MISSING"
echo "-- mem.log gpu_active trend:"
grep -a 'gpu_active_kb' "$OUT/mem.log" | awk 'NR==1 || NR%6==0'

# informational only: does the single prompt already match the reference?
rc_diff=-
if [ "$rc_ours" -eq 0 ]; then
  python "$OUT/run_bench.py" bench.compare_probs diff \
    tests/golden/gpt-oss-120b/ref_probs.json "$OUT/ours.json" > "$OUT/diff.log" 2>&1
  rc_diff=$?
  echo "== diff (non-strict, informativo) exit $rc_diff ($(date +%T))"
  tail -n 10 "$OUT/diff.log"
fi

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff"
echo "== end $(date +%T)"
