#!/usr/bin/env bash
# job: S17-4
# timeout: 110
# scopo: S17 passo 3/4, terza esecuzione del confronto dopo due morti del
#   caricamento MoE: S17-2 torch.OutOfMemoryError a ~24.9 GiB tenuti, S17-3
#   SIGKILL del rank a ~25.5 GiB tenuti (layer 32/36), mentre il probe
#   S17-3 allocava 31 GiB per GPU in chunk da 1 GiB su GPU libera.
#   Nuovi nel codice: _expert_budget cappata a total-7GiB (mem_get_info) +
#   empty_cache prima dello store. Nel job:
#   1) probe2 per GPU: mem_get_info, chunk da 1 GiB fino a OOM, poi il
#      PATTERN reale (dense 1.5 GiB + buffer da 264 MB fino a OOM): se il
#      pattern rende ~25 GiB e il chunk 31, il collo di bottiglia e' il
#      pattern di many-buffer e il cappetto basta;
#   2) sampler RSS/free ogni 10 s (capire se il killer e' host o GPU);
#   3) suite tests/gpu; 4) ours+diff+ppl via run_bench.py (come S17-3);
#   5) dopo ours: righe kernel oom/journalctl; dump layer SOLO se il modello
#      carica ma un confronto numerico non torna (header ffn_moe_out gia'
#      validati in S17-3).
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

python - > "$OUT/probe.log" 2>&1 <<'PYEOF'
import gc
import torch
for r in (0, 1):
    p = torch.xpu.get_device_properties(r)
    line = f"probe GPU {r}: {p.name} total {p.total_memory / 2**30:.2f} GiB"
    try:
        free, total = torch.xpu.mem_get_info(f"xpu:{r}")
        line += f", mem_get_info free {free / 2**30:.2f} total {total / 2**30:.2f} GiB"
    except Exception as e:
        line += f", mem_get_info unavailable: {type(e).__name__}"
    print(line, flush=True)
    keep, got = [], 0
    try:
        while True:
            keep.append(torch.empty(1 << 30, dtype=torch.uint8, device=f"xpu:{r}"))
            got += 1
    except (torch.OutOfMemoryError, RuntimeError):
        pass
    print(f"probe GPU {r}: 1 GiB chunks -> allocati {got} GiB prima di OOM", flush=True)
    del keep
    gc.collect()
    torch.xpu.empty_cache()
    # real pattern: dense-like 1.5 GiB + 264 MB store buffers until OOM
    dense = torch.empty(int(1.5 * 2**30), dtype=torch.uint8, device=f"xpu:{r}")
    bufs, n = [], 0
    try:
        while True:
            bufs.append(torch.empty(264 * 1000 * 1000, dtype=torch.uint8, device=f"xpu:{r}"))
            n += 1
    except (torch.OutOfMemoryError, RuntimeError):
        pass
    gib = (n * 264 * 1000 * 1000 + dense.numel()) / 2**30
    print(f"probe GPU {r}: dense+264MB pattern -> {n} buffer, {gib:.2f} GiB tenuti prima di OOM", flush=True)
    del bufs, dense
    gc.collect()
    torch.xpu.empty_cache()
PYEOF
rc_probe=$?
echo "== probe exit $rc_probe ($(date +%T))"
grep -a probe "$OUT/probe.log" | tail -n 8

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
grep -a -E 'MoE: .*budget|OOM at|load done' "$OUT/ours.log" | head -n 6
grep -a -v -E "CCL_WARN|^W1|store: " "$OUT/ours.log" | tail -n 8

echo "-- kernel oom lines:"
journalctl -k --no-pager --since "@$T0" 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 25 || true
dmesg 2>/dev/null | grep -aiE 'oom|killed process' | tail -n 10 || echo "JOURNAL_UNAVAILABLE"

python "$OUT/run_bench.py" bench.compare_probs diff --strict \
  tests/golden/gpt-oss-120b/ref_probs.json "$OUT/ours.json" > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T))"
tail -n 15 "$OUT/diff.log"

timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$M" \
  -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1
rc_ppl=$?
echo "== ppl exit $rc_ppl ($(date +%T))"
grep -a -E 'MoE: .*budget|OOM at|load done' "$OUT/ppl.log" | head -n 6
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

echo "== summary: probe=$rc_probe pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp"
echo "== end $(date +%T)"
