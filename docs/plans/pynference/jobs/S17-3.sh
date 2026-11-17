#!/usr/bin/env bash
# job: S17-3
# timeout: 110
# scopo: S17 passo 3/4 dopo il fix OOM (job S17-2: ExpertStore a 24.6 GiB
#   allocati su GPU 1 con torch che segnava 6.7 GiB "free" -> i GB usable
#   sono piu bassi di D8 misurato a GPU idle, o c'e frammentazione del
#   driver):
#   1) probe di capacita: alloca chunk da 1 GiB per GPU finche va OOM,
#      cosi il report ha i byte realmente allocabili per GPU;
#   2) suite tests/gpu (moe.py/gptoss.py toccati dal fix: ladder del budget
#      in _expert_store_ladder + log per-blocco in ExpertStore(verbose));
#   3) compare_probs ours --ref (--decode fused) + diff --strict e bench/ppl
#      8x512 vs ref_ppl.txt (C8), via run_bench.py (Tokenizer->load_tokenizer
#      a runtime, bench/ non toccato). Il caricamento ora riprova con budget
#      -10% a ogni OutOfMemoryError (floor 8 GB) e stampa ogni buffer.
#   4) se un confronto non torna: dump (2.5) con --n-cpu-moe 34; attenzione:
#      llama.cpp (openai-moe.cpp:131) chiama il nodo FFN `ffn_moe_out`, non
#      `ffn_out` (per questo in S17-2 la lista header era vuota).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
T=docs/plans/pynference/tools/llama_ref.sh
export GPTOSS_GGUF=$M

python - > "$OUT/probe.log" 2>&1 <<'PYEOF'
import gc
import torch
for r in (0, 1):
    props = torch.xpu.get_device_properties(r)
    print(f"probe GPU {r}: {props.name} total {props.total_memory / 2**30:.2f} GiB", flush=True)
    keep = []
    got = 0
    try:
        while True:
            keep.append(torch.empty(1 << 30, dtype=torch.uint8, device=f"xpu:{r}"))
            got += 1
    except torch.OutOfMemoryError:
        pass
    print(f"probe GPU {r}: allocati {got} GiB prima di OOM", flush=True)
    del keep
    gc.collect()
    torch.xpu.empty_cache()
PYEOF
echo "== probe exit $? ($(date +%T))"
grep -a probe "$OUT/probe.log"

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
grep -a -E 'OOM at|store:|resident|load done' "$OUT/ours.log" | head -n 20
grep -a -v -E "CCL_WARN|^W1|store: " "$OUT/ours.log" | tail -n 8

python "$OUT/run_bench.py" bench.compare_probs diff --strict \
  tests/golden/gpt-oss-120b/ref_probs.json "$OUT/ours.json" > "$OUT/diff.log" 2>&1
rc_diff=$?
echo "== diff strict exit $rc_diff ($(date +%T))"
tail -n 15 "$OUT/diff.log"

timeout 3000 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$M" \
  -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1
rc_ppl=$?
echo "== ppl exit $rc_ppl ($(date +%T))"
grep -a -E 'OOM at|resident' "$OUT/ppl.log" | head -n 6
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

if [ "$rc_pytest" -ne 0 ] || [ "$rc_ours" -ne 0 ] || [ "$rc_diff" -ne 0 ] \
   || [ "$rc_ppl" -ne 0 ] || [ "$rc_pplcmp" -ne 0 ]; then
  echo "== something failed: layer dump for step 4, n-cpu-moe 34 ($(date +%T))"
  bash $T dump $M "The capital of France is" "$OUT/dump.txt" --n-cpu-moe 34
  rc_dump=$?
  echo "== dump exit $rc_dump ($(date +%T))"
  for h in attn_out-0 attn_norm-1 ffn_moe_out-34 ffn_moe_out-35; do
    grep -a -m 1 "$h = " "$OUT/dump.txt" || echo "MISSING_HEADER $h"
  done
  echo "-- header stems present:"
  grep -a -o -E 'common_debug_cb_eval: +[A-Za-z_]+(-[0-9]+)?' "$OUT/dump.txt" \
    | sed 's/.*: //' | sed 's/-[0-9]*$//' | sort -u | head -n 30
fi

echo "== summary: pytest=$rc_pytest ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp"
echo "== end $(date +%T)"
