#!/usr/bin/env bash
# job: S21-1
# timeout: 90
# scopo: S21 passo 4 — Qwen3.8-27B contro llama.cpp.
#   (1) suite tests/gpu completa: deltanet.py corretto in S21 (due bug CPU-only
#       nel ramo history e nel blocco token finale), i test GPU dei kernel
#       DeltaNet (S20) vanno rigirati (nota report S21, passo 2-3);
#   (2) dump llama-eval-callback (llama_ref.sh) su un prompt breve, e gli
#       stessi tensori nostri sullo stesso prompt (ours_dump.py: monkeypatch
#       di _layer_recurrent/_layer_attn e _ffn, rank 0, tensori a piena
#       larghezza quindi comparabili): attn_norm/attn_residual/attn_post_norm/
#       ffn_out/post_ffn per layer 0 (DeltaNet) e 3 (attention), via la riga
#       "sum =" del dump (cmp_dump.py, diagnostico, limite 2%);
#   (3) compare_probs ours --ref (ids dal golden) + diff --strict (C8);
#   (4) bench/ppl.py 8x512 GPL-3 vs ref_ppl.txt (C8 <=1%).
#   Chat, regress qwen35-27b, regress gemma4: passo 5, prossimo job.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/gguf/Qwen3.8-27B-Q8_0.gguf
G=tests/golden/qwen35-27b
T=docs/plans/pynference/tools/llama_ref.sh
PROMPT="The capital of France is Paris."

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

# ---- 1) GPU suite (deltanet.py changed since S20) ----
python -m pytest -q tests/gpu > "$OUT/pytest.log" 2>&1
rc_pytest=$?
echo "== pytest gpu exit $rc_pytest ($(date +%T))"
tail -n 3 "$OUT/pytest.log"
[ "$rc_pytest" -eq 0 ] || grep -a '^FAILED' "$OUT/pytest.log" | head -n 12

# ---- 2) llama.cpp reference dump of one short prompt ----
bash "$T" dump "$M" "$PROMPT" "$OUT/dump.txt"
rc_dump=$?
echo "== llama dump exit $rc_dump ($(date +%T))"
[ "$rc_dump" -eq 0 ] && grep -a "number of input tokens" "$OUT/dump.txt" | head -n 1

# ---- 2b) our intermediates on the same prompt (same token ids if parseable) ----
cat > "$OUT/ours_dump.py" <<'PYEOF'
"""Capture Qwen35 prefill intermediates for the llama.cpp dump prompt (S21-4).

Wraps _layer_recurrent/_layer_attn and _ffn to save, for layers 0 and 3, the
residual-stream tensors llama-eval-callback names attn_norm, attn_residual,
attn_post_norm, ffn_out, post_ffn. The wrapper recomputes the two norms with
the same rms_norm the model uses; the block deltas are the exact model
outputs. All are full-width [T, n_embd] on every rank (ssm_out/attn_output/
ffn_down are all-reduced), so their sums match the reference dump 1:1.
Only rank 0 keeps copies and writes the JSON.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from pynference.gguf import GGUFFile  # noqa: E402
from pynference.models import load_model  # noqa: E402
from pynference.models.gptoss import rms_norm  # noqa: E402
from pynference.models.qwen35 import Qwen35  # noqa: E402
from pynference.parallel import TP  # noqa: E402
from pynference.tokenizer import load_tokenizer  # noqa: E402

model_path, prompt, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
dump_txt = sys.argv[4] if len(sys.argv) > 4 else ""
WANT = {0, 3}


def ref_ids(path):
    """Input token ids printed by llama-eval-callback, or None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    for i, ln in enumerate(lines):
        m = re.search(r"number of input tokens = (\d+)", ln)
        if not m:
            continue
        n, ids = int(m.group(1)), []
        for ln2 in lines[i + 1:]:
            m2 = re.match(r"^\s*[\d.]+\s+I\s+(\d+)\s*$", ln2)
            if m2:
                ids.append(int(m2.group(1)))
            elif ids:
                break
            if len(ids) == n:
                return ids
    return None


g = GGUFFile(model_path)
tok = load_tokenizer(g)
ids = ref_ids(dump_txt) if dump_txt else None
own = tok.encode(prompt)
if ids is None:
    print(f"ours_dump: no ids parsed from dump, using our tokenizer: {own}", flush=True)
    ids = own
else:
    print(f"ours_dump: llama ids {ids}\nours_dump: our ids  {own}", flush=True)

orig_rec = Qwen35._layer_recurrent
orig_attn = Qwen35._layer_attn
orig_ffn = Qwen35._ffn
cap = {}


def _capture_norms(self, il, x):
    L, eps = self.layers[il], self.cfg.eps
    cap[f"attn_norm-{il}"] = rms_norm(x, L.attn_norm, eps)


def rec(self, il, x):
    blk = orig_rec(self, il, x)
    if self.tp.is_main and il in WANT:
        _capture_norms(self, il, x)
        cap[f"attn_residual-{il}"] = x + blk
    return blk


def attn(self, il, x, pos, pos0):
    blk = orig_attn(self, il, x, pos, pos0)
    if self.tp.is_main and il in WANT:
        _capture_norms(self, il, x)
        cap[f"attn_residual-{il}"] = x + blk
    return blk


def ffn(self, il, x):
    d = orig_ffn(self, il, x)
    if self.tp.is_main and il in WANT:
        L, eps = self.layers[il], self.cfg.eps
        cap[f"attn_post_norm-{il}"] = rms_norm(x, L.post_attn_norm, eps)
        cap[f"ffn_out-{il}"] = d
        cap[f"post_ffn-{il}"] = x + d
    return d


Qwen35._layer_recurrent = rec
Qwen35._layer_attn = attn
Qwen35._ffn = ffn

tp = TP()
model = load_model(g, tp, max_ctx=1024)
model.forward(ids)
if tp.is_main:
    res = {}
    for name, t in sorted(cap.items()):
        d = t.double()
        res[name] = {"shape": list(t.shape), "sum": d.sum().item(),
                     "mean": d.mean().item(), "maxabs": d.abs().max().item()}
        print(f"ours {name}: shape {list(t.shape)} sum {res[name]['sum']:+.4f}", flush=True)
    json.dump(res, open(out_json, "w"), indent=1)
tp.close()
PYEOF
torchrun --nproc_per_node=2 "$OUT/ours_dump.py" "$M" "$PROMPT" "$OUT/ours_layers.json" "$OUT/dump.txt" \
  > "$OUT/ours_dump.log" 2>&1
rc_odump=$?
echo "== ours dump exit $rc_odump ($(date +%T))"
grep -a -v -E "CCL_WARN|^W1|UserWarning|stage.copy_" "$OUT/ours_dump.log" | tail -n 6

# ---- 2c) compare the two dumps (diagnostic: relative diff of node sums) ----
cat > "$OUT/cmp_dump.py" <<'PYEOF'
"""Compare llama-eval-callback node sums with our captured sums (S21-4)."""
import json
import re
import sys

WANT = [f"{stem}-{il}" for il in (0, 3)
        for stem in ("attn_norm", "attn_residual", "attn_post_norm", "ffn_out", "post_ffn")]
REL = 0.02   # diagnostic bound: fp-order noise is <1e-3, a layer bug is >>2%


def parse_dump(path):
    sums, cur, shape = {}, None, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.search(r"common_debug_cb_eval:\s+(\S+) = ", ln)
            if m:
                cur = m.group(1)
                s = re.search(r"= \{([0-9, ]+)\}\s*$", ln)
                shape = s.group(1) if s else "?"
                continue
            m = re.search(r"sum = ([-+0-9.eE]+)", ln)
            if m and cur:
                sums.setdefault(cur, (float(m.group(1)), shape))
                cur = None
    return sums


dumps = parse_dump(sys.argv[1])
ours = json.load(open(sys.argv[2]))
rc = 0
for name in WANT:
    if name not in dumps:
        print(f"{name}: MISSING in llama dump")
        rc = 1
        continue
    if name not in ours:
        print(f"{name}: MISSING in our capture")
        rc = 1
        continue
    r, rshape = dumps[name]
    o = ours[name]["sum"]
    rel = abs(o - r) / max(abs(r), 1e-6)
    ok = rel <= REL
    rc |= 0 if ok else 1
    print(f"{name}: ours {o:+.4f} shape {ours[name]['shape']} | ref {r:+.4f} shape {{{rshape}}} "
          f"| rel {rel:.2e} {'ok' if ok else 'DIVERGE'} (limit {REL:.0%})")
print("RES dump compare:", "ok" if rc == 0 else "check above")
sys.exit(rc)
PYEOF
python "$OUT/cmp_dump.py" "$OUT/dump.txt" "$OUT/ours_layers.json" > "$OUT/dumpcmp.log" 2>&1
rc_dumpcmp=$?
echo "== dump compare exit $rc_dumpcmp ($(date +%T))"
cat "$OUT/dumpcmp.log"

# ---- 3) greedy probs vs llama-server golden (C8) ----
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

# ---- 4) perplexity vs golden (C8 <= 1%) ----
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

echo "== summary: pytest=$rc_pytest dump=$rc_dump odump=$rc_odump dumpcmp=$rc_dumpcmp ours=$rc_ours diff=$rc_diff ppl=$rc_ppl pplcmp=$rc_pplcmp"
echo "== end $(date +%T)"
