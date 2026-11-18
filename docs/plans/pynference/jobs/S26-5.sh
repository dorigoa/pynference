#!/usr/bin/env bash
# job: S26-5
# timeout: 90
# scopo: (S26 passo 1) attribuzione del FAIL di S26-4 + C8 + velocità.
#   S26-4: SIGABRT sparito (ours exit 0), ppl 1.6095 = baseline, ma regress
#   rossa (chat-it top-10 ids differ at step 16, dp max 4.92e-05) e decode
#   saltato. "TOGGLE NOT SEEN" è un artefatto: regress.sh tuba il stdout di
#   torchrun in `tail -n 6`, le righe store (verbose=True) vengono tagliate.
#   Dal 6ec5f8d (S18-4, ultima regress gpt-oss verde) nessun cambiamento al
#   codice condiviso tocca i logit gpt-oss: lo spostamento viene dal regime
#   S26 (finestra allocator e/o residenza 21 GiB/rank). Questo job attribuisce
#   (A: regime S26 ripetuto + C8 vs llama.cpp; B: controllo regime S18 via
#   monkeypatch), misura la velocità passo 1 e chiude con regress gemma4.
set -uo pipefail
OUT=$PYI_JOB_OUT
MODEL=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
G=tests/golden/gpt-oss-120b
echo "== start $(date +%T)"
grep -E "GPUActive|MemAvailable|MemTotal" /proc/meminfo

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

# bench/ rebinder for non-Gemma models, same as scripts/regress.sh (S17).
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

# Controllo S26-5: compare_probs ours nel regime di memoria pre-S26, solo
# via monkeypatch (nessuna modifica al repo): window expandable_segments
# disattivata (allocazioni specchiate, il percorso S17/S18) e
# EXPERT_HOST_RESERVE_BYTES a 24 GiB (store ~16 GiB/rank, residenza S18).
cat > "$OUT/ours_s18.py" <<'PYEOF'
"""S26-5 control: compare_probs ours in the pre-S26 memory regime.

Monkeypatches only (no repo changes): ExpertStore's expandable_segments
window disabled (mirrored allocations, the S17/S18 path) and
GPTOss.EXPERT_HOST_RESERVE_BYTES back to 24 GiB (store ~16 GiB/rank, the
S18 residency). Rebinds the GGUF tokenizer factory like scripts/regress.sh
(S17). Expects to reproduce the S18-3/S18-4 dp profile bit for bit."""
import os
import sys

sys.path.insert(0, os.getcwd())
import pynference.moe as moe  # noqa: E402

moe._expandable_segments_supported = lambda: False

from pynference.models.gptoss import GPTOss  # noqa: E402

GPTOss.EXPERT_HOST_RESERVE_BYTES = 24 << 30

from pynference.tokenizer import load_tokenizer  # noqa: E402
import bench.compare_probs as cp  # noqa: E402

cp.Tokenizer = load_tokenizer
cp.main()
PYEOF

# Checker = parte probs del checker di regress.sh (C10), ppl escluso (misurato
# a parte). Diagnostica: dopo un top-10 ids mismatch continua a scandire i
# passi e accumula |dp| sugli id comuni (profilo dp completo).
cat > "$OUT/check.py" <<'PYEOF'
"""S26-5: ours_now vs the golden ours_baseline. The probs part of the
regress.sh checker (ppl measured separately), with one diagnostic change:
after a top-10 id mismatch the scan continues and |dp| accumulates over the
shared top-10 ids, so the full dp profile survives the first flip. PASS/FAIL
problems are reported at the same first index as in regress.sh."""
import json
import sys

base_path, now_path = sys.argv[1:3]
problems = []
base = json.load(open(base_path))
now = json.load(open(now_path))
if len(base) != len(now):
    problems.append(f"prompt count: {len(base)} baseline, {len(now)} now")
for b, n in zip(base, now):
    if b["name"] != n["name"]:
        problems.append(f"prompt order: {b['name']} vs {n['name']}")
        continue
    if b["prompt"] != n["prompt"]:
        problems.append(f"{b['name']}: prompt ids differ")
        continue
    if len(b["steps"]) != len(n["steps"]):
        problems.append(f"{b['name']}: step count {len(b['steps'])} vs {len(n['steps'])}")
    max_dp = 0.0
    ids_reported = False
    for i, (bs, ns) in enumerate(zip(b["steps"], n["steps"])):
        if bs["id"] != ns["id"]:
            problems.append(f"{b['name']}: greedy token differs at step {i}")
            break
        pb = {t["id"]: t["p"] for t in bs["top"]}
        pn = {t["id"]: t["p"] for t in ns["top"]}
        if pb.keys() != pn.keys() and not ids_reported:
            problems.append(f"{b['name']}: top-10 ids differ at step {i}")
            ids_reported = True
        common = pb.keys() & pn.keys()
        if common:
            max_dp = max([max_dp] + [abs(pb[k] - pn[k]) for k in common])
    print(f"{b['name']}: max |dp| {max_dp:.2e} over top-10")
    if max_dp > 1e-3:
        problems.append(f"{b['name']}: max |dp| {max_dp:.2e} > 1e-3")
if problems:
    for p in problems:
        print("FAIL:", p)
    sys.exit(1)
print("PASS")
PYEOF

# ---- A: regime S26 corrente, stdout intero (niente tail: marker store)
echo "== A ours S26 regime full-log ($(date +%T))"
timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
    --ref "$G/ref_probs.json" -m "$MODEL" --decode fused -o "$OUT/ours_s26.json" \
    > "$OUT/cp_s26.log" 2>&1 &
APID=$!
poll "$APID" "$OUT/mem_A.log"
wait $APID; AC=$?
echo "== A ours exit $AC ($(date +%T))"
echo "A toggle markers: $(grep -c 'expandable_segments on' "$OUT/cp_s26.log") (atteso >=1)"
grep -E "store: layer|store: expandable|store: pinned|budget" "$OUT/cp_s26.log" | head -6
summarize "$OUT/mem_A.log"
python "$OUT/check.py" "$G/ours_baseline.json" "$OUT/ours_s26.json"
echo "== A vs baseline rc $? ($(date +%T)) (atteso: FAIL step 16 come S26-4)"
python bench/compare_probs.py diff "$G/ref_probs.json" "$OUT/ours_s26.json" --strict \
    > "$OUT/c8_s26.log" 2>&1; C8=$?
echo "== A C8 diff --strict rc $C8 ($(date +%T))"
grep -v -E "CCL_WARN|^W1" "$OUT/c8_s26.log" | tail -n 15

# ---- B: controllo regime S18 (specchio + riserva 24 GiB), wrapper monkeypatch
echo "== B ours S18-regime control ($(date +%T))"
timeout 900 torchrun --nproc_per_node=2 "$OUT/ours_s18.py" ours \
    --ref "$G/ref_probs.json" -m "$MODEL" --decode fused -o "$OUT/ours_s18.json" \
    > "$OUT/cp_s18.log" 2>&1 &
BPID=$!
poll "$BPID" "$OUT/mem_B.log"
wait $BPID; BC=$?
echo "== B ours exit $BC ($(date +%T))"
echo "B toggle markers: $(grep -c 'expandable_segments' "$OUT/cp_s18.log") (atteso 0)"
grep -E "store: layer|store: pinned|budget" "$OUT/cp_s18.log" | head -4
summarize "$OUT/mem_B.log"
python "$OUT/check.py" "$G/ours_baseline.json" "$OUT/ours_s18.json"
echo "== B vs baseline rc $? ($(date +%T)) (atteso: PASS, dp = S18-3/4)"

# ---- C: velocità (regime S26), forme S18-3
echo "== decode graph short ctx 8192 ($(date +%T))"
timeout 1800 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
    -m "$MODEL" --mode graph --ctx 8192 --steps 32 --no-profile \
    > "$OUT/graph_short.log" 2>&1 &
GPID=$!
poll "$GPID" "$OUT/mem_gs.log"
wait $GPID; GC=$?
echo "== graph short exit $GC ($(date +%T))"
grep -a -E 'MoE: .*budget|MoE: .*resident|store: expandable|wall time|store: hits' \
    "$OUT/graph_short.log" | head -n 8
summarize "$OUT/mem_gs.log"

echo "== decode graph prompt 2000 tok ($(date +%T))"
timeout 1800 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.decode_profile \
    -m "$MODEL" --mode graph --ctx 8192 --prompt-tokens 2000 --steps 16 --no-profile \
    > "$OUT/graph_long.log" 2>&1 &
LPID=$!
poll "$LPID" "$OUT/mem_gl.log"
wait $LPID; LC=$?
echo "== graph long exit $LC ($(date +%T))"
grep -a -E 'load done|prefill|wall time|store: hits' "$OUT/graph_long.log" | head -n 8
summarize "$OUT/mem_gl.log"

# ---- D: regressione Gemma (C10; 2.9.2 la vuole verde nell'ultimo job)
echo "== regress gemma4 ($(date +%T))"
timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
echo "== gemma4 exit $? ($(date +%T))"
grep -E "max \|dp\||ppl: baseline|PASS|FAIL" "$OUT/regress_gemma4.log" | tail -n 10

echo "== summary A_rc=$AC C8_rc=$C8 B_rc=$BC gs=$GC gl=$LC $(date +%T)"
grep -E "GPUActive|MemAvailable" /proc/meminfo
echo "== end $(date +%T)"
