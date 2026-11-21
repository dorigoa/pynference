#!/usr/bin/env bash
# job: S27-20
# timeout: 45
# scopo: S27 passo 5 (diagnostica 2.5, 1a occorrenza dei rossi S27-19) —
#   separare "stato finestra" da "degradazione numerica PERSISTENTE"
#   dell'ambiente PC nuovo (driver NVIDIA apparsi 2026-11-21 18:39; in
#   S27-19 col workaround export CUDA_VISIBLE_DEVICES=-1: suite GPU e
#   gemma VERDI, gemma bit-esatta vs golden 15/11, MA gate C8 q4 FUSED
#   rosso (KL 1.74e-2; era PASS 3.48e-3 in S27-17 finestra pulita vecchio
#   ambiente) e gpt-oss rosso/ppl variabile fra processi (1.6164 vs
#   1.8419; gate era PASS in S27-14)). In S27-19 nessun torchrun è
#   crashato ⇒ i rossi non sono la contaminazione nota di S27-16.
#   Questa finestra misura SOLO i gate, due processi identici per modello
#   (cross-processo), lavoro per primo, niente suite/re-base/regress.
# Lettura (decisioni, pre-registrate):
#   - q4 gate VERDE e gpt-oss gate VERDE anche qui ⇒ rossi S27-19 =
#     stato finestra/contaminazione residua ⇒ ripetere il job finale
#     (matrice S27-19) in un'altra finestra (S27-21).
#   - q4 e/o gpt-oss gate ROSSI di nuovo ⇒ degrado persistente del
#     percorso esperti-swap nel nuovo ambiente (gemma dense bit-esatta
#     esclude i kernel densi) ⇒ l'ambiente va sistemato fuori dal
#     perimetro della scheda ⇒ Bloccata (2.6).
# Fasi (tutto embedded, regola S27-10):
#   (0) diag ambiente CPU (torch/triton come in S27-19) + sonda device
#       NVIDIA + export CUDA_VISIBLE_DEVICES=-1 (workaround).
#   (1) q4 FUSED: ours(wrapper)+gate C8+ ppl su f1; ours+gate su f2;
#       cmpjj f1-vs-f2 (stesso comando, due processi).
#   (2) gpt-oss FUSED: ours(wrapper)+gate+ppl su o1; ours+gate su o2;
#       cmpjj o1-vs-o2.
#   (3) gemma GRAPH: ours + gate + ppl + cmp1 vs golden (guardia
#       numerica dense; atteso = S27-19: ppl 10064.7540 esatto, raw
#       3.30e-04, id tutti identici).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

# ---- (0) diagnostica ambiente + workaround -----------------------------------
cat > "$OUT/envdiag.py" <<'DIAGEOF'
"""Print torch/triton backend state; never raise (diagnostic only)."""
import os

print("CUDA_VISIBLE_DEVICES=", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
try:
    import torch
    print("torch", torch.__version__, "version.cuda=", torch.version.cuda)
    print("cuda.is_available=", torch.cuda.is_available(),
          "xpu.is_available=", torch.xpu.is_available())
except Exception as e:  # noqa: BLE001
    print("torch diag error:", type(e).__name__, e)
try:
    from triton.runtime import driver
    print("triton driver.active =", driver.active)
except Exception as e:  # noqa: BLE001
    print("triton driver diag:", type(e).__name__, e)
DIAGEOF
echo "== diag PRE-export $(date +%T)"
python "$OUT/envdiag.py" 2>&1 | tail -n 6
python -m pip list 2>/dev/null | grep -iE "^(torch|triton|nvidia|level|intel)" | head -n 12
echo "== nvidia probe $(date +%T)"
ls /dev/nvidia* 2>&1 | head -3
cat /proc/driver/nvidia/version 2>/dev/null | head -1 || true
ls /sys/bus/pci/drivers/ 2>/dev/null | grep -iE "nvidia" | head -3 || true
export CUDA_VISIBLE_DEVICES=-1
echo "== diag POST-export $(date +%T)"
python "$OUT/envdiag.py" 2>&1 | tail -n 3

M=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
MQ=$M
GQ=tests/golden/qwen4exp
MG=/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf
GG=tests/golden/gemma4
MO=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
GO=tests/golden/gpt-oss-120b

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

# ---- wrapper S17 (verbatim da scripts/regress.sh righe 30-43) ---------------
cat > "$OUT/run_bench.py" <<'RBEOF'
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
RBEOF

# ---- cmpjj verbatim da S27-16 (json vs json, niente ppl, soglie 1e-3/id) -----
cat > "$OUT/cmpjj.py" <<'CJEOF'
import json, sys

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
    for i, (bs, ns) in enumerate(zip(b["steps"], n["steps"])):
        if bs["id"] != ns["id"]:
            problems.append(f"{b['name']}: greedy token differs at step {i}")
            break
        pb = {t["id"]: t["p"] for t in bs["top"]}
        pn = {t["id"]: t["p"] for t in ns["top"]}
        if pb.keys() != pn.keys():
            problems.append(f"{b['name']}: top-10 ids differ at step {i}")
            break
        max_dp = max([max_dp] + [abs(pb[k] - pn[k]) for k in pb])
    print(f"{b['name']}: max |dp| {max_dp:.2e} over top-10")
    if max_dp > 1e-3:
        problems.append(f"{b['name']}: max |dp| {max_dp:.2e} > 1e-3")

if problems:
    for p in problems:
        print("FAIL:", p)
    sys.exit(1)
print("PASS")
CJEOF

# ---- cmp verbatim da scripts/regress.sh (67-113) — solo guardia gemma --------
cat > "$OUT/cmp2.py" <<'CMPEOF'
import json, re, sys

base_path, now_path, ppl_base_path, ppl_now_path = sys.argv[1:5]
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
    for i, (bs, ns) in enumerate(zip(b["steps"], n["steps"])):
        if bs["id"] != ns["id"]:
            problems.append(f"{b['name']}: greedy token differs at step {i}")
            break
        pb = {t["id"]: t["p"] for t in bs["top"]}
        pn = {t["id"]: t["p"] for t in ns["top"]}
        if pb.keys() != pn.keys():
            problems.append(f"{b['name']}: top-10 ids differ at step {i}")
            break
        max_dp = max([max_dp] + [abs(pb[k] - pn[k]) for k in pb])
    print(f"{b['name']}: max |dp| {max_dp:.2e} over top-10")
    if max_dp > 1e-3:
        problems.append(f"{b['name']}: max |dp| {max_dp:.2e} > 1e-3")

pp = re.compile(r"PPL = ([0-9.]+)")
b_ppl = float(pp.search(open(ppl_base_path).read()).group(1))
n_ppl = float(pp.search(open(ppl_now_path).read()).group(1))
rel = abs(n_ppl - b_ppl) / b_ppl
print(f"ppl: baseline {b_ppl:.4f}, now {n_ppl:.4f}, rel diff {rel * 100:.3f}%")
if rel > 1e-3:
    problems.append(f"ppl rel diff {rel * 100:.3f}% > 0.1%")

if problems:
    for p in problems:
        print("FAIL:", p)
    sys.exit(1)
print("PASS")
CMPEOF

gate() { # gate REFJSON JSON LOG: C8 strict, stampa coda, ritorna rc
  python bench/compare_probs.py diff --strict "$1" "$2" > "$3" 2>&1
  local rc=$?
  clean "$3" | tail -n 6
  return $rc
}

# ---- (1) q4 FUSED: f1 ours+gate+ppl, f2 ours+gate, cmpjj ---------------------
q4run() { # q4run TAG: ours --decode fused (wrapper) -> $OUT/$TAG.json
  local tag=$1
  timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
    bench.compare_probs ours --ref "$GQ/ref_probs.json" -m "$MQ" \
    --decode fused -o "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "== q4 $tag(fused,wrapper) exit $rc ($(date +%T)) (atteso 0)"
  if [ $rc -ne 0 ]; then clean "$OUT/$tag.log" | tail -n 40
  else clean "$OUT/$tag.log" | tail -n 4; fi
  return $rc
}
q4ppl() { # q4ppl TAG: bench.ppl -> $OUT/$TAG.log (PPL = ...)
  local tag=$1
  timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl \
    -m "$MQ" -c 512 --chunks 8 > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "== q4 $tag ppl exit $rc ($(date +%T)) (atteso 0)"
  clean "$OUT/$tag.log" | grep -a "PPL =" | tail -n 1
  return $rc
}

q4run f1; rc_f1=$?
echo "== q4 gate f1 $(date +%T) (atteso 0 = C8, vecchio ambiente 0 in S27-17)"
gate "$GQ/ref_probs.json" "$OUT/f1.json" "$OUT/diff_q4_f1.log"; rc_g1=$?
q4ppl pplq4; rc_pq=$?
q4run f2; rc_f2=$?
echo "== q4 gate f2 $(date +%T) (atteso 0 = C8)"
gate "$GQ/ref_probs.json" "$OUT/f2.json" "$OUT/diff_q4_f2.log"; rc_g2=$?
python "$OUT/cmpjj.py" "$OUT/f1.json" "$OUT/f2.json" \
  > "$OUT/cmp_q4_f1f2.log" 2>&1
rc_cj=$?
echo "== q4 cmpjj f1-vs-f2 exit $rc_cj ($(date +%T))"
cat "$OUT/cmp_q4_f1f2.log"
grep -a "resident\|host cache" "$OUT/f1.log" | tail -n 2
grep -a "resident\|host cache" "$OUT/f2.log" | tail -n 2

# ---- (2) gpt-oss FUSED: o1 ours+gate+ppl, o2 ours+gate, cmpjj ----------------
gosrun() { # gosrun TAG: ours --decode fused (wrapper) -> $OUT/$TAG.json
  local tag=$1
  timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
    bench.compare_probs ours --ref "$GO/ref_probs.json" -m "$MO" \
    --decode fused -o "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "== gpt-oss $tag(fused,wrapper) exit $rc ($(date +%T)) (atteso 0)"
  if [ $rc -ne 0 ]; then clean "$OUT/$tag.log" | tail -n 40
  else clean "$OUT/$tag.log" | tail -n 4; fi
  return $rc
}
gosrun o1; rc_o1=$?
echo "== gpt-oss gate o1 $(date +%T) (atteso 0 = C8, vecchio ambiente 0 in S27-14)"
gate "$GO/ref_probs.json" "$OUT/o1.json" "$OUT/diff_gptoss_o1.log"; rc_go1=$?
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl \
  -m "$MO" -c 512 --chunks 8 > "$OUT/ppl_gptoss.log" 2>&1
rc_po=$?
echo "== gpt-oss ppl exit $rc_po ($(date +%T)) (atteso 0)"
clean "$OUT/ppl_gptoss.log" | grep -aE "\[1\]|PPL =" | tail -n 2
gosrun o2; rc_o2=$?
echo "== gpt-oss gate o2 $(date +%T) (atteso 0 = C8)"
gate "$GO/ref_probs.json" "$OUT/o2.json" "$OUT/diff_gptoss_o2.log"; rc_go2=$?
python "$OUT/cmpjj.py" "$OUT/o1.json" "$OUT/o2.json" \
  > "$OUT/cmp_gptoss_o1o2.log" 2>&1
rc_co=$?
echo "== gpt-oss cmpjj o1-vs-o2 exit $rc_co ($(date +%T))"
cat "$OUT/cmp_gptoss_o1o2.log"
grep -a "local experts resident" "$OUT/o1.log" | tail -n 1
grep -a "local experts resident" "$OUT/o2.log" | tail -n 1

# ---- (3) gemma guardia numerica: ours + gate + ppl + cmp1 --------------------
timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
  --ref "$GG/ref_probs.json" -m "$MG" --decode graph -o "$OUT/ours_gemma.json" \
  > "$OUT/ours_gemma.log" 2>&1
rc_g=$?
echo "== gemma ours(graph) exit $rc_g ($(date +%T)) (atteso 0)"
if [ $rc_g -ne 0 ]; then clean "$OUT/ours_gemma.log" | tail -n 40
else clean "$OUT/ours_gemma.log" | tail -n 4; fi
echo "== gemma gate $(date +%T) (atteso 0)"
gate "$GG/ref_probs.json" "$OUT/ours_gemma.json" "$OUT/diff_gemma.log"; rc_gg=$?
timeout 900 torchrun --nproc_per_node=2 bench/ppl.py -m "$MG" -c 512 \
  --chunks 8 > "$OUT/ppl_gemma.log" 2>&1
rc_pg=$?
echo "== gemma ppl exit $rc_pg ($(date +%T)) (atteso 0)"
clean "$OUT/ppl_gemma.log" | grep -a "PPL =" | tail -n 1
python "$OUT/cmp2.py" "$GG/ours_baseline.json" "$OUT/ours_gemma.json" \
  "$GG/ours_ppl.txt" "$OUT/ppl_gemma.log" > "$OUT/cmp_gemma.log" 2>&1
rc_cg=$?
echo "== gemma cmp1 vs golden exit $rc_cg ($(date +%T)) (atteso 0, = S27-19)"
cat "$OUT/cmp_gemma.log"

echo "== summary: qf1=$rc_f1 qg1=$rc_g1 qppl=$rc_pq qf2=$rc_f2 qg2=$rc_g2 qcmp=$rc_cj of1=$rc_o1 og1=$rc_go1 oppl=$rc_po of2=$rc_o2 og2=$rc_go2 ocmp=$rc_co gmours=$rc_g gmgate=$rc_gg gmppl=$rc_pg gmcmp=$rc_cg"
echo "== end $(date +%T)"
