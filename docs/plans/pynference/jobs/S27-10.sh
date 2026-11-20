#!/usr/bin/env bash
# job: S27-10
# timeout: 80
# scopo: (S27 passo 5, dopo la diagnosi di S27-9) due cose indipendenti.
#   (1) qwen4exp: il path di decode del passo 4 (driver segmentato, eager E
#       graph) devia dalla golden S27-4 (path pre-passo-4) e il gate C8 strict
#       sul grafo è rosso (KL fino a 1,10e-1). Metodo 2.5: A/ B dei layer.
#       Si esegue il modulo modello PRE-passo-4 (git show 507c831^) e il
#       modulo corrente nello stesso repo/PROMPTS, con hook che registrano
#       per ogni decode step: output di hc_combine per posizione (attn/MoE),
#       PLE, embedding, router logits, uscita QSA, logits. Il primo record
#       che diverge localizza il componente. In più: gate C8 + passo-vs-golden
#       del path FUSED (eager, no graph) per capire se il bug è del driver o
#       della cattura, e confronto fused-vs-graph (json S27-9).
#   (2) gemma4 (C10): il modello è stato spostato (regress.sh:11 hardcodail
#       vecchio path, file fuori scheda → non modificabile): riesegui qui i
#       COMANDI EQUIVALENTI di regress.sh gemma4 (ours graph + ppl + blocco di
#       confronto della golden) col path nuovo /nvme/gguf/gemma-4/... —
#       deviazione documentata nel report.
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
MQ=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
GQ=tests/golden/qwen4exp
MG=/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf
GG=tests/golden/gemma4
Q9=docs/plans/pynference/jobs/S27-9.out

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

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

cat > "$OUT/cmp.py" <<'PYEOF'
"""Regress-style comparison (scripts/regress.sh block) between two ours jsons
(+ ppl vs golden ppl): cmp BASE_JSON NOW_JSON PPL_BASE PPL_NOW"""
import json, re, sys

base_path, now_path, ppl_base_path, ppl_now_path = sys.argv[1:5]
problems = []
base = json.load(open(base_path))
now = json.load(open(now_path))
for b, n in zip(base, now):
    if b["name"] != n["name"] or b["prompt"] != n["prompt"]:
        problems.append(f"{b['name']}: prompt mismatch"); continue
    max_dp = 0.0
    for i, (bs, ns) in enumerate(zip(b["steps"], n["steps"])):
        pb = {t["id"]: t["p"] for t in bs["top"]}
        pn = {t["id"]: t["p"] for t in ns["top"]}
        ids_same = pb.keys() == pn.keys()
        if bs["id"] != ns["id"]:
            print(f"{b['name']}: greedy differs at step {i}, ids same: {ids_same}")
        if not ids_same:
            print(f"{b['name']}: top-10 ids differ at step {i}")
            break
        max_dp = max([max_dp] + [abs(pb[k] - pn[k]) for k in pb])
    print(f"{b['name']}: max |dp| {max_dp:.2e} over top-10")
pp = re.compile(r"PPL = ([0-9.]+)")
if ppl_base_path and ppl_now_path:
    b_ppl = float(pp.search(open(ppl_base_path).read()).group(1))
    n_ppl = float(pp.search(open(ppl_now_path).read()).group(1))
    rel = abs(n_ppl - b_ppl) / b_ppl
    print(f"ppl: baseline {b_ppl:.4f}, now {n_ppl:.4f}, rel diff {rel * 100:.3f}%")
    if rel > 1e-3:
        problems.append(f"ppl rel {rel * 100:.3f}% > 0.1%")
print("PROBLEMS" if problems else "ALIGNED")
PYEOF

# ---- 1) qwen4exp A/B dei layer (pre-passo-4 vs corrente, eager) -------------
git show 507c831^:pynference/models/qwen4exp.py > "$OUT/qwen4exp_v0.py" || exit 1
wc -l "$OUT/qwen4exp_v0.py"

for MODE in old new; do
  S2710_OLD_MOD="$OUT/qwen4exp_v0.py" timeout 1500 torchrun --nproc_per_node=2 \
    scratch/s2710_ab.py "$MODE" "$MQ" "$OUT" > "$OUT/ab_$MODE.log" 2>&1
  ra=$?
  echo "== ab $MODE exit $ra ($(date +%T)) (atteso 0)"
  clean "$OUT/ab_$MODE.log" | tail -n 4
done
python scratch/s2710_diff.py "$OUT" > "$OUT/ab_diff.log" 2>&1; rd=$?
echo "== ab diff exit $rd ($(date +%T))"
cat "$OUT/ab_diff.log"

# ---- 2) qwen4exp: gate C8 + golden-vs-passi sul path FUSED (eager) ----------
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$GQ/ref_probs.json" -m "$MQ" --decode fused -o "$OUT/ours_q4_fused.json" \
  > "$OUT/ours_q4_fused.log" 2>&1
rc_f=$?
echo "== q4 ours(fused) exit $rc_f ($(date +%T)) (atteso 0)"
clean "$OUT/ours_q4_fused.log" | tail -n 4
python "$OUT/run_bench.py" bench.compare_probs diff --strict "$GQ/ref_probs.json" \
  "$OUT/ours_q4_fused.json" > "$OUT/diff_q4_fused.log" 2>&1
rc_df=$?
echo "== q4 diff strict (fused) exit $rc_df ($(date +%T)) (atteso 0 = gate)"
cat "$OUT/diff_q4_fused.log"
python "$OUT/cmp.py" "$GQ/ours_baseline.json" "$OUT/ours_q4_fused.json" "" "" \
  > "$OUT/cmp_q4_fused_vs_golden.log" 2>&1
echo "== q4 fused vs golden (json S27-4):"
cat "$OUT/cmp_q4_fused_vs_golden.log"
if [ -f "$Q9/ours_q4.json" ]; then
  python "$OUT/cmp.py" "$Q9/ours_q4.json" "$OUT/ours_q4_fused.json" "" "" \
    > "$OUT/cmp_q4_graph_vs_fused.log" 2>&1
  echo "== q4 graph(S27-9) vs fused:"
  cat "$OUT/cmp_q4_graph_vs_fused.log"
fi

# ---- 3) gemma4 C10: comandi equivalenti a regress.sh, path nuovo ------------
timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
  --ref "$GG/ref_probs.json" -m "$MG" --decode graph -o "$OUT/ours_gemma.json" \
  > "$OUT/ours_gemma.log" 2>&1
rc_g=$?
echo "== gemma ours(graph) exit $rc_g ($(date +%T)) (atteso 0)"
clean "$OUT/ours_gemma.log" | tail -n 4
timeout 900 torchrun --nproc_per_node=2 bench/ppl.py -m "$MG" -c 512 --chunks 8 \
  > "$OUT/ppl_gemma.log" 2>&1
rc_pg=$?
echo "== gemma ppl exit $rc_pg ($(date +%T)) (atteso 0)"
grep -a "Final estimate" "$OUT/ppl_gemma.log"
python "$OUT/cmp.py" "$GG/ours_baseline.json" "$OUT/ours_gemma.json" \
  "$GG/ours_ppl.txt" "$OUT/ppl_gemma.log" > "$OUT/cmp_gemma.log" 2>&1
rc_cg=$?
echo "== gemma cmp exit $rc_cg ($(date +%T))"
cat "$OUT/cmp_gemma.log"

echo "== summary: ab_old/new=see_logs diff=$rd q4fused_gate=$rc_df gemma=$((rc_g + rc_pg + rc_cg))"
echo "== end $(date +%T)"
