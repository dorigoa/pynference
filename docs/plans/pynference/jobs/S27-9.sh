#!/usr/bin/env bash
# job: S27-9
# timeout: 100
# scopo: (S27 passo 5, dopo la diagnosi di S27-8) chiudere le regressioni.
#   (1) `pytest -q tests/gpu` (2.9.2: job finale della sezione).
#   (2) gemma4 (C10): diagnostica del percorso modello (in S27-8
#       "missing model or golden files" istantaneo; golden regolari in
#       repo, mancava il modello sul PC) + `regress.sh gemma4`.
#   (3) qwen4exp: gate C8 strict sul path GRAFO (--decode graph ora
#       reale; in S27-4 il flag era inerte e il gate giro' su eager) +
#       `regress.sh qwen4exp` contro la golden rigenerata di S27-4 ora
#       installata. Se regress e' rossa col gate verde: regola S17/S18 →
#       rigenera la golden da questo gate (copia in $OUT da committare)
#       e ri-esegui regress (atteso PASS).
#   (4) gpt-oss-120b: spostamento di regime noto (RAM 63,7→121,5 GiB:
#       ppl rel 0,180% vs golden, dp ≤1,45e-4, id identici — nota S27-8).
#       Regola S26-5→S26-6: gate `diff --strict` + ppl vs ref_ppl.txt
#       (≤1%) su --decode fused (come regress.sh); se verde rigenerare
#       `ours_baseline.json`+`ours_ppl.txt` (copia in $OUT da committare)
#       e `regress.sh gpt-oss-120b` (atteso PASS).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
MQ=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
GQ=tests/golden/qwen4exp
MO=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
GO=tests/golden/gpt-oss-120b

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

pplcmp() { # pplcmp LOG REF: PPL ours vs llama, limite 1% (block S27-4)
  python - "$1" "$2" <<'PYEOF'
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
}

rebase() { # rebase OURS_JSON PPL_LOG GOLDEN_DIR: golden dal gate (regola S17/S18)
  cp "$1" "$3/ours_baseline.json" || return 1
  python - "$2" > "$3/ours_ppl.txt" <<'PYEOF' || return 1
import re, sys
m = re.search(r"PPL = ([0-9.]+)", open(sys.argv[1]).read())
if not m:
    sys.exit(3)
print(f"PPL = {m.group(1)}")
PYEOF
  cp "$3/ours_baseline.json" "$3/ours_ppl.txt" "$OUT/"   # torna sul Mac da committare
}

# ---- 1) suite GPU completa (2.9.2) -----------------------------------------
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1; rg=$?
echo "== pytest gpu exit $rg ($(date +%T))"
tail -n 2 "$OUT/gpu_all.log"

# ---- 2) gemma4: diagnostica modello + regress (C10) --------------------------
{
  ls -la /nvme/gguf/ 2>&1 | head -n 20
  ls -la /nvme/gguf/gemma-4-31B/ 2>&1 | head -n 10
  df -h /nvme 2>&1 | tail -n 2
  timeout 60 find /nvme -maxdepth 4 -iname '*gemma-4-31B*' 2>/dev/null | head -n 10
} > "$OUT/gemma_diag.log" 2>&1
cat "$OUT/gemma_diag.log"
timeout 1800 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1; r4=$?
echo "== regress gemma4 exit $r4 ($(date +%T)) (atteso 0)"
clean "$OUT/regress_gemma4.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10

# ---- 3) qwen4exp: gate C8 sul grafo + regress contro la golden S27-4 ---------
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$GQ/ref_probs.json" -m "$MQ" --decode graph -o "$OUT/ours_q4.json" > "$OUT/ours_q4.log" 2>&1
rc_oq=$?
echo "== q4 ours(graph) exit $rc_oq ($(date +%T))"
clean "$OUT/ours_q4.log" | tail -n 4
python "$OUT/run_bench.py" bench.compare_probs diff --strict "$GQ/ref_probs.json" "$OUT/ours_q4.json" \
  > "$OUT/diff_q4.log" 2>&1
rc_dq=$?
echo "== q4 diff strict exit $rc_dq ($(date +%T)) (atteso 0)"
cat "$OUT/diff_q4.log"
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$MQ" \
  -c 512 --chunks 8 > "$OUT/ppl_q4.log" 2>&1
rc_pq=$?
echo "== q4 ppl exit $rc_pq ($(date +%T)) (atteso 0)"
grep -a "Final estimate" "$OUT/ppl_q4.log"
pplcmp "$OUT/ppl_q4.log" "$GQ/ref_ppl.txt"; rc_cq=$?
echo "== q4 pplcmp exit $rc_cq ($(date +%T)) (atteso 0)"

timeout 1800 bash scripts/regress.sh qwen4exp > "$OUT/regress_q4_r1.log" 2>&1; rq=$?
echo "== regress qwen4exp r1 exit $rq ($(date +%T)) (atteso 0)"
clean "$OUT/regress_q4_r1.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL|top-10" | tail -n 12
if [ "$rq" -ne 0 ] && [ "$rc_dq" -eq 0 ] && [ "$rc_cq" -eq 0 ]; then
  echo "== q4: regress rossa col gate verde → rigenero la golden dal gate (regola S17/S18)"
  rebase "$OUT/ours_q4.json" "$OUT/ppl_q4.log" "$GQ"; rc_bq=$?
  echo "== q4 rebase exit $rc_bq ($(date +%T))"
  if [ "$rc_bq" -eq 0 ]; then
    timeout 1800 bash scripts/regress.sh qwen4exp > "$OUT/regress_qwen4exp.log" 2>&1; rq=$?
    echo "== regress qwen4exp r2 exit $rq ($(date +%T)) (atteso 0)"
    clean "$OUT/regress_qwen4exp.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10
  fi
fi
[ -f "$OUT/regress_qwen4exp.log" ] || cp "$OUT/regress_q4_r1.log" "$OUT/regress_qwen4exp.log"

# ---- 4) gpt-oss: gate C8 (fused) + re-baseline + regress ---------------------
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.compare_probs ours \
  --ref "$GO/ref_probs.json" -m "$MO" --decode fused -o "$OUT/ours_gos.json" > "$OUT/ours_gos.log" 2>&1
rc_og=$?
echo "== goss ours(fused) exit $rc_og ($(date +%T))"
clean "$OUT/ours_gos.log" | tail -n 4
python "$OUT/run_bench.py" bench.compare_probs diff --strict "$GO/ref_probs.json" "$OUT/ours_gos.json" \
  > "$OUT/diff_gos.log" 2>&1
rc_dg=$?
echo "== goss diff strict exit $rc_dg ($(date +%T)) (atteso 0)"
cat "$OUT/diff_gos.log"
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl -m "$MO" \
  -c 512 --chunks 8 > "$OUT/ppl_gos.log" 2>&1
rc_pg=$?
echo "== goss ppl exit $rc_pg ($(date +%T)) (atteso 0)"
grep -a "Final estimate" "$OUT/ppl_gos.log"
pplcmp "$OUT/ppl_gos.log" "$GO/ref_ppl.txt"; rc_cg=$?
echo "== goss pplcmp exit $rc_cg ($(date +%T)) (atteso 0, limite 1%)"

ro=3
if [ "$rc_dg" -eq 0 ] && [ "$rc_cg" -eq 0 ]; then
  echo "== goss: gate verde → re-baseline golden al regime RAM nuova (regola S26-5→S26-6)"
  rebase "$OUT/ours_gos.json" "$OUT/ppl_gos.log" "$GO"; rc_bg=$?
  echo "== goss rebase exit $rc_bg ($(date +%T))"
  if [ "$rc_bg" -eq 0 ]; then
    timeout 1800 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1; ro=$?
    echo "== regress gpt-oss exit $ro ($(date +%T)) (atteso 0)"
    clean "$OUT/regress_gptoss.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10
  fi
else
  echo "== regress gpt-oss SKIP (gate non verde: diff=$rc_dg pplcmp=$rc_cg)"
fi

echo "== summary: gpu=$rg gemma4=$r4 q4gate=$((rc_dq + rc_cq)) q4regress=$rq gossgate=$((rc_dg + rc_cg)) gossregress=$ro"
echo "== end $(date +%T)"
