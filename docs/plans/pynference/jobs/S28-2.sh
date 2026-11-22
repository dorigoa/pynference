#!/usr/bin/env bash
# job: S28-2
# timeout: 90
# scopo: S28 passo 1, ripresa dopo S28-1 — regress.sh VERDI su gemma4 e
#   qwen35-27b (paths corretti in scripts/regress.sh, mossa dal rebuild PC),
#   gate C8 come criterio per i due MoE (istruzione S27 "Note per le sezioni
#   successive": regressione rigida intrinsecamente fragile per ULP
#   cross-processo) + una fotografia regress.sh per MoE (attesa FAIL
#   documentata, S27-12…24), suite tests/gpu per ultima (2.9.2).
# Struttura: guardia (A) fatti di filesystem NVIDIA (verbatim S27-24);
#   canaro = regress.sh gemma4 per prima (C10 + ambiente): se rosso NUMERICO
#   ⇒ CANARY-RED, salta tutto, exit 0; poi qwen35-27b; poi gate MoE; poi
#   suite. Nessun golden viene riscritto (S28 = sola validazione).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

rc_g=255 rc_q=255 rc_goo=255 rc_gog=255 rc_gop=255 rc_gor=255
rc_qo=255 rc_qg=255 rc_qp=255 rc_qr=255 rc_gp=255 ENVOK=0 CANARY=0

# ---- sonda ambiente (solo CPU, fail-fast; verbatim S27-24) -------------------
echo "== nvidia probe $(date +%T)"
NV_FILES=$(ls /dev/nvidia* 2>/dev/null | head -n 6)
NV_VER=$(head -n 1 /proc/driver/nvidia/version 2>/dev/null)
NV_MODS=$(lsmod 2>/dev/null | grep -ci nvidia || true)
echo "nvidia device files: ${NV_FILES:-(none)}"
echo "nvidia version: ${NV_VER:-(none)}"
echo "nvidia kernel modules: ${NV_MODS:-0}"
if [ -n "$NV_FILES" ] || [ -n "$NV_VER" ] || [ "${NV_MODS:-0}" != "0" ]; then
  echo "== ENV-BROKEN: tracce NVIDIA su filesystem — matrice saltata $(date +%T)"
  echo "== summary: env=$ENVOK canary=$CANARY gemma=$rc_g qwen35=$rc_q gogate=$rc_gog goppl=$rc_gop goregress=$rc_gor q4gate=$rc_qg q4ppl=$rc_qp q4regress=$rc_qr gpu=$rc_gp"
  echo "== end $(date +%T)"
  exit 0
fi
echo "== ENV OK: nessun segno NVIDIA su filesystem $(date +%T)"
ENVOK=1

M=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
MQ=$M
GQ=tests/golden/qwen4exp
MG=/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf
GG=tests/golden/gemma4
MO=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
GO=tests/golden/gpt-oss-120b

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

# ---- wrapper S17 (verbatim da scripts/regress.sh) -----------------------------
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

# ---- 0) CANARO + C10: regress.sh gemma4 (atteso 0, bit-esatto) ---------------
timeout 900 bash scripts/regress.sh gemma4 > "$OUT/regress_gemma4.log" 2>&1
rc_g=$?
echo "== regress gemma4 exit $rc_g ($(date +%T)) (atteso 0 = canaro + C10)"
clean "$OUT/regress_gemma4.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 8
if [ $rc_g -ne 0 ]; then
  echo "== CANARY-RED: gemma regress rossa — resto saltato $(date +%T)"
  echo "== summary: env=$ENVOK canary=$CANARY gemma=$rc_g qwen35=$rc_q gogate=$rc_gog goppl=$rc_gop goregress=$rc_gor q4gate=$rc_qg q4ppl=$rc_qp q4regress=$rc_qr gpu=$rc_gp"
  echo "== end $(date +%T)"
  exit 0
fi
CANARY=1

# ---- 1) regress.sh qwen35-27b (denso, golden S21: atteso 0) -------------------
timeout 900 bash scripts/regress.sh qwen35-27b > "$OUT/regress_qwen35.log" 2>&1
rc_q=$?
echo "== regress qwen35-27b exit $rc_q ($(date +%T)) (atteso 0)"
clean "$OUT/regress_qwen35.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 8

# ---- 2) gpt-oss: gate C8 (criterio) + fotografia regress (attesa FAIL) --------
timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
  bench.compare_probs ours --ref "$GO/ref_probs.json" -m "$MO" \
  --decode fused -o "$OUT/ours_gptoss.json" > "$OUT/ours_gptoss.log" 2>&1
rc_goo=$?
python bench/compare_probs.py diff --strict "$GO/ref_probs.json" \
  "$OUT/ours_gptoss.json" > "$OUT/diff_gptoss.log" 2>&1
rc_gog=$?
timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl \
  -m "$MO" -c 512 --chunks 8 > "$OUT/ppl_gptoss.log" 2>&1
rc_gop=$?
echo "== gpt-oss ours/gate/ppl exit $rc_goo/$rc_gog/$rc_gop ($(date +%T)) (attesi 0/0/0)"
clean "$OUT/diff_gptoss.log" | tail -n 8
clean "$OUT/ppl_gptoss.log" | grep -a "PPL =" | tail -n 1
echo "ref ppl: $(cat "$GO/ref_ppl.txt")"
timeout 1800 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1
rc_gor=$?
echo "== regress gpt-oss (fotografia, atteso FAIL documentato) exit $rc_gor ($(date +%T))"
clean "$OUT/regress_gptoss.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 8

# ---- 3) q4exp: gate C8 FUSED (criterio) + fotografia regress (attesa FAIL) ----
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
  bench.compare_probs ours --ref "$GQ/ref_probs.json" -m "$MQ" \
  --decode fused -o "$OUT/ours_q4.json" > "$OUT/ours_q4.log" 2>&1
rc_qo=$?
python bench/compare_probs.py diff --strict "$GQ/ref_probs.json" \
  "$OUT/ours_q4.json" > "$OUT/diff_q4.log" 2>&1
rc_qg=$?
timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl \
  -m "$MQ" -c 512 --chunks 8 > "$OUT/ppl_q4.log" 2>&1
rc_qp=$?
echo "== q4 ours/gate/ppl exit $rc_qo/$rc_qg/$rc_qp ($(date +%T)) (attesi 0/0/0)"
clean "$OUT/diff_q4.log" | tail -n 8
clean "$OUT/ppl_q4.log" | grep -a "PPL =" | tail -n 1
echo "ref ppl: $(cat "$GQ/ref_ppl.txt")"
timeout 1800 bash scripts/regress.sh qwen4exp > "$OUT/regress_q4.log" 2>&1
rc_qr=$?
echo "== regress qwen4exp (fotografia, atteso FAIL documentato) exit $rc_qr ($(date +%T))"
clean "$OUT/regress_q4.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 8

# ---- 4) suite tests/gpu (2.9.2, per ultima) -----------------------------------
python -m pytest -q tests/gpu > "$OUT/pytest_gpu.log" 2>&1
rc_gp=$?
echo "== gpu tests exit $rc_gp ($(date +%T)) (atteso 0)"
tail -n 2 "$OUT/pytest_gpu.log"

echo "== summary: env=$ENVOK canary=$CANARY gemma=$rc_g qwen35=$rc_q gogate=$rc_gog goppl=$rc_gop goregress=$rc_gor q4gate=$rc_qg q4ppl=$rc_qp q4regress=$rc_qr gpu=$rc_gp"
echo "== end $(date +%T)"
