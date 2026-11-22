#!/usr/bin/env bash
# job: S27-23
# timeout: 115
# scopo: S27 passo 5 — job FINALE dopo la riparazione completa dell'ambiente
#   PC (blocco S27-18…22: driver NVIDIA comparsi ⇒ Triton "2 active drivers"
#   + gate C8 MoE rossi + jitter cross-processo/persistenti; riparazione
#   verificata alla ripresa 2026-11-22 quater via scripts/cpu.sh: driver
#   NVIDIA rimosso E ambiente Python ripristinato — torch 2.15.0+xpu ok).
#   Matrice del job finale S27-21/22 VERBATIM con le stesse DUE guardie
#   fail-fast DENTRO il job (pre-registrate in "Problemi aperti"):
#   (A) sonda ambiente SU FATTI DI FILESYSTEM (`/dev/nvidia*`,
#       `/proc/driver/nvidia/version`, `lsmod | grep -ci nvidia`): se uno
#       qualsiasi è presente ⇒ stampa "== ENV-BROKEN", salta TUTTA la
#       matrice ed esce (exit 0, summary con marker). CORREZIONE vs S27-22:
#       `gpu-window.sh` (09db136) ora esporta CUDA_VISIBLE_DEVICES=-1 nei
#       job, quindi la sonda `triton.runtime.driver.active` passerebbe per
#       costruzione anche a driver presente — non è più una guardia valida.
#       La sonda triton resta nel log come sola diagnostica.
#   (B) ordine fasi: canaro gemma PRIMA della matrice q4/gpt-oss. Se il
#       canaro è rosso (gate C8 rosso, oppure cmp1 vs golden rosso anche
#       dopo re-base own-window + 2a run) ⇒ stampa "== CANARY-RED", salta
#       q4/gpt-oss/graph60/suite ed esce (exit 0, summary).
#   Fasi poi VERBATIM S27-21 (solo riordinate):
#   (0') gemma4 (path nuovo /nvme/gguf/gemma-4/, deviazione S27-10): ours
#        graph + gate + ppl + cmp1 vs golden (atteso 0 = canaro, atteso
#        bit-esatto: max|dp| <= 3,3e-4, id identici, ppl 10064.7540); se
#        rosso con gate verde: re-base condizionale + 2a run (S27-14 f.4).
#   (1') q4 FUSED: ours(wrapper S17) + diff --strict (gate C8, atteso 0)
#        + ppl; se gate verde: RE-BASE golden own-window + DUE regressioni
#        fused own-window (attesi 0 e 0). Poi UNA `regress.sh qwen4exp`
#        (graph) come fotografia: ATTESA FAIL (deviazione documentata).
#   (2') gpt-oss-120b: `regress.sh gpt-oss-120b` (atteso 0, golden re-base
#        S27-14); se rosso: gate+ppl+re-base condizionale + regress 2a.
#   (3') velocita graph @60 own-window (fotografia, atteso ~9,4-9,5).
#   (4') suite `tests/gpu` completa (2.9.2) — PER ULTIMA.
# Golden ribasate: cp in $OUT/golden_* E b64+gzip nel log (lezione S27-14).
# Tutto embedded (regola S27-10).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"

# rc pre-iniziali (summary valido anche sugli exit anticipati)
rc_fg=255 rc_f1=255 rc_pq=255 rb_q=255 rc_f2=255 rc_r2=255 rc_f3=255 rc_r3=255
rc_rx=255 rc_g=255 rc_gg=255 rc_pg=255 rc_cg=255 rc_cg2=255 rc_go=255
rc_go2=255 r1=255 rc_gp=255 ENVOK=0 CANARY=0

# ---- sonda ambiente (solo CPU, fail-fast) -----------------------------------
# Guardia (A) su FATTI DI FILESYSTEM: la finestra ora esporta sempre
# CUDA_VISIBLE_DEVICES=-1 ⇒ la sonda triton non discrimina più (vedi header).
echo "== nvidia probe $(date +%T)"
NV_FILES=$(ls /dev/nvidia* 2>/dev/null | head -n 6)
NV_VER=$(head -n 1 /proc/driver/nvidia/version 2>/dev/null)
NV_MODS=$(lsmod 2>/dev/null | grep -ci nvidia || true)
echo "nvidia device files: ${NV_FILES:-(none)}"
echo "nvidia version: ${NV_VER:-(none)}"
echo "nvidia kernel modules: ${NV_MODS:-0}"
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
echo "== diag (informativo: la finestra esporta gia' CUDA_VISIBLE_DEVICES=-1) $(date +%T)"
python "$OUT/envdiag.py" > "$OUT/envdiag_pre.log" 2>&1
tail -n 6 "$OUT/envdiag_pre.log"
python -m pip list 2>/dev/null | grep -iE "^(torch|triton|nvidia|level|intel)" | head -n 12
if [ -n "$NV_FILES" ] || [ -n "$NV_VER" ] || [ "${NV_MODS:-0}" != "0" ]; then
  echo "== ENV-BROKEN: tracce NVIDIA su filesystem (files/version/modules) — matrice saltata $(date +%T)"
  echo "== summary: env=$ENVOK canary=$CANARY q4f1=$rc_f1 q4gate=$rc_fg q4ppl=$rc_pq q4rebase=$rb_q q4f2=$rc_f2 q4r2=$rc_r2 q4f3=$rc_f3 q4r3=$rc_r3 q4graphshot=$rc_rx gemmaours=$rc_g gemmagate=$rc_gg gemppl=$rc_pg gemmacmp1=$rc_cg gemmacmp2=$rc_cg2 gptoss=$rc_go gptoss2=$rc_go2 graph60=$r1 gpu=$rc_gp"
  echo "== end $(date +%T)"
  exit 0
fi
echo "== ENV OK: nessun segno NVIDIA su filesystem (file assenti, version assente, moduli 0)"
ENVOK=1

M=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
MQ=$M
GQ=tests/golden/qwen4exp
MG=/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf
GG=tests/golden/gemma4
MO=/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
GO=tests/golden/gpt-oss-120b

poll() { # poll PID LOG: campiona meminfo ogni 3 s in LOG finché PID vivo
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

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

emit() { # emit SRC TAG: copia in $OUT/golden_TAG + gzip/base64 nel log
  cp "$1" "$OUT/golden_$2"
  echo "== golden $2 b64 start"
  gzip -c "$1" | base64 | sed 's/^/B64 /'
  echo "== golden $2 b64 end"
}

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

# ---- cmp verbatim dal blocco python di scripts/regress.sh (67-113) ----------
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

# ---- 0') gemma4 CANARO: ours graph + gate + ppl + cmp1 (+ re-base cond.) ----
timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
  --ref "$GG/ref_probs.json" -m "$MG" --decode graph -o "$OUT/ours_gemma.json" \
  > "$OUT/ours_gemma.log" 2>&1
rc_g=$?
echo "== gemma ours(graph) exit $rc_g ($(date +%T)) (atteso 0)"
if [ $rc_g -ne 0 ]; then clean "$OUT/ours_gemma.log" | tail -n 40
else clean "$OUT/ours_gemma.log" | tail -n 4; fi
python bench/compare_probs.py diff --strict "$GG/ref_probs.json" \
  "$OUT/ours_gemma.json" > "$OUT/diff_gemma.log" 2>&1
rc_gg=$?
echo "== gemma diff strict exit $rc_gg ($(date +%T)) (atteso 0 = gate C8)"
clean "$OUT/diff_gemma.log" | tail -n 8
timeout 900 torchrun --nproc_per_node=2 bench/ppl.py -m "$MG" -c 512 \
  --chunks 8 > "$OUT/ppl_gemma.log" 2>&1
rc_pg=$?
echo "== gemma ppl exit $rc_pg ($(date +%T)) (atteso 0)"
clean "$OUT/ppl_gemma.log" | grep -a "PPL =" | tail -n 1
python "$OUT/cmp2.py" "$GG/ours_baseline.json" "$OUT/ours_gemma.json" \
  "$GG/ours_ppl.txt" "$OUT/ppl_gemma.log" > "$OUT/cmp_gemma.log" 2>&1
rc_cg=$?
echo "== gemma cmp1 vs golden exit $rc_cg ($(date +%T)) (atteso 0 = canaro)"
cat "$OUT/cmp_gemma.log"
if [ $((rc_g + rc_pg + rc_gg)) -eq 0 ] && [ $rc_cg -ne 0 ]; then
  # gate verde + C10 rossa ⇒ deriva fra finestre: re-base + 2a run (S27-14 f.4)
  cp "$OUT/ours_gemma.json" "$GG/ours_baseline.json"
  grep -a "PPL =" "$OUT/ppl_gemma.log" | tail -n 1 > "$GG/ours_ppl.txt"
  echo "== gemma golden rebased (own-window) $(date +%T)"
  emit "$OUT/ours_gemma.json" gemma4_baseline.json
  cp "$GG/ours_ppl.txt" "$OUT/golden_gemma4_ppl.txt"
  echo "== golden gemma4_ppl.txt: $(cat "$GG/ours_ppl.txt")"
  timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
    --ref "$GG/ref_probs.json" -m "$MG" --decode graph \
    -o "$OUT/ours_gemma2.json" > "$OUT/ours_gemma2.log" 2>&1
  rc_g2=$?
  timeout 900 torchrun --nproc_per_node=2 bench/ppl.py -m "$MG" -c 512 \
    --chunks 8 > "$OUT/ppl_gemma2.log" 2>&1
  rc_pg2=$?
  echo "== gemma ours2/ppl2 exit $rc_g2/$rc_pg2 ($(date +%T)) (atteso 0/0)"
  python "$OUT/cmp2.py" "$GG/ours_baseline.json" "$OUT/ours_gemma2.json" \
    "$GG/ours_ppl.txt" "$OUT/ppl_gemma2.log" > "$OUT/cmp_gemma2.log" 2>&1
  rc_cg2=$?
  echo "== gemma cmp2 (own-window, atteso 0) exit $rc_cg2 ($(date +%T))"
  cat "$OUT/cmp_gemma2.log"
fi
# canaro verde = cmp1 0; oppure cmp2 0 con cmp1 rosso (deriva fra finestre
# sana: own-window chiuso; le golden ribasate vanno committate a mano).
if [ $rc_cg -eq 0 ] || { [ $rc_cg -ne 0 ] && [ $rc_cg2 -eq 0 ]; }; then
  CANARY=1
else
  echo "== CANARY-RED: gemma non bit-esatta in finestra sana — matrice saltata $(date +%T)"
  echo "== summary: env=$ENVOK canary=$CANARY q4f1=$rc_f1 q4gate=$rc_fg q4ppl=$rc_pq q4rebase=$rb_q q4f2=$rc_f2 q4r2=$rc_r2 q4f3=$rc_f3 q4r3=$rc_r3 q4graphshot=$rc_rx gemmaours=$rc_g gemmagate=$rc_gg gemppl=$rc_pg gemmacmp1=$rc_cg gemmacmp2=$rc_cg2 gptoss=$rc_go gptoss2=$rc_go2 graph60=$r1 gpu=$rc_gp"
  echo "== end $(date +%T)"
  exit 0
fi

# ---- 1') q4 FUSED: ours + gate C8 + ppl + re-base + 2 regressioni ------------
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
python bench/compare_probs.py diff --strict "$GQ/ref_probs.json" \
  "$OUT/f1.json" > "$OUT/diff_q4_fused.log" 2>&1
rc_fg=$?
echo "== q4 diff strict (fused) exit $rc_fg ($(date +%T)) (atteso 0 = gate C8)"
clean "$OUT/diff_q4_fused.log" | tail -n 8
q4ppl pplq4; rc_pq=$?
if [ $((rc_f1 + rc_fg + rc_pq)) -eq 0 ]; then
  cp "$OUT/f1.json" "$GQ/ours_baseline.json"
  grep -a "PPL =" "$OUT/pplq4.log" | tail -n 1 > "$GQ/ours_ppl.txt"
  echo "== q4 golden rebased (FUSED, own-window) $(date +%T)"
  emit "$OUT/f1.json" qwen4exp_baseline.json
  cp "$GQ/ours_ppl.txt" "$OUT/golden_qwen4exp_ppl.txt"
  echo "== golden qwen4exp_ppl.txt: $(cat "$GQ/ours_ppl.txt")"
  rb_q=0
else
  echo "== q4 re-base SKIPPED (gate non verde) $(date +%T)"
fi
# regressione fused #2 e #3 own-window (comandi verbatim regress.sh, solo
# il flag --decode: graph -> fused; deviazione documentata).
q4run f2; rc_f2=$?
q4ppl pplq4b; rc_pq2=$?
python "$OUT/cmp2.py" "$GQ/ours_baseline.json" "$OUT/f2.json" \
  "$GQ/ours_ppl.txt" "$OUT/pplq4b.log" > "$OUT/cmp_q4_f2.log" 2>&1
rc_r2=$?
echo "== q4 regress-fused #2 exit $rc_r2 ($(date +%T)) (atteso 0)"
cat "$OUT/cmp_q4_f2.log"
q4run f3; rc_f3=$?
q4ppl pplq4c; rc_pq3=$?
python "$OUT/cmp2.py" "$GQ/ours_baseline.json" "$OUT/f3.json" \
  "$GQ/ours_ppl.txt" "$OUT/pplq4c.log" > "$OUT/cmp_q4_f3.log" 2>&1
rc_r3=$?
echo "== q4 regress-fused #3 exit $rc_r3 ($(date +%T)) (atteso 0)"
cat "$OUT/cmp_q4_f3.log"
# fotografia: regress.sh qwen4exp nel config attuale esegue il path graph
# (riga 17, "flag inerte" scritto in S26) — ATTESA FAIL, deviazione.
timeout 1800 bash scripts/regress.sh qwen4exp > "$OUT/regress_qwen4exp.log" 2>&1
rc_rx=$?
echo "== regress qwen4exp (GRAPH, fotografia, atteso FAIL) exit $rc_rx ($(date +%T))"
clean "$OUT/regress_qwen4exp.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10

# ---- 2') gpt-oss-120b: regress.sh (golden re-base S27-14) + condizionale ----
timeout 1800 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1
rc_go=$?
echo "== regress gpt-oss-120b exit $rc_go ($(date +%T)) (atteso 0)"
clean "$OUT/regress_gptoss.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10
if [ $rc_go -ne 0 ]; then
  # C10 rossa ⇒ gate C8 own-window; se verde: re-base + regress 2a (S27-14 f.5)
  timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
    bench.compare_probs ours --ref "$GO/ref_probs.json" -m "$MO" \
    --decode fused -o "$OUT/ours_gptoss.json" > "$OUT/ours_gptoss.log" 2>&1
  rc_o=$?
  python bench/compare_probs.py diff --strict "$GO/ref_probs.json" \
    "$OUT/ours_gptoss.json" > "$OUT/diff_gptoss.log" 2>&1
  rc_gog=$?
  timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl \
    -m "$MO" -c 512 --chunks 8 > "$OUT/ppl_gptoss.log" 2>&1
  rc_po=$?
  echo "== gpt-oss ours/gate/ppl exit $rc_o/$rc_gog/$rc_po ($(date +%T)) (attesi 0/0/0)"
  clean "$OUT/diff_gptoss.log" | tail -n 8
  if [ $((rc_o + rc_gog + rc_po)) -eq 0 ]; then
    cp "$OUT/ours_gptoss.json" "$GO/ours_baseline.json"
    grep -a "PPL =" "$OUT/ppl_gptoss.log" | tail -n 1 > "$GO/ours_ppl.txt"
    echo "== gpt-oss golden rebased (own-window) $(date +%T)"
    emit "$OUT/ours_gptoss.json" gpt-oss-120b_baseline.json
    cp "$GO/ours_ppl.txt" "$OUT/golden_gpt-oss-120b_ppl.txt"
    echo "== golden gpt-oss-120b_ppl.txt: $(cat "$GO/ours_ppl.txt")"
    timeout 1800 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss2.log" 2>&1
    rc_go2=$?
    echo "== regress gpt-oss-120b #2 (own-window, atteso 0) exit $rc_go2 ($(date +%T))"
    clean "$OUT/regress_gptoss2.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10
  else
    echo "== gpt-oss re-base SKIPPED (gate non verde) $(date +%T)"
  fi
fi

# ---- 3') velocità graph @ ~60 token (fotografia finale) ----------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --no-profile \
  > "$OUT/graph60.log" 2>&1 &
P1=$!
poll "$P1" "$OUT/meminfo_graph60.log"
wait $P1; r1=$?
echo "== graph60 exit $r1 ($(date +%T)) (atteso 0)"
clean "$OUT/graph60.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|host cache"
clean "$OUT/graph60.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph60.log"

# ---- 4') suite GPU completa — PER ULTIMA (anti-contaminazione) ---------------
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
rc_gp=$?
echo "== pytest gpu exit $rc_gp ($(date +%T)) (atteso 0)"
clean "$OUT/gpu_all.log" | tail -n 2

echo "== summary: env=$ENVOK canary=$CANARY q4f1=$rc_f1 q4gate=$rc_fg q4ppl=$rc_pq q4rebase=$rb_q q4f2=$rc_f2 q4r2=$rc_r2 q4f3=$rc_f3 q4r3=$rc_r3 q4graphshot=$rc_rx gemmaours=$rc_g gemmagate=$rc_gg gemppl=$rc_pg gemmacmp1=$rc_cg gemmacmp2=$rc_cg2 gptoss=$rc_go gptoss2=$rc_go2 graph60=$r1 gpu=$rc_gp"
echo "== end $(date +%T)"
