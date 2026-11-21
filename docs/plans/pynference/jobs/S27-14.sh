#!/usr/bin/env bash
# job: S27-14
# timeout: 120
# scopo: (S27 passo 5) completare il gate C8 graph mancato in S27-13,
#   isolare la deriva replay-vs-eager (2.5) con un test stesso-processo a
#   stato identico, e chiudere le regressioni C10 con re-base own-window
#   SOLO dove il gate C8 vs llama è verde (regola S17/S18, come S27-9).
#   Novità vs S27-13:
#   - (0) `ours q4 --decode graph` col WRAPPER run_bench.py di regress.sh
#     (S17): S27-13 lo lanciava WITHOUT wrapper, ma bench/compare_probs.py
#     riga 68 costruisce sempre `Tokenizer(GGUFFile)` (Gemma-only) →
#     candidato numero uno al crash post-3s; log completi (`tail -n 60`) e
#     non `tail -n 3` (lezione S27-13: `jobs/S27-13.out/` mai arrivato).
#   - (1) microtest "stato identico, eager vs replay": dal MEDESIMO
#     snapshot post-prefill (delta conv/state, PLE, buffer QSA + n, n_past)
#     runno il passo t0 = greedy(prefill) in eager (×2), graph (capture +
#     replay, e seconda replay), eager post-capture; confronto logits e
#     STATO COMPLETO buffer-per-buffer con righe divergenti → isola il
#     componente della deriva senza dipendere dai dump A/B cross-processo.
#   - (4) confronto gemma con il blocco di regress.sh VERBATIM (cmp.py di
#     S27-13 mascherava: niente problems.append sugli id, niente exit 1).
#   - Le golden ribasate finiscono ANCHE nel log in base64 (gzip), perché
#     `.out` non è sincronizzato in modo affidabile.
# Fasi: (0) ours q4 graph + diff --strict (gate C8 graph) + ppl q4;
#   (1) microtest; (2) suite tests/gpu; (3) se gate q4 verde: re-base
#   golden q4 (graph) + `regress.sh qwen4exp`; (4) gemma4: gate C8 vs
#   llama + C10 equivalenti path nuovo (cmp verbatim) + re-base condizionale
#   con seconda run own-window; (5) gpt-oss: gate C8 (fused) + re-base +
#   `regress.sh gpt-oss-120b`; (6) velocità graph @60 own-window.
# Tutto embedded (regola S27-10: sul PC niente .git e niente scratch/).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
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

emit() { # emit TAG FILE: copia in $OUT/golden_TAG + gzip/base64 nel log
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

# ---- microtest stesso-processo (embedded) ------------------------------------
cat > "$OUT/s2714_mt.py" <<'MTEOF'
"""S27-14: stato identico, eager vs replay (diagnosi 2.5). TP2, un processo.

Per prompt (primi 2 di ref_probs): reset + prefill, snapshot COMPLETO dello
stato su CPU (delta conv/state, PLE, tutti i buffer QSA + n, n_past). Da lì
ogni variante riparte dal MEDESIMO snapshot e fa un solo passo, il token
t0 = greedy(prefill):
  e1, e2 : eager          (pavimento: run eager ripetuti bit-identici?)
  g1     : use_graph=True (capture + replay del primo passo)
  g2     : replay         (grafo già catturato: determinismo della replay?)
  e3     : eager dopo capture (il capture non deve sporcare l'eager)
Confronta top-50 logits e STATO completo (per ogni buffer: max|d| e righe
divergenti) ⇒ isola il componente divergente senza dump cross-processo.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pynference.gguf import GGUFFile  # noqa: E402
from pynference.models import load_model  # noqa: E402
from pynference.parallel import TP  # noqa: E402

QB = ("pool", "idx_keys", "idx_pos", "kv_k", "kv_v")


def grab(model):
    """Stato GPU completo -> CPU (native dtype, confronto bit-esatto)."""
    d = {"_n_past": np.array([int(model.n_past)])}
    for il, (c, s) in sorted(model.delta_cache.items()):
        d[f"d{il}_conv"] = c.detach().cpu()
        d[f"d{il}_state"] = s.detach().cpu()
    for il, st in sorted(model.qsa_cache.items()):
        d[f"q{il}_n"] = st["n"].detach().cpu()
        for k in QB:
            t = st.get(k)
            if torch.is_tensor(t):
                d[f"q{il}_{k}"] = t.detach().cpu()
    if model.ple_state is not None:
        d["ple"] = model.ple_state.detach().cpu()
    return d


def restore(model, d):
    model.n_past = int(d["_n_past"][0])
    for il, (c, s) in sorted(model.delta_cache.items()):
        c.copy_(d[f"d{il}_conv"].to(c.device))
        s.copy_(d[f"d{il}_state"].to(s.device))
    for il, st in sorted(model.qsa_cache.items()):
        st["n"].fill_(int(d[f"q{il}_n"][0]))
        for k in QB:
            t = st.get(k)
            if torch.is_tensor(t):
                t.copy_(d[f"q{il}_{k}"].to(t.device))
    if model.ple_state is not None:
        model.ple_state.copy_(d["ple"].to(model.ple_state.device))


def topk(logits):
    p = torch.softmax(logits.detach().float().double().reshape(-1), -1)
    v, i = p.topk(50)
    return i.cpu().numpy(), v.cpu().numpy()


def cmp_logits(tag, a, b):
    (ia, pa), (ib, pb) = a, b
    dp = float(np.abs(pa - pb).max())
    nd = int((ia != ib).sum())
    print(f"{tag}: greedy {int(ia[0])} vs {int(ib[0])}; top_p max|d| "
          f"{dp:.3e}; top_id {nd} elems differ", flush=True)


def cmp_state(tag, a, b):
    nbad = 0
    for k in sorted(a):
        ta, tb = a[k], b[k]
        if isinstance(ta, np.ndarray) or isinstance(tb, np.ndarray):
            if not np.array_equal(ta, tb):
                print(f"  {tag} {k}: {ta!r} vs {tb!r}")
                nbad += 1
            continue
        if ta.shape != tb.shape:
            print(f"  {tag} {k}: SHAPE {tuple(ta.shape)} vs {tuple(tb.shape)}")
            nbad += 1
            continue
        if torch.equal(ta, tb):
            continue
        fa, fb = ta.float(), tb.float()
        if fa.dim() == 0:
            print(f"  {tag} {k}: {float(fa):g} vs {float(fb):g}")
        else:
            per = (fa - fb).abs().amax(dim=tuple(range(1, fa.dim())))
            nz = (per > 0).nonzero().flatten()
            rows = (f"rows {int(nz[0])}..{int(nz[-1])} ({int(nz.numel())})"
                    if nz.numel() else "-")
            print(f"  {tag} {k}: max|d| {float(per.max()):.3e} {rows}")
        nbad += 1
    print(f"  {tag}: {nbad} buffers differ" if nbad
          else f"  {tag}: state identical", flush=True)


def main():
    model_path, ref_path = sys.argv[1:3]
    tp = TP()
    prompts = [(e["name"], e["prompt"])
               for e in json.load(open(ref_path))][:2]
    model = load_model(GGUFFile(model_path), tp, max_ctx=1024)
    model.fast_decode = True
    for name, ids in prompts:
        model.reset()
        model.use_graph = False
        lp = model.forward(ids)
        sp = grab(model)
        t0 = int(torch.softmax(lp.double(), -1).reshape(-1).argmax())
        if tp.is_main:
            print(f"--- {name}: prefill {len(ids)} tok, n_past={model.n_past}, "
                  f"t0={t0}", flush=True)

        def step(mode):
            restore(model, sp)
            model.use_graph = mode == "g"
            lg = model.forward([t0])
            st = grab(model)
            model.use_graph = False
            return lg, st

        l_e1, s_e1 = step("e")
        l_e2, _ = step("e")
        l_g1, s_g1 = step("g")
        l_g2, _ = step("g")
        l_e3, _ = step("e")
        if tp.is_main:
            cmp_logits(f"{name} e1-vs-e2 (floor eager)",
                       *[topk(x) for x in (l_e1, l_e2)])
            cmp_logits(f"{name} g1-vs-g2 (floor replay)",
                       *[topk(x) for x in (l_g1, l_g2)])
            cmp_logits(f"{name} e1-vs-g1 (REPLAY vs EAGER)",
                       *[topk(x) for x in (l_e1, l_g1)])
            cmp_logits(f"{name} e1-vs-e3 (eager post-capture)",
                       *[topk(x) for x in (l_e1, l_e3)])
            cmp_state(f"{name} eager-vs-replay", s_e1, s_g1)
        del sp, s_e1, s_g1
    tp.close()


main()
MTEOF

# ---- 0) ours q4 graph COL WRAPPER + gate C8 graph + ppl ----------------------
timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
  bench.compare_probs ours --ref "$GQ/ref_probs.json" -m "$MQ" \
  --decode graph -o "$OUT/ours_q4_graph.json" > "$OUT/ours_q4_graph.log" 2>&1
rc_gq=$?
echo "== q4 ours(graph,wrapper) exit $rc_gq ($(date +%T)) (atteso 0)"
clean "$OUT/ours_q4_graph.log" | tail -n 60
clean "$OUT/ours_q4_graph.log" | grep -a -iE 'error|traceback|assert' -A2 \
  | head -n 40
python bench/compare_probs.py diff --strict "$GQ/ref_probs.json" \
  "$OUT/ours_q4_graph.json" > "$OUT/diff_q4_graph.log" 2>&1
rc_dq=$?
echo "== q4 diff strict (graph) exit $rc_dq ($(date +%T)) (atteso 0 = gate C8)"
clean "$OUT/diff_q4_graph.log" | tail -n 8
timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl \
  -m "$MQ" -c 512 --chunks 8 > "$OUT/ppl_q4.log" 2>&1
rc_pq=$?
echo "== q4 ppl exit $rc_pq ($(date +%T)) (atteso 0)"
clean "$OUT/ppl_q4.log" | grep -a "PPL =" | tail -n 1

# ---- 1) microtest stesso-processo -------------------------------------------
timeout 2400 torchrun --nproc_per_node=2 "$OUT/s2714_mt.py" "$MQ" \
  "$GQ/ref_probs.json" > "$OUT/mt.log" 2>&1
rc_mt=$?
echo "== mt exit $rc_mt ($(date +%T)) (atteso 0)"
clean "$OUT/mt.log" | tail -n 90

# ---- 2) suite GPU completa ---------------------------------------------------
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1
rc_gp=$?
echo "== pytest gpu exit $rc_gp ($(date +%T)) (atteso 0)"
clean "$OUT/gpu_all.log" | tail -n 2

# ---- 3) re-base golden q4 (SOLO se gate C8 graph verde) + regress ------------
rb_q=255
if [ $((rc_gq + rc_dq)) -eq 0 ]; then
  cp "$OUT/ours_q4_graph.json" "$GQ/ours_baseline.json"
  grep -a "PPL =" "$OUT/ppl_q4.log" | tail -n 1 > "$GQ/ours_ppl.txt"
  echo "== q4 golden rebased (graph, own-window) $(date +%T)"
  emit "$OUT/ours_q4_graph.json" qwen4exp_baseline.json
  cp "$GQ/ours_ppl.txt" "$OUT/golden_qwen4exp_ppl.txt"
  echo "== golden qwen4exp_ppl.txt: $(cat "$GQ/ours_ppl.txt")"
  rb_q=0
else
  echo "== q4 re-base SKIPPED (gate non verde) $(date +%T)"
fi
timeout 1800 bash scripts/regress.sh qwen4exp > "$OUT/regress_qwen4exp.log" 2>&1
rc_rq=$?
echo "== regress qwen4exp exit $rc_rq ($(date +%T)) (atteso 0)"
clean "$OUT/regress_qwen4exp.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL|greedy|top-10" | tail -n 12

# ---- 4) gemma4: gate C8 vs llama + C10 equivalente (cmp verbatim) ------------
timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
  --ref "$GG/ref_probs.json" -m "$MG" --decode graph -o "$OUT/ours_gemma.json" \
  > "$OUT/ours_gemma.log" 2>&1
rc_g=$?
echo "== gemma ours(graph) exit $rc_g ($(date +%T)) (atteso 0)"
clean "$OUT/ours_gemma.log" | tail -n 60
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
echo "== gemma cmp1 (vs golden 15/11) exit $rc_cg ($(date +%T))"
cat "$OUT/cmp_gemma.log"
rc_cg2=255
if [ $((rc_g + rc_pg + rc_gg)) -eq 0 ] && [ $rc_cg -ne 0 ]; then
  # gate C8 vs llama VERDE e C10 rossa ⇒ deriva fra finestre (codice gemma
  # invariato, non è nel perimetro S27): re-base own-window + 2a run.
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

# ---- 5) gpt-oss-120b: gate C8 (fused) + re-base own-window + regress ---------
timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
  bench.compare_probs ours --ref "$GO/ref_probs.json" -m "$MO" \
  --decode fused -o "$OUT/ours_gptoss.json" > "$OUT/ours_gptoss.log" 2>&1
rc_o=$?
echo "== gpt-oss ours(fused) exit $rc_o ($(date +%T)) (atteso 0)"
clean "$OUT/ours_gptoss.log" | tail -n 60
python bench/compare_probs.py diff --strict "$GO/ref_probs.json" \
  "$OUT/ours_gptoss.json" > "$OUT/diff_gptoss.log" 2>&1
rc_gog=$?
echo "== gpt-oss diff strict exit $rc_gog ($(date +%T)) (atteso 0 = gate C8)"
clean "$OUT/diff_gptoss.log" | tail -n 8
timeout 900 torchrun --nproc_per_node=2 "$OUT/run_bench.py" bench.ppl \
  -m "$MO" -c 512 --chunks 8 > "$OUT/ppl_gptoss.log" 2>&1
rc_po=$?
echo "== gpt-oss ppl exit $rc_po ($(date +%T)) (atteso 0)"
clean "$OUT/ppl_gptoss.log" | grep -a "PPL =" | tail -n 1
rb_g=255
if [ $((rc_o + rc_gog + rc_po)) -eq 0 ]; then
  # gate C8 vs llama VERDE: golden own-window (numeri di S27-13 0,193% =
  # deriva fra finestre, codice invariato dalla re-base S27-9) + regress
  # che runna una 2a volta own-window e confronto con la golden ribasata.
  cp "$OUT/ours_gptoss.json" "$GO/ours_baseline.json"
  grep -a "PPL =" "$OUT/ppl_gptoss.log" | tail -n 1 > "$GO/ours_ppl.txt"
  echo "== gpt-oss golden rebased (own-window) $(date +%T)"
  emit "$OUT/ours_gptoss.json" gpt-oss-120b_baseline.json
  cp "$GO/ours_ppl.txt" "$OUT/golden_gpt-oss-120b_ppl.txt"
  echo "== golden gpt-oss-120b_ppl.txt: $(cat "$GO/ours_ppl.txt")"
  rb_g=0
else
  echo "== gpt-oss re-base SKIPPED (gate non verde) $(date +%T)"
fi
timeout 1800 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1
rc_go=$?
echo "== regress gpt-oss-120b exit $rc_go ($(date +%T)) (atteso 0)"
clean "$OUT/regress_gptoss.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10

# ---- 6) velocità graph @ ~60 token (guardia own-window) ----------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --no-profile \
  > "$OUT/graph60.log" 2>&1 &
P1=$!
poll "$P1" "$OUT/meminfo_graph60.log"
wait $P1; r1=$?
echo "== graph60 exit $r1 ($(date +%T))"
clean "$OUT/graph60.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|host cache"
clean "$OUT/graph60.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph60.log"

echo "== summary: q4ours=$rc_gq q4gate=$rc_dq q4ppl=$rc_pq mt=$rc_mt gpu=$rc_gp q4rebase=$rb_q q4exp=$rc_rq gemmaours=$rc_g gemmagate=$rc_gg gemppl=$rc_pg gemmacmp1=$rc_cg gemmacmp2=$rc_cg2 gptours=$rc_o gptgate=$rc_gog gptppl=$rc_po gptrebase=$rb_g gptoss=$rc_go graph60=$r1"
echo "== end $(date +%T)"
