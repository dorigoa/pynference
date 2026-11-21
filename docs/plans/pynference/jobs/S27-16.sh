#!/usr/bin/env bash
# job: S27-16
# timeout: 60
# S27-16 — batteria di determinismo senza confondenti + mt3 (diagnosi 2.5,
# esiti S27-15). Nessun gate/re-base: solo misurare. Le batterie girano PER
# PRIME (in S27-15 mt2 crasho` prima, e gf1/qG1 potrebbero essere run sporche);
# mt3 per ULTIMO, così un crash non contamina nulla.
# (1) gemma own-window: 3x fused + 2x graph; cmp gf1-gf2, gf2-gf3, gf1-gf3
#     (gf1 outlier di prima-run vs eager intrinsecamente variabile),
#     gg1-gg2, gf1-gg1, gg1-vs-golden, gf1-vs-golden.
# (2) q4 own-window: 2x graph + 2x fused (wrapper S17); cmp qf1-qf2 (manca
#     in S27-15), qg1-qg2, qf1-qg1, qf2-qg2, qf1-qg2, qf1-vs-golden,
#     qg1-vs-golden; righe store per run (set residente uguale?).
# (3) mt3 = mt2 con il fix di restore per st["n"] (tensore 0-dim: copy_ al
#     posto di fill_(int(...[0]))) + cmp_state anche eager-vs-eager e
#     replay-vs-replay. Terza e ultima esecuzione del microtest (2.5).
# Tutto embedded (regola S27-10: sul PC niente .git e niente scratch/).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
M=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
GQ=tests/golden/qwen4exp
MG=/nvme/gguf/gemma-4/gemma-4-31B-it-Q8_0.gguf
GG=tests/golden/gemma4

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

# ---- cmpjj: blocco di scripts/regress.sh (67-113) SENZA la parte ppl --------
# due output `ours` (o ours vs golden) a confronto: greedy/top-10 id identici,
# max |dp| <= 1e-3.
cat > "$OUT/cmpjj.py" <<'JJEOF'
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
JJEOF

# ---- 1) gemma: determinismo fused vs graph, 3 fused per l'outlier -----------
gemma_run() { # gemma_run MODE TAG
  timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
    --ref "$GG/ref_probs.json" -m "$MG" --decode "$1" \
    -o "$OUT/$2.json" > "$OUT/$2.log" 2>&1
  local rc=$?
  echo "== gemma $2 ($1) exit $rc ($(date +%T)) (atteso 0)"
  clean "$OUT/$2.log" | tail -n 3
  return $rc
}
gemma_run fused gf1; rc_gf1=$?
gemma_run fused gf2; rc_gf2=$?
gemma_run fused gf3; rc_gf3=$?
gemma_run graph gg1; rc_gg1=$?
gemma_run graph gg2; rc_gg2=$?

gcmp() { # gcmp TAG A B EXPECT
  python "$OUT/cmpjj.py" "$OUT/$A.json" "$OUT/$B.json" \
    > "$OUT/cmp_$TAG.log" 2>&1
  local rc=$?
  echo "== gemma cmp $TAG exit $rc ($(date +%T)) ($EXPECT)"
  clean "$OUT/cmp_$TAG.log" | tail -n 8
  return $rc
}
gcmp gf12 "$OUT/gf1.json" "$OUT/gf2.json" "0 se eager deterministico"; rc_c1=$?
gcmp gf23 "$OUT/gf2.json" "$OUT/gf3.json" "0 se eager deterministico"; rc_c2=$?
gcmp gf13 "$OUT/gf1.json" "$OUT/gf3.json" "gf1 outlier?"; rc_c3=$?
gcmp gg12 "$OUT/gg1.json" "$OUT/gg2.json" "0 se replay deterministica"; rc_c4=$?
gcmp gf1g1 "$OUT/gf1.json" "$OUT/gg1.json" "fused vs graph"; rc_c5=$?
python "$OUT/cmpjj.py" "$GG/ours_baseline.json" "$OUT/gg1.json" \
  > "$OUT/cmp_gg1gold.log" 2>&1; rc_c6=$?
echo "== gemma gg1-vs-golden exit $rc_c6 ($(date +%T)) (stabilita fra finestre)"
clean "$OUT/cmp_gg1gold.log" | tail -n 8
python "$OUT/cmpjj.py" "$GG/ours_baseline.json" "$OUT/gf1.json" \
  > "$OUT/cmp_gf1gold.log" 2>&1; rc_c7=$?
echo "== gemma gf1-vs-golden exit $rc_c7 ($(date +%T)) (golden su path graph)"
clean "$OUT/cmp_gf1gold.log" | tail -n 8

# ---- 2) q4: determinismo graph vs fused (own-window, wrapper S17) ------------
q4_run() { # q4_run MODE TAG
  timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
    bench.compare_probs ours --ref "$GQ/ref_probs.json" -m "$M" \
    --decode "$1" -o "$OUT/$2.json" > "$OUT/$2.log" 2>&1
  local rc=$?
  echo "== q4 $2 ($1) exit $rc ($(date +%T)) (atteso 0)"
  clean "$OUT/$2.log" | tail -n 3
  clean "$OUT/$2.log" | grep -a -E "store: hits|host cache (ON|OFF)" | tail -n 2
  return $rc
}
q4_run graph qG1; rc_qg1=$?
q4_run graph qG2; rc_qg2=$?
q4_run fused qF1; rc_qf1=$?
q4_run fused qF2; rc_qf2=$?

qcmp() { # qcmp TAG A B EXPECT
  python "$OUT/cmpjj.py" "$OUT/$A.json" "$OUT/$B.json" \
    > "$OUT/cmp_$TAG.log" 2>&1
  local rc=$?
  echo "== q4 cmp $TAG exit $rc ($(date +%T)) ($EXPECT)"
  clean "$OUT/cmp_$TAG.log" | tail -n 8
  return $rc
}
qcmp qf12 "$OUT/qF1.json" "$OUT/qF2.json" "0 se eager q4 deterministico"; rc_q1=$?
qcmp qg12 "$OUT/qG1.json" "$OUT/qG2.json" "0 se replay deterministica"; rc_q2=$?
qcmp qf1g1 "$OUT/qF1.json" "$OUT/qG1.json" "fused vs graph (prima coppie)"; rc_q3=$?
qcmp qf2g2 "$OUT/qF2.json" "$OUT/qG2.json" "fused vs graph (seconde coppie)"; rc_q4=$?
qcmp qf1g2 "$OUT/qF1.json" "$OUT/qG2.json" "incrocio"; rc_q5=$?
python "$OUT/cmpjj.py" "$GQ/ours_baseline.json" "$OUT/qF1.json" \
  > "$OUT/cmp_qf1gold.log" 2>&1; rc_q6=$?
echo "== q4 qF1-vs-golden exit $rc_q6 ($(date +%T)) (golden eager S27-4)"
clean "$OUT/cmp_qf1gold.log" | tail -n 8
python "$OUT/cmpjj.py" "$GQ/ours_baseline.json" "$OUT/qG1.json" \
  > "$OUT/cmp_qg1gold.log" 2>&1; rc_q7=$?
echo "== q4 qG1-vs-golden exit $rc_q7 ($(date +%T)) (golden eager S27-4)"
clean "$OUT/cmp_qg1gold.log" | tail -n 8

# ---- 3) mt3: microtest "stato identico, eager vs replay" (TERZA ed ult.) ----
# Fix rispetto a mt2: restore di st["n"] con copy_ (tensore 0-dim: mt2 usava
# fill_(int(d[...][0])) -> IndexError). In piu` cmp_state e1-vs-e2 e g1-vs-g2.
cat > "$OUT/mt3.py" <<'MTEOF'
"""S27-16: "stato identico, eager vs replay" (diagnosi 2.5), TP2, un processo.

Terza esecuzione (mt crasho` per InferenceMode in S27-14, mt2 per l'indice
[0] su tensore 0-dim in S27-15). Loop completo dentro torch.inference_mode();
restore di st["n"] con copy_.

Per prompt (primi 2 di ref_probs): reset + prefill, snapshot COMPLETO dello
stato su CPU. Da li` ogni variante riparte dal MEDESIMO snapshot e fa un solo
passo, il token t0 = greedy(prefill):
  e1, e2 : eager          (pavimento: run eager ripetuti bit-identici?)
  g1     : use_graph=True (capture + replay del primo passo)
  g2     : replay         (grafo gia` catturato: determinismo della replay?)
  e3     : eager dopo capture (il capture non deve sporcare l'eager)
Confronta top-50 logits e STATO completo (per ogni buffer: max|d| e righe
divergenti): e-vs-e, g-vs-g, e-vs-g.
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
        st["n"].copy_(d[f"q{il}_n"].to(st["n"].device))
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
    with torch.inference_mode():
        for name, ids in prompts:
            model.reset()
            model.use_graph = False
            lp = model.forward(ids)
            sp = grab(model)
            t0 = int(torch.softmax(lp.double(), -1).reshape(-1).argmax())
            if tp.is_main:
                print(f"--- {name}: prefill {len(ids)} tok, n_past="
                      f"{model.n_past}, t0={t0}", flush=True)

            def step(mode):
                restore(model, sp)
                model.use_graph = mode == "g"
                lg = model.forward([t0])
                st = grab(model)
                model.use_graph = False
                return lg, st

            l_e1, s_e1 = step("e")
            l_e2, s_e2 = step("e")
            l_g1, s_g1 = step("g")
            l_g2, s_g2 = step("g")
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
                cmp_state(f"{name} eager-vs-eager", s_e1, s_e2)
                cmp_state(f"{name} replay-vs-replay", s_g1, s_g2)
                cmp_state(f"{name} eager-vs-replay", s_e1, s_g1)
            del sp, s_e1, s_e2, s_g1, s_g2
    tp.close()


main()
MTEOF

timeout 2400 torchrun --nproc_per_node=2 "$OUT/mt3.py" "$M" \
  "$GQ/ref_probs.json" > "$OUT/mt3.log" 2>&1
rc_mt3=$?
echo "== mt3 exit $rc_mt3 ($(date +%T)) (atteso 0)"
clean "$OUT/mt3.log" | grep -a -E '^(---|  )|floor|REPLAY|post-capture|buffers differ|state identical' | head -n 120
clean "$OUT/mt3.log" | grep -a -iE 'error|traceback|assert' -A2 | head -n 30

echo "== summary: gf1=$rc_gf1 gf2=$rc_gf2 gf3=$rc_gf3 gg1=$rc_gg1 gg2=$rc_gg2 c1=$rc_c1 c2=$rc_c2 c3=$rc_c3 c4=$rc_c4 c5=$rc_c5 c6=$rc_c6 c7=$rc_c7 qg1=$rc_qg1 qg2=$rc_qg2 qf1=$rc_qf1 qf2=$rc_qf2 q1=$rc_q1 q2=$rc_q2 q3=$rc_q3 q4=$rc_q4 q5=$rc_q5 q6=$rc_q6 q7=$rc_q7 mt3=$rc_mt3"
echo "== end $(date +%T)"
