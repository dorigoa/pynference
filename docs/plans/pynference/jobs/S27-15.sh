#!/usr/bin/env bash
# job: S27-15
# timeout: 90
# S27-15 — batteria di determinismo (diagnosi 2.5, esiti S27-14).
# Nessun gate/re-base: solo misurare DOVE nasce la variabilità run-to-run.
# (0) mt2: microtest S27-14 CORRETTO — tutto il loop dentro
#     torch.inference_mode() (forward/decode q4 sono @torch.inference_mode,
#     i buffer nati in forward sono tensori inference: S27-14 crashava nel
#     restore fuori moda). Misure: e1-vs-e2 pavimento eager; g1-vs-g2
#     determinismo replay a stato identico; e1-vs-g1 replay vs eager;
#     e1-vs-e3 eager dopo capture; cmp_state buffer-per-buffer.
# (1) gemma own-window: 2x --decode fused + 2x --decode graph, confronti
#     json-vs-json (cmpjj.py = blocco di regress.sh senza ppl): fused1-vs-
#     fused2, graph1-vs-graph2, fused1-vs-graph1.
# (2) q4: 2x ours graph (wrapper S17) + 1x ours fused, stessi confronti.
# Lettura: graph diverge, fused no ⇒ non-determinismo della replay; il
# componente divergente di mt2 dice se è uno stato o il calcolo.
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
# due output `ours` a confronto: greedy/top-10 id identici, max |dp| <= 1e-3.
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

# ---- mt2: microtest S27-14 con il fix InferenceMode --------------------------
cat > "$OUT/mt2.py" <<'MTEOF'
"""S27-15: "stato identico, eager vs replay" (diagnosi 2.5), TP2, un processo.

Identico a S27-14 tranne il FIX: tutto il loop (snapshot/restore/step/confronti)
DENTRO torch.inference_mode(). forward/decode di qwen4exp sono decorati
@torch.inference_mode(), quindi i buffer nati in prefill (delta_cache, statici
QSA, PLE) sono tensori inference e S27-14 crashava nel restore fuori moda.

Per prompt (primi 2 di ref_probs): reset + prefill, snapshot COMPLETO dello
stato su CPU. Da lì ogni variante riparte dal MEDESIMO snapshot e fa un solo
passo, il token t0 = greedy(prefill):
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

# ---- 0) microtest corretto ---------------------------------------------------
timeout 2400 torchrun --nproc_per_node=2 "$OUT/mt2.py" "$M" \
  "$GQ/ref_probs.json" > "$OUT/mt2.log" 2>&1
rc_mt2=$?
echo "== mt2 exit $rc_mt2 ($(date +%T)) (atteso 0)"
clean "$OUT/mt2.log" | grep -a -E '^(---|  )|floor|REPLAY|post-capture|buffers differ|state identical' | head -n 80
clean "$OUT/mt2.log" | grep -a -iE 'error|traceback|assert' -A2 | head -n 30

# ---- 1) gemma: determinismo fused vs graph (own-window) ----------------------
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
gemma_run graph gg1; rc_gg1=$?
gemma_run graph gg2; rc_gg2=$?

python "$OUT/cmpjj.py" "$OUT/gf1.json" "$OUT/gf2.json" \
  > "$OUT/cmp_gf_f.log" 2>&1; rc_c1=$?
echo "== gemma fused1-vs-fused2 exit $rc_c1 ($(date +%T)) (atteso 0 se eager deterministico)"
clean "$OUT/cmp_gf_f.log" | tail -n 8
python "$OUT/cmpjj.py" "$OUT/gg1.json" "$OUT/gg2.json" \
  > "$OUT/cmp_g_g.log" 2>&1; rc_c2=$?
echo "== gemma graph1-vs-graph2 exit $rc_c2 ($(date +%T)) (0 se replay deterministica)"
clean "$OUT/cmp_g_g.log" | tail -n 8
python "$OUT/cmpjj.py" "$OUT/gf1.json" "$OUT/gg1.json" \
  > "$OUT/cmp_f_g.log" 2>&1; rc_c3=$?
echo "== gemma fused1-vs-graph1 exit $rc_c3 ($(date +%T)) (atteso 0)"
clean "$OUT/cmp_f_g.log" | tail -n 8

# ---- 2) q4: determinismo graph vs fused (own-window, wrapper S17) ------------
q4_run() { # q4_run MODE TAG
  timeout 1500 torchrun --nproc_per_node=2 "$OUT/run_bench.py" \
    bench.compare_probs ours --ref "$GQ/ref_probs.json" -m "$M" \
    --decode "$1" -o "$OUT/$2.json" > "$OUT/$2.log" 2>&1
  local rc=$?
  echo "== q4 $2 ($1) exit $rc ($(date +%T)) (atteso 0)"
  clean "$OUT/$2.log" | tail -n 3
  return $rc
}
q4_run graph qG1; rc_qg1=$?
q4_run graph qG2; rc_qg2=$?
q4_run fused qF1; rc_qf1=$?

python "$OUT/cmpjj.py" "$OUT/qG1.json" "$OUT/qG2.json" \
  > "$OUT/cmp_q_g_g.log" 2>&1; rc_c4=$?
echo "== q4 graph1-vs-graph2 exit $rc_c4 ($(date +%T)) (0 se replay deterministica)"
clean "$OUT/cmp_q_g_g.log" | tail -n 8
python "$OUT/cmpjj.py" "$OUT/qF1.json" "$OUT/qG1.json" \
  > "$OUT/cmp_q_f_g.log" 2>&1; rc_c5=$?
echo "== q4 fused1-vs-graph1 exit $rc_c5 ($(date +%T)) (atteso 0 se replay fedele)"
clean "$OUT/cmp_q_f_g.log" | tail -n 8

echo "== summary: mt2=$rc_mt2 gf1=$rc_gf1 gf2=$rc_gf2 gg1=$rc_gg1 gg2=$rc_gg2 c_gff=$rc_c1 c_gg=$rc_c2 c_fg=$rc_c3 qg1=$rc_qg1 qg2=$rc_qg2 qf1=$rc_qf1 c_qgg=$rc_c4 c_qfg=$rc_c5"
echo "== end $(date +%T)"
