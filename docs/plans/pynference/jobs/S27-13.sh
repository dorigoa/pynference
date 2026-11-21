#!/usr/bin/env bash
# job: S27-13
# timeout: 120
# scopo: (S27 passo 5) validazione del fix a `_capture` (commit bc6b07e:
#   ripristino COMPLETO delle righe QSA sporcate dai 2 passi di warmup:
#   `idx_keys`/`kv_k`/`kv_v` su {n0,n0+1}, righe `pool` del fold,
#   `idx_pos` al sentinel) + batteria finale della sezione.
#   (0) A/B graph-vs-fused con snapshot, script IDENTICI a S27-12 (embedded):
#       atteso "== all steps bit-identical" — con il fix ogni riga torna ai
#       valori pre-warmup, quindi anche le righe assolute > n dei dump
#       matchano e la soglia 0 esatto può passare. Se resta rosso sul solo
#       top_p con stato OK ⇒ la deriva è della replay in sé (kernel
#       registrati vs eager): si apre il test stesso-processo
#       "post-capture eager vs replay".
#   (1) pytest -q tests/gpu suite COMPLETA (2.9.2; atteso 142 passed 7 skipped).
#   (2) gate C8 strict qwen4exp sul path GRAPH + `regress.sh qwen4exp`
#       (golden eager S27-4 installata: PASS ⇒ replay == golden).
#   (3) gemma4 C10: comandi EQUIVALENTI di regress.sh col modello al nuovo
#       percorso /nvme/gguf/gemma-4/ (deviazione documentata in "Deviazioni").
#   (4) regress.sh gpt-oss-120b (moe.py toccato al passo 3; golden re-base
#       in S27-9, atteso PASS).
#   (5,6) velocità finale graph @60/@2000 (--no-profile) per i "Valori".
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

# ---- script A/B + diff + cmp (embedded, identici a S27-12/S27-11) ----------
cat > "$OUT/s2713_ab.py" <<'ABEOF'
"""S27-13 (= S27-12): A/B fused-vs-graph con snapshot di stato passo-passo.

Fase GRAPH prima (fedele a S27-9: reset per prompt, capture alla prima
decode; registra la sequenza greedy), fase FUSED dopo che replaya gli
stessi token: input identici a ogni passo. Dopo prefill ("p") e dopo ogni
passo (0..NSL-1) dump npz su rank0."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pynference.gguf import GGUFFile  # noqa: E402
from pynference.models import load_model  # noqa: E402
from pynference.parallel import TP  # noqa: E402

NSL = 5


def snap(model, logits, path, tp):
    if not tp.is_main:
        return
    d = {}
    p = torch.softmax(logits.detach().float().double().reshape(-1), -1)
    v, i = p.topk(50)
    d["top_id"] = i.cpu().numpy()
    d["top_p"] = v.cpu().numpy()
    qk = sorted(model.qsa_cache)
    dl = sorted(model.delta_cache)
    for tag, il in (("q0", qk[0]), ("qL", qk[-1])):
        st = model.qsa_cache[il]
        n = int(st["n"])
        lo = max(0, n - 8)
        d[tag + "_n"] = np.array([n])
        d[tag + "_pool"] = st["pool"].float().cpu().numpy()
        d[tag + "_idx_keys"] = st["idx_keys"][lo:n + 16].float().cpu().numpy()
        d[tag + "_idx_pos"] = st["idx_pos"][lo:n + 16].cpu().numpy()
        d[tag + "_kv_k"] = st["kv_k"][lo:n + 16, 0].float().cpu().numpy()
        d[tag + "_kv_v"] = st["kv_v"][lo:n + 16, 0].float().cpu().numpy()
    for tag, il in (("d0", dl[0]), ("dL", dl[-1])):
        c, s = model.delta_cache[il]
        d[tag + "_conv"] = c.float().cpu().numpy()
        if tag == "d0" or s.numel() <= (1 << 20):
            d[tag + "_state"] = s.float().cpu().numpy()
        else:
            d[tag + "_state"] = s.reshape(-1)[:1024].float().cpu().numpy()
            d[tag + "_state_norm"] = np.array([float(s.norm()),
                                               float(s.abs().max())])
    if model.ple_state is not None:
        d["ple"] = model.ple_state.float().cpu().numpy()
    if getattr(model, "_res", None) is not None:   # decode buffers: post-first-step
        d["ple_hist"] = np.asarray(model._ple_hist, dtype=np.int64)
        d["ple_emb"] = model._ple_emb.reshape(-1)[:64].float().cpu().numpy()
        d["res"] = model._res.float().cpu().numpy()
        d["ffn_in"] = model._ffn_in.float().cpu().numpy()
        d["inj0"] = model._inj[0].float().cpu().numpy()
        d["injL"] = model._inj[-1].float().cpu().numpy()
    np.savez(path, **d)


def main():
    model_path, ref_path, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    tp = TP()
    prompts = [(e["name"], e["prompt"]) for e in json.load(open(ref_path))]
    model = load_model(GGUFFile(model_path), tp, max_ctx=1024)
    model.fast_decode = True
    if tp.is_main:
        kinds = {}
        for il, L in enumerate(model.layers):
            kinds.setdefault(L.kind, []).append(il)
        print("layers:", {k: (len(v), v[0], v[-1]) for k, v in kinds.items()},
              flush=True)
        q = model.layers[kinds["qsa"][0]].qsa
        print(f"qsa cp={q.compress} idx_dim={q.idx_dim} "
              f"n_head_kv={q.n_head_kv} head_dim={q.head_dim} "
              f"top_pool={q.n_top_pool}", flush=True)
        print(f"delta n_value_l={model.n_value_l} ssm={model.ssm} "
              f"conv_k={model.conv_k} qkv_l={model._qkv_l} "
              f"ple={model.ple_state is not None}", flush=True)
    seqs = {}
    for mode, use_graph in (("graph", True), ("fused", False)):
        model.use_graph = use_graph
        for name, ids in prompts:
            model.reset()
            logits = model.forward(ids)
            snap(model, logits, f"{out_dir}/{mode}_{name}_p.npz", tp)
            toks = []
            for st in range(NSL):
                nxt = (seqs[name][st] if mode == "fused"
                       else int(torch.softmax(logits.double(), -1)
                                .reshape(-1).argmax()))
                if mode == "graph":
                    toks.append(nxt)
                logits = model.forward([nxt])
                snap(model, logits, f"{out_dir}/{mode}_{name}_{st}.npz", tp)
            if mode == "graph":
                seqs[name] = toks
            if tp.is_main:
                print(f"{mode} {name}: done, greedy {toks if mode == 'graph' else seqs[name]}",
                      flush=True)
    tp.close()


main()
ABEOF

cat > "$OUT/s2713_diff.py" <<'DIFEOF'
"""S27-13 (= S27-12): diff graph-vs-fused passo-passo: per ogni passo la lista
dei componenti con diff > 0 esatto (l'attesa dopo il fix è bit-identità)."""
import os
import sys

import numpy as np

out_dir = sys.argv[1]
steps = ["p"] + [str(i) for i in range(5)]
names = sorted({f[len("graph_"):-len("_p.npz")]
                for f in os.listdir(out_dir)
                if f.startswith("graph_") and f.endswith("_p.npz")})
bad_any = 0
for name in names:
    for st in steps:
        gp = f"{out_dir}/graph_{name}_{st}.npz"
        fp = f"{out_dir}/fused_{name}_{st}.npz"
        g = dict(np.load(gp))
        f = dict(np.load(fp))
        bad = []
        for k in g:
            a, b = g[k], f[k]
            if a.shape != b.shape:
                bad.append(f"{k} SHAPE {a.shape} vs {b.shape}")
                continue
            if np.array_equal(a, b):
                continue
            if np.issubdtype(a.dtype, np.floating):
                bad.append(f"{k} max|d| {np.abs(a - b).max():.3e}")
            else:
                n = int((a != b).sum())
                bad.append(f"{k} {n} elems differ"
                           + (f", greedy {a[0]} vs {b[0]}" if k == "top_id"
                              and n else ""))
        if bad:
            bad_any += 1
            print(f"{name} step {st}: " + "; ".join(bad[:8])
                  + (f" (+{len(bad) - 8})" if len(bad) > 8 else ""))
        else:
            print(f"{name} step {st}: OK")
print("== any step diverges" if bad_any else "== all steps bit-identical")
DIFEOF

cat > "$OUT/cmp.py" <<'CMP'
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
CMP

# ---- 0) A/B graph-vs-fused (validazione del fix) ----------------------------
timeout 2400 torchrun --nproc_per_node=2 "$OUT/s2713_ab.py" "$MQ" \
  "$GQ/ref_probs.json" "$OUT/snaps" > "$OUT/ab.log" 2>&1
rc_ab=$?
echo "== ab exit $rc_ab ($(date +%T)) (atteso 0)"
clean "$OUT/ab.log" | tail -n 4
python "$OUT/s2713_diff.py" "$OUT/snaps" > "$OUT/diff.log" 2>&1
rc_d=$?
echo "== diff exit $rc_d ($(date +%T))"
cat "$OUT/diff.log"

# ---- 1) suite GPU completa -------------------------------------------------
timeout 2400 python -m pytest -q tests/gpu > "$OUT/gpu_all.log" 2>&1; rg=$?
echo "== pytest gpu exit $rg ($(date +%T)) (atteso 0)"
clean "$OUT/gpu_all.log" | tail -n 2

# ---- 2) gate C8 strict qwen4exp sul grafo + regress -------------------------
timeout 1500 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
  --ref "$GQ/ref_probs.json" -m "$MQ" --decode graph -o "$OUT/ours_q4_graph.json" \
  > "$OUT/ours_q4_graph.log" 2>&1
rc_gq=$?
echo "== q4 ours(graph) exit $rc_gq ($(date +%T)) (atteso 0)"
clean "$OUT/ours_q4_graph.log" | tail -n 3
python bench/compare_probs.py diff --strict "$GQ/ref_probs.json" \
  "$OUT/ours_q4_graph.json" > "$OUT/diff_q4_graph.log" 2>&1
rc_dq=$?
echo "== q4 diff strict (graph) exit $rc_dq ($(date +%T)) (atteso 0 = gate C8)"
clean "$OUT/diff_q4_graph.log" | tail -n 6
timeout 1800 bash scripts/regress.sh qwen4exp > "$OUT/regress_qwen4exp.log" 2>&1; rq=$?
echo "== regress qwen4exp exit $rq ($(date +%T)) (atteso 0)"
clean "$OUT/regress_qwen4exp.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL|greedy|top-10" | tail -n 12

# ---- 3) gemma4 C10: comandi equivalenti a regress.sh, path nuovo ------------
timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours \
  --ref "$GG/ref_probs.json" -m "$MG" --decode graph -o "$OUT/ours_gemma.json" \
  > "$OUT/ours_gemma.log" 2>&1
rc_g=$?
echo "== gemma ours(graph) exit $rc_g ($(date +%T)) (atteso 0)"
clean "$OUT/ours_gemma.log" | tail -n 3
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

# ---- 4) regress gpt-oss-120b -------------------------------------------------
timeout 1800 bash scripts/regress.sh gpt-oss-120b > "$OUT/regress_gptoss.log" 2>&1; ro=$?
echo "== regress gpt-oss-120b exit $ro ($(date +%T)) (atteso 0)"
clean "$OUT/regress_gptoss.log" | grep -aE "max \|dp\||ppl: baseline|PASS|FAIL" | tail -n 10

# ---- 5) velocità finale graph @ ~60 token -----------------------------------
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

# ---- 6) velocità finale graph @ ~2000 token ----------------------------------
timeout 3000 torchrun --nproc_per_node=2 bench/decode_profile.py \
  -m "$M" --mode graph --ctx 8192 --steps 32 --prompt-tokens 2000 --no-profile \
  > "$OUT/graph2000.log" 2>&1 &
P2=$!
poll "$P2" "$OUT/meminfo_graph2000.log"
wait $P2; r2=$?
echo "== graph2000 exit $r2 ($(date +%T))"
clean "$OUT/graph2000.log" | grep -a -E "^\[r[0-9]\] mem|^\[graph\]|store: hits|host cache"
clean "$OUT/graph2000.log" | grep -a -iE "error|traceback|assert" | head -n 10
summarize "$OUT/meminfo_graph2000.log"

echo "== summary: ab=$rc_ab diff=$rc_d gpu=$rg q4gate=$((rc_gq + rc_dq)) q4exp=$rq gemma=$((rc_g + rc_pg + rc_cg)) gptoss=$ro graph60=$r1 graph2000=$r2"
echo "== end $(date +%T)"
