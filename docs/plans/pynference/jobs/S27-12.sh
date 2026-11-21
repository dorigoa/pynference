#!/usr/bin/env bash
# job: S27-12
# timeout: 45
# scopo: diagnosi 2.5 della deriva del REPLAY su qwen4exp (S27-11: driver
#   eager e path fused allineati al golden; il confronto a 3 vie sul Mac
#   dice che graph == fused al passo 0 per tutti i prompt e diverge dal
#   passo 1: max|dp| 1.9e-4..2.8e-3). Un solo processo TP2, due fasi:
#   GRAPH prima (fedele a S27-9: reset per prompt, cattura alla prima
#   decode; i token greedy della fase graph vengono registrati) e FUSED
#   dopo (reset+prefill, eager twin, stessi token della fase graph — cosi'
#   gli INPUT restano identici a ogni passo anche dopo il fork greedy).
#   Dopo il prefill e dopo ogni passo (5) dump npz per (mode, prompt):
#   top-50 logits; prima layer QSA: n, pool piena, righe idx_keys/idx_pos/
#   kv_k/kv_v attorno a n; ultimo QSA: n+pool; primo layer delta: conv+
#   state pieni; ultimo delta: slice+norme; PLE state/hist/emb; _res,
#   _ffn_in, _inj[0], _inj[-1]. Diff numpy nel job: per passo, la lista
#   dei componenti con max|d| > 0 esatto. Lettura: se al passo 0 lo STATO
#   e' identico ma i logits n. -> la replay/calcoli in-grafo divergono; se
#   un componente di stato diverge per primo -> residuo del warmup di
#   _capture (che sporca righe QSA e non le ripristina) o ripristino
#   incompleto. Tutto embedded (regola appresa da S27-10: mai git, mai
#   scratch/ nel job).
set -uo pipefail
echo "== start $(date +%T)"
OUT=$PYI_JOB_OUT
mkdir -p "$OUT"
MQ=/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
GQ=tests/golden/qwen4exp

clean() { grep -a -v -E "CCL_WARN|^W1|UserWarning|warn|stage.copy_" "$1"; }

cat > "$OUT/s2712_ab.py" <<'ABEOF'
"""S27-12: A/B fused-vs-graph con snapshot di stato passo-passo (2.5).

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

cat > "$OUT/s2712_diff.py" <<'DIFEOF'
"""S27-12: diff graph-vs-fused passo-passo: per ogni passo la lista dei
componenti con diff > 0 esatto (l'attesa e' bit-identita')."""
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

# ---- A/B fused-vs-graph + diff ------------------------------------------------
timeout 2400 torchrun --nproc_per_node=2 "$OUT/s2712_ab.py" "$MQ" \
  "$GQ/ref_probs.json" "$OUT/snaps" > "$OUT/ab.log" 2>&1
rc_ab=$?
echo "== ab exit $rc_ab ($(date +%T)) (atteso 0)"
clean "$OUT/ab.log" | tail -n 14
python "$OUT/s2712_diff.py" "$OUT/snaps" > "$OUT/diff.log" 2>&1
rc_d=$?
echo "== diff exit $rc_d ($(date +%T))"
cat "$OUT/diff.log"

echo "== summary: ab=$rc_ab diff=$rc_d"
echo "== end $(date +%T)"
