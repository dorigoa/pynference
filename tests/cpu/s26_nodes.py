"""S26 step-5 diagnostic: replay the S23-1 dump prompt through the real
Qwen4Exp driver and compare every cb()-dumped node against the llama.cpp
golden dumps (tests/golden/qwen4exp/{dump_nodes,qsa_dump_nodes}.json), to
find the first diverging node (plan 2.5).

Modes (argv[1]):
  cpu  world=1 TP1 stub, 4-layer cap, MoE replaced by the chain-test math
       (dequant only the routed experts) — runs under scripts/cpu.sh.
       Validates the model's loaders and forward wiring (l_last-0/1/2 and
       the blk.3 QSA chain through the real loader, never covered before).
  xpu  real TP() (torchrun, world=2) + real ExpertStore, full 48 layers —
       the exact S26-8 regime, one prefill of the 5 dump tokens. TP-local
       tensors (QSA per-head parts under the parity split) are skipped at
       world>1; everything replicated or all-reduced is compared.

Prints one NODE line per golden node in dataflow order plus FIRST_DIVERGING.
"""
import json
import os
import pathlib
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
if MODE == "cpu":
    os.environ.setdefault("PYI_QWEN4EXP_LAYERS", "4")

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pynference.gguf import GGUFFile
from pynference.models import load_model
from pynference.models.qwen4exp import (PLE_TABLE, hc_combine,
                                        ple_gather_embedding,
                                        ple_row_indices)
from pynference.quant.ref import ref_dequant

MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
GOLDEN = ROOT / "tests" / "golden" / "qwen4exp"
DUMP_TOKENS = [760, 6511, 314, 9338, 369]   # "The capital of France is"

N_EXPS, TOP_K, FFN_EXP, N_EMBD = 512, 10, 640, 2560
TOPK_MIN = 6.103515625e-5                   # 2**-14, llama.cpp renorm floor

NODES = (json.loads((GOLDEN / "dump_nodes.json").read_text())["nodes"]
         + json.loads((GOLDEN / "qsa_dump_nodes.json").read_text())["nodes"])
# dump_nodes.json lists come first so same-name lookups keep the blk.0 file's
# priority; l_last-1/-2 only exist in the qsa file.


def node(name, call=0):
    hits = [n for n in NODES if n["name"] == name]
    return hits[call] if call < len(hits) else None


# golden nodes of the QSA prefill, dataflow order -> QSA.forward parts key
QSA_ORDER = [("indexer_k_raw", 0, "indexer_k_raw"),
             ("indexer_pool_k_new", 0, "indexer_pool_k_new"),
             ("indexer_k", 0, "indexer_k"),
             ("indexer_q", 0, "indexer_q"),
             ("indexer_score", 0, "indexer_score"),
             ("Qcur_full", 0, "Qcur_full"),
             ("Qcur_reshaped", 0, "Qcur_reshaped"),
             ("Qcur_normed", 0, "Qcur_normed"),
             ("Qcur", 0, "Qcur"),
             ("Vcur", 0, "Vcur"),
             ("Kcur", 0, "Kcur"),
             ("Kcur_normed", 0, "Kcur_normed"),
             ("Kcur", 1, "Kcur_rope"),
             ("gate_reshaped", 0, "gate_reshaped"),
             ("attn_pregate", 0, "attn_pregate"),
             ("gate_sigmoid", 0, "gate_sigmoid"),
             ("attn_gated", 0, "attn_gated"),
             ("attn_output", 0, "attn_output")]
# rank-local at world=2 (heads split by parity): no global sample alignment
TP_LOCAL_QSA = {"Qcur_full", "Qcur_reshaped", "Qcur_normed", "Qcur", "Vcur",
                "Kcur", "Kcur_normed", "gate_reshaped", "attn_pregate",
                "gate_sigmoid", "attn_gated"}


def metrics(ours, nd):
    """(cos, rel-sum-vs-L1, max|sample diff|, count of finite-mismatches)."""
    try:
        o = np.asarray(ours.detach().float().cpu().numpy(),
                       dtype=np.float64).reshape(len(nd["rows"]), -1)
    except ValueError:
        return None
    samp = np.array(nd["rows"], dtype=np.float64)
    if samp.shape[1] != o.shape[1]:
        # llama.cpp dumps first3+last3 columns per sampled row (duplicated
        # when the row is narrow); mirror that construction exactly.
        if o.shape[1] < 6:
            return None      # dump row-count/width unalignable with this tensor
        cols = list(range(3)) + list(range(o.shape[1] - 3, o.shape[1]))
        ours_s = o[:, cols]
    else:
        ours_s = o
    fin = np.isfinite(ours_s) & np.isfinite(samp)
    bad = int((np.isfinite(ours_s) != np.isfinite(samp)).sum())
    a, b = ours_s[fin], samp[fin]
    if a.size < 3:
        return None
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    d = float(o[np.isfinite(o)].sum() - np.sum(np.array(
        nd["sum"], dtype=np.float64)))
    rel = abs(d) / max(float(np.abs(o[np.isfinite(o)]).sum()), 1e-9)
    return cos, rel, float(np.abs(a - b).max()), bad


# ------------------------------------------------------------- CPU MoE stub

class CpuMoE:
    """Chain-test _moe math for any layer: softmax top-10 renormalized with
    the 2**-14 floor, plain SwiGLU, scalar-sigmoid-gated shared expert;
    dequantizes only the experts the dump prompt routes to."""

    def __init__(self, g, il):
        self.g, self.il, self._c = g, il, {}

    def _deq(self, name, r0, r1):
        raw = np.ascontiguousarray(self.g.rows(name, r0, r1))
        return torch.from_numpy(ref_dequant(raw, self.g.tensors[name].type))

    def _f32(self, name, shape):
        return torch.from_numpy(
            self.g.raw(name).view(np.float32).reshape(*shape).copy())

    def _expert(self, e):
        key = (self.il, e)
        if key not in self._c:
            il = self.il
            self._c[key] = (
                self._deq(f"blk.{il}.ffn_gate_exps.weight", e * FFN_EXP, (e + 1) * FFN_EXP),
                self._deq(f"blk.{il}.ffn_up_exps.weight", e * FFN_EXP, (e + 1) * FFN_EXP),
                self._deq(f"blk.{il}.ffn_down_exps.weight", e * N_EMBD, (e + 1) * N_EMBD))
        return self._c[key]

    def compute(self, x):
        il = self.il
        T = x.shape[0]
        w_router = self._f32(f"blk.{il}.ffn_gate_inp.weight", (N_EXPS, N_EMBD))
        w_shexp = self._f32(f"blk.{il}.ffn_gate_inp_shexp.weight", (N_EMBD,))
        w_sh_g = self._deq(f"blk.{il}.ffn_gate_shexp.weight", 0, FFN_EXP)
        w_sh_u = self._deq(f"blk.{il}.ffn_up_shexp.weight", 0, FFN_EXP)
        w_sh_d = self._deq(f"blk.{il}.ffn_down_shexp.weight", 0, N_EMBD)
        probs = torch.softmax(x @ w_router.T, dim=-1)
        vals, ids = torch.topk(probs, TOP_K, dim=-1)
        wts = vals / vals.sum(-1, keepdim=True).clamp(min=TOPK_MIN)
        y = torch.zeros(T, N_EMBD)
        for t in range(T):
            xt = x[t]
            for j in range(TOP_K):
                wg, wu, wd = self._expert(int(ids[t, j]))
                act = torch.nn.functional.silu(xt @ wg.T) * (xt @ wu.T)
                y[t] += wts[t, j] * (act @ wd.T)
        shared = (torch.nn.functional.silu(x @ w_sh_g.T) * (x @ w_sh_u.T))
        return y + (shared @ w_sh_d.T) * torch.sigmoid(x @ w_shexp).unsqueeze(-1)

    def prefill(self, x):
        return self.compute(x)

    def decode(self, x):
        return self.compute(x.reshape(1, -1))[0]


class TP1:
    world, rank, is_main = 1, 0, True
    device = torch.device("cpu")

    @staticmethod
    def split(n):
        return 0, n

    @staticmethod
    def all_reduce(t):
        return t

    @staticmethod
    def all_gather(t):
        return t


# ------------------------------------------------------------------- replay

def replay(model, out):
    """One fresh prefill of DUMP_TOKENS mirroring Qwen4Exp.forward, capturing
    every golden node in dataflow order."""
    dev = model.tp.device
    T = len(DUMP_TOKENS)
    cap = []

    ids = torch.tensor(DUMP_TOKENS, dtype=torch.long, device=dev)
    pos = torch.arange(T, dtype=torch.long, device=dev)
    x = model.embed_tokens(ids)
    res = x.unsqueeze(1).expand(T, model.hc_n, model.n_embd).reshape(T, model.hc_dim)
    cap.append(("hc_init", 0, res))

    for il, L in enumerate(model.layers):
        if il > 3:
            break                       # no golden nodes past blk.3
        if model.ple is not None and il == model.ple_layer:
            pids = list(DUMP_TOKENS)
            idx = ple_row_indices(pids, model._ple_hist, **model.ple_cfg)
            emb = ple_gather_embedding(model.g, PLE_TABLE, idx, device=dev)
            nheads = (model.ple_cfg["ngram"] - 1) * model.ple_cfg["heads_per_ngram"]
            emb = emb.view(len(pids), nheads, model.ple_dim).reshape(
                len(pids), model.n_embd)
            res, model.ple_state, pp = model.ple.forward(res, emb, model.ple_state)
            model._ple_hist = np.concatenate(
                [model._ple_hist, np.asarray(pids, dtype=np.int64)])
            for key in ("ple_gate", "ple_gated_value", "ple_conv_out"):
                cap.append((f"{key}-{il}", 0, pp[key]))

        mixed, inj, pa = L.mix_attn.forward(res)
        for key in ("hc_norm", "hc_gate", "hc_mixed", "hc_inject"):
            cap.append((f"{key}-{il}", 0, pa[key]))

        if L.kind == "delta":
            blk = model._delta_block(il, L, mixed)
            if il == 0:
                cap.append(("linear_attn_out-0", 0, blk))
        else:
            out_t, state, qp = L.qsa.forward(mixed, pos, model.qsa_cache[il])
            model.qsa_cache[il] = state
            blk = model.tp.all_reduce(out_t)
            if il == 3:
                for name, call, pkey in QSA_ORDER:
                    cap.append((f"{name}-{il}", call, qp[pkey]))
        res = hc_combine(res, blk, inj)
        if il == 0:
            cap.append(("hc_combine-0", 0, res))

        mixed2, inj2, pb = L.mix_ffn.forward(res)
        ffn = model._ffn_block(il, mixed2)
        if il == 0:
            cap.append(("ffn_out-0", 0, ffn))
        res = hc_combine(res, ffn, inj2)
        cap.append((f"l_last-{il}", 0, res))
    return cap


def main():
    if MODE == "xpu":
        from pynference.parallel import TP
        tp = TP()
        g = GGUFFile(MODEL)
        model = load_model(g, tp, max_ctx=8192)
    else:
        g = GGUFFile(MODEL)
        model = load_model(g, TP1(), max_ctx=256)
        model.moe = [CpuMoE(g, il) for il in range(model.n_layer)]
        print(f"loaded: n_layer={model.n_layer} moe=cpu-stub", flush=True)

    if MODE == "xpu":                      # collective balance: both ranks
        model.reset()
        lg_p = model.forward(list(DUMP_TOKENS))          # 48-layer prefill
        lg_d = model.forward([DUMP_TOKENS[-1]])          # 1-token decode
        if model.tp.rank == 0:
            for tag, lg in (("prefill", lg_p), ("decode", lg_d)):
                fin = torch.isfinite(lg)
                p = torch.softmax(lg.float(), -1)
                top = torch.topk(p, 5)
                print(f"PROBE {tag}: finite={int(fin.sum())}/{lg.numel()} "
                      f"top5={[(int(i), round(float(v), 4)) for v, i in
                               zip(top.values, top.indices)]}", flush=True)
        model.reset()                     # fresh caches for the node replay

    # Production regime: Qwen4Exp.forward carries @torch.inference_mode(), so
    # the lazily created TP._buf collective buffers are inference tensors; the
    # probe forward above created them. replay() calls the submodules directly
    # and must hold the same mode (S26-10: all_reduce's b.copy_ on an
    # inference tensor raised outside InferenceMode).
    with torch.inference_mode():
        cap = replay(model, None)
    if model.tp.rank != 0:
        return                            # rank 0 prints the comparison
    first = None
    n_cmp = 0
    for name, call, t in cap:
        base = name.rsplit("-", 1)[0] if not name.endswith("init") else name
        nd = node(name, call)
        if nd is None:
            print(f"NODE {name}#{call} SKIP no-golden", flush=True)
            continue
        w = model.tp.world
        if w > 1 and base in TP_LOCAL_QSA:
            print(f"NODE {name}#{call} SKIP tp-local", flush=True)
            continue
        m = metrics(t, nd)
        if m is None:
            print(f"NODE {name}#{call} SKIP shape/empty", flush=True)
            continue
        cos, rel, maxabs, bad = m
        n_cmp += 1
        ok = cos >= 0.995 and bad == 0
        if not ok and first is None:
            first = name
        print(f"NODE {name}#{call} cos={cos:.6f} dsumL1={rel:.2e} "
              f"maxabs={maxabs:.3e} fin_bad={bad} {'OK' if ok else 'FAIL'}",
              flush=True)
    print(f"compared {n_cmp} nodes; FIRST_DIVERGING: {first or 'none'}",
          flush=True)


if __name__ == "__main__":
    main()
