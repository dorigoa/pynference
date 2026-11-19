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
       tensors (QSA per-head parts under the contiguous split) are skipped at
       world>1; everything replicated or all-reduced is compared (QSA heads
       split CONTIGUOUSLY, rank r -> heads [r*12, (r+1)*12) — see
       test_s26_pairing.py). Plus a synthetic 12q x 1kv SDPA probe (the
       world=2 _attend shape) against a plain f32 reference.
  cpu2 TP2-emulation on CPU (S26-11 follow-up): a rank-local model per rank
       r in-process, blk.3's QSA fed the replicated hc_mixed-3 from the
       cpu-replay (no collectives live inside QSA.forward — only its output
       is all-reduced), every rank-local part compared against the
       contiguous slice of the TP1 parts, and rank partials summed and
       compared to the attn_output-3 dump. Localizes an attn_output-3 TP2
       divergence to the split math (FAIL here) or to the GPU regime (PASS
       here).

Prints one NODE line per golden node in dataflow order plus FIRST_DIVERGING.
"""
import json
import os
import pathlib
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
if MODE in ("cpu", "cpu2"):
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
# rank-local at world=2 (heads split contiguously): no global sample alignment
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


# --------------------------------------------------------- TP2 emulation (cpu2)

class TP2:
    """In-process stub of rank `rank` of a world-2 TP group on CPU. QSA has
    no collectives inside its forward (wo's partial is all-reduced outside),
    so the identity all_reduce never sees traffic in the cpu2 path."""
    world, device = 2, torch.device("cpu")

    def __init__(self, rank):
        self.rank = rank
        self.is_main = rank == 0

    def split(self, n):
        return self.rank * (n // 2), n // 2

    @staticmethod
    def all_reduce(t):
        return t

    @staticmethod
    def all_gather(t):
        return t


# parts key -> how the TP1 part maps to rank r's part under the contiguous
# QSA split (rank r -> heads [r*H/world, (r+1)*H/world); kv-heads 2 -> r)
PART_KIND = {"indexer_k_raw": "rep", "indexer_pool_k_new": "rep",
             "indexer_k": "rep", "indexer_q": "rep", "indexer_score": "rep",
             "Qcur_full": "flat2d", "gate_reshaped": "flat2d",
             "attn_pregate": "flat2d", "gate_sigmoid": "flat2d",
             "attn_gated": "flat2d",
             "Qcur_reshaped": "head3d", "Qcur_normed": "head3d",
             "Qcur": "head3d",
             "Kcur": "kvflat", "Vcur": "kvflat",
             "Kcur_normed": "kv3d", "Kcur_rope": "kv3d"}


def slice_ref(t, kind, r, world=2, hd=256, n_head=24):
    T = t.shape[0]
    if kind == "rep":
        return t
    if kind == "flat2d":
        nl = n_head // world
        return t.reshape(T, n_head, -1)[:, r * nl:(r + 1) * nl].reshape(T, -1)
    if kind == "head3d":
        nl = t.shape[1] // world
        return t[:, r * nl:(r + 1) * nl]
    if kind == "kvflat":
        return t.reshape(T, -1, hd)[:, [r]].reshape(T, -1)
    if kind == "kv3d":
        return t[:, [r]]
    raise ValueError(kind)


def dcmp(a, b):
    """(cos, maxabs, fin_bad) over the finite entries only — indexer_score
    carries -inf visibility padding, inf-inf would NaN the whole metric."""
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    if a.numel() != b.numel():
        return None
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    bad = int((fa != fb).sum())
    a, b = a[fa & fb], b[fa & fb]
    if a.numel() < 3:
        return None
    cos = float(a @ b / max(float(a.norm() * b.norm()), 1e-30))
    return cos, float((a - b).abs().max()), bad


def main_cpu2():
    g = GGUFFile(MODEL)
    m1 = load_model(g, TP1(), max_ctx=256)
    m1.moe = [CpuMoE(g, il) for il in range(m1.n_layer)]
    with torch.inference_mode():
        cap1 = replay(m1, None)
    del m1.moe
    # the TP1 blk.3 parts, keyed by golden (name, call) exactly as QSA_ORDER
    ref = {}
    for nm, call, t in cap1:
        ref[(nm, call)] = t
    mixed3 = ref[("hc_mixed-3", 0)]
    T = len(DUMP_TOKENS)
    pos = torch.arange(T, dtype=torch.long)
    partials = {}
    fails = 0
    for r in (0, 1):
        mr = load_model(g, TP2(r), max_ctx=256)
        mr.reset()
        with torch.inference_mode():
            out_r, _, qp = mr.layers[3].qsa.forward(mixed3, pos,
                                                    mr.qsa_cache[3])
        del mr
        for name, call, pkey in QSA_ORDER:
            if pkey == "attn_output":
                continue
            rt = qp[pkey]
            exp = slice_ref(ref[(f"{name}-3", call)], PART_KIND[pkey], r)
            m = dcmp(rt, exp)
            if m is None:
                print(f"PART {pkey} r{r} SKIP shape", flush=True)
                continue
            cos, maxabs, bad = m
            ok = cos >= 0.999 and bad == 0
            fails += not ok
            print(f"PART {pkey} r{r} cos={cos:.6f} maxabs={maxabs:.3e} "
                  f"fin_bad={bad} {'OK' if ok else 'FAIL'}", flush=True)
        partials[r] = out_r
    full = partials[0] + partials[1]
    m = metrics(full, node("attn_output-3", 0))
    if m is None:
        print("SUM attn_output-3 SKIP", flush=True)
        return
    cos, rel, maxabs, bad = m
    ok = cos >= 0.995 and bad == 0
    print(f"SUM attn_output-3 cos={cos:.6f} dsumL1={rel:.2e} "
          f"maxabs={maxabs:.3e} fin_bad={bad} {'OK' if ok else 'FAIL'}",
          flush=True)
    verdict = ("OK (split math verified, divergence is GPU-specific)"
               if ok and not fails else
               "FAIL (split math broken, debug on CPU)")
    print(f"CPU2_VERDICT: {verdict}", flush=True)


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
    if MODE == "cpu2":
        main_cpu2()
        return
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

        # Synthetic 12q x 1kv probe — the _attend shape at world=2 — real
        # (xpu) SDPA backend vs plain f32 attention with modular GQA pairing
        # q-head h -> kv-head h % n_kv. Guards the S26-11 attn_output-3
        # suspect (xpu flash + enable_gqa broadcast) without model weights.
        L3 = model.layers[3].qsa
        dev = model.tp.device
        TT = 5
        gen = torch.Generator().manual_seed(7)
        sq = torch.randn(TT, L3.n_head, L3.head_dim, generator=gen).to(dev)
        sk = torch.randn(TT, L3.n_head_kv, L3.head_dim, generator=gen).to(dev)
        sv = torch.randn(TT, L3.n_head_kv, L3.head_dim, generator=gen).to(dev)
        sm = torch.zeros(TT, TT, device=dev)
        sm[:, -1] = float("-inf")          # drop the last key for all queries
        with torch.inference_mode():
            got = L3._attend(sq, sk, sv, sm).float()
            qh, kh, vh = (t.transpose(0, 1) for t in (sq, sk, sv))
            rep = torch.arange(L3.n_head, device=dev) % L3.n_head_kv
            sc = qh @ kh[rep].transpose(-1, -2) * L3.kq_scale + sm
            ref5 = (sc.softmax(-1) @ vh[rep]).transpose(0, 1).float()
        d = float((got - ref5).abs().max())
        cos5 = float((got.reshape(-1) @ ref5.reshape(-1)) /
                     max(float(got.norm() * ref5.norm()), 1e-30))
        print(f"SDPA_PROBE r{model.tp.rank} shape=({L3.n_head}q,"
              f"{L3.n_head_kv}kv) cos={cos5:.6f} maxabs={d:.3e}", flush=True)

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
