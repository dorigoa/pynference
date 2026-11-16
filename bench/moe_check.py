#!/usr/bin/env python
"""S15-1: synthetic 32-expert MoE layer (C5) on two GPUs vs a CPU reference.

    torchrun --nproc_per_node=2 bench/moe_check.py

For each type (Q8_0, MXFP4, IQ4_XS; K = FFN = 2880, 2816 for IQ4_XS whose
blocks are 256 wide) and each residency budget (full VRAM vs 0 = everything
in pinned RAM) the layer decodes tokens and prefills a chunk; both are
compared against a numpy/gguf-py f32 reference that sees all 32 experts
(both ranks compute it and must agree after the all_reduce). At budget 0 the
run also measures the H2D miss bandwidth from the store's byte counters.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pynference.moe import ExpertGroup, ExpertStore, MoE
from pynference.parallel import TP
from pynference.quant import make_qweight
from pynference.quant.ref import ref_dequant
from pynference.quant.testing import make_test_blocks

N_EXPERTS, TOP_K = 32, 4
# name, k, ffn, activation (IQ4_XS blocks are 256 wide: 2880 % 256 != 0)
CFGS = [("Q8_0", 2880, 2880, "swiglu"),
        ("MXFP4", 2880, 2880, "swiglu_oai"),
        ("IQ4_XS", 2816, 2816, "swiglu")]
PROJ = {"gate": 0, "up": 1, "down": 2}
E_SHARED = 100  # raw bytes source for the shared expert (not a routed id)
DECODE_TOL, PREFILL_TOL, BUDGET_TOL, RANK_TOL = 5e-4, 3e-2, 1e-5, 1e-5
ACT_STD = 5.0  # target pre-clip gate/up std: swiglu_oai's operating range (±7)


def make_data(qi, qtype, k, ffn, tokens_n, bw_tokens):
    """Router/bias/tokens/bw-token arrays, identical on every rank.

    Token std is scaled per type so the pre-clip gate/up std is ~ACT_STD:
    synthetic MXFP4 weights are heavy-tailed (E8M0 exponents span 2^±10,
    rms_w ~ 1.5e3), so unscaled gate/up would hit ~±2e4 and swiglu_oai would
    saturate at its ±7 limit everywhere; bf16 prefill rounding then flips
    values near zero and diverges ~0.1 from f32 (job S15-3). The router is
    divided by the same factor, so router logits (and the top-k margin) are
    unchanged; the bias keeps its size relative to the logits.
    """
    rng = np.random.default_rng(7777 + qi)
    router = rng.standard_normal((N_EXPERTS, k)).astype(np.float32)
    bias = (0.3 * rng.standard_normal(N_EXPERTS)).astype(np.float32)
    tokens = (0.3 * rng.standard_normal((tokens_n, k))).astype(np.float32)
    bw = (0.3 * rng.standard_normal((bw_tokens, k))).astype(np.float32)
    g = ref_dequant(raw_bytes(qi, qtype, k, ffn, 0, "gate"), qtype)
    rms_w = float(np.sqrt(np.square(g).mean()))
    s = ACT_STD / (0.3 * math.sqrt(k) * rms_w)
    return router / s, bias, (tokens * s).astype(np.float32), (bw * s).astype(np.float32)


def raw_bytes(qi, qtype, k, ffn, e, name):
    n, kk = (ffn, k) if name != "down" else (k, ffn)
    return make_test_blocks(qtype, n, kk, seed=1000003 * qi + 31 * e + PROJ[name])


def route(tokens, router, bias):
    """numpy f32 routing (identical inputs on every rank)."""
    logits = tokens @ router.T + bias
    s = np.sort(logits, axis=1)  # guard: rank-independent top-k in f32
    gap = (s[:, -TOP_K] - s[:, -TOP_K - 1]).min()
    assert gap > 1e-3, f"router margin too small ({gap}) — pick another seed"
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    idx = np.argsort(-p, axis=1)[:, :TOP_K]
    w = np.take_along_axis(p, idx, 1)
    w = (w / w.sum(1, keepdims=True)).astype(np.float32)
    return idx, w


def _swiglu(g, u):
    return g / (1.0 + np.exp(-g)) * u


def _swiglu_oai(g, u, alpha=1.702, limit=7.0010):
    g, u = np.clip(g, -limit, limit), np.clip(u, -limit, limit)
    return g / (1.0 + np.exp(-alpha * g)) * (u + limit)


ACT = {"swiglu": _swiglu, "swiglu_oai": _swiglu_oai}


def reference(qi, qtype, k, ffn, act_name, tokens, router, bias):
    """f32 CPU forward over all experts, one dequantized expert at a time."""
    act = ACT[act_name]
    idx, w = route(tokens, router, bias)
    out = np.zeros((tokens.shape[0], k), np.float32)
    for e in range(N_EXPERTS):
        pos = np.where(idx == e)
        if pos[0].size == 0:
            continue
        rows = pos[0]
        wg = ref_dequant(raw_bytes(qi, qtype, k, ffn, e, "gate"), qtype)
        wu = ref_dequant(raw_bytes(qi, qtype, k, ffn, e, "up"), qtype)
        wd = ref_dequant(raw_bytes(qi, qtype, k, ffn, e, "down"), qtype)
        x = tokens[rows]
        y = act(x @ wg.T, x @ wu.T) @ wd.T
        out[rows] += y * w[rows, pos[1]][:, None]
    sg = ref_dequant(raw_bytes(qi, qtype, k, ffn, E_SHARED, "gate"), qtype)
    su = ref_dequant(raw_bytes(qi, qtype, k, ffn, E_SHARED, "up"), qtype)
    sd = ref_dequant(raw_bytes(qi, qtype, k, ffn, E_SHARED, "down"), qtype)
    out += act(tokens @ sg.T, tokens @ su.T) @ sd.T
    return out


def _rel(a, b):
    b = torch.as_tensor(b, dtype=torch.float32).to(a.device)
    return ((a - b).abs().max() / b.abs().max()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--bw-tokens", type=int, default=64)
    args = ap.parse_args()
    tp = TP()
    fails = []
    for qi, (qtype, k, ffn, act_name) in enumerate(CFGS):
        router_np, bias_np, tokens_np, bw_np = make_data(
            qi, qtype, k, ffn, args.tokens, args.bw_tokens)  # same on every rank

        def raw(e, name, qi=qi, qtype=qtype, k=k, ffn=ffn):
            return raw_bytes(qi, qtype, k, ffn, e, name)

        groups = [ExpertGroup(name, 0, N_EXPERTS,
                              ffn if name != "down" else k,
                              k if name != "down" else ffn, qtype,
                              lambda e, nm=name: raw(e, nm))
                  for name in ("gate", "up", "down")]
        shared = tuple(make_qweight(raw(e, nm), qtype,
                                    ffn if nm != "down" else k,
                                    k if nm != "down" else ffn, tp.device)
                       for e, nm in [(E_SHARED, "gate"), (E_SHARED, "up"),
                                     (E_SHARED, "down")])
        router = torch.from_numpy(router_np).to(tp.device)
        bias = torch.from_numpy(bias_np).to(tp.device)
        tokens = torch.from_numpy(tokens_np).to(tp.device)
        triple = sum(g.slot_bytes for g in groups)

        outs, stores = {}, {}
        for bname, budget in [("full", N_EXPERTS * triple), ("zero", 0)]:
            st = ExpertStore(groups, tp, budget)
            moe = MoE(st, 0, top_k=TOP_K, router_w=router, router_bias=bias,
                      activation=act_name, shared=shared)
            d = torch.stack([moe.decode(tokens[t]) for t in range(args.tokens)])
            p = moe.prefill(tokens)
            outs[bname], stores[bname] = (d, p), st
            tp.barrier()

        ref = reference(qi, qtype, k, ffn, act_name, tokens_np, router_np, bias_np)
        (df, pf), (dz, pz) = outs["full"], outs["zero"]
        checks = [
            ("decode", _rel(df, ref), DECODE_TOL),
            ("prefill", _rel(pf, ref), PREFILL_TOL),
            ("decode zero-vs-full", _rel(dz, df), BUDGET_TOL),
            ("prefill zero-vs-full", _rel(pz, pf), BUDGET_TOL),
        ]
        ag = tp.all_gather(df).view(args.tokens, tp.world, k)
        if tp.world > 1:
            checks.append(("rank agreement",
                           (ag[:, 0] - ag[:, 1]).abs().max().item()
                           / ag.abs().max().item(), RANK_TOL))
        for name, err, tol in checks:
            ok = err < tol
            fails += [] if ok else [f"{qtype} {name}"]
            print(f"RES rank{tp.rank} {qtype:6s} {name:22s} rel={err:.2e} "
                  f"tol={tol:.0e} {'OK' if ok else 'FAIL'}", flush=True)

        if qtype == "Q8_0":  # miss bandwidth, everything in RAM (C5)
            st = stores["zero"]
            st.reset_stats()
            bw = torch.from_numpy(bw_np).to(tp.device)
            t0 = time.perf_counter()
            for t in range(args.bw_tokens):
                moe.decode(bw[t])
            torch.xpu.synchronize()
            dt = time.perf_counter() - t0
            s = st.stats()
            print(f"RES rank{tp.rank} Q8_0 miss bw {s['bytes'] / dt / 1e9:.2f} GB/s "
                  f"({s['bytes'] / 1e6:.0f} MB in {dt:.2f} s, {s['misses']} misses, "
                  f"{s['hits']} hits)", flush=True)

    tp.barrier()
    if fails:
        print(f"FAIL rank{tp.rank}: {fails}", flush=True)
        sys.exit(1)
    print(f"ALL OK rank{tp.rank}", flush=True)


if __name__ == "__main__":
    main()
