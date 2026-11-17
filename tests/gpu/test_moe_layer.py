"""S15 step 4: GPU tests for the MoE layer (router + bias, top-k, resolve
batching, indexed GEMVs, swiglu/swiglu_oai, weighted sum, shared expert,
prefill grouping) against a numpy reference built from gguf-py dequantized
weights. World 1 (TP without torchrun), synthetic Q8_0 experts.

Run only inside a GPU job (2.4); skipped when the GPUs are hidden.
"""
import numpy as np
import pytest
import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

from pynference.moe import ExpertGroup, ExpertStore, MoE  # noqa: E402
from pynference.parallel import TP  # noqa: E402
from pynference.quant import make_qweight, row_bytes  # noqa: E402
from pynference.quant.ref import ref_dequant  # noqa: E402
from pynference.quant.testing import make_test_blocks  # noqa: E402

K, FFN, NE, TOPK, T = 128, 64, 8, 3, 6
REL_TOL, BF16_TOL = 5e-4, 3e-2

_rng = np.random.default_rng(11)
ROUTER = _rng.standard_normal((NE, K)).astype(np.float32)
BIAS = (0.5 * _rng.standard_normal(NE)).astype(np.float32)
TOKENS = (0.3 * _rng.standard_normal((T, K))).astype(np.float32)


def _raw(e, n, k, tag):
    return make_test_blocks("Q8_0", n, k, seed=4000 + 13 * tag + e)


def _groups(layer=0):
    return [
        ExpertGroup("gate", layer, NE, FFN, K, "Q8_0",
                    lambda e, l=layer: _raw(e, FFN, K, 10 * l + 1)),
        ExpertGroup("up", layer, NE, FFN, K, "Q8_0",
                    lambda e, l=layer: _raw(e, FFN, K, 10 * l + 2)),
        ExpertGroup("down", layer, NE, K, FFN, "Q8_0",
                    lambda e, l=layer: _raw(e, K, FFN, 10 * l + 3)),
    ]


def _triple_bytes():
    return (FFN * row_bytes("Q8_0", K) * 2 + K * row_bytes("Q8_0", FFN))


def _deq(e, n, k, tag):
    return ref_dequant(_raw(e, n, k, tag), "Q8_0")  # [n, k] f32


def _reference(act):
    """numpy f32 MoE over all experts incl. shared expert and router bias."""
    logits = TOKENS @ ROUTER.T + BIAS
    s = np.sort(logits, axis=1)  # routing must be unambiguous (f32 both sides)
    assert (s[:, -TOPK] - s[:, -TOPK - 1]).min() > 1e-3, "router margin too small"
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    idx = np.argsort(-p, axis=1)[:, :TOPK]
    w = np.take_along_axis(p, idx, 1)
    w /= w.sum(1, keepdims=True)
    out = np.zeros((T, K), np.float32)
    for e in range(NE):
        rows = np.where((idx == e).any(1))[0]
        if not rows.size:
            continue
        wg, wu = _deq(e, FFN, K, 1), _deq(e, FFN, K, 2)
        wd = _deq(e, K, FFN, 3)
        x = TOKENS[rows]
        a = act(x @ wg.T, x @ wu.T)
        for j, t in enumerate(rows):
            out[t] += w[t, int(np.where(idx[t] == e)[0][0])] * (a[j] @ wd.T)
    sg, su = _deq(999, FFN, K, 7), _deq(999, FFN, K, 8)
    sd = _deq(999, K, FFN, 9)
    out += act(TOKENS @ sg.T, TOKENS @ su.T) @ sd.T
    return out


def _swiglu_np(g, u):
    return g / (1.0 + np.exp(-g)) * u


def _swiglu_oai_np(g, u, alpha=1.702, limit=7.0):
    # ggml swiglu_oai_f32 (verified in S17): gate clipped only on the high side, +1 shift
    g, u = np.minimum(g, limit), np.clip(u, -limit, limit)
    return g / (1.0 + np.exp(-alpha * g)) * (u + 1.0)


@pytest.fixture(scope="module")
def tp():
    return TP()


def _moe(tp, budget, activation="swiglu", swap=2):
    st = ExpertStore(_groups(), tp, budget, swap_slots=swap)
    shared = tuple(make_qweight(_raw(999, n, k, tag), "Q8_0", n, k, tp.device)
                   for n, k, tag in [(FFN, K, 7), (FFN, K, 8), (K, FFN, 9)])
    m = MoE(st, 0, top_k=TOPK, router_w=torch.from_numpy(ROUTER).to(tp.device),
            router_bias=torch.from_numpy(BIAS).to(tp.device),
            activation=activation, shared=shared)
    return st, m


def _rel(a, b):
    b = torch.as_tensor(b, dtype=torch.float32).to(a.device)
    return ((a - b).abs().max() / b.abs().max()).item()


def test_decode_vs_cpu_reference(tp):
    _, m = _moe(tp, NE * _triple_bytes())
    ref = _reference(_swiglu_np)
    for t in range(T):
        y = m.decode(torch.from_numpy(TOKENS[t]).to(tp.device))
        assert _rel(y, ref[t]) < REL_TOL, t
    assert m.store.stats()["misses"] == 0  # everything resident


def test_decode_swiglu_oai_vs_cpu_reference(tp):
    _, m = _moe(tp, NE * _triple_bytes(), activation="swiglu_oai")
    ref = _reference(_swiglu_oai_np)
    y = torch.stack([m.decode(torch.from_numpy(TOKENS[t]).to(tp.device))
                     for t in range(T)])
    assert _rel(y, ref) < REL_TOL


def test_zero_budget_batches_misses_and_matches_full(tp):
    st0, m0 = _moe(tp, 0, swap=2)  # all local experts swap in 2-slot blocks
    _, mf = _moe(tp, NE * _triple_bytes())
    dev = tp.device
    y0 = torch.stack([m0.decode(torch.from_numpy(TOKENS[t]).to(dev)) for t in range(T)])
    yf = torch.stack([mf.decode(torch.from_numpy(TOKENS[t]).to(dev)) for t in range(T)])
    assert _rel(y0, yf) < 1e-5  # same bytes, same kernels
    assert st0.stats()["misses"] > 0
    assert _rel(y0, _reference(_swiglu_np)) < REL_TOL


def test_partial_residency_more_misses_than_slots_matches_full(tp):
    # S17-8: with residents AND more misses than swap slots, decode repeated
    # the residents in every miss block (a resident summed 2-3 times); zero
    # budget (above) has no residents and could not see it
    st, m = _moe(tp, 2 * _triple_bytes(), swap=1)
    _, mf = _moe(tp, NE * _triple_bytes())
    dev = tp.device
    res = st.maps[0].resident
    multi = 0
    for t in range(T):
        x = torch.from_numpy(TOKENS[t]).to(dev)
        _, idx = torch.topk(m._logits(x.reshape(1, -1))[0], TOPK)
        ids = idx.tolist()
        multi += any(e in res for e in ids) and sum(e not in res for e in ids) > 1
        assert _rel(m.decode(x), mf.decode(x)) < 1e-5, t
    assert multi > 0, "no token exercised residents + >1 miss with 1 swap slot"


def test_swap_pools_pin_rows_on_demand(tp):
    # S17 (jobs S17-4/6): eager HostPool pinned ~7 GB per rank at load, and
    # lazy whole-pool pinning on the first miss still fed the host OOM
    # killer. The store must pin nothing while building and, after any
    # number of misses, nothing beyond its staging ring (data correctness
    # across reuse: test_zero_budget... + test_store_pinned_ring_bounded).
    st, m = _moe(tp, 0, swap=2)  # budget 0: every local expert swaps in
    assert st.pinned_bytes() == 0
    assert st.pinned_planned_bytes() == NE * _triple_bytes()
    dev = tp.device
    for t in range(T):
        m.decode(torch.from_numpy(TOKENS[t]).to(dev))
    assert st.stats()["misses"] > 0
    assert 0 < st.pinned_bytes() <= st._ring_slots * st._ring_bytes
    assert all(p.allocated_bytes == 0 for p in st.pools.values())


def test_prefill_matches_decode_and_reference(tp):
    _, m = _moe(tp, NE * _triple_bytes())
    p = m.prefill(torch.from_numpy(TOKENS).to(tp.device))
    d = torch.stack([m.decode(torch.from_numpy(TOKENS[t]).to(tp.device))
                     for t in range(T)])
    assert _rel(p, d) < BF16_TOL  # prefill runs bf16 (C2 M>1 path)
    assert _rel(p, _reference(_swiglu_np)) < BF16_TOL


def test_prefill_resolves_each_expert_once(tp):
    st, m = _moe(tp, 0, swap=2)
    logits = TOKENS @ ROUTER.T + BIAS
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    used = set(np.argsort(-p, axis=1)[:, :TOPK].reshape(-1).tolist())
    m.prefill(torch.from_numpy(TOKENS).to(tp.device))
    s = st.stats()
    assert s == {"hits": 0, "misses": len(used),
                 "bytes": len(used) * _triple_bytes()}
