"""S15 step 2: expert ownership, residency plan, SlotMap LRU, pinned pool,
routing, activations and prefill grouping — pure functions, world simulated.
"""
import numpy as np
import pytest
import torch

from pynference.moe import (ExpertGroup, HostPool, MoE, SlotMap, ACTIVATIONS,
                            expert_owner, group_by_expert, local_experts,
                            plan_residency, route_weights, swiglu, swiglu_oai)
from pynference.quant import row_bytes


def _group(n_experts=8, n=4, k=128, qtype="Q8_0", layer=0, name="gate", rng=None):
    rng = rng or np.random.default_rng(0)
    rb = row_bytes(qtype, k)
    store = {e: rng.integers(0, 256, n * rb, dtype=np.uint8)
             for e in range(n_experts)}
    return ExpertGroup(name, layer, n_experts, n, k, qtype, lambda e: store[e])


# ------------------------------------------------------------- ownership

def test_ownership_round_robin():
    assert [expert_owner(e, 2) for e in range(4)] == [0, 1, 0, 1]
    assert local_experts(6, 0, 2) == [0, 2, 4]
    assert local_experts(6, 1, 2) == [1, 3, 5]
    assert local_experts(5, 0, 1) == [0, 1, 2, 3, 4]
    loc = local_experts(32, 1, 2)
    assert len(loc) == 16 and all(e % 2 == 1 for e in loc)


# ------------------------------------------------------------- residency plan

def test_plan_zero_budget_keeps_swap():
    R, X = plan_residency([100, 100], n_local=4, budget_bytes=0, swap_slots=2)
    assert R == [0, 0] and X == [2, 2]  # all in RAM, still runnable


def test_plan_full_budget_no_swap():
    # 2 layers x 4 local experts x 100 + swap 2*2*100 = exactly fits
    R, X = plan_residency([100, 100], n_local=4, budget_bytes=800, swap_slots=2)
    assert R == [4, 4] and X == [0, 0]


def test_plan_partial_spreads_layers_and_fits_budget():
    R, X = plan_residency([100, 100], n_local=4, budget_bytes=700, swap_slots=2)
    assert R == [2, 1] and X == [2, 2]
    used = sum((r + x) * 100 for r, x in zip(R, X))
    assert used == 700
    assert R[1] <= R[0] <= R[1] + 1  # even coverage, expert-major fill


def test_plan_even_distribution_exact():
    # swap first (400), then 300 left fills experts 0..0 of both layers...
    R, X = plan_residency([100, 100], n_local=4, budget_bytes=600, swap_slots=2)
    assert R == [1, 1] and X == [2, 2]
    assert sum((r + x) * 100 for r, x in zip(R, X)) <= 600


def test_plan_refunds_swap_when_layer_complete():
    R, X = plan_residency([100], n_local=2, budget_bytes=400, swap_slots=2)
    assert R == [2] and X == [0]


def test_plan_empty_local():
    R, X = plan_residency([100], n_local=0, budget_bytes=1000, swap_slots=2)
    assert R == [0] and X == [0]


# ------------------------------------------------------------- SlotMap LRU

def test_slotmap_hit_miss_and_lru_order():
    m = SlotMap(resident_ids=[0, 1], swap_slots=2)  # slots 0,1 resident; 2,3 swap
    s, miss = m.resolve([0, 2])
    assert s == [0, 2] and miss == [(2, 2)]
    s, miss = m.resolve([3, 4])  # 3 takes never-used slot 3; 4 then evicts 2
    assert s == [3, 2] and miss == [(3, 3), (4, 2)]
    assert m.expert == [0, 1, 4, 3]
    s, miss = m.resolve([3, 5])  # 3 hit refreshes its ts (now the newest)
    assert s == [3, 2] and miss == [(5, 2)]  # so 5 evicts 2, not 3
    assert m.expert == [0, 1, 5, 3]


def test_slotmap_two_misses_never_share_a_slot():
    m = SlotMap([], swap_slots=3)
    s, miss = m.resolve([7, 8, 9])
    assert len(set(s)) == 3 and len(miss) == 3
    assert [e for e, _ in miss] == [7, 8, 9]


def test_slotmap_too_many_fresh_misses():
    m = SlotMap([], swap_slots=1)
    with pytest.raises(ValueError, match="swap slots"):
        m.resolve([1, 2])


def test_slotmap_resident_never_evicted():
    m = SlotMap(resident_ids=[0], swap_slots=1)
    for e in range(10):
        slots, _ = m.resolve([e])
    assert m.expert[0] == 0


# ------------------------------------------------------------- ExpertGroup / HostPool

def test_group_slot_bytes_real_dims():
    # S15-1 synthetic dims: K = 2880, N = 2880 — but IQ4_XS blocks are 256
    # wide, so its variant uses K = N = 2816 (11 blocks); see report Deviazioni.
    for qtype, k, rb in [("Q8_0", 2880, 3060), ("MXFP4", 2880, 1530),
                         ("IQ4_XS", 2816, 1496)]:
        g = _group(qtype=qtype, n=k, k=k)
        assert g.slot_bytes == k * rb


def test_group_bytes_validation():
    g = _group()
    g.raw_fn = lambda e: np.zeros(7, np.uint8)
    with pytest.raises(ValueError, match="expected"):
        g.expert_bytes(0)


def test_host_pool_rows_and_pin_state():
    g = _group(n_experts=6)
    pool = HostPool(g, [1, 3, 5])
    pool.fill()
    view = pool.buf.view(3, g.slot_bytes)
    assert view[1].numpy().tolist() == list(g.raw_fn(3))
    assert pool.row == {1: 0, 3: 1, 5: 2}
    assert pool.nbytes == 3 * g.slot_bytes
    # GPUs hidden here: the pool stays pageable instead of crashing (moe._pin)
    assert pool.buf.is_pinned() == (torch.xpu.device_count() > 0)


# ------------------------------------------------------------- routing / activation

def test_route_weights_renormalize():
    logits = torch.tensor([2.0, 1.0, 0.0, 3.0, -1.0, 4.0])
    ids, w = route_weights(logits, 3)
    assert ids == [5, 3, 0]
    p = torch.softmax(logits, 0)[[5, 3, 0]]
    assert torch.allclose(torch.tensor(w), (p / p.sum()), atol=1e-6)
    assert abs(sum(w) - 1.0) < 1e-6


def test_swiglu_matches_formula():
    g = torch.linspace(-3, 3, 9)
    u = torch.linspace(2, -2, 9)
    assert torch.allclose(swiglu(g, u), torch.nn.functional.silu(g) * u)


def test_swiglu_oai_clamps_and_scales():
    alpha, limit = 1.702, 7.0010
    g = torch.tensor([-10.0, 0.0, 10.0])
    u = torch.tensor([8.0, -8.0, 0.0])
    got = swiglu_oai(g, u)
    gc, uc = g.clamp(-limit, limit), u.clamp(-limit, limit)
    want = gc * torch.sigmoid(alpha * gc) * (uc + limit)
    assert torch.allclose(got, want)
    assert got[0] < 0 and got[1] == 0  # odd-ish in gate, zero at gate 0
    assert (got <= limit * (2 * limit)).all()
    assert set(ACTIVATIONS) == {"swiglu", "swiglu_oai"}


# ------------------------------------------------------------- prefill grouping

def test_group_by_expert_local_filtering():
    # 3 tokens, top-2; world=2 rank=1 -> local odd experts
    ids = [[1, 2], [3, 1], [0, 3]]
    w = [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]]
    got = group_by_expert(ids, w, local={1, 3})
    assert list(got) == [1, 3]  # ascending expert order
    rows1, w1 = got[1]
    assert rows1.tolist() == [0, 1] and w1.tolist() == pytest.approx([0.6, 0.3])
    rows3, w3 = got[3]
    assert rows3.tolist() == [1, 2] and w3.tolist() == pytest.approx([0.7, 0.5])
    assert rows1.dtype == np.int64 and w1.dtype == np.float32


# ------------------------------------------------------------- MoE construction

def _fake_store():
    from types import SimpleNamespace
    gate = _group(name="gate", n=32)                # [32, 128] (Q8_0: k % 32)
    up = _group(name="up", n=32)                    # [32, 128]
    down = _group(name="down", n=128, k=32)         # [128, 32]
    layers = {0: {"gate": gate, "up": up, "down": down}}
    return SimpleNamespace(tp=SimpleNamespace(rank=0, world=1), layers=layers,
                           local_ids={0: [0, 2, 4, 6]}, maps={}, bufs={})


def _moe(store=None, **kw):
    args = dict(top_k=2, router_w=torch.randn(8, 128), router_bias=torch.randn(8))
    args.update(kw)
    return MoE(store or _fake_store(), 0, **args)


def test_moe_construction_ok():
    m = _moe()
    assert (m.k, m.ffn, m.n_experts, m.top_k) == (128, 32, 8, 2)
    assert m.local == {0, 2, 4, 6}
    g, u = torch.tensor([1.0, -2.0]), torch.tensor([0.5, 3.0])
    assert torch.allclose(_moe(activation="swiglu").act(g, u), swiglu(g, u))
    assert torch.allclose(_moe(activation="swiglu_oai").act(g, u), swiglu_oai(g, u))


def test_moe_construction_rejects_bad_inputs():
    store = _fake_store()
    del store.layers[0]["up"]
    with pytest.raises(ValueError, match="missing expert groups"):
        _moe(store)
    store = _fake_store()
    store.layers[0]["down"] = _group(name="down", n=32, k=128)  # gate-shaped
    with pytest.raises(ValueError, match=r"gate/up must be"):
        _moe(store)
    with pytest.raises(ValueError, match="router_w must be"):
        _moe(router_w=torch.randn(8, 128, dtype=torch.float64))
    with pytest.raises(ValueError, match="router_w must be"):
        _moe(router_w=torch.randn(6, 128))
    with pytest.raises(ValueError, match="router_bias must be"):
        _moe(router_bias=torch.randn(4))


# ------------------------------------------------------------- bench seed guards

def _load_bench():
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("moe_check",
                                                  root / "bench" / "moe_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_seeds_route_unambiguously_and_raw_is_deterministic():
    mc = _load_bench()
    assert [c[0] for c in mc.CFGS] == ["Q8_0", "MXFP4", "IQ4_XS"]
    for qi, (qtype, k, ffn, _act) in enumerate(mc.CFGS):
        rng = np.random.default_rng(7777 + qi)  # same draw order as the bench
        router = rng.standard_normal((mc.N_EXPERTS, k)).astype(np.float32)
        bias = (0.3 * rng.standard_normal(mc.N_EXPERTS)).astype(np.float32)
        toks = (0.3 * rng.standard_normal((32, k))).astype(np.float32)
        idx, w = mc.route(toks, router, bias)  # asserts a safe 4th-vs-5th margin
        assert idx.shape == (32, mc.TOP_K)
        assert w.sum(1).max() < 1.0 + 1e-5
        # all experts get routed to, so every slot path runs in the job
        assert len(set(idx.reshape(-1).tolist())) == mc.N_EXPERTS
        raw = mc.raw_bytes(qi, qtype, k, ffn, 5, "down")
        assert raw.shape == (k, row_bytes(qtype, ffn))
        assert np.array_equal(raw, mc.raw_bytes(qi, qtype, k, ffn, 5, "down"))
