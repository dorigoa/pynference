"""S15 step 2: expert ownership, residency plan, SlotMap LRU, pinned pool,
routing, activations and prefill grouping — pure functions, world simulated.
"""
import numpy as np
import pytest
import torch

from pynference.moe import (ExpertGroup, HostPool, MoE, SlotMap, ACTIVATIONS,
                            expert_owner, group_by_expert, local_experts,
                            plan_residency, route_weights, swiglu, swiglu_oai,
                            topk_weights)
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


def test_host_pool_lazy_pins_rows_on_demand():
    # S17 (job S17-4): the store must not pin the whole swap pool at load
    # time; a lazy pool allocates only on the first expert a miss needs.
    g = _group(n_experts=6)
    pool = HostPool(g, [1, 3, 5], lazy=True)
    assert pool.buf is None and pool.allocated_bytes == 0
    assert pool.nbytes == 3 * g.slot_bytes          # planned, for the C5 check
    row3 = pool.row_bytes(3)
    view = pool.buf.view(3, g.slot_bytes)
    assert pool.allocated_bytes == pool.nbytes      # one alloc for the pool
    assert row3.numpy().tolist() == list(g.raw_fn(3))
    assert view[0].tolist() == [0] * g.slot_bytes   # expert 1 not asked, not loaded
    assert pool.row_bytes(1).numpy().tolist() == list(g.raw_fn(1))
    # eager default unchanged: buffer exists right away
    assert HostPool(g, [1, 3, 5]).buf is not None


def test_host_pool_accepts_2d_raw_fn():
    # quant.testing.make_test_blocks returns [n, row_bytes]; expert_bytes
    # flattens it, so fill keeps the pool view 2D (S15-1 crashed on GPU).
    g = _group(n_experts=6)
    rb = row_bytes(g.qtype, g.k)
    flat_fn = g.raw_fn
    g.raw_fn = lambda e: flat_fn(e).reshape(g.n, rb)
    pool = HostPool(g, [0, 2])
    pool.fill()
    flat = pool.buf.view(2, g.slot_bytes)
    for i, e in enumerate([0, 2]):
        assert flat[i].numpy().tolist() == list(flat_fn(e))


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


def test_swiglu_oai_matches_ggml():
    # S17: exact ggml swiglu_oai_f32 (ops.cpp:3388-3450, llama-graph.cpp:2143-2149):
    # x = min(gate, 7) (high side only), y = clamp(up, -7, 7),
    # out = x / (1 + exp(-1.702*x)) * (y + 1)   (alpha 1.702, limit 7.0)
    alpha, limit = 1.702, 7.0
    g = torch.tensor([-10.0, -3.0, 0.0, 3.0, 10.0])
    u = torch.tensor([9.0, -9.0, 0.0, 8.0, -8.0])
    got = swiglu_oai(g, u)
    gc, uc = g.clamp(max=limit), u.clamp(-limit, limit)
    want = gc / (1.0 + torch.exp(-alpha * gc)) * (uc + 1.0)
    assert torch.allclose(got, want, atol=1e-6)
    # gate clipped only above: out(10, 0) == limit * (0 + 1) * sigmoid(1.702*7)
    # (sigmoid 11.9 is 0.9999933, not exactly 1, in float32: tolerance 1e-4)
    assert abs(float(swiglu_oai(torch.tensor([10.0]), torch.tensor([0.0]))) - limit) < 1e-4
    # gate not clipped below: huge negative gate -> ~0, not -limit (the old S15 form)
    assert abs(float(swiglu_oai(torch.tensor([-100.0]), torch.tensor([0.0])))) < 1e-8
    # up shifted by +1 (not +limit): out(0, u) == 0, out(1, 100) uses y = 7 -> 8x
    assert float(swiglu_oai(torch.tensor([0.0]), torch.tensor([123.0]))) == 0.0
    big = float(swiglu_oai(torch.tensor([1.0]), torch.tensor([100.0])))
    assert abs(big - float(swiglu_oai(torch.tensor([1.0]), torch.tensor([limit])))) < 1e-5
    assert set(ACTIVATIONS) == {"swiglu", "swiglu_oai"}


def test_topk_weights_renorm_matches_route_weights():
    logits = torch.tensor([[2.0, 1.0, 0.0, 3.0, -1.0, 4.0]])
    w, idx = topk_weights(logits, 3, "renorm")
    assert idx[0].tolist() == [5, 3, 0]
    p = torch.softmax(logits, -1)[0, [5, 3, 0]]
    assert torch.allclose(w[0], p / p.sum(), atol=1e-6)
    ids, wl = route_weights(logits[0], 3)
    assert ids == idx[0].tolist() and torch.allclose(torch.tensor(wl), w[0], atol=1e-6)


def test_topk_weights_logits_softmaxes_only_the_topk():
    # GPT-OSS (llama-graph.cpp SOFTMAX_WEIGHT): top-k on the logits, softmax
    # over the k only, no renormalization. Mathematically equivalent to
    # softmax-all + renorm (softmax is order-preserving), but we run the
    # exact llama.cpp op sequence for C8.
    logits = torch.tensor([[2.0, 1.0, 0.0, 3.0, -1.0, 4.0]])
    w, idx = topk_weights(logits, 3, "logits")
    assert idx[0].tolist() == [5, 3, 0]
    assert torch.allclose(w[0], torch.softmax(logits[0, [5, 3, 0]], 0), atol=1e-6)
    w_renorm, _ = topk_weights(logits, 3, "renorm")
    assert torch.allclose(w[0], w_renorm[0], atol=1e-6)  # same up to rounding
    with pytest.raises(ValueError, match="unknown routing"):
        topk_weights(logits, 2, "argmax")


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


def test_moe_routing_and_expert_bias_validation():
    m = _moe(routing="logits", activation="swiglu_oai")
    assert m.routing == "logits" and m.act.keywords["limit"] == 7.0
    with pytest.raises(ValueError, match="unknown routing"):
        _moe(routing="softmax")
    E, ffn, k = 8, 32, 128  # gate/up [E, ffn], down [E, k], float32
    good = (torch.zeros(E, ffn), torch.zeros(E, ffn), torch.zeros(E, k))
    got = _moe(expert_bias=good).eb
    assert all(t is r for t, r in zip(got, good))
    with pytest.raises(ValueError, match="expert_bias must be"):
        _moe(expert_bias=(torch.zeros(E, ffn),) * 3)  # down must be [E, k]
    with pytest.raises(ValueError, match="expert_bias must be"):
        _moe(expert_bias=(torch.zeros(E, ffn), torch.zeros(E, ffn)))
    with pytest.raises(ValueError, match="expert_bias must be"):
        _moe(expert_bias=(torch.zeros(E, ffn, dtype=torch.float16),
                          torch.zeros(E, ffn, dtype=torch.float16),
                          torch.zeros(E, k, dtype=torch.float16)))


# ------------------------------------------------------------- decode batching

class _SlotStore:
    """CPU stand-in for ExpertStore: the real SlotMap, one fp32 weight per slot."""

    def __init__(self, W, resident, swap):
        from types import SimpleNamespace
        self.W = W
        self.tp = SimpleNamespace(rank=0, world=1, all_reduce=lambda y: y)
        self.layers = {0: {"gate": _group(name="gate", n=32), "up": _group(name="up", n=32),
                           "down": _group(name="down", n=128, k=32)}}
        self.local_ids = {0: list(range(8))}
        self.maps = {0: SlotMap(resident, swap)}
        S = len(resident) + swap
        self.bufs = {0: {p: torch.zeros(S, *w.shape[1:]) for p, w in W.items()}}
        for s, e in enumerate(resident):
            for p, w in W.items():
                self.bufs[0][p][s] = w[e]
        self.resolved = []

    def resolve(self, layer, ids):
        self.resolved.append(list(ids))
        slots, misses = self.maps[layer].resolve(list(ids))
        for e, s in misses:
            for p, w in self.W.items():
                self.bufs[layer][p][s] = w[e]
        return torch.tensor(slots, dtype=torch.int32)


def _gemv_indexed_cpu(qtype, w_stack, x, slots, out):
    out.copy_(torch.stack([w_stack[s] @ x[i] for i, s in enumerate(slots.tolist())]))


def test_decode_counts_each_resident_once_with_many_misses(monkeypatch):
    # S17-8: residents were repeated in every block of misses, so with more
    # misses than swap slots a resident expert was summed several times
    import pynference.moe as moe_mod
    monkeypatch.setattr(moe_mod, "gemv_indexed", _gemv_indexed_cpu)
    g = torch.Generator().manual_seed(3)
    W = {"gate": torch.randn(8, 32, 128, generator=g) * 0.1,
         "up": torch.randn(8, 32, 128, generator=g) * 0.1,
         "down": torch.randn(8, 128, 32, generator=g) * 0.1}
    bias = torch.zeros(8)
    bias[0] = 10.0                                  # expert 0 (resident) always picked
    store = _SlotStore(W, resident=[0], swap=1)     # X = 1: 3 misses -> 3 blocks
    m = MoE(store, 0, top_k=4, router_w=torch.randn(8, 128, generator=g) * 0.1,
            router_bias=bias)
    for t in range(4):
        x = torch.randn(128, generator=g)
        w, idx = topk_weights(m._logits(x.reshape(1, -1)), 4)
        ref = sum(wt * (W["down"][e] @ swiglu(W["gate"][e] @ x, W["up"][e] @ x))
                  for wt, e in zip(w[0].tolist(), idx[0].tolist()))
        store.resolved.clear()
        y = m.decode(x)
        assert len(store.resolved) == 3              # [0, miss], [miss], [miss]
        assert sum(ids.count(0) for ids in store.resolved) == 1
        torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-6)


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
        # data from the bench itself (incl. per-type token scaling)
        router, bias, toks, _bw = mc.make_data(qi, qtype, k, ffn, 32, 4)
        idx, w = mc.route(toks, router, bias)  # asserts a safe 4th-vs-5th margin
        assert idx.shape == (32, mc.TOP_K)
        assert w.sum(1).max() < 1.0 + 1e-5
        # all experts get routed to, so every slot path runs in the job
        assert len(set(idx.reshape(-1).tolist())) == mc.N_EXPERTS
        raw = mc.raw_bytes(qi, qtype, k, ffn, 5, "down")
        assert raw.shape == (k, row_bytes(qtype, ffn))
        assert np.array_equal(raw, mc.raw_bytes(qi, qtype, k, ffn, 5, "down"))
