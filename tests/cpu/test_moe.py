"""S15 step 2: expert ownership, residency plan, SlotMap LRU, pinned pool,
routing, activations and prefill grouping — pure functions, world simulated.
"""
import numpy as np
import pytest
import torch

from pynference.moe import (ExpertGroup, HostPool, SlotMap, ACTIVATIONS,
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
