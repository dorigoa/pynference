"""S15 step 3: GPU tests for ExpertStore (resident buffers, LRU swap slots,
async copies, resolve, stats). Synthetic expert groups, world 1 (TP without
torchrun). Byte equality checks the GGUF layout copy byte for byte.

Run only inside a GPU job (2.4); skipped when the GPUs are hidden.
"""
import numpy as np
import pytest
import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

from pynference.moe import ExpertGroup, ExpertStore
from pynference.parallel import TP
from pynference.quant import row_bytes

K, FFN, N_EXPERTS = 128, 64, 8


def _raw(e: int, n: int, k: int, qtype: str, tag: int) -> np.ndarray:
    rb = row_bytes(qtype, k)
    return np.random.default_rng(9000 + 97 * tag + e).integers(0, 256, n * rb, np.uint8)


def _groups(layer=0):
    return [
        ExpertGroup("gate", layer, N_EXPERTS, FFN, K, "Q8_0",
                    lambda e, l=layer: _raw(e, FFN, K, "Q8_0", 10 * l + 1)),
        ExpertGroup("up", layer, N_EXPERTS, FFN, K, "Q8_0",
                    lambda e, l=layer: _raw(e, FFN, K, "Q8_0", 10 * l + 2)),
        ExpertGroup("down", layer, N_EXPERTS, K, FFN, "Q8_0",
                    lambda e, l=layer: _raw(e, K, FFN, "Q8_0", 10 * l + 3)),
    ]


def _slot_bytes():
    return {"gate": FFN * row_bytes("Q8_0", K), "up": FFN * row_bytes("Q8_0", K),
            "down": K * row_bytes("Q8_0", FFN)}


def _triple_bytes():
    return sum(_slot_bytes().values())


@pytest.fixture(scope="module")
def tp():
    return TP()


def _store(tp, budget, swap=2, layer=0):
    return ExpertStore(_groups(layer), tp, budget, swap_slots=swap)


def test_full_residency_no_miss(tp):
    st = _store(tp, budget=N_EXPERTS * _triple_bytes())
    assert st.memory_bytes() == N_EXPERTS * _triple_bytes()  # R = 8, X = 0
    slots = st.resolve(0, list(range(N_EXPERTS)))
    assert slots.tolist() == list(range(N_EXPERTS)) and st.stats()["misses"] == 0
    for name, tag in [("gate", 1), ("up", 2), ("down", 3)]:
        buf = st.bufs[0][name]
        n, k = (FFN, K) if name != "down" else (K, FFN)
        for e in range(N_EXPERTS):
            want = torch.from_numpy(_raw(e, n, k, "Q8_0", tag)).to(buf.device)
            assert torch.equal(buf[e].reshape(-1), want), (name, e)


def test_zero_budget_lru_and_content(tp):
    st = _store(tp, budget=0, swap=2)
    assert st.maps[0].S == 2 and st.maps[0].R == 0
    st.resolve(0, [0])
    st.resolve(0, [1])
    st.resolve(0, [2])  # evicts 0: slot 0 oldest
    s = st.stats()
    assert s["misses"] == 3 and s["bytes"] == 3 * _triple_bytes() and s["hits"] == 0
    buf = st.bufs[0]["gate"]
    assert torch.equal(buf[0].reshape(-1),
                       torch.from_numpy(_raw(2, FFN, K, "Q8_0", 1)).to(buf.device))
    assert torch.equal(buf[1].reshape(-1),
                       torch.from_numpy(_raw(1, FFN, K, "Q8_0", 1)).to(buf.device))
    assert st.resolve(0, [1]).tolist() == [1]  # swap hit refreshes LRU
    assert st.stats()["hits"] == 1
    st.reset_stats()
    assert st.stats() == {"hits": 0, "misses": 0, "bytes": 0}


def test_partial_residency_hits_and_stats(tp):
    # 3 resident + 1 swap: 4 experts worth of budget
    st = _store(tp, budget=4 * _triple_bytes(), swap=1)
    assert st.maps[0].R == 3 and st.maps[0].S == 4
    assert st.resolve(0, [1, 7]).tolist() == [1, 3]  # 1 resident, 7 -> swap slot 3
    assert st.resolve(0, [7, 1]).tolist() == [3, 1]  # both now loaded, no traffic
    assert st.stats() == {"hits": 3, "misses": 1, "bytes": _triple_bytes()}
    buf = st.bufs[0]["down"]
    assert torch.equal(buf[3].reshape(-1),
                       torch.from_numpy(_raw(7, K, FFN, "Q8_0", 3)).to(buf.device))


def test_too_many_fresh_misses_raises(tp):
    st = _store(tp, budget=0, swap=2)  # only 2 slots exist at once
    with pytest.raises(ValueError, match="swap slots"):
        st.resolve(0, [5, 6, 7])


def test_non_local_id_raises(tp):
    st = _store(tp, budget=0)
    with pytest.raises(ValueError, match="not owned"):
        st.resolve(0, [99])


def test_two_layers_budget_spread(tp):
    groups = _groups(0) + _groups(1)
    t = _triple_bytes()
    st = ExpertStore(groups, tp, 8 * t + 2 * t, swap_slots=1)
    # swap reserved first (2t), 8t left: expert-major fill stays even, 4+4
    assert st.maps[0].R == 4 and st.maps[0].S == 5
    assert st.maps[1].R == 4 and st.maps[1].S == 5
    # S17 ring staging: build pins nothing (eager pools fed the host OOM
    # killer in S17-4; whole-pool pinning on first miss did too in S17-6).
    assert st.pinned_bytes() == 0
    assert st.pinned_planned_bytes() == 8 * t  # 4 non-residents per layer
    for e in (4, 5, 6, 7):  # swap_slots=1: one fresh miss per resolve
        st.resolve(0, [e])
        st.resolve(1, [e])
    ring = st._ring_slots * st._ring_bytes
    assert st.pinned_bytes() == ring  # just the ring, never the pools
    assert all(p.allocated_bytes == 0 for p in st.pools.values())


def test_store_pinned_ring_bounded(tp):
    """Job S17-6: the xe driver holds system RAM for locked host memory, so
    a store must never pin more than its staging ring, no matter how many
    distinct experts the decode misses; bytes must stay exact across ring
    buffer reuse."""
    st = _store(tp, budget=0)  # swap=2: fresh experts keep evicting
    pool = st.pools[0, "gate"]
    assert st.pinned_bytes() == 0 and pool.buf is None
    for e in (2, 3, 4, 5):  # 4 experts x 3 projections = 12 stages > ring
        st.resolve(0, [e])
    ring = st._ring_slots * st._ring_bytes
    assert st.pinned_bytes() == ring
    assert all(b.is_pinned() for b in st._ring)  # inside a GPU job
    assert pool.buf is None and pool.allocated_bytes == 0  # store never pins
    # exact bytes for the last expert after ring wraparound:
    slot = st.maps[0].slot[5]
    for name in ("gate", "up", "down"):
        want = torch.from_numpy(st.layers[0][name].expert_bytes(5))
        got = st.bufs[0][name][slot].reshape(-1)
        torch.xpu.synchronize()  # the side-stream copies must have landed
        assert torch.equal(got.cpu(), want)


def test_host_cache_rows_copy_exact(tp):
    """S27-3: when the host allowance covers the whole non-resident set,
    the store page-locks and fills the pools at load (host cache ON), swap
    misses copy straight from the pinned rows, and the staging ring is
    never allocated."""
    t = _triple_bytes()
    st = ExpertStore(_groups(), tp, 4 * t, swap_slots=4, host_pin_bytes=8 * t)
    assert st.host_cache
    assert st.pinned_bytes() == 8 * t  # all 8 experts are pool rows already
    assert all(p.allocated_bytes == p.nbytes for p in st.pools.values())
    gate = st.pools[0, "gate"]
    assert torch.equal(gate.cached_row(4),
                       torch.from_numpy(st.layers[0]["gate"].expert_bytes(4)))
    slots = [st.resolve(0, [e]).tolist()[0] for e in (4, 5, 6, 7)]
    assert st._ring is None  # host cache path: no ring staging at all
    torch.xpu.synchronize()
    for e, slot in zip((4, 5, 6, 7), slots):
        for name in ("gate", "up", "down"):
            want = torch.from_numpy(st.layers[0][name].expert_bytes(e))
            got = st.bufs[0][name][slot].reshape(-1)
            assert torch.equal(got.cpu(), want), (name, e)


def test_host_cache_off_below_allowance(tp):
    """A too-small allowance keeps the S17 ring behavior: pools untouched,
    misses staged through the ring."""
    st = ExpertStore(_groups(), tp, 0, swap_slots=2, host_pin_bytes=1)
    assert not st.host_cache
    st.resolve(0, [5])
    assert st._ring is not None
    assert all(p.allocated_bytes == 0 for p in st.pools.values())
