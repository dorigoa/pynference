"""S17 step 1: GPT-OSS MoE GGUF wiring (router reshape, expert row slicing,
per-expert biases, MoE construction) on pure numpy/torch, no GPUs.

Runs on the PC where /nvme/gguf holds the real file; skipped elsewhere.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pynference.gguf import GGUFFile
from pynference.moe import MoE
from pynference.models.gptoss import moe_expert_bias, moe_expert_groups, moe_router

MODEL = os.environ.get("GPTOSS_GGUF", "/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf")

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="gpt-oss GGUF not present")

E, N, K = 128, 2880, 2880  # facts.json: experts, ffn rows, hidden/inputs


@pytest.fixture(scope="module")
def g():
    f = GGUFFile(MODEL)
    yield f
    f.close()


def test_expert_groups_match_gguf(g):
    groups = moe_expert_groups(g)
    assert len(groups) == 36 * 3
    assert sorted({gr.name for gr in groups}) == ["down", "gate", "up"]
    gr = groups[0]
    assert (gr.layer, gr.n_experts, gr.n, gr.k, gr.qtype) == (0, E, N, K, "MXFP4")
    assert gr.slot_bytes == N * 1530  # MXFP4: 2880/32*17 = 1530 B per row
    # expert 7 bytes == the flat slice [7*slot, 8*slot) of the fused tensor
    raw = g.raw("blk.0.ffn_gate_exps.weight")
    assert np.array_equal(gr.expert_bytes(7),
                          raw[7 * gr.slot_bytes:8 * gr.slot_bytes])
    # rows() slice == the same bytes (rows = outputs, expert-major)
    rows = g.rows("blk.0.ffn_gate_exps.weight", 7 * N, 8 * N)
    assert rows.shape == (N, 1530)
    assert np.array_equal(rows.reshape(-1), gr.expert_bytes(7))


def test_router_shapes_and_element_order(g):
    w, b = moe_router(g, 3)
    assert w.shape == (E, K) and w.dtype == torch.float32
    assert b.shape == (E,) and b.dtype == torch.float32
    # ggml order [K, E]: flat index h + e*K == router_w[e, h]
    flat = np.frombuffer(bytes(g.raw("blk.3.ffn_gate_inp.weight")), np.uint8).view(np.float32)
    assert w[9, 1234].item() == float(flat[1234 + 9 * K])
    fb = np.frombuffer(bytes(g.raw("blk.3.ffn_gate_inp.bias")), np.uint8).view(np.float32)
    assert b[9].item() == float(fb[9])


def test_expert_bias_shapes_and_order(g):
    gb, ub, db = moe_expert_bias(g, 5)
    for t in (gb, ub, db):
        assert t.shape == (E, N) and t.dtype == torch.float32
    for t, name in ((gb, "gate"), (ub, "up"), (db, "down")):
        flat = np.frombuffer(bytes(g.raw(f"blk.5.ffn_{name}_exps.bias")),
                             np.uint8).view(np.float32)
        assert t[3, 77].item() == float(flat[77 + 3 * N])  # [expert, output row]


def test_moe_construction_real_tensors_cpu(g):
    groups = {gr.name: gr for gr in moe_expert_groups(g)[:3]}
    store = SimpleNamespace(tp=SimpleNamespace(rank=0, world=1), layers={0: groups},
                            local_ids={0: list(range(E))}, maps={}, bufs={})
    w, b = moe_router(g, 0)
    m = MoE(store, 0, top_k=4, router_w=w, router_bias=b,
            activation="swiglu_oai", routing="logits", expert_bias=moe_expert_bias(g, 0))
    assert (m.k, m.ffn, m.n_experts, m.top_k) == (K, N, E, 4)
    assert m.routing == "logits" and m.act.keywords["limit"] == 7.0
    assert m.local == set(range(E))
    # router logits on a fixed input: exact GPT-OSS routing weights sum to 1
    x = torch.zeros(1, K)
    logits = m._logits(x)
    assert logits.shape == (1, E) and torch.allclose(logits[0], b, atol=0)
    from pynference.moe import topk_weights
    wts, idx = topk_weights(logits, 4, "logits")
    assert abs(float(wts[0].sum()) - 1.0) < 1e-6
    assert idx[0].tolist() == torch.topk(b, 4).indices.tolist()
