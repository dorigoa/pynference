"""CPU tests for the qwen4exp MoE loader (plan S26 step 2): fused expert
groups, router, shared expert with scalar gate, wired into a generic MoE.

The model lives only on the PC (/nvme); the whole file is gated on it. The
loader is checked against tests/golden/qwen4exp/facts.json (the S23 dump
inventory): 48 layers x (gate, up, down), the per-(layer, projection) mixed
types of the "UD" quant, expert e = rows [e*n, (e+1)*n) of the fused
ggml [k, n, E] tensor (same layout as GPT-OSS), router [E, k] row order of
the ggml [k, E] matrix. No kernels run: construction only (make_qweight for
the shared expert just repacks bytes, q8_0_to_device is pure torch).
Routing ("softmax") and shared-gate numerics live in tests/cpu/test_moe.py,
where the MoE is generic. See report S26.
"""
import json
import os
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pynference.gguf import GGUFFile
from pynference.models.qwen4exp import (MOE_FFN, MOE_N_EXPERTS, MOE_TOP_K,
                                        moe_expert_groups, moe_gate_shexp,
                                        moe_layer, moe_router,
                                        moe_shared_expert)
from pynference.moe import MoE
from pynference.quant import row_bytes

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen4exp"
MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL),
                                reason=f"model file {MODEL} not present")

FACTS = json.loads((GOLDEN / "facts.json").read_text(encoding="utf-8"))
N_LAYER = FACTS["metadata"]["qwen4exp.block_count"]


@pytest.fixture(scope="module")
def g():
    f = GGUFFile(MODEL)
    yield f
    f.close()


def test_expert_groups_match_facts_inventory(g):
    groups = moe_expert_groups(g)
    assert len(groups) == N_LAYER * 3
    by_proj = {}
    for gr in groups:
        by_proj.setdefault(gr.name, []).append(gr)
    for proj, want_n, want_k in (("gate", MOE_FFN, 2560), ("up", MOE_FFN, 2560),
                                 ("down", 2560, MOE_FFN)):
        gs = by_proj[proj]
        assert [gr.layer for gr in gs] == list(range(N_LAYER))
        assert all((gr.n, gr.k, gr.n_experts) == (want_n, want_k, MOE_N_EXPERTS)
                   for gr in gs)
        # per-layer quant types are exactly the facts.json list ("UD" mix)
        want_types = FACTS["per_layer_types"][f"ffn_{proj}_exps.weight"]
        assert [gr.qtype for gr in gs] == want_types
        assert all(gr.slot_bytes == want_n * row_bytes(gr.qtype, want_k)
                   for gr in gs)


def test_expert_slice_is_the_fused_row_block(g):
    # expert e of blk.0.ffn_gate_exps.weight = rows [e*n, (e+1)*n) flat
    name = "blk.0.ffn_gate_exps.weight"
    n = g.tensors[name].shape[1]
    gate = {gr.layer: gr for gr in moe_expert_groups(g) if gr.name == "gate"}[0]
    for e in (0, 7, MOE_N_EXPERTS - 1):
        got = gate.raw_fn(e)
        want = g.rows(name, e * n, (e + 1) * n)
        assert np.array_equal(np.asarray(got), want)
    down = {gr.layer: gr for gr in moe_expert_groups(g) if gr.name == "down"}[0]
    nd = g.tensors["blk.0.ffn_down_exps.weight"].shape[1]
    assert np.array_equal(np.asarray(down.raw_fn(3)),
                          g.rows("blk.0.ffn_down_exps.weight", 3 * nd, 4 * nd))


def test_router_is_f32_E_by_k_in_ggml_order(g):
    w = moe_router(g, 0)
    assert w.shape == (MOE_N_EXPERTS, 2560) and w.dtype == torch.float32
    # ggml [k, E]: flat[e*k + j] is router weight (e, j) -> row-major [E, k]
    flat = np.asarray(g.raw("blk.0.ffn_gate_inp.weight")).view(np.float32)
    rng = np.random.default_rng(26)
    es = rng.integers(0, MOE_N_EXPERTS, 32)
    js = rng.integers(0, 2560, 32)
    assert np.array_equal(w[es, js].numpy(), flat[es * 2560 + js])


def test_shared_expert_shapes_types_and_gate(g):
    gate, up, down, gw = moe_shared_expert(g, 0, torch.device("cpu"))
    t = g.tensors["blk.0.ffn_gate_shexp.weight"]
    assert (gate.N, gate.K, gate.qtype) == (MOE_FFN, 2560, t.type)
    assert (up.N, up.K) == (MOE_FFN, 2560)
    assert (down.N, down.K, down.qtype) == (2560, MOE_FFN,
                                            g.tensors["blk.0.ffn_down_shexp.weight"].type)
    assert gw.shape == (2560,) and gw.dtype == torch.float32
    assert torch.equal(gw, moe_gate_shexp(g, 0))


def test_moe_layer_wires_flash_next_routing(g):
    # construction only: a store-shaped namespace over the real groups
    groups = moe_expert_groups(g)
    layers = {}
    for gr in groups:
        layers.setdefault(gr.layer, {})[gr.name] = gr
    store = SimpleNamespace(tp=SimpleNamespace(rank=0, world=1), layers=layers,
                            local_ids={il: list(range(MOE_N_EXPERTS))
                                       for il in range(N_LAYER)},
                            maps={}, bufs={})
    m = moe_layer(g, store, 1, torch.device("cpu"))
    assert isinstance(m, MoE)
    assert (m.k, m.ffn, m.n_experts, m.top_k) == (2560, MOE_FFN, MOE_N_EXPERTS, MOE_TOP_K)
    assert m.routing == "softmax"
    assert m.router_bias is None and m.eb is None  # no biases anywhere
    assert not m._fused_act  # plain swiglu, not swiglu_oai
    assert m.shared is not None and m.shared_gate.shape == (2560,)
    assert m.local == set(range(MOE_N_EXPERTS))
