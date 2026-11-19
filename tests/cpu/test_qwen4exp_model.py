"""S26 step 3 (CPU): the full Qwen4Exp driver (load / forward / decode /
reset / memory_report) on the real GGUF, TP1, a 4-layer cap and the MoE
skipped (PYI_QWEN4EXP_SKIP_MOE — the ExpertStore page-locks real expert RAM,
and the CPU test only checks the dense wiring). Layers 0..2 are DeltaNet
(blk.1 carries PLE), blk.3 is the first compressed QSA layer, so one forward
exercises every block kind, both HC mixers, PLE, the delta state and the QSA
cache. Logits are NOT gold here (experts are zeroed): this checks shape,
finiteness, determinism and context limits. The gold comparison against
llama.cpp --strict and the C8 prefill==decode bitwise check are the GPU job
S26-7. The chain-level math (per-block orientation, TP split) is already
verified in test_qwen4exp_chain.py against the dump.
"""
import os

# The constructor reads these at load time; set them before load_model runs
# so the scaffold stays tiny and CPU-friendly.
os.environ["PYI_QWEN4EXP_LAYERS"] = "4"
os.environ["PYI_QWEN4EXP_SKIP_MOE"] = "1"

import pytest
import torch

from pynference.gguf import GGUFFile
from pynference.models import load_model

MODEL = ("/nvme/Strata-data/models/unsloth-UD-IQ4_XS-v31/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")

needs_model = pytest.mark.skipif(not os.path.exists(MODEL),
                                 reason=f"model file {MODEL} not present")

DUMP_TOKENS = [760, 6511, 314, 9338, 369]   # "The capital of France is" (S23-1)
MAX_CTX = 128


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


@pytest.fixture(scope="module")
def model():
    m = load_model(GGUFFile(MODEL), TP1(), max_ctx=MAX_CTX)
    assert m.n_layer == 4          # env cap applied
    assert m.ple is not None       # blk.1 is inside the cap
    assert m.ple_layer == 1
    return m


@needs_model
def test_prefill_logits(model):
    model.reset()
    lg = model.forward(DUMP_TOKENS)
    assert lg.shape == (model.n_vocab,)
    assert torch.isfinite(lg).all()
    assert model.n_past == len(DUMP_TOKENS)


@needs_model
def test_decode_path(model):
    # Prefill three tokens, then a single-token decode (T == 1 routes the QSA
    # layer through QSA.decode; the delta layers take their replacement state).
    model.reset()
    first = model.forward(DUMP_TOKENS[:3])
    assert first.shape == (model.n_vocab,) and torch.isfinite(first).all()
    step = model.forward([DUMP_TOKENS[3]])
    assert step.shape == (model.n_vocab,) and torch.isfinite(step).all()
    assert model.n_past == 4


@needs_model
def test_cache_kinds(model):
    # blk.3 is the first compressed (QSA) layer; 0/1/2 are DeltaNet.
    assert 3 in model.qsa_cache
    assert 0 in model.delta_cache and 1 in model.delta_cache
    assert 3 not in model.delta_cache


@needs_model
def test_reset_determinism(model):
    model.reset()
    lg1 = model.forward(DUMP_TOKENS)
    model.reset()
    lg2 = model.forward(DUMP_TOKENS)
    assert torch.equal(lg1, lg2)     # C8: prefill is reproducible bit-for-bit


@needs_model
def test_context_limit(model):
    model.reset()
    with pytest.raises(ValueError):
        model.forward(list(range(MAX_CTX + 72)))


@needs_model
def test_memory_report_keys(model):
    rep = model.memory_report()
    assert set(rep) == {"weights", "experts", "cache", "scratch",
                        "pinned_host", "device_total"}
    assert rep["experts"] == 0        # store skipped on CPU
    assert rep["pinned_host"] == 0
    assert rep["cache"] == model.kv_bytes()
    assert rep["device_total"] == (rep["weights"] + rep["experts"]
                                   + rep["cache"] + rep["scratch"])
