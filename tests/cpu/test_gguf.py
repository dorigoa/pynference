"""CPU tests for pynference.gguf on the real Gemma-4-31B file (skipped if absent)."""
import os
from collections import Counter

import pytest

from pynference.gguf import GGUFFile

MODEL = "/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf"

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")


@pytest.fixture(scope="module")
def g():
    return GGUFFile(MODEL)


def test_arch_and_layers(g):
    assert g.arch == "gemma4"
    assert g.get("block_count") == 60


def test_tensor_count(g):
    assert len(g.tensors) == 833


def test_tensor_types(g):
    assert Counter(t.type for t in g.tensors.values()) == Counter({"F32": 422, "Q8_0": 411})
