"""CPU tests of our tokenizer/template against the Gemma-4 goldens from job S6-1.

Goldens live in tests/golden/gemma4/ and come from llama.cpp (llama-server on
the Q8_0 model); see bench/llama_golden.py. Tests 1-2 need the GGUF file and
are skipped when it is absent. Test 3 (contract C8) only reads JSON in git.
"""
import importlib.util
import json
import os
import pathlib

import pytest

from pynference.gguf import GGUFFile
from pynference.tokenizer import Tokenizer, chat_prompt

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "gemma4"
MODEL = "/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf"

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

_spec = importlib.util.spec_from_file_location("compare_probs_mod", ROOT / "bench" / "compare_probs.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def load(name):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tok():
    return Tokenizer(GGUFFile(MODEL))


@needs_model
def test_tokenizer_matches_golden(tok):
    for case in load("tokens.json"):
        assert tok.encode(case["text"]) == case["ids"], case["text"]


@needs_model
def test_chat_prompt_matches_template(tok):
    # llama-server was queried with empty kwargs and produced the thinking-mode
    # rendering (system block with \x3c|think|>, no thought channel suffix); that
    # matches our chat_prompt(messages, thinking=True). See report S6 notes.
    for case in load("template.json"):
        ours = chat_prompt(case["messages"], thinking=True)
        assert ours == case["prompt"], case["name"]
        assert tok.encode(ours) == tok.encode(case["prompt"]), case["name"]


def test_ours_baseline_respects_c8():
    ref = load("ref_probs.json")
    ours = load("ours_baseline.json")
    assert cp.strict_problems(ref, ours) == []
