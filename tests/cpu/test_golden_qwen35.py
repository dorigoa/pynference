"""CPU tests: BPETokenizer + chat.render against the Qwen3.8-27B goldens.

`tokens.json` / `template.json` / `ref_probs.json` come from job S14-1
(llama-server /tokenize and /apply-template on Qwen3.8-27B-Q8_0). The model
files live only on the PC (/nvme), so these tests skip elsewhere. JSON files
on disk escape the first char of special tokens as \\u003c (plan 2.1);
json.load decodes them back.
"""
import json
import os
import pathlib

import pytest

from pynference.chat import render
from pynference.gguf import GGUFFile
from pynference.tokenizer import BPETokenizer, load_tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "qwen35-27b"
GOLDEN_OSS = ROOT / "tests" / "golden" / "gpt-oss-120b"
QWEN_MODEL = "/nvme/gguf/Qwen3.8-27B-Q8_0.gguf"
OSS_MODEL = "/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf"

needs_qwen = pytest.mark.skipif(not os.path.exists(QWEN_MODEL),
                                reason=f"model file {QWEN_MODEL} not present")
needs_both = pytest.mark.skipif(not (os.path.exists(QWEN_MODEL) and os.path.exists(OSS_MODEL)),
                                reason="both GGUFs (Qwen + GPT-OSS) required")

CASES = json.loads((GOLDEN / "tokens.json").read_text(encoding="utf-8"))
QWEN_TPL = json.loads((GOLDEN / "template.json").read_text(encoding="utf-8"))
OSS_TPL = json.loads((GOLDEN_OSS / "template.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tok():
    g = GGUFFile(QWEN_MODEL)
    t = BPETokenizer(g)
    yield t
    g.close()


@needs_qwen
@pytest.mark.parametrize("case", CASES, ids=[c["text"][:24] for c in CASES])
def test_token_ids_match_golden(tok, case):
    assert tok.encode(case["text"], parse_special=False) == case["ids"]


@needs_qwen
def test_metadata(tok):
    assert tok.pre == "qwen35"
    assert not tok.add_bos
    assert tok.bos_id == 248044 and tok.eos_id == 248046
    assert tok.eog_ids == {248044, 248046}


@needs_qwen
def test_special_tokens_roundtrip(tok):
    text = "Hi<|im_start|>user\nCiao<|im_end|>\n<think>ok</think>"
    ids = tok.encode(text)
    assert tok.special_ids["<|im_start|>"] in ids and tok.special_ids["<|im_end|>"] in ids
    assert tok.special_ids["<think>"] in ids and tok.special_ids["</think>"] in ids
    assert tok.decode(ids, special=True) == text


@needs_qwen
def test_load_tokenizer_selects_bpe():
    g = GGUFFile(QWEN_MODEL)
    try:
        assert isinstance(load_tokenizer(g), BPETokenizer)
    finally:
        g.close()


@needs_qwen
@pytest.mark.parametrize("case", CASES[:6], ids=[c["text"][:24] for c in CASES[:6]])
def test_decode_roundtrip(tok, case):
    assert tok.decode(tok.encode(case["text"], parse_special=False)) == case["text"]


@needs_qwen
@pytest.mark.parametrize("case", QWEN_TPL, ids=[c["name"] for c in QWEN_TPL])
def test_render_matches_llama_server_template(case):
    # llama-server /apply-template does not prepend BOS (add_bos=False for
    # qwen35), and the Qwen template itself does not emit {{ bos_token }}:
    # the rendered prompt must match the golden verbatim.
    g = GGUFFile(QWEN_MODEL)
    try:
        text = render(g, case["messages"], add_generation_prompt=True, **case["kwargs"])
    finally:
        g.close()
    assert text == case["prompt"], f"case {case['name']}"


@needs_qwen
def test_thinking_variants_differ(tok):
    base = next(c for c in QWEN_TPL if c["name"] == "chat-it")
    off = next(c for c in QWEN_TPL if c["name"] == "chat-it-thinking-0")
    assert base["prompt"] == next(c for c in QWEN_TPL if c["name"] == "chat-it-thinking-1")["prompt"]
    # thinking on: the template pre-opens <think> and the model closes it;
    # thinking off: the template emits <think>...</think> right away, so
    # only the OFF prompt carries the closing tag.
    think_end = tok.special_ids["</think>"]
    ids_off = tok.encode(off["prompt"], add_bos=False)
    ids_base = tok.encode(base["prompt"], add_bos=False)
    assert tok.special_ids["<think>"] in ids_off
    assert think_end in ids_off and think_end not in ids_base


@needs_both
@pytest.mark.parametrize("case", OSS_TPL, ids=[c["name"] for c in OSS_TPL])
def test_render_matches_llama_server_template_gpt_oss(case):
    g = GGUFFile(OSS_MODEL)
    try:
        text = render(g, case["messages"], add_generation_prompt=True, **case["kwargs"])
    finally:
        g.close()
    assert text == case["prompt"], f"case {case['name']}"
