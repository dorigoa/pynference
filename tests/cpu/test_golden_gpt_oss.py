"""CPU tests: BPETokenizer (S14) against the GPT-OSS-120B goldens.

`tokens.json` comes from job S13-1 (llama-server /tokenize). The model file
lives only on the PC (/nvme), so the tests that open it skip elsewhere.
JSON files on disk escape the first char of special tokens as \\u003c
(plan 2.1); json.load decodes them back.
"""
import json
import os
import pathlib

import pytest

from pynference.gguf import GGUFFile
from pynference.tokenizer import BPETokenizer, load_tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "gpt-oss-120b"
MODEL = "/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf"

needs_model = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")

CASES = json.loads((GOLDEN / "tokens.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tok():
    g = GGUFFile(MODEL)
    t = BPETokenizer(g)
    yield t
    g.close()


@needs_model
@pytest.mark.parametrize("case", CASES, ids=[c["text"][:24] for c in CASES])
def test_token_ids_match_golden(tok, case):
    assert tok.encode(case["text"], parse_special=False) == case["ids"]


@needs_model
def test_metadata(tok):
    assert tok.pre == "gpt-4o"
    assert not tok.add_bos
    assert tok.bos_id == 199998 and tok.eos_id == 200002
    # harmony: <|return|> (eos) and <|call|> are EOG by text rule,
    # <|end|> is not (llama-vocab.cpp workaround)
    assert tok.eog_ids == {199999, 200002, 200012}


@needs_model
def test_special_tokens_roundtrip(tok):
    text = "<|start|>system<|message|>Hi<|end|>user<|message|>Ciao"
    ids = tok.encode(text)
    assert tok.special_ids["<|start|>"] in ids and tok.special_ids["<|message|>"] in ids
    assert tok.decode(ids, special=True) == text
    for i in ids:  # every emitted token maps back to its exact text
        assert tok.token_bytes(i, special=True) == tok.tokens[i].encode() or tok.types[i] == 1


@needs_model
def test_load_tokenizer_selects_class(tok):
    g = GGUFFile(MODEL)
    try:
        assert isinstance(load_tokenizer(g), BPETokenizer)
    finally:
        g.close()


@needs_model
@pytest.mark.parametrize("case", CASES[:6], ids=[c["text"][:24] for c in CASES[:6]])
def test_decode_roundtrip(tok, case):
    assert tok.decode(tok.encode(case["text"], parse_special=False)) == case["text"]
