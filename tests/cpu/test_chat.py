"""CPU tests for pynference.chat.render against the llama-server golden
(`tests/golden/gemma4/template.json`) and on small inline templates."""
import json
import os

import jinja2
import pytest
from jinja2.exceptions import SecurityError

from pynference.chat import render
from pynference.gguf import GGUFFile

MODEL = "/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf"
GOLDEN = "tests/golden/gemma4/template.json"

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"model file {MODEL} not present")


@pytest.fixture(scope="module")
def gguf() -> GGUFFile:
    return GGUFFile(MODEL)


def test_render_matches_llama_server_golden(gguf):
    # The golden was captured from llama-server with empty kwargs; its output
    # corresponds to thinking enabled (see report S6: the prompt carries the
    # think token and no thought-channel suffix). llama.cpp strips the BOS
    # from the rendered template, we keep it (Hugging Face semantics): the
    # template itself emits {{ bos_token }}, hence add_bos=False when encoding.
    bos = gguf.kv["tokenizer.ggml.tokens"][int(gguf.kv["tokenizer.ggml.bos_token_id"])]
    for case in json.load(open(GOLDEN)):
        kwargs = dict(case["kwargs"])
        kwargs.setdefault("enable_thinking", True)
        text = render(gguf, case["messages"], add_generation_prompt=True, **kwargs)
        assert text == bos + case["prompt"], f"case {case['name']}"
        assert text.count(bos) == 1


def test_template_text_tokenises_without_extra_bos(gguf):
    from pynference.tokenizer import Tokenizer

    tok = Tokenizer(gguf)
    case = json.load(open(GOLDEN))[0]
    text = render(gguf, case["messages"], enable_thinking=True)
    ids = tok.encode(text, add_bos=False)
    assert ids[0] == tok.bos_id                    # the template text brings its own BOS
    assert ids[1] != tok.bos_id                    # exactly one
    assert tok.encode(text)[:2] == [tok.bos_id, tok.bos_id]   # the default would double it


def test_inline_template_context_and_filters():
    out = render("{{ messages[0]['content'] }}|{{ add_generation_prompt }}|{{ tools }}|{{ 'x' | tojson }}",
                 [{"role": "user", "content": "hi"}])
    assert out == "hi|True|[]|\"x\""


def test_inline_template_trim_lstrip_blocks():
    assert render("{% if add_generation_prompt %}\nx\n{% endif %}\n", []) == "x\n"


def test_raise_exception_global():
    with pytest.raises(jinja2.TemplateError, match="boom"):
        render("{{ raise_exception('boom') }}", [])


def test_sandbox_blocks_dunder_access():
    with pytest.raises(SecurityError):
        render("{{ messages.__class__.__mro__ }}", [{"role": "user", "content": "x"}])
