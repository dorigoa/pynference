"""Chat prompt rendering from a model's Jinja chat template (the GGUF's
`tokenizer.chat_template`).

`render` builds the prompt the way Hugging Face's `apply_chat_template` does:
`jinja2.sandbox.ImmutableSandboxedEnvironment` with `trim_blocks` and
`lstrip_blocks`, the `raise_exception` and `strftime_now` globals and the
`tojson` filter, plus the usual `messages` / `add_generation_prompt` /
`tools` context variables.

The rendered text may itself contain the BOS token: callers must tokenise it
with `Tokenizer.encode(..., add_bos=False)` so it is not doubled.
"""
from __future__ import annotations

import datetime
import json
from typing import Any

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _raise_exception(*args) -> None:
    raise jinja2.TemplateError(" ".join(str(a) for a in args))


def _strftime_now(fmt: str) -> str:
    return datetime.datetime.now().strftime(fmt)


def _tojson(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _environment() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    env.filters["tojson"] = _tojson
    return env


def render(g_or_template, messages: list[dict], add_generation_prompt: bool = True,
           **kwargs) -> str:
    """Render `messages` with a chat template: either a `GGUFFile` (its
    `tokenizer.chat_template` and the BOS/EOS token strings are used) or a
    template string. Extra `kwargs` become template variables."""
    if hasattr(g_or_template, "kv"):
        g = g_or_template
        template = g.kv.get("tokenizer.chat_template")
        if not template:
            raise ValueError("this GGUF has no tokenizer.chat_template")
        tokens = list(g.kv.get("tokenizer.ggml.tokens", []))
        ctx: dict[str, Any] = {}
        for var, key in (("bos_token", "tokenizer.ggml.bos_token_id"),
                         ("eos_token", "tokenizer.ggml.eos_token_id")):
            i = g.kv.get(key)
            if i is not None and int(i) < len(tokens):
                ctx[var] = tokens[int(i)]
    else:
        template = g_or_template
        ctx = {}
    ctx.update({"tools": [], "messages": messages,
                "add_generation_prompt": add_generation_prompt})
    ctx.update(kwargs)
    return _environment().from_string(template).render(ctx)
