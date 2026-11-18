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


# ---- reasoning separation over a token stream (S14) ----
#
# The byte-level BPE families keep the model's reasoning inside the answer
# stream: GPT-OSS (harmony) writes it in the `analysis` channel
# (`<|channel|>analysis<|message|>...<channel|>` before the `final` one),
# Qwen types it after the `<think>` the template pre-opens, until `</think>`.
# The tags below are those vocabularies' special tokens; the first char is
# written as an escape so no chat marker appears literally in the sources
# (plan 2.1). Gemma 4 needs no splitter: its template strips the thought
# channel, so its answers carry no reasoning to separate.

HARMONY_START = "\x3c|start|>"
HARMONY_CHANNEL = "\x3c|channel|>"
HARMONY_MESSAGE = "\x3c|message|>"
HARMONY_CONSTRAIN = "\x3c|constrain|>"
HARMONY_CLOSE = "\x3cchannel|>"
HARMONY_STOPS = ("\x3c|return|>", "\x3c|call|>", "\x3c|end|>")
QWEN_THINK_END = "\x3c/think>"


def _tag_at_end_prefix(text: str, tags: tuple[str, ...]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of a
    tag: the part of a possible tag split across stream chunks."""
    for L in range(min(len(text), max(len(t) for t in tags) - 1), 0, -1):
        if any(t.startswith(text[-L:]) for t in tags):
            return L
    return 0


class ReasoningSplitter:
    """Splits a streamed assistant answer into (reasoning, content) per model
    family: `gpt-oss` routes the harmony `analysis` channel to reasoning and
    any other channel (e.g. `final`) to content; `qwen` treats everything
    before `</think>` as reasoning. `push(piece)` returns the
    (reasoning, content) deltas safe to emit now, holding back the tail of a
    possible tag; `close()` flushes what a cut stream left. Works on text
    decoded with special tokens (control tags included): the state machine
    swallows the channel/noise tags.

    `thinking=False` is for a prompt rendered without thinking (Qwen
    `enable_thinking=False`): the template already closed the think block in
    the prompt, so a qwen answer is content from its first token. gpt-oss is
    unaffected (its channels say where each part goes)."""

    FAMILIES = {"gpt-4o": "gpt-oss", "qwen35": "qwen"}

    _TAGS = {"gpt-oss": (HARMONY_CHANNEL, HARMONY_MESSAGE, HARMONY_CONSTRAIN,
                         HARMONY_CLOSE, HARMONY_START, *HARMONY_STOPS),
             "qwen": (QWEN_THINK_END,)}

    def __init__(self, family: str, thinking: bool = True):
        if family not in self.FAMILIES.values():
            raise ValueError(f"no reasoning splitter for family {family!r}")
        self.family = family
        self.buf = ""            # text not yet classifiable (possible tag tail)
        self.desc_text = ""      # gpt-oss: channel descriptor being read
        self.idle_text = ""      # gpt-oss: text seen outside any channel
        self.desc = False        # gpt-oss: between <|channel|> and <|message|>
        if family == "qwen":
            self.state = "thinking" if thinking else "answer"
        else:
            self.state = "idle"

    @classmethod
    def from_tokenizer(cls, tok) -> "ReasoningSplitter | None":
        """Splitter for the tokenizer's family (`tokenizer.ggml.pre`), or None
        when the family keeps reasoning out of the answer stream."""
        fam = cls.FAMILIES.get(getattr(tok, "pre", None))
        return cls(fam) if fam is not None else None

    def push(self, text: str) -> tuple[str, str]:
        """Feed one streamed piece of the answer text."""
        self.buf += text
        return self._drain(final=False)

    def close(self) -> tuple[str, str]:
        """End of stream: flush whatever was held back."""
        return self._drain(final=True)

    def split_all(self, text: str) -> tuple[str, str]:
        """Whole answer in one call: (reasoning, content)."""
        r1, c1 = self.push(text)
        r2, c2 = self.close()
        return r1 + r2, c1 + c2

    # ---- internals ----

    def _drain(self, final: bool) -> tuple[str, str]:
        r_out: list[str] = []
        c_out: list[str] = []
        if self.family == "qwen":
            self._drain_qwen(r_out, c_out, final)
        else:
            self._drain_harmony(r_out, c_out, final)
        return "".join(r_out), "".join(c_out)

    def _drain_qwen(self, r_out: list[str], c_out: list[str], final: bool) -> None:
        while True:
            i = self.buf.find(QWEN_THINK_END)
            if i < 0:
                break
            if i:
                (r_out if self.state == "thinking" else c_out).append(self.buf[:i])
            self.buf = self.buf[i + len(QWEN_THINK_END):]
            self.state = "answer"
        if self.state == "answer":
            if self.buf:
                c_out.append(self.buf)
                self.buf = ""
            return
        hold = 0 if final else _tag_at_end_prefix(self.buf, self._TAGS["qwen"])
        keep = self.buf[:len(self.buf) - hold]
        if keep:
            r_out.append(keep)
            self.buf = self.buf[len(keep):]

    def _drain_harmony(self, r_out: list[str], c_out: list[str], final: bool) -> None:
        tags = self._TAGS["gpt-oss"]
        while True:
            hit = [(self.buf.find(t), t) for t in tags if t in self.buf]
            if not hit:
                break
            i, tag = min(hit)
            self._harmony_emit(self.buf[:i], r_out, c_out)
            self.buf = self.buf[i + len(tag):]
            if tag == HARMONY_CHANNEL:
                # what came before is `<|start|>role` noise: drop it
                self.desc, self.desc_text, self.idle_text = True, "", ""
            elif tag == HARMONY_MESSAGE and self.desc:
                desc = self.desc_text.strip()
                self.state = "reasoning" if desc.startswith("analysis") else "content"
                self.desc = False
            elif tag == HARMONY_CLOSE or tag == HARMONY_START or tag in HARMONY_STOPS:
                # close the current channel: between messages, or a stream that
                # restarted a block without closing the previous one
                self._harmony_flush(c_out)
                self.state, self.desc = "idle", False
            # HARMONY_CONSTRAIN / stray HARMONY_MESSAGE: noise
        hold = 0 if final else _tag_at_end_prefix(self.buf, tags)
        keep = self.buf[:len(self.buf) - hold]
        if keep:
            if self.desc:
                self.desc_text += keep
            else:
                self._harmony_emit(keep, r_out, c_out)
            self.buf = self.buf[len(keep):]
        if final and self.state == "idle":
            self._harmony_flush(c_out)

    def _harmony_emit(self, text: str, r_out: list[str], c_out: list[str]) -> None:
        if not text:
            return
        if self.desc:
            self.desc_text += text       # channel name, not content yet
        elif self.state == "reasoning":
            r_out.append(text)
        elif self.state == "content":
            c_out.append(text)
        else:
            self.idle_text += text        # outside channels: noise or fallback

    def _harmony_flush(self, c_out: list[str]) -> None:
        """A stream that never opened a channel is plain content; one that
        closed channels carries only `<|start|>role` noise while idle."""
        if self.idle_text:
            head = self.idle_text.split(HARMONY_START, 1)[0]
            if head:
                c_out.append(head)
            self.idle_text = ""
