"""Tokenizer built from GGUF metadata.

Implements the "gemma4" tokenizer as llama.cpp does: SentencePiece-style BPE
(spaces become U+2581, no pre-splitting except newline runs, merges ranked by
their order in the file) with <0xXX> byte fallback. Special tokens (control,
user-defined, unknown) are matched literally in the text first.

Also implements the byte-level "gpt2" tokenizer (GPT-OSS with the `gpt-4o`
pre-tokenizer, Qwen3.5 with `qwen35`): the text is split by a pre-tokenizer
regex (`regex` package), each piece is mapped byte-by-byte through the GPT-2
byte -> unicode map, merged by BPE rank and looked up in the vocabulary.
`load_tokenizer` picks the class from `tokenizer.ggml.model`.
"""
from __future__ import annotations

import heapq
import re

import numpy as np
import regex

from .gguf import GGUFFile

NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
SPACE = "▁"
_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _bpe_merge(sym: list[str], ranks: dict[tuple[str, str], int]) -> list[str]:
    """Merge adjacent symbols by lowest merge rank (ties: leftmost), as llama.cpp's llm_tokenizer_bpe."""
    n = len(sym)
    if n < 2:
        return sym
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    nxt[-1] = -1
    heap = []

    def push(left, right):
        if left >= 0 and right >= 0:
            r = ranks.get((sym[left], sym[right]))
            if r is not None:
                heapq.heappush(heap, (r, left, right, sym[left] + sym[right]))

    for i in range(n - 1):
        push(i, i + 1)
    while heap:
        _, left, right, text = heapq.heappop(heap)
        if not sym[left] or not sym[right] or sym[left] + sym[right] != text or nxt[left] != right:
            continue
        sym[left] = text
        sym[right] = ""
        nxt[left] = nxt[right]
        if nxt[right] >= 0:
            prev[nxt[right]] = left
        push(prev[left], left)
        push(left, nxt[left])
    return [s for s in sym if s]


class Tokenizer:
    def __init__(self, g: GGUFFile):
        model = g.kv.get("tokenizer.ggml.model")
        if model != "gemma4":
            raise NotImplementedError(f"tokenizer model {model!r} not supported yet")
        self.tokens: list[str] = list(g.kv["tokenizer.ggml.tokens"])
        self.types = np.asarray(g.kv["tokenizer.ggml.token_type"], dtype=np.int32)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}  # last duplicate wins
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(g.kv["tokenizer.ggml.merges"]):
            sep = m.find(" ", 1)
            if sep > 0:
                self.ranks.setdefault((m[:sep], m[sep + 1:]), i)
        self.bos_id = int(g.kv["tokenizer.ggml.bos_token_id"])
        self.eos_id = int(g.kv["tokenizer.ggml.eos_token_id"])
        self.add_bos = bool(g.kv.get("tokenizer.ggml.add_bos_token", True))
        special = [i for i, t in enumerate(self.types) if t in (CONTROL, USER_DEFINED, UNKNOWN)]
        special.sort(key=lambda i: -len(self.tokens[i]))
        self._special_re = re.compile("(" + "|".join(re.escape(self.tokens[i]) for i in special) + ")")
        self.special_ids = {self.tokens[i]: i for i in reversed(special)}
        # end-of-generation tokens
        self.eog_ids = {self.eos_id} | {self.token_to_id[t] for t in ("<turn|>", "<|tool_response>")
                                        if t in self.token_to_id}

    # ---- encoding ----
    def encode(self, text: str, add_bos: bool | None = None, parse_special: bool = True) -> list[int]:
        out: list[int] = []
        if self.add_bos if add_bos is None else add_bos:
            out.append(self.bos_id)
        parts = self._special_re.split(text) if parse_special else [text]
        for i, part in enumerate(parts):
            if i % 2 == 1:
                out.append(self.special_ids[part])
            elif part:
                self._encode_raw(part.replace(" ", SPACE), out)
        return out

    def _encode_raw(self, text: str, out: list[int]):
        for word in re.findall(r"[^\n]+|\n+", text):
            if word[0] == "\n" and word in self.token_to_id:
                out.append(self.token_to_id[word])
                continue
            for piece in self._bpe(word):
                tid = self.token_to_id.get(piece)
                if tid is not None:
                    out.append(tid)
                else:
                    for b in piece.encode("utf-8"):
                        bt = self.token_to_id.get(f"<0x{b:02X}>")
                        if bt is not None:
                            out.append(bt)

    def _bpe(self, word: str) -> list[str]:
        return _bpe_merge(list(word), self.ranks)

    # ---- decoding ----
    def token_bytes(self, tid: int, special: bool = True) -> bytes:
        t, typ = self.tokens[tid], self.types[tid]
        if typ == BYTE:
            m = _BYTE_TOKEN.fullmatch(t)
            return bytes([int(m.group(1), 16)]) if m else b""
        if typ in (CONTROL, UNKNOWN, UNUSED):
            return t.encode() if special else b""
        return t.replace(SPACE, " ").encode("utf-8")

    def decode(self, ids, special: bool = False) -> str:
        return b"".join(self.token_bytes(i, special) for i in ids).decode("utf-8", "replace")


def _bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte -> unicode map (as llama.cpp's unicode_byte_to_utf8_map)."""
    printable = set(range(0x21, 0x7F)) | set(range(0xA1, 0xAD)) | set(range(0xAE, 0x100))
    b2u, n = {}, 0
    for b in range(256):
        if b in printable:
            b2u[b] = chr(b)
        else:
            b2u[b] = chr(256 + n)
            n += 1
    return b2u


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {c: b for b, c in BYTE_ENCODER.items()}

# pre-tokenizer regexes, verbatim from tokenizer.json (llama-vocab.cpp comments)
_PRE_RE = {
    "gpt-4o": regex.compile(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "qwen35": regex.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
        r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
}

# token texts that llama.cpp treats as end-of-generation (llama-vocab.cpp)
_EOG_TEXTS = frozenset({
    "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>",
    "<|flush|>", "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>",
    "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
    "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>",
    "[e~[", "<|ifm|im_end|>",
})


class BPETokenizer:
    """Byte-level BPE tokenizer for `tokenizer.ggml.model == "gpt2"` GGUFs
    (GPT-OSS `gpt-4o` pre-tokenizer, Qwen `qwen35`), as llama.cpp does."""

    def __init__(self, g: GGUFFile):
        model = g.kv.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise NotImplementedError(f"tokenizer model {model!r} is not byte-level BPE")
        pre = g.kv.get("tokenizer.ggml.pre")
        if pre not in _PRE_RE:
            raise NotImplementedError(f"pre-tokenizer {pre!r} not supported")
        self.pre = pre
        self._pre_re = _PRE_RE[pre]
        self.tokens: list[str] = list(g.kv["tokenizer.ggml.tokens"])
        self.types = np.asarray(g.kv["tokenizer.ggml.token_type"], dtype=np.int32)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}  # last duplicate wins
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(g.kv["tokenizer.ggml.merges"]):
            a, sep, b = m.partition(" ")
            if sep:
                self.ranks.setdefault((a, b), i)
        self.bos_id = int(g.kv["tokenizer.ggml.bos_token_id"])
        self.eos_id = int(g.kv["tokenizer.ggml.eos_token_id"])
        self.add_bos = bool(g.kv.get("tokenizer.ggml.add_bos_token", False))
        special = [i for i, t in enumerate(self.types) if t in (CONTROL, USER_DEFINED, UNKNOWN)]
        special.sort(key=lambda i: -len(self.tokens[i]))
        self._special_re = re.compile("(" + "|".join(re.escape(self.tokens[i]) for i in special) + ")")
        self.special_ids = {self.tokens[i]: i for i in reversed(special)}
        # end-of-generation tokens: llama.cpp's text rules, the harmony
        # exception (with <|return|> and <|call|> EOG, <|end|> is not) and eos
        eog_texts = {t for t in _EOG_TEXTS if t in self.token_to_id}
        if {"<|return|>", "<|call|>", "<|end|>"} <= eog_texts:
            eog_texts.remove("<|end|>")
        eog_texts.add(self.tokens[self.eos_id])
        self.eog_ids = {self.token_to_id[t] for t in eog_texts}

    # ---- encoding ----
    def encode(self, text: str, add_bos: bool | None = None, parse_special: bool = True) -> list[int]:
        out: list[int] = []
        if self.add_bos if add_bos is None else add_bos:
            out.append(self.bos_id)
        parts = self._special_re.split(text) if parse_special else [text]
        for i, part in enumerate(parts):
            if i % 2 == 1:
                out.append(self.special_ids[part])
            elif part:
                self._encode_raw(part, out)
        return out

    def _encode_raw(self, text: str, out: list[int]):
        pos = 0
        pieces: list[str] = []
        for m in self._pre_re.finditer(text):
            if m.start() > pos:  # keep any text the regex left out as its own piece
                pieces.append(text[pos:m.start()])
            pieces.append(m.group())
            pos = m.end()
        if pos < len(text):
            pieces.append(text[pos:])
        for piece in pieces:
            encoded = "".join(BYTE_ENCODER[b] for b in piece.encode("utf-8"))
            for sym in _bpe_merge(list(encoded), self.ranks):
                tid = self.token_to_id.get(sym)
                if tid is not None:
                    out.append(tid)
                else:
                    for ch in sym:  # every single byte token is in a byte-level vocab
                        bt = self.token_to_id.get(ch)
                        if bt is not None:
                            out.append(bt)

    # ---- decoding ----
    def token_bytes(self, tid: int, special: bool = True) -> bytes:
        t, typ = self.tokens[tid], self.types[tid]
        if typ in (CONTROL, UNKNOWN, UNUSED):
            return t.encode() if special else b""
        if typ == USER_DEFINED:
            return t.encode()
        return bytes(BYTE_DECODER[c] for c in t)

    def decode(self, ids, special: bool = False) -> str:
        return b"".join(self.token_bytes(i, special) for i in ids).decode("utf-8", "replace")


def load_tokenizer(g: GGUFFile) -> Tokenizer | BPETokenizer:
    """Tokenizer for a GGUF, chosen by `tokenizer.ggml.model`."""
    model = g.kv.get("tokenizer.ggml.model")
    if model == "gemma4":
        return Tokenizer(g)
    if model == "gpt2":
        return BPETokenizer(g)
    raise NotImplementedError(f"tokenizer model {model!r} not supported")


class StreamDecoder:
    """Turns a stream of token ids into text, holding back incomplete UTF-8 sequences."""

    def __init__(self, tok: Tokenizer | BPETokenizer, special: bool = False):
        self.tok, self.special, self.buf = tok, special, b""

    def push(self, tid: int) -> str:
        self.buf += self.tok.token_bytes(tid, self.special)
        try:
            s, self.buf = self.buf.decode("utf-8"), b""
            return s
        except UnicodeDecodeError as e:
            if e.reason == "unexpected end of data":
                s, self.buf = self.buf[:e.start].decode("utf-8"), self.buf[e.start:]
                return s
            s, self.buf = self.buf.decode("utf-8", "replace"), b""
            return s


def chat_prompt(messages: list[dict], thinking: bool = False) -> str:
    """Gemma 4 chat format (text only, no tools), as in the GGUF's chat template.

    The BOS token is not included: `Tokenizer.encode` adds it."""
    out = []
    if thinking or (messages and messages[0]["role"] in ("system", "developer")):
        out.append("<|turn>system\n")
        if thinking:
            out.append("<|think|>\n")
        if messages and messages[0]["role"] in ("system", "developer"):
            out.append(messages[0]["content"].strip())
            messages = messages[1:]
        out.append("<turn|>\n")
    for m in messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        content = m["content"] if role == "model" else m["content"].strip()
        out.append(f"<|turn>{role}\n{content}<turn|>\n")
    out.append("<|turn>model\n")
    if not thinking:
        out.append("<|channel>thought\n<channel|>")
    return "".join(out)


def main():
    import sys
    import time
    t = time.time()
    tok = Tokenizer(GGUFFile(sys.argv[1]))
    print(f"loaded {len(tok.tokens)} tokens, {len(tok.ranks)} merges in {time.time() - t:.1f}s; "
          f"eog={sorted(tok.eog_ids)}", file=sys.stderr)
    for text in sys.argv[2:]:
        print(",".join(map(str, tok.encode(text))))


if __name__ == "__main__":
    main()
