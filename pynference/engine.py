"""Generation engine (contract C4): sessions with prefix reuse, driven by
commands replayed on every rank.

`Session` keeps the exact list of ids the model has consumed, so `prefill`
can feed only the new tail when the new prompt extends the old one. `Engine`
broadcasts each rank-0 command on the TP control channel (`broadcast_obj`)
and executes it everywhere; since the logits are identical on the ranks and
sampling is deterministic from the seed, all ranks generate the same tokens
without further messages."""
from __future__ import annotations

from .sampling import Sampler
from .tokenizer import StreamDecoder


class Session:
    """One conversation over one model instance (contract C4)."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.tokens: list[int] = []          # exactly the ids fed to the model
        self.last_stop: int | None = None    # stop id that ended the last generate
        self.last_truncated: bool = False    # it ended on max_tokens: everything fed came back

    def reset(self) -> None:
        self.model.reset()
        self.tokens = []
        self.last_stop = None

    def prefill(self, ids: list[int]):
        """Logits of the last of `ids`. If `ids` extends `tokens`, only the new
        tail goes through the model; otherwise the context is reset and `ids`
        are all re-prefilled."""
        ids = list(ids)
        if ids[: len(self.tokens)] != self.tokens:
            self.reset()
        new = ids[len(self.tokens):]
        if not new:                       # nothing new: replay everything (rare)
            self.reset()
            new = ids
        self.tokens = ids
        return self.model.forward(new)

    def generate(self, ids: list[int], max_tokens: int, sampling, stop_ids,
                 stop_strings=(), on_token=None) -> list[int]:
        """Prefill `ids`, then sample at most `max_tokens` tokens.

        `sampling` is a dict (contract C4: temp, top_k, top_p, min_p, seed) or
        a `Sampler`. Stops at a token in `stop_ids` (not returned, remembered
        in `last_stop`) or when the decoded text contains one of
        `stop_strings` (tokens from the match on are trimmed). `on_token(id,
        text)` sees every generated token as it arrives. After the call
        `tokens` equals `ids` + the tokens fed to the model: normally
        `ids + result`, but a stop string may cut the result earlier than the
        model's progress (the fed part stays in the context). So the next call
        can extend `tokens` exactly only when `last_stop` is set (the stop id
        still has to be fed) or `last_truncated` is True (`max_tokens` cut it:
        `result` covers everything fed)."""
        sampler = Sampler(**sampling) if isinstance(sampling, dict) else sampling
        stop_ids = set(stop_ids)
        logits = self.prefill(ids)
        generated: list[int] = []
        out = StreamDecoder(self.tok)
        self.last_stop = None
        self.last_truncated = False
        while len(generated) < max_tokens:
            nxt = sampler(logits)
            if nxt in stop_ids:
                self.last_stop = nxt
                break
            if on_token is not None:
                on_token(nxt, out.push(nxt))
            else:
                out.push(nxt)
            generated.append(nxt)
            if stop_strings:
                cut = self._stop_string_cut(generated, stop_strings)
                if cut is not None:
                    del generated[cut:]
                    break
            logits = self.model.forward([nxt])
            self.tokens.append(nxt)
        else:                               # no break: cut at max_tokens
            self.last_truncated = True
        return generated

    def _stop_string_cut(self, generated: list[int], stop_strings) -> int | None:
        """Tokens to keep so the decoded text ends before the first stop string."""
        text = self.tok.decode(generated)
        hit = min((i for i in (text.find(s) for s in stop_strings) if i >= 0), default=-1)
        if hit < 0:
            return None
        cut = len(generated)
        while cut > 0 and len(self.tok.decode(generated[:cut])) > hit:
            cut -= 1
        return cut


class Engine:
    """Executes commands on every rank (contract C4).

    `run(cmd)` on rank 0 broadcasts and executes; `follow()` on the other
    ranks executes what arrives until `{"op": "stop"}`. With `tp=None` the
    engine is single-process (no broadcast) — used by the CPU tests."""

    def __init__(self, model, tok, tp=None):
        self.model, self.tok, self.tp = model, tok, tp
        self.session = Session(model, tok)

    def run(self, cmd: dict, on_token=None):
        if self.tp is not None:
            self.tp.broadcast_obj(cmd)
        return self._execute(cmd, on_token)

    def follow(self) -> None:
        while True:
            cmd = self.tp.broadcast_obj(None)
            if cmd["op"] == "stop":
                return
            self._execute(cmd)

    def _execute(self, cmd: dict, on_token=None):
        op = cmd["op"]
        if op == "generate":
            return self.session.generate(cmd["ids"], cmd["max_tokens"], cmd["sampling"],
                                         cmd.get("stop_ids", ()), cmd.get("stop_strings", ()),
                                         on_token)
        if op in ("reset", "stop"):
            if op == "reset":
                self.session.reset()
            return None
        raise ValueError(f"unknown engine command {op!r}")
