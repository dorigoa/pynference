"""Command line: interactive chat and plain generation.

Two GPUs (tensor split):
  torchrun --nproc_per_node=2 -m pynference.cli chat -m MODEL.gguf
One GPU:
  python -m pynference.cli chat -m MODEL.gguf

  python -m pynference.cli generate -m MODEL.gguf -p "Ciao" --chat   (as pynference.generate)

Chat (`chat` subcommand): rank 0 reads the user's lines (stdin, or `--script`
FILE with one turn per line) and broadcasts commands over the engine control
channel (C4); the other ranks just `follow()`. Slash commands: /reset,
/system TEXT, /exit. After every answer one line of statistics.
`--show-reasoning` prints the model's reasoning before the answer (GPT-OSS
analysis channel, Qwen think block); `--no-reasoning` renders the prompt
without thinking. Families without a splitter (Gemma 4) are unaffected.

The conversation ids are kept token-exact: after an answer that filled the
context (closed by a stop id, or cut by `-n`) the next user turn is appended
at token level (`continuation_ids`): the tokens that close the assistant turn
the way the template would, then the template's own user block for the new
turn. This is needed because a template that strips thought channels (Gemma
4: `strip_thinking`, and past reasoning is never replayed) cannot re-render a
thinking answer: a full re-render would neither extend the previous prompt
nor keep the answer in it. When no continuation can be built the session
re-encodes and re-prefills the whole prompt — correctness first. `[stats]`
prints new prompt tokens / total prompt tokens per answer.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from . import generate
from .chat import ReasoningSplitter, render as render_chat
from .engine import Engine
from .gguf import GGUFFile
from .models import load_model
from .parallel import TP
from .tokenizer import StreamDecoder, Tokenizer, load_tokenizer


def _chat_argv(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m pynference.cli chat",
                                 description="interactive chat with the model")
    ap.add_argument("-m", "--model", required=True, help="GGUF file (first shard if split)")
    ap.add_argument("--system", default=None, help="system message")
    ap.add_argument("--script", default=None, help="read user turns from a file, one per line")
    ap.add_argument("--ctx", type=int, default=8192, help="max context length")
    ap.add_argument("-n", "--n-predict", type=int, default=256, help="max tokens per answer")
    ap.add_argument("--temp", type=float, default=None, help="default: model's (0 = greedy)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--decode", choices=["graph", "fused", "torch"], default="graph",
                    help="decode path (as in pynference.generate)")
    ap.add_argument("--show-reasoning", action="store_true",
                    help="show the model's reasoning (GPT-OSS analysis channel, Qwen think block)")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="render without thinking (Qwen enable_thinking=False; Gemma default template)")
    return ap.parse_args(argv)


def _user_turns(args):
    """User turns, one per line; empty lines ignored."""
    if args.script:
        with open(args.script, encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = None
    i = 0
    while True:
        if lines is not None:
            if i >= len(lines):
                return
            line = lines[i]
            i += 1
        else:
            try:
                line = input("you> ")
            except EOFError:
                return
        line = line.strip()
        if line:
            yield line


def continuation_ids(g, tok, tkwargs, user_turn: str, answer: str,
                     last_stop: int | None, last_truncated: bool):
    """Token ids that append `user_turn`'s user block to the context the
    previous answer left, or None if no token-exact continuation is known
    (the caller re-encodes the whole prompt).

    The answer must have filled the context: closed by a stop id (which is
    replayed first) or cut at `max_tokens` (it fed exactly what it returned).
    The user block is taken from a fresh render of a one-message conversation
    minus the shared head (BOS + system block); before it, the still-open
    pieces of the assistant turn are closed the way the template would
    (channel close for a thought cut by `-n`, then `<turn|>`)."""
    if last_truncated:
        close = "\n<channel|>" if answer.count("<|channel>") > answer.count("<channel|>") else ""
        close_ids = tok.encode(close + "<turn|>\n", add_bos=False)
    elif last_stop is not None and tok.decode([last_stop], special=True) == "<turn|>":
        close_ids = [last_stop] + tok.encode("\n", add_bos=False)
    else:
        return None
    block = render_chat(g, [{"role": "user", "content": user_turn}],
                        add_generation_prompt=True, **tkwargs)
    head = render_chat(g, [], add_generation_prompt=False, **tkwargs)
    if not block.startswith(head):
        return None
    return close_ids + tok.encode(block[len(head):], add_bos=False)


def _chat(argv: list[str]) -> int:
    args = _chat_argv(argv)
    tp = TP()
    log = (lambda *a, **k: print(*a, **k, file=sys.stderr, flush=True)) if tp.is_main else (lambda *a, **k: None)
    g = GGUFFile(args.model)
    tok = load_tokenizer(g)
    log(f"[pynference] {g.kv.get('general.name')} ({g.arch}), {tp.world} GPU(s)")
    model = load_model(g, tp, max_ctx=args.ctx)
    model.fast_decode = args.decode != "torch"
    model.use_graph = args.decode == "graph"
    t = time.perf_counter()
    model.forward([tok.bos_id] * 8)       # warm-up: Triton compile/autotune, graph capture
    model.forward([tok.bos_id])
    model.reset()
    torch.xpu.synchronize(tp.device)
    log(f"[pynference] loaded in {model.load_seconds:.1f}s + warm-up {time.perf_counter() - t:.1f}s")

    engine = Engine(model, tok, tp)
    if not tp.is_main:
        engine.follow()                   # ranks > 0: just replay rank 0's commands
        tp.close()
        return 0

    temp = args.temp if args.temp is not None else float(g.kv.get("general.sampling.temp", 1.0))
    top_k = args.top_k if args.top_k is not None else int(g.kv.get("general.sampling.top_k", 40))
    top_p = args.top_p if args.top_p is not None else float(g.kv.get("general.sampling.top_p", 1.0))
    sampling = {"temp": temp, "top_k": top_k, "top_p": top_p, "min_p": 0.0, "seed": args.seed}
    stop_ids = sorted(tok.eog_ids)
    # Thinking on by default: for Gemma it is also the only mode where turn
    # k+1's prompt extends turn k's (no thought-channel suffix), which is what
    # prefix reuse needs (report S6/S7); --no-reasoning renders without it
    # (Qwen templates: enable_thinking=False; Gemma: its default block).
    thinking = not args.no_reasoning
    if isinstance(tok, Tokenizer):
        tkwargs = {"enable_thinking": True} if thinking else {}
    else:
        tkwargs = {"enable_thinking": thinking}

    def system_msgs():
        return [{"role": "system", "content": args.system}] if args.system else []

    messages = system_msgs()
    ids: list[int] = []                   # exact tokens in the model (rank 0's view)
    text_prev: str | None = None          # last rendered prompt
    answer_prev: str | None = None        # its answer text
    stream_out = [False]
    # reasoning families: split the stream so reasoning is shown only with
    # --show-reasoning (and never mixed into the answer); tag text needs
    # special tokens in, hence the own special=True decoder. The state
    # machine restarts every answer: `sp[0]` is set before each run.
    probe = ReasoningSplitter.from_tokenizer(tok)
    raw_dec = StreamDecoder(tok, special=True) if probe is not None else None
    sp: list[ReasoningSplitter | None] = [None]
    in_thinking = [False]

    def on_token(tid: int, text: str) -> None:
        if sp[0] is None:
            print(text, end="", flush=True)
            stream_out[0] = True
            return
        r, c = sp[0].push(raw_dec.push(tid))
        if args.show_reasoning and r:
            if not in_thinking[0]:
                print("[reasoning]", flush=True)
                in_thinking[0] = True
            print(r, end="", flush=True)
            stream_out[0] = True
        if c:
            if in_thinking[0]:
                print(flush=True)
                in_thinking[0] = False
            print(c, end="", flush=True)
            stream_out[0] = True

    def reset():
        nonlocal messages, ids, text_prev, answer_prev
        engine.run({"op": "reset"})
        messages = system_msgs()
        ids, text_prev, answer_prev = [], None, None

    for turn in _user_turns(args):
        if turn == "/exit":
            break
        if turn == "/reset":
            reset()
            print("[reset]", flush=True)
            continue
        if turn.startswith("/system"):
            args.system = turn[len("/system"):].strip() or None
            reset()
            print("[system updated]", flush=True)
            continue
        messages = messages + [{"role": "user", "content": turn}]
        text = render_chat(g, messages, add_generation_prompt=True, **tkwargs)
        last = engine.session
        if (text_prev is not None and answer_prev is not None
                and (last.last_stop is not None                    # answer closed by a stop id
                     or last.last_truncated)                       # ... or cut at -n: all of it fed
                and text.startswith(text_prev + answer_prev)):
            # the re-render extends the prompt: stop id (if any), then the new tail
            tail = text[len(text_prev + answer_prev):]
            closed = [] if last.last_stop is None else [last.last_stop]
            ids = list(last.tokens) + closed + tok.encode(tail, add_bos=False)
        else:
            # token-level continuation for templates that cannot re-render
            # the answer (thought channels stripped); None: re-encode all
            cont = (continuation_ids(g, tok, tkwargs, turn, answer_prev,
                                     last.last_stop, last.last_truncated)
                    if text_prev is not None else None)
            ids = list(last.tokens) + cont if cont is not None else tok.encode(text, add_bos=False)
        prev = last.tokens
        new_prompt = len(ids) - len(prev) if ids[:len(prev)] == prev and ids != prev else len(ids)
        prompt_len = len(ids)
        t0 = time.perf_counter()
        sp[0] = ReasoningSplitter(probe.family) if probe is not None else None
        in_thinking[0] = False
        generated = engine.run({"op": "generate", "ids": ids, "max_tokens": args.n_predict,
                                "sampling": sampling, "stop_ids": stop_ids}, on_token=on_token)
        dt = time.perf_counter() - t0
        if sp[0] is not None:                     # flush a stream cut by a tag tail
            r, c = sp[0].close()
            if args.show_reasoning and r:
                print(r, end="", flush=True)
                stream_out[0] = True
            if c:
                if in_thinking[0]:
                    print(flush=True)
                    in_thinking[0] = False
                print(c, end="", flush=True)
                stream_out[0] = True
        if stream_out[0]:
            print(flush=True)
            stream_out[0] = False
        answer = tok.decode(generated)
        messages = messages + [{"role": "assistant", "content": answer}]
        ids = list(engine.session.tokens)
        text_prev, answer_prev = text, answer
        print(f"[stats] prompt new {new_prompt}/{prompt_len} tok, {len(generated)} tok in "
              f"{dt:.2f}s = {len(generated) / max(dt, 1e-9):.1f} tok/s", flush=True)

    engine.run({"op": "stop"})
    tp.close()
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "generate":
        return generate.main(argv[1:]) or 0
    if argv and argv[0] == "chat":
        return _chat(argv[1:])
    print("usage: python -m pynference.cli {chat,generate} ...", file=sys.stderr)
    print("  chat     -m MODEL [--system T] [--script FILE] [--ctx N] [-n M] [--temp T] "
          "[--top-k K] [--top-p P] [--seed S] [--decode D] [--show-reasoning] [--no-reasoning]",
          file=sys.stderr)
    print("  generate same options as python -m pynference.generate", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
