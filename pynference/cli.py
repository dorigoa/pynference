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

The conversation ids are kept token-exact: the next prompt re-rendered by the
template extends the previous one, so only the new tail is prefilled
(`Session.prefill` prefix reuse). When that extension breaks (rare tokenizer
round-trip edges) the session re-prefills everything — correctness first.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from . import generate
from .chat import render as render_chat
from .engine import Engine
from .gguf import GGUFFile
from .models import load_model
from .parallel import TP
from .tokenizer import Tokenizer


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


def _chat(argv: list[str]) -> int:
    args = _chat_argv(argv)
    tp = TP()
    log = (lambda *a, **k: print(*a, **k, file=sys.stderr, flush=True)) if tp.is_main else (lambda *a, **k: None)
    g = GGUFFile(args.model)
    tok = Tokenizer(g)
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
    # enable_thinking=True: the only mode where turn k+1's prompt extends turn k's
    # (no thought-channel suffix), which is what prefix reuse needs. See report S6/S7.
    tkwargs = {"enable_thinking": True}

    def system_msgs():
        return [{"role": "system", "content": args.system}] if args.system else []

    messages = system_msgs()
    ids: list[int] = []                   # exact tokens in the model (rank 0's view)
    text_prev: str | None = None          # last rendered prompt
    answer_prev: str | None = None        # its answer text
    stream_out = [False]

    def on_token(_tid: int, text: str) -> None:
        print(text, end="", flush=True)
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
        if (text_prev is not None and answer_prev is not None
                and engine.session.last_stop is not None          # an answer closed by a stop id
                and text.startswith(text_prev + answer_prev)):
            # extend token-exactly: what closed the answer, then the new tail
            tail = text[len(text_prev + answer_prev):]
            ids = ids + [engine.session.last_stop] + tok.encode(tail, add_bos=False)
        else:
            ids = tok.encode(text, add_bos=False)
        prev = engine.session.tokens
        new_prompt = len(ids) - len(prev) if ids[:len(prev)] == prev and ids != prev else len(ids)
        t0 = time.perf_counter()
        generated = engine.run({"op": "generate", "ids": ids, "max_tokens": args.n_predict,
                                "sampling": sampling, "stop_ids": stop_ids}, on_token=on_token)
        dt = time.perf_counter() - t0
        if stream_out[0]:
            print(flush=True)
            stream_out[0] = False
        answer = tok.decode(generated)
        messages = messages + [{"role": "assistant", "content": answer}]
        ids = list(engine.session.tokens)
        text_prev, answer_prev = text, answer
        print(f"[stats] prompt new {new_prompt}/{len(ids)} tok, {len(generated)} tok in "
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
    print("  chat     -m MODEL [--system T] [--script FILE] [--ctx N] [-n M] "
          "[--temp T] [--top-k K] [--top-p P] [--seed S] [--decode D]", file=sys.stderr)
    print("  generate same options as python -m pynference.generate", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
