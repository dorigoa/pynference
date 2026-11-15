"""OpenAI-compatible HTTP server (contract D3): chat completions, one request
at a time, text only.

Two GPUs (tensor split):
  torchrun --nproc_per_node=2 -m pynference.server -m MODEL.gguf --host 127.0.0.1 --port 8090
One GPU:
  python -m pynference.server -m MODEL.gguf

Rank 0 runs a `ThreadingHTTPServer`; every generation goes through the
`Engine` (C4), so the other ranks replay exactly the same commands. One
request generates at a time (a lock); the others wait. SIGINT/SIGTERM on rank
0 shuts the server down cleanly: in-flight request finished, `{"op": "stop"}`
broadcast, process group closed — the follower ranks then exit by themselves.

Endpoints:
  GET  /health              -> {"status": "ok", "model": ID}
  GET  /v1/models           -> OpenAI model list (the one loaded model)
  POST /v1/chat/completions -> OpenAI chat completion; `stream: true` uses SSE
Parameters: `messages`, `max_tokens`/`max_completion_tokens`, `temperature`,
`top_p`, `top_k`, `min_p`, `seed`, `stop` (string or list), `stream`.
Sampling defaults come from the GGUF (as in cli/generate); a request overrides
what it sets. The prompt is the model's chat format built by
`tokenizer.chat_prompt`, tokenized exactly like `pynference.generate --chat`,
so a greedy request matches `cli generate` on the same prompt.
Unknown paths give 404, bad requests 400, both with an OpenAI-style JSON
error body. If a streaming client disappears mid-generation the text is
dropped but the generation runs to the end: the ranks must stay in lockstep.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .engine import Engine
from .gguf import GGUFFile
from .models import load_model
from .parallel import TP
from .tokenizer import Tokenizer, chat_prompt

CREATED = int(time.time())


def _error(message: str, param: str | None = None) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error",
                      "param": param, "code": None}}


def _num(body: dict, key: str, lo: float, hi: float, default):
    v = body.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be a number")
    if not lo <= float(v) <= hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}]")
    return float(v)


def _int(body: dict, key: str, lo: int, hi: int, default):
    v = body.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{key} must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}]")
    return v


def validate_body(body, defaults: dict) -> dict:
    """Parse/validate a /v1/chat/completions body; raises ValueError (400)."""
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("messages must be a non-empty list")
    for m in msgs:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str) \
                or not isinstance(m.get("content"), str):
            raise ValueError("each message needs a string 'role' and 'content'")
    if "n" in body and _int(body, "n", 1, 1, 1) != 1:
        raise ValueError("only n=1 is supported")
    if "max_completion_tokens" in body:
        max_tokens = _int(body, "max_completion_tokens", 1, 1 << 30, defaults["max_tokens"])
    else:
        max_tokens = _int(body, "max_tokens", 1, 1 << 30, defaults["max_tokens"])
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean")
    stop = body.get("stop")
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        raise ValueError("stop must be a string or a list of strings")
    if any(not s for s in stop):
        raise ValueError("stop strings must be non-empty")
    sampling = {"temp": _num(body, "temperature", 0.0, 100.0, defaults["temp"]),
                "top_k": _int(body, "top_k", 0, 1 << 30, defaults["top_k"]),
                "top_p": _num(body, "top_p", 0.0, 1.0, defaults["top_p"]),
                "min_p": _num(body, "min_p", 0.0, 1.0, defaults["min_p"]),
                "seed": _int(body, "seed", 0, 1 << 62, defaults["seed"])}
    return {"messages": msgs, "max_tokens": max_tokens, "stream": stream,
            "stop": stop, "sampling": sampling}


def _stop_cut(tok, ids: list[int], stop_strings) -> int | None:
    """Token count to keep so the decoded text ends before the first stop
    string — identical rule to `Session._stop_string_cut`."""
    text = tok.decode(ids)
    hit = min((i for i in (text.find(s) for s in stop_strings) if i >= 0), default=-1)
    if hit < 0:
        return None
    cut = len(ids)
    while cut > 0 and len(tok.decode(ids[:cut])) > hit:
        cut -= 1
    return cut


def _unflushable_len(text: str, stop_strings) -> int:
    """Length of the suffix of `text` that is a proper prefix of a stop string:
    the first characters of a future match, not yet safe to send."""
    for L in range(min(len(text), max((len(s) for s in stop_strings), default=1) - 1), 0, -1):
        tail = text[-L:]
        if any(s.startswith(tail) and len(tail) < len(s) for s in stop_strings):
            return L
    return 0


class PynferenceServer:
    """HTTP front end over an `Engine`. `make_server(...)` builds it; the
    generation lock keeps one request in flight; `stopping` makes new requests
    answer 503 while the server shuts down."""

    def __init__(self, engine: Engine, tok, model_id: str, defaults: dict,
                 prompt_ids, host: str = "127.0.0.1", port: int = 0):
        self.engine, self.tok = engine, tok
        self.model_id, self.defaults, self.prompt_ids = model_id, defaults, prompt_ids
        self.gen_lock = threading.Lock()
        self.stopping = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):   # quiet: one line per request instead
                pass

            def _json(self, status: int, obj: dict) -> None:
                data = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._json(200, {"status": "ok", "model": outer.model_id})
                elif self.path == "/v1/models":
                    self._json(200, {"object": "list", "data": [
                        {"id": outer.model_id, "object": "model",
                         "created": CREATED, "owned_by": "pynference"}]})
                else:
                    self._json(404, _error(f"unknown path {self.path!r}"))

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self._json(404, _error(f"unknown path {self.path!r}"))
                    return
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n).decode("utf-8")) if n else None
                    params = validate_body(body, outer.defaults)
                except (ValueError, json.JSONDecodeError) as e:
                    self._json(400, _error(str(e)))
                    return
                if outer.stopping:
                    self._json(503, _error("server is shutting down"))
                    return
                try:
                    ids = outer.prompt_ids(params["messages"])
                except Exception as e:
                    self._json(400, _error(f"could not build the prompt: {e}"))
                    return
                model = outer.engine.model
                max_ctx = getattr(model, "max_ctx", None)
                if max_ctx and len(ids) >= max_ctx:
                    self._json(400, _error(f"prompt of {len(ids)} tokens does not fit "
                                           f"the context ({max_ctx})"))
                    return
                max_tokens = params["max_tokens"]
                if max_ctx:
                    max_tokens = min(max_tokens, max_ctx - len(ids) - 1)
                cmd = {"op": "generate", "ids": ids, "max_tokens": max_tokens,
                       "sampling": params["sampling"],
                       "stop_ids": sorted(outer.tok.eog_ids),
                       "stop_strings": params["stop"]}
                with outer.gen_lock:
                    if outer.stopping:
                        self._json(503, _error("server is shutting down"))
                    elif params["stream"]:
                        self._stream(cmd, params["stop"], len(ids))
                    else:
                        self._complete(cmd, len(ids))

            def _complete(self, cmd: dict, prompt_tokens: int) -> None:
                generated = outer.engine.run(cmd)
                self._json(200, outer._completion(cmd, generated, prompt_tokens))

            def _stream(self, cmd: dict, stop_strings, prompt_tokens: int) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                cid = _new_id()
                created = int(time.time())
                acc: list[int] = []
                sent = [0]
                gone = [False]

                def sse(obj) -> None:
                    if gone[0]:
                        return
                    try:
                        self.wfile.write(b"data: " + json.dumps(
                            obj, ensure_ascii=False).encode() + b"\n\n")
                        self.wfile.flush()
                    except OSError:
                        gone[0] = True   # client left: keep generating (rank sync), drop text

                def chunk(delta: dict, finish: str | None) -> dict:
                    return {"id": cid, "object": "chat.completion.chunk", "created": created,
                            "model": outer.model_id,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

                sse(chunk({"role": "assistant", "content": ""}, None))

                def on_token(tid: int, _text: str) -> None:
                    acc.append(tid)
                    cut = _stop_cut(outer.tok, acc, stop_strings) if stop_strings else None
                    if cut is not None:
                        keep = outer.tok.decode(acc[:cut])
                    else:
                        keep = outer.tok.decode(acc)
                        if stop_strings:
                            keep = keep[:len(keep) - _unflushable_len(keep, stop_strings)]
                    if len(keep) > sent[0]:
                        sse(chunk({"content": keep[sent[0]:]}, None))
                        sent[0] = len(keep)

                generated = outer.engine.run(cmd, on_token=on_token)
                full = outer.tok.decode(generated)
                finish = "length" if outer.engine.session.last_truncated else "stop"
                if len(full) > sent[0]:
                    sse(chunk({"content": full[sent[0]:]}, None))
                sse(chunk({}, finish))
                if not gone[0]:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                outer._log(cmd, prompt_tokens, len(generated), finish, stream=True)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def _completion(self, cmd: dict, generated: list[int], prompt_tokens: int) -> dict:
        finish = "length" if self.engine.session.last_truncated else "stop"
        cid, created = _new_id(), int(time.time())
        n = len(generated)
        self._log(cmd, prompt_tokens, n, finish, stream=False)
        return {"id": cid, "object": "chat.completion", "created": created,
                "model": self.model_id,
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": self.tok.decode(generated)},
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                          "total_tokens": prompt_tokens + n}}

    def _log(self, cmd: dict, prompt_tokens: int, n: int, finish: str, stream: bool) -> None:
        print(f"[pynference] POST /v1/chat/completions -> {n} tok "
              f"(prompt {prompt_tokens}, finish {finish}{', stream' if stream else ''})",
              file=sys.stderr, flush=True)


def _new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def run_server(server: PynferenceServer, engine: Engine, tp) -> None:
    """Serve until SIGINT/SIGTERM: stop accepting, let the in-flight request
    finish, broadcast `stop` so the follower ranks exit, close the group."""
    def _stop(signum, frame):
        server.stopping = True
        threading.Thread(target=server.httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.serve_forever()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with server.gen_lock:              # wait out the in-flight generation
        if tp is not None:
            tp.broadcast_obj({"op": "stop"})
    if tp is not None:
        tp.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pynference.server",
                                 description="OpenAI-compatible chat server")
    ap.add_argument("-m", "--model", required=True, help="GGUF file (first shard if split)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--ctx", type=int, default=8192, help="max context length")
    ap.add_argument("-n", "--max-tokens", type=int, default=512,
                    help="completion cap when the request omits max_tokens")
    ap.add_argument("--seed", type=int, default=1234, help="default seed")
    ap.add_argument("--decode", choices=["graph", "fused", "torch"], default="graph",
                    help="decode path (as in pynference.generate)")
    args = ap.parse_args(argv)

    tp = TP()
    log = (lambda *a, **k: print(*a, **k, file=sys.stderr, flush=True)) if tp.is_main \
        else (lambda *a, **k: None)
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
    log(f"[pynference] loaded in {model.load_seconds:.1f}s + warm-up "
        f"{time.perf_counter() - t:.1f}s")

    engine = Engine(model, tok, tp)
    if not tp.is_main:
        engine.follow()               # ranks > 0: replay rank 0's commands until "stop"
        tp.close()
        return 0

    defaults = {"temp": float(g.kv.get("general.sampling.temp", 1.0)),
                "top_k": int(g.kv.get("general.sampling.top_k", 40)),
                "top_p": float(g.kv.get("general.sampling.top_p", 1.0)),
                "min_p": 0.0, "seed": args.seed, "max_tokens": args.max_tokens}
    server = PynferenceServer(
        engine, tok,
        model_id=args.model.rsplit("/", 1)[-1].removesuffix(".gguf"),
        defaults=defaults,
        prompt_ids=lambda msgs: tok.encode(chat_prompt(msgs)),
        host=args.host, port=args.port)
    log(f"[pynference] serving on http://{args.host}:{server.port} "
        f"(model id {server.model_id!r})")
    run_server(server, engine, tp)
    log("[pynference] server stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
