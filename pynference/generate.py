"""Text generation from the command line.

Two GPUs (tensor split):
  torchrun --nproc_per_node=2 -m pynference.generate -m MODEL.gguf -p "Ciao" --chat
One GPU:
  python -m pynference.generate -m MODEL.gguf -p "Ciao"

Every rank runs the same loop on the same logits; sampling happens on the CPU
with the same seed, so all ranks pick the same token without extra messages.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from .gguf import GGUFFile
from .models import load_model
from .parallel import TP
from .sampling import Sampler
from .tokenizer import StreamDecoder, Tokenizer, chat_prompt


def top_probs(tok: Tokenizer, logits: torch.Tensor, n: int) -> list[tuple[str, float]]:
    p = torch.softmax(logits.float(), dim=-1)
    v, i = p.topk(n)
    return [(tok.tokens[int(t)], round(float(pv), 5)) for pv, t in zip(v, i)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="GGUF file (first shard if split)")
    ap.add_argument("-p", "--prompt", default="Ciao! Presentati in due frasi.")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat format")
    ap.add_argument("--system", default=None, help="system message (with --chat)")
    ap.add_argument("-n", "--n-predict", type=int, default=256)
    ap.add_argument("--temp", type=float, default=None, help="default: model's (0 = greedy)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ctx", type=int, default=8192, help="max context length")
    ap.add_argument("--decode", choices=["graph", "fused", "torch"], default="graph",
                    help="decode path: fused kernels captured as XPUGraph (default), fused eager, plain torch")
    ap.add_argument("--n-probs", type=int, default=0,
                    help="print the top-N token probabilities at every step (JSON lines on stderr)")
    args = ap.parse_args(argv)

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
    mem = torch.xpu.memory_allocated(tp.device) / 2**30
    log(f"[pynference] loaded in {model.load_seconds:.1f}s + warm-up {time.perf_counter() - t:.1f}s, "
        f"{mem:.2f} GiB on GPU {tp.rank} (KV cache {model.kv_bytes() / 2**30:.2f} GiB)")

    temp = args.temp if args.temp is not None else float(g.kv.get("general.sampling.temp", 1.0))
    top_k = args.top_k if args.top_k is not None else int(g.kv.get("general.sampling.top_k", 40))
    top_p = args.top_p if args.top_p is not None else float(g.kv.get("general.sampling.top_p", 1.0))
    sampler = Sampler(temp, top_k, top_p, args.seed)

    text = args.prompt
    if args.chat:
        msgs = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": text}]
        text = chat_prompt(msgs)
    ids = tok.encode(text)
    log(f"[pynference] prompt {len(ids)} tokens, temp={temp} top_k={top_k} top_p={top_p}")

    torch.xpu.synchronize(tp.device)
    t0 = time.perf_counter()
    logits = model.forward(ids)
    torch.xpu.synchronize(tp.device)
    t_prompt = time.perf_counter() - t0

    out = StreamDecoder(tok)
    generated = []
    t1 = time.perf_counter()
    for i in range(args.n_predict):
        if args.n_probs and tp.is_main:
            print(json.dumps({"step": i, "top": top_probs(tok, logits, args.n_probs)}, ensure_ascii=False),
                  file=sys.stderr, flush=True)
        nxt = sampler(logits)
        if nxt in tok.eog_ids:
            break
        generated.append(nxt)
        if tp.is_main:
            print(out.push(nxt), end="", flush=True)
        if i + 1 < args.n_predict:
            logits = model.forward([nxt])
    t_gen = time.perf_counter() - t1
    if tp.is_main:
        print(flush=True)
    n = len(generated)
    log(f"[pynference] prompt: {len(ids)} tok in {t_prompt:.2f}s ({len(ids) / t_prompt:.1f} tok/s); "
        f"generation: {n} tok in {t_gen:.2f}s ({n / max(t_gen, 1e-9):.2f} tok/s)")
    tp.close()


if __name__ == "__main__":
    main()
