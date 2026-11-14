"""Perplexity computed exactly like llama.cpp's llama-perplexity (default mode).

The text is tokenized once (with BOS), cut into chunks of --ctx tokens; each
chunk is evaluated from an empty cache with its first token replaced by BOS,
and only the second half of each chunk is scored. The result is directly
comparable with `llama-perplexity -f FILE -c CTX --chunks N`.

  torchrun --nproc_per_node=2 bench/ppl.py -m MODEL.gguf -f /usr/share/common-licenses/GPL-3 -c 512 --chunks 8
"""
import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pynference.gguf import GGUFFile  # noqa: E402
from pynference.models import load_model  # noqa: E402
from pynference.parallel import TP  # noqa: E402
from pynference.tokenizer import Tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-f", "--file", default="/usr/share/common-licenses/GPL-3")
    ap.add_argument("-c", "--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=-1)
    args = ap.parse_args()

    tp = TP()
    g = GGUFFile(args.model)
    tok = Tokenizer(g)
    model = load_model(g, tp, max_ctx=args.ctx)
    tokens = tok.encode(open(args.file, encoding="utf-8").read())
    n_ctx = args.ctx
    n_chunk = len(tokens) // n_ctx if args.chunks < 0 else min(args.chunks, len(tokens) // n_ctx)
    first = n_ctx // 2
    if tp.is_main:
        print(f"{len(tokens)} tokens, {n_chunk} chunks of {n_ctx}", flush=True)
    nll, nll2, count = 0.0, 0.0, 0
    t0 = time.time()
    for i in range(n_chunk):
        chunk = tokens[i * n_ctx:(i + 1) * n_ctx]
        if tok.add_bos:
            chunk = [tok.bos_id] + chunk[1:]
        model.reset()
        logits = model.forward(chunk, chunk=n_ctx, all_logits=True)          # [n_ctx, n_vocab]
        lp = torch.log_softmax(logits[first:n_ctx - 1].double(), dim=-1)
        target = torch.tensor(tokens[i * n_ctx + first + 1:(i + 1) * n_ctx], device=lp.device)
        v = -lp.gather(1, target[:, None])[:, 0]
        nll += v.sum().item()
        nll2 += (v * v).sum().item()
        count += v.numel()
        if tp.is_main:
            print(f"[{i + 1}]{math.exp(nll / count):.4f},", end="", flush=True)
    m = nll / count
    sd = math.sqrt(max(nll2 / count - m * m, 0) / (count - 1))
    if tp.is_main:
        print(f"\nFinal estimate: PPL = {math.exp(m):.4f} +/- {sd * math.exp(m):.5f}  "
              f"({count} scored tokens, {time.time() - t0:.1f}s)")
    tp.close()


if __name__ == "__main__":
    main()
