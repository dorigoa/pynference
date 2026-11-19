"""Where does a decode step spend its time? Wall time per token vs GPU kernel time.

  torchrun --nproc_per_node=2 bench/decode_profile.py -m MODEL.gguf [--steps 32]
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pynference.gguf import GGUFFile  # noqa: E402
from pynference.models import load_model  # noqa: E402
from pynference.parallel import TP  # noqa: E402
from pynference.tokenizer import load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048,
                    help="max_ctx for the model (C9 measures with --ctx 8192)")
    ap.add_argument("--mode", choices=["torch", "fused", "graph"], default="fused",
                    help="decode path: plain torch ops, fused Triton kernels, or fused + XPUGraph")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="grow the prompt to about this many tokens (C9 also measures at ~2000)")
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()
    tp = TP()
    g = GGUFFile(args.model)
    tok = load_tokenizer(g)
    model = load_model(g, tp, max_ctx=args.ctx)

    def memline(tag):
        """Per-rank device memory summary (S27: profile wants per-GPU bytes)."""
        rep = getattr(model, "memory_report", lambda: {})()
        parts = " ".join(f"{k}={v / 2**30:.2f}G" for k, v in rep.items())
        alloc = res = -1
        if torch.device(tp.device).type == "xpu":
            alloc = torch.xpu.memory_allocated(dev)
            res = torch.xpu.memory_reserved(dev)
        print(f"[r{tp.rank}] mem {tag}: {parts} "
              f"torch_alloc={alloc / 2**30:.2f}G torch_res={res / 2**30:.2f}G",
              flush=True)
    model.fast_decode = args.mode != "torch"
    model.use_graph = args.mode == "graph"
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 4)
    if args.prompt_tokens > len(ids):   # C9: decode is also measured after a ~2000-token prompt
        while len(ids) < args.prompt_tokens:
            ids += tok.encode("The quick brown fox jumps over the lazy dog. ")
        ids = ids[:args.prompt_tokens]
    dev = tp.device
    memline("loaded")

    def run(n):
        model.reset()
        torch.xpu.synchronize(dev)
        t = time.perf_counter()
        logits = model.forward(ids)
        torch.xpu.synchronize(dev)
        pre = (time.perf_counter() - t) / len(ids)
        nxt = int(logits.argmax())
        t = time.perf_counter()
        for _ in range(n):
            logits = model.forward([nxt])
            nxt = int(logits.argmax())
        torch.xpu.synchronize(dev)
        return pre, (time.perf_counter() - t) / n

    run(4)                       # compile / autotune / capture
    pre, wall = run(args.steps)
    memline("post-decode")
    if tp.is_main:
        print(f"\n[{args.mode}] prefill {len(ids)} tokens: {pre * 1e3:.3f} ms/token ({1 / pre:.0f} token/s)", flush=True)
        print(f"[{args.mode}] wall time per token: {wall * 1e3:.1f} ms ({1 / wall:.2f} tok/s)", flush=True)
    store = getattr(model, "store", None)   # MoE models only (C5): hit rate drives decode speed
    if store is not None:
        st = store.stats()
        res = sum(m.R for m in store.maps.values())
        tot = sum(m.S for m in store.maps.values())
        if tp.is_main:
            print(f"store: hits {st['hits']} misses {st['misses']} "
                  f"copied {st['bytes'] / 2**30:.2f} GiB; resident {res}/{tot} slots", flush=True)
    if args.no_profile:
        tp.close()
        return
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        run(8)
    dev_us = defaultdict(float)
    cnt = defaultdict(int)
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_xpu_time_total", 0)
        if t > 0:
            dev_us[e.key] += t
            cnt[e.key] += e.count
    total_dev = sum(dev_us.values()) / 8 / 1e3
    if tp.is_main:
        print(f"GPU kernel time per token: {total_dev:.1f} ms")
        print(f"{'kernel':60s} {'ms/token':>9s} {'calls/token':>12s}")
        for k, v in sorted(dev_us.items(), key=lambda x: -x[1])[:25]:
            print(f"{k[:60]:60s} {v / 8 / 1e3:9.2f} {cnt[k] / 8:12.0f}")
    tp.close()


if __name__ == "__main__":
    main()
