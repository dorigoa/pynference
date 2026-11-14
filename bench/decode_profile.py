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
from pynference.tokenizer import Tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--mode", choices=["torch", "fused", "graph"], default="fused",
                    help="decode path: plain torch ops, fused Triton kernels, or fused + XPUGraph")
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()
    tp = TP()
    g = GGUFFile(args.model)
    tok = Tokenizer(g)
    model = load_model(g, tp, max_ctx=2048)
    model.fast_decode = args.mode != "torch"
    model.use_graph = args.mode == "graph"
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 4)
    dev = tp.device

    def run(n):
        model.reset()
        logits = model.forward(ids)
        nxt = int(logits.argmax())
        torch.xpu.synchronize(dev)
        t = time.perf_counter()
        for _ in range(n):
            logits = model.forward([nxt])
            nxt = int(logits.argmax())
        torch.xpu.synchronize(dev)
        return (time.perf_counter() - t) / n

    run(4)                       # compile / autotune / capture
    wall = run(args.steps)
    if tp.is_main:
        print(f"\n[{args.mode}] wall time per token: {wall * 1e3:.1f} ms ({1 / wall:.2f} tok/s)", flush=True)
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
