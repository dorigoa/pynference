"""All-reduce latency between the two GPUs (bf16 sum).

  torchrun --nproc-per-node 2 bench/allreduce.py   -> xccl (oneCCL), one process per GPU
  python bench/allreduce.py                        -> single process driving both GPUs:
                                                      via host (pinned) and via direct copy
"""
import os
import time

import torch

# element counts; 5376 = Gemma-4 hidden size, 2560 = Flash-Next hidden size
SIZES = [2560, 5376, 16384, 131072, 1 << 20, 8 << 20]


def fmt(n):
    b = n * 2
    return f"{b / 2**10:.0f} KB" if b < 2**20 else f"{b / 2**20:.0f} MB"


def iters_for(n):
    return 500 if n <= 16384 else (100 if n <= 1 << 20 else 20)


def run_xccl():
    import torch.distributed as dist
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.xpu.set_device(rank)
    dev = torch.device("xpu", rank)
    dist.init_process_group("xccl", device_id=dev)
    x = torch.full((8,), float(rank + 1), dtype=torch.bfloat16, device=dev)
    dist.all_reduce(x)
    ok = x[0].item() == sum(range(1, world + 1))
    if rank == 0:
        print(f"xccl world={world} correctness: {'OK' if ok else 'WRONG ' + str(x[0].item())}")
        print(f"{'size':>8} {'stream-ordered us':>18} {'with host sync us':>18} {'bus GB/s':>9}")
    for n in SIZES:
        x = torch.zeros(n, dtype=torch.bfloat16, device=dev)
        it = iters_for(n)
        for _ in range(10):
            dist.all_reduce(x)
        torch.xpu.synchronize(dev)
        dist.barrier()
        t = time.perf_counter()
        for _ in range(it):
            dist.all_reduce(x)
        torch.xpu.synchronize(dev)
        t_async = (time.perf_counter() - t) / it
        dist.barrier()
        t = time.perf_counter()
        for _ in range(min(it, 100)):
            dist.all_reduce(x)
            torch.xpu.synchronize(dev)
        t_sync = (time.perf_counter() - t) / min(it, 100)
        if rank == 0:
            busbw = 2 * (world - 1) / world * n * 2 / t_async / 1e9
            print(f"{fmt(n):>8} {t_async * 1e6:18.1f} {t_sync * 1e6:18.1f} {busbw:9.2f}", flush=True)
    dist.destroy_process_group()


def run_single():
    d0, d1 = torch.device("xpu", 0), torch.device("xpu", 1)

    def sync():
        torch.xpu.synchronize(d0)
        torch.xpu.synchronize(d1)

    print("single process, 2 GPUs: all-reduce emulated (a0 = a1 = a0 + a1), host sync each time")
    print(f"{'size':>8} {'via host us':>12} {'direct copy us':>15}")
    for n in SIZES:
        a0 = torch.ones(n, dtype=torch.bfloat16, device=d0)
        a1 = torch.ones(n, dtype=torch.bfloat16, device=d1)
        h0 = torch.empty(n, dtype=torch.bfloat16).pin_memory()
        h1 = torch.empty(n, dtype=torch.bfloat16).pin_memory()
        it = min(iters_for(n), 200)

        def via_host():
            h0.copy_(a0, non_blocking=True)
            h1.copy_(a1, non_blocking=True)
            sync()
            torch.add(h0, h1, out=h0)
            a0.copy_(h0, non_blocking=True)
            a1.copy_(h0, non_blocking=True)
            sync()

        def direct():
            t1 = a1.to(d0, non_blocking=True)
            a0.add_(t1)
            a1.copy_(a0, non_blocking=True)
            sync()

        res = []
        for fn in (via_host, direct):
            a0.fill_(0)
            a1.fill_(0)
            fn()
            t = time.perf_counter()
            for _ in range(it):
                fn()
            res.append((time.perf_counter() - t) / it * 1e6)
        print(f"{fmt(n):>8} {res[0]:12.1f} {res[1]:15.1f}", flush=True)


if __name__ == "__main__":
    run_xccl() if "RANK" in os.environ else run_single()
