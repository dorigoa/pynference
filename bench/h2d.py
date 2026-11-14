"""Transfer bandwidth vs transfer size: RAM->GPU, GPU->RAM, GPU->GPU.

Each cell moves at least ~1 GiB as back-to-back copies of the given size
(non_blocking, pinned host memory) and reports GB/s (1e9 B/s). "both" issues
copies to the two GPUs interleaved and reports the aggregate bandwidth.
"""
import time

import torch

KB, MB, GB = 1 << 10, 1 << 20, 1 << 30
SIZES = [256 * KB, 1 * MB, 4 * MB, 16 * MB, 64 * MB, 256 * MB]
POOL = 512 * MB        # buffer size on each side; copies cycle through it
MIN_BYTES = 1 * GB     # bytes moved per measurement
MAX_COPIES = 4096

NDEV = torch.xpu.device_count()
DEVS = [torch.device("xpu", i) for i in range(NDEV)]


def sync():
    for d in DEVS:
        torch.xpu.synchronize(d)


def timed(fn, reps=3):
    fn()
    sync()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t)
    return best


def plan(size):
    n = min(MAX_COPIES, max(1, MIN_BYTES // size))
    slots = POOL // size
    return n, [((k % slots) * size, (k % slots + 1) * size) for k in range(n)]


def gbps(nbytes, t):
    return nbytes / t / 1e9


def main():
    print(f"devices: {NDEV} x {torch.xpu.get_device_name(0)}")
    host = torch.empty(POOL, dtype=torch.uint8).pin_memory()
    host2 = torch.empty(POOL, dtype=torch.uint8).pin_memory()
    pageable = torch.empty(POOL, dtype=torch.uint8)
    host.fill_(1)
    pageable.fill_(1)
    dev = [torch.empty(POOL, dtype=torch.uint8, device=d) for d in DEVS]
    dev2 = [torch.empty(POOL, dtype=torch.uint8, device=d) for d in DEVS]
    print(f"host pinned: {host.is_pinned()}\n")

    cols = ["H2D x0", "H2D x1", "H2D both", "D2H x0", "D2H both", "H2D pageable x0", "GPU0->GPU1"]
    print(f"{'size':>8} {'copies':>6} " + " ".join(f"{c:>15}" for c in cols))
    for size in SIZES:
        n, sl = plan(size)
        tot = n * size

        def h2d(i):
            for a, b in sl:
                dev[i][a:b].copy_(host[a:b], non_blocking=True)

        def h2d_both():
            for a, b in sl:
                for i in range(NDEV):
                    dev[i][a:b].copy_(host[a:b], non_blocking=True)

        def d2h(i, dst=host):
            for a, b in sl:
                dst[a:b].copy_(dev[i][a:b], non_blocking=True)

        def d2h_both():
            for a, b in sl:
                host[a:b].copy_(dev[0][a:b], non_blocking=True)
                host2[a:b].copy_(dev[1][a:b], non_blocking=True)

        def h2d_pageable():
            for a, b in sl:
                dev[0][a:b].copy_(pageable[a:b], non_blocking=True)

        def p2p():
            for a, b in sl:
                dev2[1][a:b].copy_(dev[0][a:b], non_blocking=True)

        r = [gbps(tot, timed(lambda: h2d(0))), gbps(tot, timed(lambda: h2d(1))),
             gbps(NDEV * tot, timed(h2d_both)), gbps(tot, timed(lambda: d2h(0))),
             gbps(NDEV * tot, timed(d2h_both)),
             gbps(tot, timed(h2d_pageable, reps=2)) if size >= 4 * MB else float("nan"),
             gbps(tot, timed(p2p, reps=2)) if size >= 1 * MB else float("nan")]
        lbl = f"{size // MB} MB" if size >= MB else f"{size // KB} KB"
        print(f"{lbl:>8} {n:>6} " + " ".join(f"{v:15.2f}" for v in r), flush=True)

    # latency of a tiny synchronous copy
    for name, fn in [("H2D 4KB + sync", lambda: dev[0][:4096].copy_(host[:4096], non_blocking=True)),
                     ("D2H 4KB + sync", lambda: host[:4096].copy_(dev[0][:4096], non_blocking=True))]:
        fn()
        sync()
        t = time.perf_counter()
        for _ in range(500):
            fn()
            torch.xpu.synchronize(DEVS[0])
        print(f"\n{name}: {(time.perf_counter() - t) / 500 * 1e6:.1f} us per copy", end="")
    print()


if __name__ == "__main__":
    main()
