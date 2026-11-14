"""Per-kernel launch overhead from Python on XPU: eager ops vs graph capture/replay."""
import time

import torch
import torch.nn.functional as F

DEV = torch.device("xpu", 0)


def per_call(fn, n):
    fn()
    torch.xpu.synchronize(DEV)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize(DEV)
    return (time.perf_counter() - t) / n * 1e6


def main():
    x = torch.zeros(4096, device=DEV)
    xb = torch.randn(1, 2560, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(2560, 2560, dtype=torch.bfloat16, device=DEV)
    print(f"eager x.add_(1) (4096 elem):           {per_call(lambda: x.add_(1.0), 5000):7.2f} us/op")
    print(f"eager F.linear 1x2560 @ 2560x2560 bf16:  {per_call(lambda: F.linear(xb, w), 2000):7.2f} us/op")

    def op_sync():
        x.add_(1.0)
        torch.xpu.synchronize(DEV)
    print(f"add + synchronize round trip:           {per_call(op_sync, 1000):7.2f} us")

    graph_cls = getattr(torch.xpu, "XPUGraph", None)
    if graph_cls is None:
        print("graph capture: torch.xpu.XPUGraph not available in this build")
        return
    try:
        g = graph_cls()
        s = torch.xpu.Stream(DEV)
        s.wait_stream(torch.xpu.current_stream(DEV))
        with torch.xpu.stream(s):
            for _ in range(3):
                x.add_(1.0)
        torch.xpu.current_stream(DEV).wait_stream(s)
        with torch.xpu.graph(g):
            for _ in range(1000):
                x.add_(1.0)
        us = per_call(g.replay, 20) / 1000
        print(f"graph replay of 1000 x.add_(1):          {us:7.2f} us/op")
    except Exception as e:  # noqa: BLE001
        print(f"graph capture failed: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
