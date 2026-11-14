"""Smallest possible Triton kernel on XPU; prints the full error if compilation fails."""
import torch
import triton
import triton.language as tl


@triton.jit
def add_one(X, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + o, tl.load(X + o, mask=o < N) + 1, mask=o < N)


x = torch.zeros(1000, device="xpu")
try:
    add_one[(4,)](x, 1000, BLOCK=256)
    torch.xpu.synchronize()
    print("triton OK:", x[:3].tolist(), "sum", x.sum().item())
except Exception as e:  # noqa: BLE001
    print(str(e)[:8000])
