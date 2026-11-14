"""GPU memory bandwidth: torch read/copy, bf16 GEMV (oneDNN), Triton bf16 and int8 GEMV.

GEMV shapes are decode-time projections (batch 1). GB/s counts weight bytes
only (activations are negligible), so it is directly comparable with the
memory bandwidth of the card.
"""
import sys
import time

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception as e:  # noqa: BLE001
    triton = None
    print(f"triton import failed: {e}")

DEV = torch.device("xpu", int(sys.argv[1]) if len(sys.argv) > 1 else 0)
# (N, K): out_features, in_features
SHAPES = {"gemma4 ffn up   5376->21504": (21504, 5376),
          "gemma4 ffn down 21504->5376": (5376, 21504),
          "qwen27b qkv     5120->10240": (10240, 5120),
          "small           2560->2560": (2560, 2560)}


def timed(fn, iters):
    fn()
    torch.xpu.synchronize(DEV)
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize(DEV)
        best = min(best, (time.perf_counter() - t) / iters)
    return best


if triton is not None:
    @triton.jit
    def gemv_kernel(W, X, S, Y, N, K, stride_wn,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SCALED: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + rk, mask=rk < K, other=0.0).to(tl.float32)
            w = tl.load(W + rn[:, None] * stride_wn + rk[None, :],
                        mask=(rn[:, None] < N) & (rk[None, :] < K), other=0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        if SCALED:
            acc = acc * tl.load(S + rn, mask=rn < N, other=0.0).to(tl.float32)
        tl.store(Y + rn, acc.to(tl.bfloat16), mask=rn < N)

    CONFIGS = [(bn, bk, nw) for bn in (8, 16, 32) for bk in (256, 512) for nw in (4, 8)]

    def triton_gemv(w, x, s, y, cfg):
        bn, bk, nw = cfg
        n, k = w.shape
        gemv_kernel[(triton.cdiv(n, bn),)](w, x, s, y, n, k, w.stride(0),
                                           BLOCK_N=bn, BLOCK_K=bk, SCALED=s is not None, num_warps=nw)

    def best_triton(w, x, s, ref):
        y = torch.empty(w.shape[0], dtype=torch.bfloat16, device=DEV)
        best = (float("inf"), None)
        for cfg in CONFIGS:
            try:
                triton_gemv(w, x, s, y, cfg)
                torch.xpu.synchronize(DEV)
                err = (y.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
                if err > 2e-2:
                    print(f"    cfg {cfg}: wrong result (rel err {err:.3g})")
                    continue
                t = timed(lambda: triton_gemv(w, x, s, y, cfg), 20)
                best = min(best, (t, cfg))
            except Exception as e:  # noqa: BLE001
                print(f"    cfg {cfg}: {type(e).__name__}: {str(e)[:120]}")
        return best


def main():
    p = torch.xpu.get_device_properties(DEV)
    print(f"{DEV}: {p.name}, {p.total_memory / 2**30:.1f} GiB\n")

    x = torch.empty(256 * 2**20, dtype=torch.bfloat16, device=DEV).uniform_()  # 512 MiB
    y = torch.empty_like(x)
    nb = x.numel() * 2
    print(f"torch read  (sum, 512 MiB): {nb / timed(lambda: x.sum(dtype=torch.float32), 10) / 1e9:7.1f} GB/s")
    print(f"torch copy  (512 MiB):      {2 * nb / timed(lambda: y.copy_(x), 10) / 1e9:7.1f} GB/s  (read+write)")
    del x, y

    print(f"\n{'GEMV batch 1':30s} {'MB':>6} {'torch bf16':>11} {'triton bf16':>12} {'triton int8':>12}  best cfgs")
    for name, (n, k) in SHAPES.items():
        w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02
        xv = torch.randn(k, dtype=torch.bfloat16, device=DEV)
        ref = (w.float() @ xv.float())
        t_torch = timed(lambda: F.linear(xv[None], w), 20)
        wb = w.numel() * 2
        row = [wb / t_torch / 1e9]
        cfgs = []
        if triton is not None:
            t, c = best_triton(w, xv, None, ref)
            row.append(wb / t / 1e9)
            cfgs.append(c)
            # int8 weights, per-row scale (stand-in for Q8_0-style kernels)
            scale = (w.abs().amax(dim=1).float() / 127.0).clamp(min=1e-8)
            wq = torch.round(w.float() / scale[:, None]).to(torch.int8)
            refq = (wq.float() * scale[:, None]) @ xv.float()
            t, c = best_triton(wq, xv, scale, refq)
            row.append(wq.numel() / t / 1e9)
            cfgs.append(c)
            del wq
        print(f"{name:30s} {wb / 2**20:6.0f} " + " ".join(f"{v:11.1f} " for v in row) + f" {cfgs}", flush=True)
        del w


if __name__ == "__main__":
    main()
