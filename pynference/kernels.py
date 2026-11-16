"""Fused Triton kernels for the decode step (batch 1).

Each one replaces a handful of small torch ops, so a transformer layer needs
~11 launches instead of ~90. All of them take positions/lengths from device
tensors, so a decode step can be captured once as an XPUGraph and replayed.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _np2(n: int) -> int:
    return triton.next_power_of_2(n)


# ---------------------------------------------------------------- RMSNorm
@triton.jit
def _rmsnorm_k(X, W, Y, N, stride_x, stride_y, eps, HAS_W: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < N
    x = tl.load(X + row * stride_x + c, mask=m, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / N + eps)
    if HAS_W:
        y = y * tl.load(W + c, mask=m, other=0.0)
    tl.store(Y + row * stride_y + c, y, mask=m)


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """Row-wise RMSNorm over the last dim of a 2D tensor (float32 out)."""
    M, N = x.shape
    out = torch.empty(M, N, dtype=torch.float32, device=x.device) if out is None else out
    _rmsnorm_k[(M,)](x, w if w is not None else x, out, N, x.stride(0), out.stride(0), eps,
                     HAS_W=w is not None, BLOCK=_np2(N), num_warps=8)
    return out


@triton.jit
def _add_norm_k(X, A, W1, S, W2, XO, HO, N, eps,
                HAS_S: tl.constexpr, HAS_W2: tl.constexpr, BLOCK: tl.constexpr):
    """XO = (X + rmsnorm(A) * W1) * S;  HO = rmsnorm(XO) * W2."""
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < N
    a = tl.load(A + row * N + c, mask=m, other=0.0).to(tl.float32)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / N + eps) * tl.load(W1 + c, mask=m, other=0.0)
    x = tl.load(X + row * N + c, mask=m, other=0.0) + a
    if HAS_S:
        x = x * tl.load(S)
    tl.store(XO + row * N + c, x, mask=m)
    if HAS_W2:
        h = x * tl.rsqrt(tl.sum(x * x, 0) / N + eps) * tl.load(W2 + c, mask=m, other=0.0)
        tl.store(HO + row * N + c, h, mask=m)


def add_norm(x, a, w1, eps, scale=None, w2=None, x_out=None, h_out=None):
    """Residual update x + rmsnorm(a)*w1 (times scale), plus the next RMSNorm if w2 is given."""
    M, N = x.shape
    x_out = torch.empty_like(x) if x_out is None else x_out
    if w2 is not None and h_out is None:
        h_out = torch.empty_like(x)
    _add_norm_k[(M,)](x, a, w1, scale if scale is not None else w1, w2 if w2 is not None else w1,
                      x_out, h_out if h_out is not None else x_out, N, eps,
                      HAS_S=scale is not None, HAS_W2=w2 is not None, BLOCK=_np2(N), num_warps=8)
    return (x_out, h_out) if w2 is not None else x_out


# ---------------------------------------------------------------- head norm + RoPE (+ store into KV cache)
@triton.jit
def _norm_rope_k(X, W, COS, SIN, POS, OUT, SLOT, D, stride_xt, stride_xh, stride_ot, stride_oh, eps,
                 HALF: tl.constexpr, HAS_W: tl.constexpr, HAS_SLOT: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    m = d < D
    base = X + t * stride_xt + h * stride_xh
    x = tl.load(base + d, mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    x = x * r
    if HAS_W:
        x = x * tl.load(W + d, mask=m, other=0.0)
    if HALF > 0:
        rot = d < 2 * HALF
        p = tl.where(d < HALF, d + HALF, d - HALF)
        xp = tl.load(base + p, mask=rot, other=0.0).to(tl.float32) * r
        if HAS_W:
            xp = xp * tl.load(W + p, mask=rot, other=0.0)
        pos = tl.load(POS + t)
        i = d % HALF
        cs = tl.load(COS + pos * HALF + i, mask=rot, other=1.0)
        sn = tl.load(SIN + pos * HALF + i, mask=rot, other=0.0)
        sgn = tl.where(d < HALF, -1.0, 1.0)
        x = tl.where(rot, x * cs + sgn * xp * sn, x)
    row = t
    if HAS_SLOT:
        row = tl.load(SLOT + t)
    tl.store(OUT + row * stride_ot + h * stride_oh + d, x, mask=m)


def norm_rope(x, w, eps, cos=None, sin=None, pos=None, out=None, slot=None):
    """x [T, H, D] (strided view ok): per-head RMSNorm (weight w or none), NEOX RoPE on the
    first 2*cos.shape[1] dims at positions pos [T]. Writes to out [T, H, D], or into a KV
    cache out [H, S, D] at rows slot [T]."""
    T, H, D = x.shape
    if out is None:
        out = torch.empty(T, H, D, dtype=torch.float32, device=x.device)
    if slot is None:
        so_t, so_h = out.stride(0), out.stride(1)
    else:
        so_t, so_h = out.stride(1), out.stride(0)
    half = cos.shape[1] if cos is not None else 0
    _norm_rope_k[(T, H)](x, w if w is not None else x, cos if half else x, sin if half else x,
                         pos if half else x, out, slot if slot is not None else x, D,
                         x.stride(0), x.stride(1), so_t, so_h, eps,
                         HALF=half, HAS_W=w is not None, HAS_SLOT=slot is not None,
                         BLOCK_D=_np2(D), num_warps=4)
    return out


# ---------------------------------------------------------------- GELU(tanh)(gate) * up
@triton.jit
def _gelu_mul_k(GU, Y, F, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < F
    g = tl.load(GU + row * 2 * F + c, mask=m, other=0.0)
    u = tl.load(GU + row * 2 * F + F + c, mask=m, other=0.0)
    z = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    t = 1.0 - 2.0 / (tl.exp(2.0 * z) + 1.0)          # tanh(z)
    tl.store(Y + row * F + c, 0.5 * g * (1.0 + t) * u, mask=m)


def gelu_mul(gu: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """gu [M, 2F] = [gate | up] -> gelu_tanh(gate) * up, [M, F]."""
    M, F2 = gu.shape
    F = F2 // 2
    out = torch.empty(M, F, dtype=torch.float32, device=gu.device) if out is None else out
    _gelu_mul_k[(M, triton.cdiv(F, 1024))](gu, out, F, BLOCK=1024, num_warps=4)
    return out


# ---------------------------------------------------------------- decode attention (1 query token, GQA)
@triton.jit
def _attn_decode_k(Q, K, V, NLEN, PM, PL, PO, S, D, G, SPLIT,
                   BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    """One program per (query head, split of the keys): partial softmax over keys
    [split*SPLIT, min((split+1)*SPLIT, n)), scale 1.0. K/V cache [Hkv, S, D]."""
    hq = tl.program_id(0)
    sp = tl.program_id(1)
    nsp = tl.num_programs(1)
    hk = hq // G
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    q = tl.load(Q + hq * D + d, mask=dm, other=0.0).to(tl.float32)
    n = tl.load(NLEN)
    start = sp * SPLIT
    end = tl.minimum(start + SPLIT, n)
    m_i = tl.max(tl.full((BLOCK_S,), -float("inf"), tl.float32), 0)
    l_i = tl.sum(tl.zeros((BLOCK_S,), dtype=tl.float32), 0)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    kb = K + hk * S * D
    vb = V + hk * S * D
    for s0 in range(start, end, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        sm = s < end
        k = tl.load(kb + s[:, None] * D + d[None, :], mask=sm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        sc = tl.sum(k * q[None, :], axis=1)
        sc = tl.where(sm, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new)
        v = tl.load(vb + s[:, None] * D + d[None, :], mask=sm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new
    idx = hq * nsp + sp
    tl.store(PM + idx, m_i)
    tl.store(PL + idx, l_i)
    tl.store(PO + idx * D + d, acc, mask=dm)


@triton.jit
def _attn_combine_k(PM, PL, PO, SINKS, O, NSP, D, BLOCK_SP: tl.constexpr, BLOCK_D: tl.constexpr,
                    HAS_SINK: tl.constexpr):
    hq = tl.program_id(0)
    sp = tl.arange(0, BLOCK_SP)
    spm = sp < NSP
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    m = tl.load(PM + hq * NSP + sp, mask=spm, other=-float("inf"))
    l = tl.load(PL + hq * NSP + sp, mask=spm, other=0.0)
    mx = tl.max(m, 0)
    if HAS_SINK:
        sink = tl.load(SINKS + hq)
        mx = tl.maximum(mx, sink)
    w = tl.where(spm & (m > -float("inf")), tl.exp(m - mx), 0.0)
    o = tl.load(PO + (hq * NSP + sp)[:, None] * D + d[None, :], mask=spm[:, None] & dm[None, :], other=0.0)
    den = tl.sum(w * l, 0)
    if HAS_SINK:
        den = den + tl.exp(sink - mx)
    out = tl.sum(w[:, None] * o, axis=0) / den
    tl.store(O + hq * D + d, out, mask=dm)


def attn_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, n_len: torch.Tensor,
                n_split: int, split: int, out: torch.Tensor | None = None, ws=None,
                sinks: torch.Tensor | None = None) -> torch.Tensor:
    """q [Hq, D]; caches [Hkv, S, D]; n_len int tensor [1] = number of valid keys (slots 0..n-1).
    n_split * split must cover the largest n. sinks [Hq] (or None): softmax sinks per query head,
    added to the denominator only, as gpt-oss does (S16 map S3). Returns [Hq, D] float32."""
    Hq, D = q.shape
    Hkv, S, _ = k_cache.shape
    dev = q.device
    if ws is None:
        ws = (torch.empty(Hq * n_split, device=dev), torch.empty(Hq * n_split, device=dev),
              torch.empty(Hq * n_split * D, device=dev))
    pm, pl, po = ws
    out = torch.empty(Hq, D, dtype=torch.float32, device=dev) if out is None else out
    bs = 16 if D >= 512 else 32
    _attn_decode_k[(Hq, n_split)](q, k_cache, v_cache, n_len, pm, pl, po, S, D, Hq // Hkv, split,
                                  BLOCK_S=bs, BLOCK_D=_np2(D), num_warps=4)
    _attn_combine_k[(Hq,)](pm, pl, po, sinks if sinks is not None else q, out, n_split, D,
                           BLOCK_SP=max(_np2(n_split), 2), BLOCK_D=_np2(D), num_warps=4,
                           HAS_SINK=sinks is not None)
    return out


# ---------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    dev = torch.device("xpu", 0)
    eps = 1e-6

    def ref_rms(x, w):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return y * w if w is not None else y

    x = torch.randn(3, 5376, device=dev) * 3
    w = torch.randn(5376, device=dev)
    print("rms_norm      ", (rms_norm(x, w, eps) - ref_rms(x, w)).abs().max().item())
    a, w2, s = torch.randn_like(x), torch.randn(5376, device=dev), torch.tensor([0.7], device=dev)
    xo, ho = add_norm(x, a, w, eps, scale=s, w2=w2)
    rx = (x + ref_rms(a, w)) * s
    print("add_norm      ", (xo - rx).abs().max().item(), (ho - ref_rms(rx, w2)).abs().max().item())

    from .models.gemma4 import rope_neox, rope_table
    import numpy as np
    T, H, D = 3, 4, 512
    qkv = torch.randn(T, 2 * H * D, device=dev)
    xq = qkv[:, H * D:].view(T, H, D)                    # strided view, like q/k inside qkv
    wq = torch.randn(D, device=dev)
    ff = np.array([1.0] * 64 + [1e30] * 192, dtype=np.float32)
    cos, sin = rope_table(512, 1e6, ff, 64, dev)
    pos = torch.tensor([5, 17, 40], device=dev)
    ref = rope_neox(ref_rms(xq, wq), cos[pos], sin[pos])
    print("norm_rope     ", (norm_rope(xq, wq, eps, cos, sin, pos) - ref).abs().max().item())
    cache = torch.zeros(H, 64, D, dtype=torch.float16, device=dev)
    slot = torch.tensor([9, 3, 60], device=dev)
    norm_rope(xq, wq, eps, cos, sin, pos, out=cache, slot=slot)
    print("norm_rope→kv  ", (cache[:, slot].transpose(0, 1).float() - ref).abs().max().item())
    print("norm (no rope)", (norm_rope(xq, None, eps) - ref_rms(xq, None)).abs().max().item())

    gu = torch.randn(2, 2 * 10752, device=dev) * 3
    import torch.nn.functional as F
    print("gelu_mul      ", (gelu_mul(gu) - F.gelu(gu[:, :10752], approximate="tanh") * gu[:, 10752:]).abs().max().item())

    for Hq, Hkv, D, S, n in [(16, 8, 256, 1024, 1024), (16, 8, 256, 1024, 37), (16, 2, 512, 8192, 3000)]:
        q = torch.randn(Hq, D, device=dev)
        kc = (torch.randn(Hkv, S, D, device=dev) * 0.1).half()
        vc = torch.randn(Hkv, S, D, device=dev).half()
        G = Hq // Hkv
        sc = torch.einsum("hgd,hsd->hgs", q.view(Hkv, G, D), kc[:, :n].float())
        ref = torch.einsum("hgs,hsd->hgd", sc.softmax(-1), vc[:, :n].float()).reshape(Hq, D)
        split = 256
        nsp = triton.cdiv(S, split)
        o = attn_decode(q, kc, vc, torch.tensor([n], device=dev, dtype=torch.int32), nsp, split)
        print(f"attn_decode Hq={Hq} Hkv={Hkv} D={D} n={n}: {(o - ref).abs().max().item():.2e}")
        sk = torch.randn(Hq, device=dev) * 2.0
        m2 = torch.maximum(sc.max(-1, keepdim=True).values, sk.view(Hkv, G, 1))
        p2 = (sc - m2).exp()
        ref2 = (p2 @ vc[:, :n].float()) / (p2.sum(-1, keepdim=True) + (sk.view(Hkv, G, 1) - m2).exp())
        o2 = attn_decode(q, kc, vc, torch.tensor([n], device=dev, dtype=torch.int32), nsp, split, sinks=sk)
        print(f"attn_decode+sinks Hq={Hq} Hkv={Hkv} D={D} n={n}: {(o2 - ref2.reshape(Hq, D)).abs().max().item():.2e}")


if __name__ == "__main__":
    _selftest()
