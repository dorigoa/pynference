"""Gated DeltaNet core for the Qwen3.8-27B recurrent layers (plan S20).

CPU references for the three pieces the Triton kernels must reproduce (S19
map A, `delta-net-base.cpp` + `ggml/src/ggml-cpu/ops.cpp`):
causal depthwise conv1d with state (kernel 4, no bias, then SiLU), the gated
delta-rule recurrence over a 128x128 state per value head, and the gated RMS
norm before `ssm_out`.

Two independent references live side by side:

* vectorized torch (`conv_prefill`, `delta_prefill`, `delta_decode`, ...) —
  state held in natural layout, `state[h][i][j] = M[i][j]` with `i` the
  key/query channel and `j` the value channel;
* `*_ops_ref` numpy transcriptions of the llama.cpp CPU kernels
  (`ggml_compute_forward_ssm_conv_f32` and
  `ggml_compute_forward_gated_delta_net_one_chunk`, K=1, scalar-gate
  variant), which keep ops.cpp's transposed scratch layout
  `s_out[j*S + i] = M[i][j]` and its loops, computed in float32 like the
  C++ code. They cross-check the vectorized path on CPU (S20 step 1) and
  must stay untouched by kernel work.

Conventions: batch is one sequence (decode and chunked prefill); value head
`j` uses key/query head `j % H_k` — llama.cpp round-robin (ops.cpp:11079),
not the `j // 3` the plan first guessed (S19 map A.12); the delta-rule
output is scaled by 1/sqrt(S); the conv weight `w` is logical `[C, 4]`
(channel-major taps), which is also the raw GGUF byte order of
`ssm_conv1d` (`{4, 10240}` row-major, qwen35.cpp:87).

The Triton kernels at the bottom (`decode`, `prefill`) reproduce the
vectorized references on GPU: conv + SiLU one channel block at a time,
then one program per value head holding the whole 128x128 state in
registers. Both update `conv_state` and `state` in place, as the decode
loop needs; callers pass copies when they also want the old values.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


def _np2(n: int) -> int:
    return triton.next_power_of_2(n)


def _key_map(n_heads: int, n_key_heads: int, device: torch.device) -> torch.Tensor:
    """Index tensor mapping each value head to its key/query head (j % H_k)."""
    return torch.arange(n_heads, device=device) % n_key_heads


def _silu(x):
    return x * torch.sigmoid(x)


# ------------------------------------------------------------- conv with state
def conv_prefill(x, conv_state, w):
    """Causal depthwise conv over a token chunk, then SiLU (vectorized).

    x:          [T, C]      per-token conv input (the qkv rows, post-projection)
    conv_state: [K-1, C]    the K-1 oldest rows, oldest first
    w:          [C, K]      taps per channel
    returns:    y [T, C] after SiLU, new_state [K-1, C] (the K-1 newest rows)
    """
    ins = torch.cat([conv_state, x], dim=0)              # [K-1+T, C]
    T, K = x.shape[0], w.shape[1]
    idx = torch.arange(T, device=x.device).unsqueeze(1) + torch.arange(K, device=x.device)
    y = (ins[idx] * w.t().unsqueeze(0)).sum(dim=1)       # y[t] = sum_i in[t+i] * w[i]
    return _silu(y), ins[-(K - 1):].clone()


def conv_decode(x_new, conv_state, w):
    """One token through `conv_prefill` (T=1); x_new [C], y [C]."""
    y, new_state = conv_prefill(x_new.unsqueeze(0), conv_state, w)
    return y[0], new_state


def conv_ops_ref(x, conv_state, w):
    """Transcription of ggml_compute_forward_ssm_conv_f32 (ops.cpp:9811) + SiLU.

    numpy/float32, explicit token/channel/tap loops like the C++ rows loop.
    Same shapes and return values as `conv_prefill`.
    """
    x = np.asarray(x, dtype=np.float32)
    conv_state = np.asarray(conv_state, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    ins = np.concatenate([conv_state, x], axis=0)        # {d_conv-1+n_t, d_inner}
    ncs, nr = ins.shape
    nc = w.shape[1]
    T = x.shape[0]
    out = np.empty((T, nr), dtype=np.float32)
    for i2 in range(T):                                  # tokens
        for i1 in range(nr):                             # d_inner row
            sumf = np.float32(0.0)                       # f32 accumulator, as in ops.cpp
            for i0 in range(nc):                         # d_conv
                sumf += ins[i2 + i0, i1] * w[i1, i0]
            out[i2, i1] = sumf
    return out * (np.float32(1.0) / (np.float32(1.0) + np.exp(-out))), ins[-(nc - 1):].copy()


# ----------------------------------------------------- gated delta recurrence
def delta_prefill(q, k, v, g, beta, state):
    """Gated delta rule over a token chunk, vectorized over heads and state.

    q, k:       [T, H_k, S] post-conv, post-L2-norm
    v:          [T, H_v, S]
    g, beta:    [T, H_v]    log-decay (<= 0) and gate (in (0, 1))
    state:      [H_v, S, S] running M, natural layout state[h][i][j] = M[i][j]
    returns:    out [T, H_v, S] (scaled by 1/sqrt(S)), new_state [H_v, S, S]
    """
    T, H_k, S = q.shape
    H_v = v.shape[1]
    scale = 1.0 / (S ** 0.5)
    M = state.clone()
    km = _key_map(H_v, H_k, q.device)
    kk = k[:, km]                                        # [T, H_v, S]
    qq = q[:, km]
    out = torch.empty_like(v)
    for t in range(T):
        M = M * torch.exp(g[t]).view(-1, 1, 1)           # M *= exp(g)
        delta = (v[t] - torch.einsum('hij,hi->hj', M, kk[t])) * beta[t].view(-1, 1)
        M = M + kk[t].unsqueeze(2) * delta.unsqueeze(1)  # M[i][j] += k[i] * delta[j]
        out[t] = torch.einsum('hij,hi->hj', M, qq[t]) * scale
    return out, M


def delta_decode(q, k, v, g, beta, state):
    """One token through `delta_prefill` (T=1); q,k [H_k,S], v [H_v,S], out [H_v,S]."""
    out, new_state = delta_prefill(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                                   g.unsqueeze(0), beta.unsqueeze(0), state)
    return out[0], new_state


def delta_ops_ref(q, k, v, g, beta, state):
    """Transcription of ggml_compute_forward_gated_delta_net_one_chunk (ops.cpp:11003).

    numpy/float32, K=1 non-KDA variant: transposed scratch `s[j*S + i] =
    M[i][j]` (ops.cpp:11104), per head then per token, with the three
    per-channel loops (dot+beta, outer product, output dot) kept separate as
    in the C++; the inner `ggml_vec_*` calls become whole-row numpy ops.
    Same shapes and return values as `delta_prefill`.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    T, H_k, S = q.shape
    H_v = v.shape[1]
    scale = np.float32(1.0) / np.sqrt(np.float32(S))
    attn = np.empty((T, H_v, S), dtype=np.float32)
    new_state = np.empty_like(state)
    delta = np.empty(S, dtype=np.float32)
    for iv1 in range(H_v):                               # rows = heads, one seq
        s = np.ascontiguousarray(state[iv1].T).reshape(-1)  # copy: s[j*S+i] = M[i][j]
        kd_all = np.ascontiguousarray(k[:, iv1 % H_k])
        qd_all = np.ascontiguousarray(q[:, iv1 % H_k])
        for t in range(T):
            s *= np.exp(g[t, iv1], dtype=np.float32)     # ggml_vec_scale_f32
            kd, qd = kd_all[t], qd_all[t]
            b = beta[t, iv1]
            for j in range(S):                           # delta[j] = (v[j] - dot(row_j, k)) * beta
                row = s[j * S:(j + 1) * S]
                ssum = np.dot(row, kd)                    # ggml_vec_dot_f32, f32 in/out
                delta[j] = (v[t, iv1, j] - ssum) * b
            for j in range(S):                           # outer product: row_j += delta[j] * k
                row = s[j * S:(j + 1) * S]
                row += kd * delta[j]                     # ggml_vec_mad_f32
            for j in range(S):                           # attn[j] = dot(row_j, q) * scale
                row = s[j * S:(j + 1) * S]
                attn[t, iv1, j] = np.dot(row, qd) * scale
        new_state[iv1] = s.reshape(S, S).T               # undo the transposed scratch
    return attn, new_state


# ------------------------------------------------------------------ norms
def l2norm_qk(x, eps):
    """Per-head L2 norm of q/k rows: x / sqrt(sum(x^2) + eps) (Mappa A.8)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def norm_gated(out, z, w_norm, eps):
    """RMSNorm over each head's S channels with weight, gated by SiLU (Mappa A.10).

    out:     [..., S] per-head slices (e.g. [H_v, S] for one token)
    z:       same shape as out, the `attn_gate` slice for the token
    w_norm:  [S] ssm_norm weight
    returns: same shape as out
    """
    n = out.shape[-1]
    return out * torch.rsqrt((out * out).sum(dim=-1, keepdim=True) / n + eps) \
        * w_norm * _silu(z)


def norm_gated_ops_ref(out, z, w_norm, eps):
    """numpy/float32 per-head transcription of build_norm_gated (qwen35.cpp:275)."""
    out = np.asarray(out, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    w_norm = np.asarray(w_norm, dtype=np.float32)
    n = out.shape[-1]
    rms = np.sqrt((out * out).sum(-1) / np.float32(n) + np.float32(eps))
    y = (out / rms[..., None]) * w_norm
    return y * (z / (np.float32(1.0) + np.exp(-z)))


# ------------------------------------------------------------------ decode kernel
@triton.jit
def _conv_decode_k(X, ST, W, Y, C, stride_st, K: tl.constexpr, BLOCK_C: tl.constexpr):
    """One token through the causal depthwise conv + SiLU; `conv_state` in place.

    X: [C] raw conv input; ST: [K-1, C] rows oldest first, shifted after the
    output is written; W: [C, K] taps; Y: [C] post-SiLU output.
    """
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    m = c < C
    x = tl.load(X + c, mask=m, other=0.0)
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for i in tl.static_range(K - 1):
        acc += tl.load(ST + i * stride_st + c, mask=m, other=0.0) \
            * tl.load(W + c * K + i, mask=m, other=0.0)
    acc += x * tl.load(W + c * K + (K - 1), mask=m, other=0.0)
    tl.store(Y + c, acc * tl.sigmoid(acc), mask=m)
    for u in tl.static_range(K - 1):      # row u <- old row u+1 (or x): reads
        if u + 1 < K - 1:                 # only rows this program has not
            v = tl.load(ST + (u + 1) * stride_st + c, mask=m, other=0.0)
        else:
            v = x
        tl.store(ST + u * stride_st + c, v, mask=m)


@triton.jit
def _delta_decode_k(Y, G, BETA, ST, OUT, H_K, S, eps, scale, BLOCK_S: tl.constexpr):
    """One token of the gated delta rule, one program per value head.

    The whole [S, S] state of the head lives in registers and is written back
    in place. Y: [C] post-conv rows, split q | k | v (widths H_k*S, H_k*S,
    H_v*S); G, BETA: [H_v]; OUT: [H_v, S]. q/k get the per-head L2 norm of
    Mappa A.8; heads share key/query heads as `h % H_K` (ops.cpp:11079).
    """
    h = tl.program_id(0)
    km = h % H_K
    s = tl.arange(0, BLOCK_S)
    m = s < S
    q = tl.load(Y + km * S + s, mask=m, other=0.0)
    k = tl.load(Y + (H_K + km) * S + s, mask=m, other=0.0)
    v = tl.load(Y + (2 * H_K + h) * S + s, mask=m, other=0.0)
    q = q * tl.rsqrt(tl.sum(q * q, 0) + eps)
    k = k * tl.rsqrt(tl.sum(k * k, 0) + eps)
    ii = tl.arange(0, BLOCK_S)[:, None]
    m2 = (ii < S) & (s[None, :] < S)
    base = ST + h * S * S + ii * S + s[None, :]
    M = tl.load(base, mask=m2, other=0.0) * tl.exp(tl.load(G + h))
    delta = (v - tl.sum(M * k[:, None], 0)) * tl.load(BETA + h)
    M = M + k[:, None] * delta[None, :]
    tl.store(OUT + h * S + s, tl.sum(M * q[:, None], 0) * scale, mask=m)
    tl.store(base, M, mask=m2)


def decode(x_new, conv_state, w, g, beta, state, n_key_heads, qk_eps=1e-6):
    """One token through conv+SiLU and the gated delta rule (Triton, f32).

    x_new [C] raw conv input (the qkv rows), conv_state [K-1, C] and
    state [H_v, S, S] are updated **in place**; w [C, K]; g/beta [H_v].
    Returns the raw delta output [H_v, S] scaled by 1/sqrt(S): `norm_gated`
    with the `attn_gate` slice and `ssm_out` come later, in the model.
    """
    C = x_new.shape[0]
    H_v, S, S2 = state.shape
    assert S == S2
    y = torch.empty_like(x_new)
    _conv_decode_k[(triton.cdiv(C, 1024),)](
        x_new, conv_state, w, y, C, conv_state.stride(0),
        K=w.shape[1], BLOCK_C=1024, num_warps=4)
    out = torch.empty(H_v, S, dtype=torch.float32, device=x_new.device)
    _delta_decode_k[(H_v,)](
        y, g, beta, state, out, n_key_heads, S, qk_eps, S ** -0.5,
        BLOCK_S=_np2(S), num_warps=8)
    return out


# --------------------------------------------------- fused decode gates / norm (S22)
@triton.jit
def _delta_gates_k(AB, A_LOG, DT, G, BETA, H, BLOCK: tl.constexpr):
    """g = a_log * softplus(alpha + dt_bias), beta = sigmoid(beta_raw) (Mappa
    A.4); the softplus threshold 20 matches F.softplus. AB [2H] = [alpha |
    beta], one token's rows of the merged `alpha_beta` projection."""
    i = tl.arange(0, BLOCK)
    m = i < H
    x = tl.load(AB + i, mask=m, other=0.0) + tl.load(DT + i, mask=m, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tl.store(G + i, tl.load(A_LOG + i, mask=m, other=0.0) * sp, mask=m)
    tl.store(BETA + i, tl.sigmoid(tl.load(AB + H + i, mask=m, other=0.0)), mask=m)


def gates(ab, a_log, dt_bias):
    """(g, beta), each [H_v], replacing the torch chain the S21 decode used."""
    H = a_log.shape[0]
    g = torch.empty(H, dtype=torch.float32, device=ab.device)
    beta = torch.empty_like(g)
    _delta_gates_k[(1,)](ab, a_log, dt_bias, g, beta, H, BLOCK=_np2(H), num_warps=4)
    return g, beta


@triton.jit
def _norm_gated_k(Y, Z, W, OUT, S, eps, stride_z, BLOCK_S: tl.constexpr):
    """Per-head RMSNorm over S channels with weight, gated by SiLU — the
    decode shape (one program per value head); `z` may be a row-strided
    view of the [H_v, S] `attn_gate` GEMV output."""
    h = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    m = s < S
    y = tl.load(Y + h * S + s, mask=m, other=0.0)
    z = tl.load(Z + h * stride_z + s, mask=m, other=0.0)
    n = y * tl.rsqrt(tl.sum(y * y, 0) / S + eps) * tl.load(W + s, mask=m, other=0.0)
    tl.store(OUT + h * S + s, n * z * tl.sigmoid(z), mask=m)


def norm_gated_decode(y, z, w_norm, eps, out=None):
    """`norm_gated` for one token: y/z [H_v, S] float32 (z rows may stride
    over a wider GEMV output), same result as the torch `norm_gated`."""
    H_v, S = y.shape
    out = torch.empty(H_v, S, dtype=torch.float32, device=y.device) if out is None else out
    _norm_gated_k[(H_v,)](y, z, w_norm, out, S, eps, z.stride(0),
                          BLOCK_S=_np2(S), num_warps=4)
    return out


# ----------------------------------------------------------------- prefill kernel
@triton.jit
def _conv_prefill_k(X, ST, STN, W, Y, T, C, stride_x, stride_st, stride_stn,
                    K: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    """Causal depthwise conv + SiLU over a token chunk, grid over channel/token blocks.

    X/Y: [T, C]; W: [C, K]. ST: [K-1, C] is read-only (programs across token
    blocks would race on an in-place shift); the K-1 newest rows of the
    padded input go to STN, written only by the `pid_t == 0` programs.
    """
    cb = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tb = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    mc = cb < C
    acc = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
    for i in tl.static_range(K):
        r = tb + i                                     # row of the padded input
        h = tl.load(ST + r[:, None] * stride_st + cb[None, :],
                    mask=(r < K - 1)[:, None] & mc[None, :], other=0.0)
        # `(tb < T)` too: the last token block would otherwise read past X's
        # rows (garbage fed to sigmoid; masked from the store, but the load
        # itself is out of bounds — caught under TRITON_INTERPRET in S21)
        xv = tl.load(X + (r - (K - 1))[:, None] * stride_x + cb[None, :],
                     mask=(r >= K - 1)[:, None] & (tb < T)[:, None] & mc[None, :],
                     other=0.0)
        acc += (h + xv) * tl.load(W + cb[None, :] * K + i, mask=mc[None, :], other=0.0)
    y = acc * tl.sigmoid(acc)
    tl.store(Y + tb[:, None] * C + cb[None, :], y, mask=(tb < T)[:, None] & mc[None, :])
    if tl.program_id(1) == 0:
        # `r`/`xv` are still live from the loop above with block shapes; the
        # history branch needs fresh names or Triton rejects the redefinition.
        for u in tl.static_range(K - 1):
            # block-shaped row index: a scalar `r2` mask compiles away on GPU
            # but the TRITON_INTERPRET CPU path rejects the scalar & tensor
            # mask combination (S21 test of the model layer); same semantics
            r2 = tl.full([BLOCK_C], T + u, tl.int32)
            hv = tl.load(ST + r2 * stride_st + cb, mask=(r2 < K - 1) & mc, other=0.0)
            xv2 = tl.load(X + (r2 - (K - 1)) * stride_x + cb,
                          mask=(r2 >= K - 1) & mc, other=0.0)
            tl.store(STN + u * stride_stn + cb, hv + xv2, mask=mc)


@triton.jit
def _delta_prefill_k(Y, G, BETA, ST, OUT, T, H_K, S, eps, scale,
                     stride_y, stride_g, stride_out, BLOCK_S: tl.constexpr):
    """Gated delta rule over a chunk: sequential in t, one program per value head.

    The head's whole [S, S] state stays in registers across the t loop and is
    written back in place (single writer per head). Y: [T, C] post-conv rows
    split q | k | v; G/BETA: [T, H_v]; OUT: [T, H_v, S]. Same per-token math
    as `_delta_decode_k`.
    """
    h = tl.program_id(0)
    km = h % H_K
    s = tl.arange(0, BLOCK_S)
    m = s < S
    ii = tl.arange(0, BLOCK_S)[:, None]
    m2 = (ii < S) & (s[None, :] < S)
    base = ST + h * S * S + ii * S + s[None, :]
    M = tl.load(base, mask=m2, other=0.0)
    q0 = km * S
    k0 = (H_K + km) * S
    v0 = (2 * H_K + h) * S
    for t in range(T):
        q = tl.load(Y + t * stride_y + q0 + s, mask=m, other=0.0)
        k = tl.load(Y + t * stride_y + k0 + s, mask=m, other=0.0)
        v = tl.load(Y + t * stride_y + v0 + s, mask=m, other=0.0)
        q = q * tl.rsqrt(tl.sum(q * q, 0) + eps)
        k = k * tl.rsqrt(tl.sum(k * k, 0) + eps)
        M = M * tl.exp(tl.load(G + t * stride_g + h))
        delta = (v - tl.sum(M * k[:, None], 0)) * tl.load(BETA + t * stride_g + h)
        M = M + k[:, None] * delta[None, :]
        tl.store(OUT + t * stride_out + h * S + s,
                 tl.sum(M * q[:, None], 0) * scale, mask=m)
    tl.store(base, M, mask=m2)


def prefill(x, conv_state, w, g, beta, state, n_key_heads, qk_eps=1e-6):
    """A token chunk through conv+SiLU and the gated delta rule (Triton, f32).

    x [T, C] raw conv input (the qkv rows); conv_state [K-1, C] and
    state [H_v, S, S] are updated **in place** (the conv result is staged in
    a second buffer, since ST is read-only in the conv kernel); w [C, K];
    g/beta [T, H_v]. Returns the raw delta output [T, H_v, S] scaled by
    1/sqrt(S), same per-token semantics as `decode`.
    """
    T, C = x.shape
    H_v, S, S2 = state.shape
    assert S == S2
    y = torch.empty(T, C, dtype=torch.float32, device=x.device)
    new_hist = torch.empty_like(conv_state)
    _conv_prefill_k[(triton.cdiv(C, 256), triton.cdiv(T, 32))](
        x, conv_state, new_hist, w, y, T, C,
        x.stride(0), conv_state.stride(0), new_hist.stride(0),
        K=w.shape[1], BLOCK_C=256, BLOCK_T=32, num_warps=4)
    conv_state.copy_(new_hist)
    out = torch.empty(T, H_v, S, dtype=torch.float32, device=x.device)
    _delta_prefill_k[(H_v,)](
        y, g, beta, state, out, T, n_key_heads, S, qk_eps, S ** -0.5,
        y.stride(0), g.stride(0), out.stride(0),
        BLOCK_S=_np2(S), num_warps=8)
    return out
