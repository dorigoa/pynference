"""Qwen3.8-Flash-Next (architecture `qwen4exp`) modules — plan S24.

One device-agnostic FP32 torch implementation: the CPU reference (tests/cpu,
torch CPU) and the GPU run the SAME code on `xpu`, like S20/S22. Layouts are
torch order [T, C]; GGML prints [C, T] with the flat stream index
`e + 2560*s + 10240*t`, so a GGML [hc*n_embd, T] tensor is our [T, hc*n_embd]
(S23 map A). Maps A/C in docs/plans/pynference/reports/ are the reference for
the math; constants live in tests/golden/qwen4exp/facts.json
(metadata.qwen4exp.ple.*).

This file currently holds the PLE (only blk.1 has it): the n-gram row hash
computed host-side, the IQ4_NL embedding-table gather, and the gated
key/value + dilated depthwise-conv block. The hyper-connection mixer
(`hc_attn_*`/`hc_ffn_*`/`output_hc_*`) is added by S24 step 3.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from pynference.quant.ref import ref_dequant
from pynference.quant.weight import NativeQWeight


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def grouped_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last (channel) dim, per residual stream, times gamma.

    x [..., C] f32, gamma [C] or broadcastable [..., C]. This is the
    RMS_NORM+MUL pair llama.cpp builds for `ple_norm_*` (and the HC mixers):
    the stored gamma is multiplied as-is, no (1+w) folding on our side
    (the converter already folded it in the file if the model wants it).
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def ple_row_indices(
    tokens,
    history,
    *,
    ngram: int,
    heads_per_ngram: int,
    multipliers,
    head_offsets,
    head_vocab_sizes,
    eos_token_id: int,
) -> np.ndarray:
    """Table rows for each token: int64 [n_heads * T], token-major, head
    index inner (`rows[t * n_heads + h]`, h = (n-2)*heads_per_ngram + g).

    Host-side replacement for the int64/xor ops ggml lacks (S23 map C,
    qwen4exp.cpp set_input 1293-1352). Window `ctx[s]`: ctx[0] is the token
    itself (its OWN eos never cuts the window); ctx[s] is the predecessor s
    steps back, read as eos when missing; an eos in the window cuts, so every
    position at-or-after it becomes eos. Then, per n-gram size n = 2..ngram,
    `mixed = XOR_s (ctx[s] * multipliers[s])` in uint64 and
    `row = mixed % head_vocab_sizes[h] + head_offsets[h]`.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    history = np.asarray(history, dtype=np.int64)
    n_heads = len(head_offsets)
    assert n_heads == (ngram - 1) * heads_per_ngram
    g = np.concatenate([history, tokens])
    T, H = tokens.shape[0], history.shape[0]

    ctx = np.full((T, ngram), eos_token_id, dtype=np.int64)
    for i in range(T):
        pos = H + i
        ctx[i, 0] = g[pos]
        cut = False
        for s in range(1, ngram):
            if cut or pos - s < 0:
                cut = True  # missing predecessor reads as eos and cuts too
            else:
                t = g[pos - s]
                if t == eos_token_id:
                    cut = True
                else:
                    ctx[i, s] = t

    mul = np.array(multipliers, dtype=np.uint64)
    offs = np.array(head_offsets, dtype=np.uint64)
    voc = np.array(head_vocab_sizes, dtype=np.uint64)
    cu = ctx.astype(np.uint64)
    rows = np.empty((T, n_heads), dtype=np.int64)
    for n in range(2, ngram + 1):
        mixed = cu[:, 0] * mul[0]  # uint64 mul/xor wrap mod 2**64 like C++
        for s in range(1, n):
            mixed = mixed ^ (cu[:, s] * mul[s])
        h0 = (n - 2) * heads_per_ngram
        sel = mixed[:, None] % voc[None, h0:h0 + heads_per_ngram]
        sel = sel + offs[None, h0:h0 + heads_per_ngram]
        rows[:, h0:h0 + heads_per_ngram] = sel
    return rows.reshape(-1)


def ple_gather_embedding(g, name: str, idx: np.ndarray, *, device) -> torch.Tensor:
    """Rows `idx` (int64, from ple_row_indices) of the n-gram table, already
    dequantized float32 [len(idx), head_dim].

    Only the selected bytes move: on CPU they go through `quant.ref`
    (gguf-py, the reference the kernels are checked against); on GPU the
    gathered bytes become a small NativeQWeight and `rows_dequant` expands
    them on device (S24 plan: pinned/mmap gather -> H2D -> dequant).
    """
    info = g.tensors[name]
    head_dim = info.shape[0]
    rows = np.concatenate([g.rows(name, int(r), int(r) + 1) for r in np.asarray(idx)])
    device = torch.device("cpu") if device is None else torch.device(device)
    if device.type == "cpu":
        return torch.from_numpy(ref_dequant(rows, info.type))
    qw = NativeQWeight(info.type, torch.from_numpy(rows).to(device), rows.shape[0], head_dim)
    return qw.rows_dequant(torch.arange(rows.shape[0], device=device))


class PLE:
    """The PLE block (qwen4exp.cpp build_ple 1417-1505), run on `res_hc`
    before blk.1's attention mixer.

    Inputs/outputs are torch [T, ...]: `res_hc` [T, hc*n_embd] (in and out),
    `emb` [T, n_embd] (gathered table rows, head dim slowest: columns
    [160h, 160h+160) = head h), `state` [hist, hc*n_embd] — the causal conv
    history ((kernel-1)*ngram = 9 rows) that the block takes and returns;
    owning that row per sequence is the engine's job (S27).

    `key`/`value` are QWeights (ple_key: n = hc*n_embd, k = n_embd;
    ple_value: n = k = n_embd); `norm_*` are [hc, n_embd] F32; `conv_w` is
    [hc*n_embd, kernel] — the GGUF [4, 10240] F32 viewed with the tap index
    last (flat file index = channel*4 + tap).
    """

    def __init__(self, key, value, norm_key, norm_query, norm_conv, conv_w, *,
                 n_embd: int, ngram: int, eps: float):
        self.key, self.value = key, value
        self.norm_key, self.norm_query, self.norm_conv = norm_key, norm_query, norm_conv
        self.conv_w = conv_w
        self.n_embd, self.ngram, self.eps = n_embd, ngram, eps
        self.hc = norm_key.shape[0]
        self.channels = self.hc * n_embd
        self.kernel = conv_w.shape[1]
        self.hist = (self.kernel - 1) * ngram
        assert conv_w.shape[0] == self.channels

    def zero_state(self, device=None) -> torch.Tensor:
        return torch.zeros(self.hist, self.channels, device=device)

    def forward(self, res_hc: torch.Tensor, emb: torch.Tensor, state: torch.Tensor):
        """-> (res_hc_out, new_state, parts).

        parts keeps the cb()-dumped intermediates for the dump comparison
        (torch layouts: ple_gate [T, hc], *_value/conv [T, hc, n_embd]).
        """
        T = res_hc.shape[0]
        C, hc = self.n_embd, self.hc

        raw_key = self.key(emb).view(T, hc, C)
        key = grouped_norm(raw_key, self.norm_key, self.eps)
        query = grouped_norm(res_hc.view(T, hc, C), self.norm_query, self.eps)
        s = (key * query).sum(-1) * (1.0 / math.sqrt(C))          # [T, hc]
        gate_in = torch.sign(s) * torch.sqrt(s.abs().clamp(min=1e-6))
        gate = torch.sigmoid(gate_in)

        value = self.value(emb)                                    # [T, C]
        gated = value.unsqueeze(1) * gate.unsqueeze(-1)            # [T, hc, C]

        normed = grouped_norm(gated, self.norm_conv, self.eps).reshape(T, self.channels)
        x = torch.cat([state, normed], 0)                          # [hist + T, hc*C]
        conv = torch.zeros(T, self.channels, dtype=res_hc.dtype, device=res_hc.device)
        for k in range(self.kernel):  # causal, dilation = ngram
            start = self.hist - (self.kernel - 1 - k) * self.ngram
            conv += x[start:start + T] * self.conv_w[:, k]
        conv_out = silu(conv).view(T, hc, C)

        res = res_hc + (gated + conv_out).reshape(T, self.channels)
        new_state = x[-self.hist:].clone()  # write-back tail (build_conv_state_at)
        parts = {
            "ple_key_raw": raw_key, "ple_key": key, "ple_value": value,
            "ple_query": query, "ple_s": s, "ple_gate_in": gate_in,
            "ple_gate": gate, "ple_gated_value": gated,
            "ple_conv_normed": normed, "ple_conv_out": conv_out,
        }
        return res, new_state, parts
