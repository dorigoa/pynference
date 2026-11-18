"""Qwen3.8-27B (GGUF arch "qwen35"), tensor-parallel.

Follows llama.cpp src/models/qwen35.cpp + delta-net-base.cpp (S19 map:
reports/S19-map.md):
- 64 main layers: `blk.64` is the MTP draft block, never executed (D4);
  is_attn[il] = (il + 1) % 4 == 0 -> 16 full-attention layers, 48 gated
  DeltaNet layers (Mappa B.11);
- DeltaNet: qkv [10240 = q 2048 | k 2048 | v 6144] + gate, causal depthwise
  conv k=4 + SiLU, per-head L2 norm on q/k, gated delta rule with scalar
  decay g = ssm_a * softplus(ssm_alpha.x + ssm_dt.bias) (ssm_a already
  holds -exp(A_log), Mappa A.4), gated per-head RMS norm with silu(z),
  ssm_out. The value head j uses the key head j mod 16 (llama.cpp
  round-robin, ops.cpp:11079, NOT j // 3 - see S19 map A.12);
- attention: attn_q [5120,12288] with q|gate interleaved per head (512
  rows: q 256, gate 256, Mappa B.2), per-head RMS q/k norms before RoPE,
  partial NEOX RoPE on the first 64 of 256 dims (M-RoPE sections collapse
  to one position for text, Mappa B.4), scale 1/sqrt(256), causal, GQA
  kv-head = h/6, output gate attn_out *= sigmoid(gate) before attn_output;
- dense SwiGLU FFN down(silu(gate.x) * (up.x)); no biases anywhere.

Tensor split (S19 map "Piano TP"): rank r takes the DeltaNet heads of
parity r (j % world == r), gathered contiguous at load, so inside a rank
the key rule stays h % (H_k / world) and `pynference.deltanet` needs no
new API; attention heads and FFN rows split contiguously; ssm_out /
attn_output / ffn_down split by input columns + all-reduce; the
vocabulary is split for the embedding lookup (masked + all-reduce) and
the logits head (all_gather).  Mappa B.9 said token_embd and output.weight
are tied; in the Qwen3.8-27B-Q8_0 checkpoint they are NOT (separate Q8_0
tensors, S21) - load output.weight when present, tie only when absent.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..gguf import GGUFFile
from ..parallel import TP
# ..quant (QLinear) and ..deltanet (Triton) are imported inside the helpers
# that need them: importing Triton at module load breaks the TRITON_INTERPRET
# contract of tests/cpu (S15/S20 reports).
from .gptoss import attn_prefill, dense_prefill, rms_norm, rope_neox  # Triton-free


@dataclass
class Qwen35Config:
    n_layer: int          # main layers; the MTP block is excluded (D4)
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int            # RoPE rotates the first n_rot of head_dim dims
    ffn: int
    rope_base: float
    eps: float
    n_vocab: int
    n_ctx_train: int
    is_attn: list[bool]
    conv_kernel: int
    ssm_state: int        # S, the delta-state square side (128)
    n_key_heads: int      # ssm.group_count (16)
    n_value_heads: int    # ssm.time_step_rank (48)

    @classmethod
    def from_gguf(cls, g: GGUFFile) -> "Qwen35Config":
        n_blocks = int(g.get("block_count"))
        mtp = int(g.get("nextn_predict_layers", 0))
        if mtp > 1:
            raise NotImplementedError(f"qwen35: {mtp} MTP blocks (D4 loads at most one, and never runs it)")
        n_layer = n_blocks - min(mtp, 1)
        interval = int(g.get("full_attention_interval", 4))
        is_attn = [(il + 1) % interval == 0 for il in range(n_blocks)]
        return cls(
            n_layer=n_layer,
            n_embd=int(g.get("embedding_length")),
            n_head=int(g.get("attention.head_count")),
            n_head_kv=int(g.get("attention.head_count_kv")),
            head_dim=int(g.get("attention.key_length")),
            n_rot=int(g.get("rope.dimension_count", g.get("attention.key_length"))),
            ffn=int(g.get("feed_forward_length")),
            rope_base=float(g.get("rope.freq_base")),
            eps=float(g.get("attention.layer_norm_rms_epsilon")),
            n_vocab=int(g.tensors["token_embd.weight"].shape[1]),
            n_ctx_train=int(g.get("context_length")),
            is_attn=is_attn[:n_layer],
            conv_kernel=int(g.get("ssm.conv_kernel")),
            ssm_state=int(g.get("ssm.state_size")),
            n_key_heads=int(g.get("ssm.group_count")),
            n_value_heads=int(g.get("ssm.time_step_rank")),
        )


def rope_table(n_rot: int, base: float, n_pos: int, device):
    """Plain NEOX cos/sin tables [n_pos, n_rot/2] - gemma4.rope_table with
    factors=None. Copied here because gemma4 imports Triton at module load
    (TRITON_INTERPRET contract, S20 note); M-RoPE sections [11,11,10,0]
    share one position for text (S19 map B.4), so partial RoPE over the
    first n_rot dims is what llama.cpp computes."""
    inv = 1.0 / base ** (np.arange(n_rot // 2, dtype=np.float64) * 2 / n_rot)
    ang = np.arange(n_pos, dtype=np.float64)[:, None] * inv[None, :]
    return (torch.from_numpy(np.cos(ang).astype(np.float32)).to(device),
            torch.from_numpy(np.sin(ang).astype(np.float32)).to(device))


def merge_runs(runs: list[list[int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, l in runs:
        if out and out[-1][0] + out[-1][1] == a:
            out[-1][1] += l
        else:
            out.append([a, l])
    return [tuple(r) for r in out]


def split_run(n: int, world: int, rank: int) -> tuple[int, int]:
    """This rank's contiguous slice of a dimension of size n as a
    (start, len) run. tp.split returns (start, end) but _q_rows/_q_cols take
    (start, len): feeding the former straight in breaks every rank > 0
    (S21-1: rank 1 read past the tensor into the next one)."""
    assert n % world == 0, (n, world)
    s = n // world
    return rank * s, s


def head_parity_rows(n_heads: int, head_dim: int, world: int, rank: int) -> list[tuple[int, int]]:
    """Output-row runs (start, len) of the head rows this rank keeps: heads
    j with j % world == rank, in global order, merged when adjacent (S19
    map: rank r gathers the parity-r heads contiguous). Covers n_heads //
    world heads exactly."""
    return merge_runs([[j * head_dim, head_dim] for j in range(rank, n_heads, world)])


def qkv_parity_runs(n_key_heads: int, n_value_heads: int, s: int, world: int, rank: int):
    """The DeltaNet head gather of one rank (S19 map "Piano TP").

    qkv output layout is [H_k*S q | H_k*S k | H_v*S v]; returns
    (qkv_runs, value_runs, value_head_ids, value_head_rows):
    - qkv_runs: rows of attn_qkv and channels of ssm_conv1d (same order),
      q/k by key-head parity, v by value-head parity - per rank
      (2*H_k/world + H_v/world) * S channels;
    - value_runs: rows of attn_gate (z), same channels as the v part;
    - value_head_ids: global value-head ids kept, in order (ssm_a,
      ssm_dt.bias, ssm_alpha/ssm_beta output rows via value_head_rows,
      and ssm_out input columns);
    Inside the rank the kernel key rule h % (H_k // world) reproduces the
    global j % H_k exactly for any world dividing H_k (tests/cpu)."""
    q = head_parity_rows(n_key_heads, s, world, rank)
    k_off = n_key_heads * s
    v_off = 2 * n_key_heads * s
    qkv = merge_runs(q + [(a + k_off, l) for a, l in q]
                     + [(a + v_off, l) for a, l in head_parity_rows(n_value_heads, s, world, rank)])
    value_runs = head_parity_rows(n_value_heads, s, world, rank)
    ids = np.arange(rank, n_value_heads, world)
    return qkv, value_runs, ids, [(int(j), 1) for j in ids]


class _Layer:
    pass


class Qwen35:
    def __init__(self, g: GGUFFile, tp: TP, max_ctx: int = 8192, kv_dtype=torch.float16):
        self.g, self.tp, self.cfg = g, tp, Qwen35Config.from_gguf(g)
        self.max_ctx, self.kv_dtype = max_ctx, kv_dtype
        if not os.environ.get("PYI_NONDETERMINISTIC"):
            torch.backends.mkldnn.deterministic = True   # same C8 rationale as gpt-oss (S17-8)
        c, dev = self.cfg, tp.device
        if c.n_head % tp.world or c.n_head_kv % tp.world:
            raise ValueError(f"attention heads not divisible by {tp.world} GPUs")
        if c.n_key_heads % tp.world or c.n_value_heads % tp.world:
            raise ValueError(f"DeltaNet heads not divisible by {tp.world} GPUs")
        self.n_head_l = c.n_head // tp.world
        self.n_kv_l = c.n_head_kv // tp.world
        self.n_key_l = c.n_key_heads // tp.world          # deltanet kernels: h % n_key_l
        self.n_value_l = c.n_value_heads // tp.world
        self._qkv_l = (2 * self.n_key_l + self.n_value_l) * c.ssm_state   # conv channels per rank
        self._scale = 1.0 / math.sqrt(c.head_dim)
        # split/n_split for kernels.attn_decode over the full KV cache (gpt-oss, S17)
        self._split_full = max(256, ((max_ctx + 63) // 64 + 15) // 16 * 16)
        self._nsp_full = (max_ctx + self._split_full - 1) // self._split_full
        self.fast_decode, self.use_graph = True, False    # C3
        self._dense_bytes = 0
        t0 = time.time()

        self.v0, self.v1 = tp.split(c.n_vocab)
        self.embed = self._q_rows("token_embd.weight", [split_run(c.n_vocab, tp.world, tp.rank)])
        # NOT tied in this checkpoint (S21): output.weight is a separate Q8_0
        # tensor at its own offset; the S21-2 tie (lm_head = embed) fed the
        # input-embedding matrix to the final hidden -> flat garbage logits.
        self.lm_head = (self._q_rows("output.weight", [split_run(c.n_vocab, tp.world, tp.rank)])
                        if "output.weight" in self.g.tensors else self.embed)  # tie if absent
        self.output_norm = self._f32("output_norm.weight")
        self.cos, self.sin = rope_table(c.n_rot, c.rope_base, max_ctx, dev)

        self.layers = []
        for il in range(c.n_layer):
            self.layers.append(self._load_layer(il))
            if tp.is_main and (il + 1) % 10 == 0:
                print(f"  loaded {il + 1}/{c.n_layer} layers ({time.time() - t0:.0f}s)", flush=True)
        self.reset()
        self._weights_bytes, self._scratch_bytes = self._dense_bytes, 0
        torch.xpu.synchronize(dev)
        self.load_seconds = time.time() - t0
        if tp.is_main:
            print(f"  qwen35: load done in {self.load_seconds:.1f}s", flush=True)

    # ---- loading ----
    def _f32(self, name: str) -> torch.Tensor:
        t = self.g.tensors[name]
        assert t.type == "F32", (name, t.type)
        self._dense_bytes += t.nbytes
        return torch.from_numpy(np.array(self.g.raw(name).view(np.float32))).to(self.tp.device)

    def _f32_gather(self, name: str, shape, sel=None) -> torch.Tensor:
        """F32 tensor reshaped to `shape` (GGUF dim0 fastest) and rows gathered by `sel`."""
        t = self.g.tensors[name]
        assert t.type == "F32", (name, t.type)
        a = np.array(self.g.raw(name).view(np.float32))
        if shape is not None:
            a = a.reshape(shape)
        if sel is not None:
            a = a[sel]
        self._dense_bytes += a.nbytes
        return torch.from_numpy(a).to(self.tp.device)

    def _check_q8(self, name: str):
        if self.g.tensors[name].type != "Q8_0":
            raise NotImplementedError(f"{name}: {self.g.tensors[name].type} weights not supported yet (Q8_0 only)")

    def _q_rows(self, name: str, runs, cols=None):
        """Q8_0 [K, N] weight; output (N) rows gathered by `runs` (start, len),
        optional contiguous input (K) column slice, block-aligned."""
        from ..quant import QLinear
        self._check_q8(name)
        raw = np.concatenate([self.g.rows(name, a, a + l) for a, l in runs], axis=0)
        if cols:
            raw = raw[:, cols[0] // 32 * 34: cols[1] // 32 * 34]
        self._dense_bytes += raw.nbytes
        return QLinear.from_q8_0(raw, raw.shape[0], raw.shape[1] // 34 * 32, self.tp.device)

    def _q_cols(self, name: str, col_runs):
        """Q8_0 weight, all rows, input (K) columns gathered by 32-aligned `col_runs`."""
        from ..quant import QLinear
        self._check_q8(name)
        K, N = self.g.tensors[name].shape
        parts = [self.g.rows(name, 0, N)[:, a // 32 * 34: (a + l) // 32 * 34] for a, l in col_runs]
        raw = np.concatenate(parts, axis=1)
        self._dense_bytes += raw.nbytes
        return QLinear.from_q8_0(raw, raw.shape[0], raw.shape[1] // 34 * 32, self.tp.device)

    def _load_layer(self, il: int) -> _Layer:
        c, tp = self.cfg, self.tp
        p = f"blk.{il}."
        L = _Layer()
        L.is_attn = c.is_attn[il]
        L.attn_norm = self._f32(p + "attn_norm.weight")
        L.post_attn_norm = self._f32(p + "post_attention_norm.weight")
        fr = split_run(c.ffn, tp.world, tp.rank)          # 8704 = 34 x 256, Q8_0-safe
        L.ffn_gate = self._q_rows(p + "ffn_gate.weight", [fr])
        L.ffn_up = self._q_rows(p + "ffn_up.weight", [fr])
        L.ffn_down = self._q_cols(p + "ffn_down.weight", [fr])
        if L.is_attn:
            # heads split contiguously: rank r takes heads r*n_head_l.., the
            # q|gate 512-row interleave stays intact (Mappa B.2)
            L.q = self._q_rows(p + "attn_q.weight", [split_run(2 * c.n_head * c.head_dim, tp.world, tp.rank)])
            L.k = self._q_rows(p + "attn_k.weight", [split_run(c.n_head_kv * c.head_dim, tp.world, tp.rank)])
            L.v = self._q_rows(p + "attn_v.weight", [split_run(c.n_head_kv * c.head_dim, tp.world, tp.rank)])
            L.q_norm = self._f32(p + "attn_q_norm.weight")
            L.k_norm = self._f32(p + "attn_k_norm.weight")
            L.o = self._q_cols(p + "attn_output.weight", [split_run(c.n_head * c.head_dim, tp.world, tp.rank)])
        else:
            qkv_runs, value_runs, head_ids, head_rows = qkv_parity_runs(
                c.n_key_heads, c.n_value_heads, c.ssm_state, tp.world, tp.rank)
            L.qkv = self._q_rows(p + "attn_qkv.weight", qkv_runs)
            L.gate = self._q_rows(p + "attn_gate.weight", value_runs)
            chans = np.concatenate([np.arange(a, a + l) for a, l in qkv_runs])
            L.conv = self._f32_gather(p + "ssm_conv1d.weight",
                                      (2 * c.n_key_heads * c.ssm_state + c.n_value_heads * c.ssm_state,
                                       c.conv_kernel), chans)   # rows [channel, tap]
            L.ssm_a = self._f32_gather(p + "ssm_a", None, head_ids)
            L.dt_bias = self._f32_gather(p + "ssm_dt.bias", None, head_ids)
            L.alpha = self._q_rows(p + "ssm_alpha.weight", head_rows)
            L.beta = self._q_rows(p + "ssm_beta.weight", head_rows)
            L.ssm_out = self._q_cols(p + "ssm_out.weight", value_runs)
            L.ssm_norm = self._f32(p + "ssm_norm.weight")
        return L

    # ---- caches (C3) ----
    def reset(self):
        """Allocate (or clear) the caches: KV [n_kv_l, max_ctx, head_dim]
        kv_dtype for attention layers; conv history [K-1, qkv_l] and delta
        state [n_value_l, S, S] float32 for DeltaNet layers. The delta
        state and conv history are read on every token, so unlike the KV
        they must be re-zeroed on reuse (gpt-oss reset only reallocates)."""
        c, dev = self.cfg, self.tp.device
        self.n_past = 0
        if getattr(self, "cache", None):
            for L, e in zip(self.layers, self.cache):
                if not L.is_attn:
                    e[0].zero_()
                    e[1].zero_()
            return
        self.cache = []
        for L in self.layers:
            if L.is_attn:
                k = torch.zeros(self.n_kv_l, self.max_ctx, c.head_dim, dtype=self.kv_dtype, device=dev)
                self.cache.append((k, torch.zeros_like(k)))
            else:
                self.cache.append((
                    torch.zeros(c.conv_kernel - 1, self._qkv_l, dtype=torch.float32, device=dev),
                    torch.zeros(self.n_value_l, c.ssm_state, c.ssm_state, dtype=torch.float32, device=dev)))

    def kv_bytes(self) -> int:
        b = 0
        for L, e in zip(self.layers, self.cache):
            if L.is_attn:
                b += e[0].numel() * e[0].element_size() * 2
            else:
                b += sum(t.numel() * t.element_size() for t in e)
        return b

    # ---- embedding and head (C3) ----
    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Embedding rows, [T] -> [T, n_embd] float32 (no scaling, Mappa B.9)."""
        local = (ids >= self.v0) & (ids < self.v1)
        rows = self.embed.rows_dequant((ids - self.v0).clamp(0, self.v1 - self.v0 - 1))
        return self.tp.all_reduce(rows * local[:, None])

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm + vocab-split head gathered: [M, n_vocab] float32 on
        every rank. No soft-cap (S19 map B.9)."""
        x = rms_norm(x, self.output_norm, self.cfg.eps)
        return self.tp.all_gather(self.lm_head(x))

    # ---- gated DeltaNet layer (S21 step 2, Mappa A) ----
    def _layer_recurrent(self, il: int, x: torch.Tensor) -> torch.Tensor:
        """One gated DeltaNet layer over a prefill chunk (Mappa A).

        RMSNorm, the qkv/gate/alpha/beta projections in fp32 (dense_prefill,
        same C8 reasoning as gpt-oss), then `deltanet.prefill`: causal conv
        + SiLU and the gated delta rule, holding the conv history and the
        delta state in `self.cache[il]` in place. The key rule inside the
        rank is `h % n_key_l` (S19 map: reproduces global `j % 16`). Then the
        gated per-head RMS norm with silu(z) and `ssm_out` + all-reduce.
        x [T, n_embd] float32 -> block delta [T, n_embd] float32."""
        from .. import deltanet   # Triton at import time only when a GPU path runs
        c, L, tp = self.cfg, self.layers[il], self.tp
        S, Hv, T = c.ssm_state, self.n_value_l, x.shape[0]
        conv_state, state = self.cache[il]
        h = rms_norm(x, L.attn_norm, c.eps)
        qkv = dense_prefill(L.qkv, h)                       # [T, _qkv_l]
        z = dense_prefill(L.gate, h)                        # [T, Hv*S]
        alpha = dense_prefill(L.alpha, h)                   # [T, Hv]
        beta = torch.sigmoid(dense_prefill(L.beta, h))      # Mappa A.3
        g = L.ssm_a * F.softplus(alpha + L.dt_bias)         # ssm_a holds -exp(A_log) (A.4)
        y = deltanet.prefill(qkv, conv_state, L.conv, g, beta, state,
                             self.n_key_l, qk_eps=c.eps)    # [T, Hv, S], scaled 1/sqrt(S)
        y = deltanet.norm_gated(y, z.view(T, Hv, S), L.ssm_norm, c.eps)
        return tp.all_reduce(dense_prefill(L.ssm_out, y.reshape(T, -1)))

    def _layer_recurrent_decode(self, il: int, x: torch.Tensor) -> torch.Tensor:
        """Decode twin of `_layer_recurrent`: x [1, n_embd] -> [1, n_embd].
        QLinear GEMVs (fp32 at M=1) + `deltanet.decode`."""
        from .. import deltanet
        c, L, tp = self.cfg, self.layers[il], self.tp
        S, Hv = c.ssm_state, self.n_value_l
        conv_state, state = self.cache[il]
        h = rms_norm(x, L.attn_norm, c.eps)
        qkv = L.qkv(h)                                      # [1, _qkv_l]
        z = L.gate(h)
        alpha = L.alpha(h)
        beta = torch.sigmoid(L.beta(h))
        g = L.ssm_a * F.softplus(alpha + L.dt_bias)
        y = deltanet.decode(qkv[0], conv_state, L.conv, g[0], beta[0], state,
                            self.n_key_l, qk_eps=c.eps)     # [Hv, S]
        y = deltanet.norm_gated(y, z[0].view(Hv, S), L.ssm_norm, c.eps)
        return tp.all_reduce(L.ssm_out(y.view(1, -1)))

    # ---- dense SwiGLU FFN (Mappa B.7/8), shared by both paths ----
    def _ffn(self, il: int, x: torch.Tensor) -> torch.Tensor:
        """post_attention_norm + SwiGLU FFN (fp32) + all-reduce:
        x [T, n_embd] -> block delta [T, n_embd] float32."""
        L, tp = self.layers[il], self.tp
        h = rms_norm(x, L.post_attn_norm, self.cfg.eps)
        g = dense_prefill(L.ffn_gate, h)
        u = dense_prefill(L.ffn_up, h)
        return tp.all_reduce(dense_prefill(L.ffn_down, F.silu(g) * u))

    def _ffn_decode(self, il: int, x: torch.Tensor) -> torch.Tensor:
        """Decode twin of `_ffn`: x [1, n_embd] -> [1, n_embd] float32."""
        L, tp = self.layers[il], self.tp
        h = rms_norm(x, L.post_attn_norm, self.cfg.eps)
        return tp.all_reduce(L.ffn_down(F.silu(L.ffn_gate(h)) * L.ffn_up(h)))

    # ---- full attention layer (S21 step 3, Mappa B) ----
    def _attention(self, il: int, q, k, v, pos: torch.Tensor, pos0: int) -> torch.Tensor:
        """q [T, Hq, D]; k, v [T, Hkv, D], already normed and roped (fp32).
        Stores k/v in the layer's KV cache and runs `attn_prefill` (dense
        causal, no sliding window, no sinks; scale 1/sqrt(head_dim), Mappa
        B.5). Returns [T, Hq*D] float32."""
        kc, vc = self.cache[il]
        T = q.shape[0]
        kc[:, pos0:pos0 + T] = k.transpose(0, 1).to(kc.dtype)
        vc[:, pos0:pos0 + T] = v.transpose(0, 1).to(vc.dtype)
        kp = torch.arange(pos0 + T, device=q.device)
        return attn_prefill(q, kc[:, :pos0 + T], vc[:, :pos0 + T], kp, pos, 0, self._scale)

    def _layer_attn(self, il: int, x: torch.Tensor, pos: torch.Tensor, pos0: int) -> torch.Tensor:
        """RMSNorm, joint q|gate projection (512 rows per head: q 256 then
        gate 256, Mappa B.2), per-head RMS q/k norms before RoPE (B.3),
        partial NEOX RoPE on the first n_rot dims (B.4), dense causal GQA
        (kv-head = h/6 contiguous, B.5), output gate attn_out *=
        sigmoid(gate) before `attn_output` (B.6) + all-reduce.
        x [T, n_embd] float32 -> block delta [T, n_embd] float32."""
        c, L, tp = self.cfg, self.layers[il], self.tp
        T, Hq, Hkv, D = x.shape[0], self.n_head_l, self.n_kv_l, c.head_dim
        h = rms_norm(x, L.attn_norm, c.eps)
        qg = dense_prefill(L.q, h).view(T, Hq, 2 * D)
        q, gate = qg[..., :D], qg[..., D:]
        k = dense_prefill(L.k, h).view(T, Hkv, D)
        v = dense_prefill(L.v, h).view(T, Hkv, D)
        q = rms_norm(q, L.q_norm, c.eps)
        k = rms_norm(k, L.k_norm, c.eps)
        cos, sin = self.cos[pos], self.sin[pos]
        q = rope_neox(q, cos, sin)
        k = rope_neox(k, cos, sin)
        a = self._attention(il, q, k, v, pos, pos0)
        a = a * torch.sigmoid(gate).reshape(T, Hq * D)
        return tp.all_reduce(dense_prefill(L.o, a))

    # ---- full attention layer, decode (T == 1) ----
    def _attention_decode(self, il: int, q, k, v, pos: torch.Tensor) -> torch.Tensor:
        """q [1, Hq, D]; k, v [1, Hkv, D], already normed and roped; pos [1]
        int tensor = absolute position. Stores k/v at slot `pos` and runs
        `kernels.attn_decode` over slots 0..pos. The kernel scores at scale
        1.0 by design (gemma4 convention), so q carries the 1/sqrt(D) here
        (same as gpt-oss). Returns [Hq, D] float32."""
        from .. import kernels as K   # Triton at call time only: module header note
        kc, vc = self.cache[il]
        Hq, D = q.shape[1], q.shape[2]
        kc.index_copy_(1, pos, k.transpose(0, 1).to(kc.dtype))
        vc.index_copy_(1, pos, v.transpose(0, 1).to(vc.dtype))
        return K.attn_decode((q * self._scale).reshape(Hq, D), kc, vc,
                             pos + 1, self._nsp_full, self._split_full)

    def _layer_attn_decode(self, il: int, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Decode twin of `_layer_attn`: x [1, n_embd] -> [1, n_embd] float32."""
        c, L, tp = self.cfg, self.layers[il], self.tp
        Hq, Hkv, D = self.n_head_l, self.n_kv_l, c.head_dim
        h = rms_norm(x, L.attn_norm, c.eps)
        qg = L.q(h).view(1, Hq, 2 * D)
        q, gate = qg[..., :D], qg[..., D:]
        k = L.k(h).view(1, Hkv, D)
        v = L.v(h).view(1, Hkv, D)
        q = rms_norm(q, L.q_norm, c.eps)
        k = rms_norm(k, L.k_norm, c.eps)
        cos, sin = self.cos[pos], self.sin[pos]
        q = rope_neox(q, cos, sin)
        k = rope_neox(k, cos, sin)
        a = self._attention_decode(il, q, k, v, pos)   # [Hq, D]
        a = a * torch.sigmoid(gate[0])
        return tp.all_reduce(L.o(a.view(1, Hq * D)))

    # ---- C3 driver ----
    @torch.inference_mode()
    def forward(self, tokens: list[int], chunk: int = 512, all_logits: bool = False) -> torch.Tensor:
        """Append tokens to the context; return the logits of the last token
        ([n_vocab]) or of every token ([len(tokens), n_vocab]).

        Layer loop (S19 map B): x += attn_block(x); x += FFN(rms_norm(x,
        post_attn_norm)); the FFN residual re-adds the pre-norm value."""
        c, dev = self.cfg, self.tp.device
        if self.n_past + len(tokens) > self.max_ctx:
            raise ValueError(f"context full ({self.n_past} + {len(tokens)} > {self.max_ctx})")
        if len(tokens) == 1 and not all_logits and self.fast_decode:
            return self.decode(tokens[0])
        outs = []
        for s in range(0, len(tokens), chunk):
            part = tokens[s:s + chunk]
            pos0 = self.n_past
            ids = torch.tensor(part, dtype=torch.long, device=dev)
            pos = torch.arange(pos0, pos0 + len(part), dtype=torch.long, device=dev)
            x = self.embed_tokens(ids)
            for il in range(c.n_layer):
                blk = (self._layer_attn(il, x, pos, pos0) if self.layers[il].is_attn
                       else self._layer_recurrent(il, x))
                x = x + blk
                x = x + self._ffn(il, x)
            self.n_past += len(part)
            if all_logits:
                outs.append(self.logits(x))
        if all_logits:
            return torch.cat(outs)
        return self.logits(x[-1:])[0]

    @torch.inference_mode()
    def decode(self, token: int) -> torch.Tensor:
        """One generated token: logits [n_vocab] float32 on every rank.
        Eager path; the XPUGraph decode is S22."""
        if self.n_past >= self.max_ctx:
            raise ValueError(f"context full ({self.n_past} >= {self.max_ctx})")
        dev = self.tp.device
        pos = torch.tensor([self.n_past], dtype=torch.long, device=dev)
        self.n_past += 1
        x = self.embed_tokens(torch.tensor([token], dtype=torch.long, device=dev))
        for il in range(self.cfg.n_layer):
            blk = (self._layer_attn_decode(il, x, pos) if self.layers[il].is_attn
                   else self._layer_recurrent_decode(il, x))
            x = x + blk
            x = x + self._ffn_decode(il, x)
        return self.logits(x)[0]

    def memory_report(self) -> dict:
        """Bytes on this GPU (C3)."""
        return {"weights": self._weights_bytes,
                "cache": self.kv_bytes(),
                "scratch": self._scratch_bytes,
                "device_total": self._weights_bytes + self.kv_bytes() + self._scratch_bytes}
