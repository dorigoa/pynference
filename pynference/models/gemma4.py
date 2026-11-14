"""Gemma 4 dense models (GGUF arch "gemma4", e.g. Gemma-4-31B), tensor-parallel.

Follows llama.cpp src/models/gemma4.cpp:
- embeddings scaled by sqrt(n_embd); output head tied to token_embd; final
  logits soft-capped (cap * tanh(x / cap));
- sliding-window layers (window 1024, rope base 10k, head dim 256) and full
  attention layers (rope base 1M with per-dimension freq factors, head dim 512,
  K projection reused as V where attn_v is absent);
- per-head RMSNorm on Q and K (with weights) and on V (without), attention
  scale 1.0;
- sandwich norms around attention and FFN (GELU-tanh gated), per-layer output
  scale.

Tensor split over `tp.world` GPUs: attention heads and FFN columns are split,
the output projections are split by input columns and summed with all-reduce;
the vocabulary is split for the embedding lookup and the output head.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import triton

from .. import kernels as K
from ..gguf import GGUFFile
from ..parallel import TP
from ..quant import QLinear


@dataclass
class Gemma4Config:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: list[int]
    is_swa: list[bool]
    head_dim: int
    head_dim_swa: int
    n_rot: int
    n_rot_swa: int
    rope_base: float
    rope_base_swa: float
    window: int
    n_ff: list[int]
    eps: float
    softcap: float
    n_vocab: int

    @classmethod
    def from_gguf(cls, g: GGUFFile) -> "Gemma4Config":
        n_layer = int(g.get("block_count"))

        def per_layer(v):
            return [int(x) for x in v] if hasattr(v, "__len__") else [int(v)] * n_layer

        if int(g.get("embedding_length_per_layer_input", 0) or 0):
            raise NotImplementedError("Gemma 4 per-layer embeddings (E2B/E4B) not supported yet")
        if int(g.get("attention.shared_kv_layers", 0) or 0):
            raise NotImplementedError("Gemma 4 shared KV layers not supported yet")
        if g.get("expert_count"):
            raise NotImplementedError("Gemma 4 MoE (26B-A4B) not supported yet")
        return cls(
            n_layer=n_layer,
            n_embd=int(g.get("embedding_length")),
            n_head=int(g.get("attention.head_count")),
            n_head_kv=per_layer(g.get("attention.head_count_kv")),
            is_swa=[bool(x) for x in g.get("attention.sliding_window_pattern")],
            head_dim=int(g.get("attention.key_length")),
            head_dim_swa=int(g.get("attention.key_length_swa")),
            n_rot=int(g.get("rope.dimension_count")),
            n_rot_swa=int(g.get("rope.dimension_count_swa")),
            rope_base=float(g.get("rope.freq_base")),
            rope_base_swa=float(g.get("rope.freq_base_swa")),
            window=int(g.get("attention.sliding_window")),
            n_ff=per_layer(g.get("feed_forward_length")),
            eps=float(g.get("attention.layer_norm_rms_epsilon")),
            softcap=float(g.get("final_logit_softcapping", 0.0) or 0.0),
            n_vocab=int(g.tensors["token_embd.weight"].shape[1]),
        )


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * w if w is not None else x


def rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [T, H, D]; cos/sin [T, R/2]. Rotates pairs (i, i + R/2) of the first R dims."""
    half = cos.shape[-1]
    x1, x2 = x[..., :half], x[..., half:2 * half]
    c, s = cos[:, None, :], sin[:, None, :]
    rot = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return rot if 2 * half == x.shape[-1] else torch.cat([rot, x[..., 2 * half:]], dim=-1)


def rope_table(n_rot: int, base: float, factors: np.ndarray | None, n_pos: int, device):
    inv = 1.0 / base ** (np.arange(n_rot // 2, dtype=np.float64) * 2 / n_rot)
    if factors is not None:
        inv = inv / factors.astype(np.float64)
    ang = np.arange(n_pos, dtype=np.float64)[:, None] * inv[None, :]
    return (torch.from_numpy(np.cos(ang).astype(np.float32)).to(device),
            torch.from_numpy(np.sin(ang).astype(np.float32)).to(device))


class _Layer:
    pass


class Gemma4:
    def __init__(self, g: GGUFFile, tp: TP, max_ctx: int = 8192, kv_dtype=torch.float16):
        self.g, self.tp, self.cfg = g, tp, Gemma4Config.from_gguf(g)
        self.max_ctx, self.kv_dtype = max_ctx, kv_dtype
        c, dev = self.cfg, tp.device
        if c.n_head % tp.world or any(h % tp.world for h in c.n_head_kv):
            raise ValueError(f"heads not divisible by {tp.world} GPUs")
        self.n_head_l = c.n_head // tp.world
        t0 = time.time()

        self.v0, self.v1 = tp.split(c.n_vocab)
        self.embed = self._qlinear("token_embd.weight", rows=(self.v0, self.v1))
        self.lm_head = self.embed                                        # tied
        if "output.weight" in g.tensors:
            self.lm_head = self._qlinear("output.weight", rows=(self.v0, self.v1))
        self.output_norm = self._f32("output_norm.weight")
        factors = np.array(g.raw("rope_freqs.weight").view(np.float32)) if "rope_freqs.weight" in g.tensors else None
        self.rope = rope_table(c.n_rot, c.rope_base, factors, max_ctx, dev)
        self.rope_swa = rope_table(c.n_rot_swa, c.rope_base_swa, None, max_ctx, dev)

        self.layers = []
        for il in range(c.n_layer):
            self.layers.append(self._load_layer(il))
            if tp.is_main and (il + 1) % 10 == 0:
                print(f"  loaded {il + 1}/{c.n_layer} layers ({time.time() - t0:.0f}s)", flush=True)
        torch.xpu.synchronize(dev)
        self.load_seconds = time.time() - t0
        self.fast_decode = True        # fused Triton kernels for single tokens
        self.use_graph = False         # capture the decode step as an XPUGraph
        self.reset()
        self._init_decode()

    # ---- loading ----
    def _f32(self, name: str) -> torch.Tensor:
        t = self.g.tensors[name]
        assert t.type == "F32", (name, t.type)
        return torch.from_numpy(np.array(self.g.raw(name).view(np.float32))).to(self.tp.device)

    def _check_q8(self, name: str):
        if self.g.tensors[name].type != "Q8_0":
            raise NotImplementedError(f"{name}: {self.g.tensors[name].type} weights not supported yet (Q8_0 only)")

    def _qlinear(self, *names: str, rows=None, cols=None) -> QLinear:
        """Q8_0 weights of one or more [N, K] tensors stacked along N; rows=(r0, r1)
        selects the same output rows of each one, cols=(c0, c1) the input columns."""
        parts = []
        for name in names:
            self._check_q8(name)
            t = self.g.tensors[name]
            K, N = t.shape
            r0, r1 = rows if rows else (0, N)
            raw = self.g.rows(name, r0, r1)
            if cols:
                raw = raw[:, cols[0] // 32 * 34: cols[1] // 32 * 34]
                K = cols[1] - cols[0]
            parts.append(raw)
        raw = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        return QLinear.from_q8_0(raw, raw.shape[0], raw.shape[1] // 34 * 32, self.tp.device)

    def _load_layer(self, il: int) -> _Layer:
        c, tp = self.cfg, self.tp
        p = f"blk.{il}."
        L = _Layer()
        L.swa = c.is_swa[il]
        L.hd = c.head_dim_swa if L.swa else c.head_dim
        L.n_kv_l = c.n_head_kv[il] // tp.world
        L.has_v = p + "attn_v.weight" in self.g.tensors
        qr = tp.split(c.n_head * L.hd)
        kr = tp.split(c.n_head_kv[il] * L.hd)
        names = [p + "attn_q.weight", p + "attn_k.weight"] + ([p + "attn_v.weight"] if L.has_v else [])
        # q, k (and v) rows of this rank, stacked
        L.qkv = self._stack_rows(names, [qr, kr, kr][:len(names)])
        L.o = self._qlinear(p + "attn_output.weight", cols=qr)
        ff_r = tp.split(c.n_ff[il])
        L.gate_up = self._stack_rows([p + "ffn_gate.weight", p + "ffn_up.weight"], [ff_r, ff_r])
        L.down = self._qlinear(p + "ffn_down.weight", cols=ff_r)
        L.ff_l = ff_r[1] - ff_r[0]
        L.attn_norm = self._f32(p + "attn_norm.weight")
        L.q_norm = self._f32(p + "attn_q_norm.weight")
        L.k_norm = self._f32(p + "attn_k_norm.weight")
        L.post_attn_norm = self._f32(p + "post_attention_norm.weight")
        L.ffn_norm = self._f32(p + "ffn_norm.weight")
        L.post_ffn_norm = self._f32(p + "post_ffw_norm.weight")
        L.out_scale = self._f32(p + "layer_output_scale.weight") if p + "layer_output_scale.weight" in self.g.tensors else None
        return L

    def _stack_rows(self, names, row_ranges) -> QLinear:
        parts = []
        for name, (r0, r1) in zip(names, row_ranges):
            self._check_q8(name)
            parts.append(self.g.rows(name, r0, r1))
        raw = np.concatenate(parts, axis=0)
        return QLinear.from_q8_0(raw, raw.shape[0], raw.shape[1] // 34 * 32, self.tp.device)

    # ---- KV cache ----
    def reset(self):
        """Allocate (or clear) the KV cache: a ring buffer of `window` positions for
        sliding layers (position p in slot p % window), max_ctx positions for
        full-attention layers. Positions always run 0..n_past-1, so slot contents
        follow from n_past and stale entries are never read."""
        c, dev = self.cfg, self.tp.device
        self.n_past = 0
        if getattr(self, "cache", None):
            return
        self.cache = []
        for L in self.layers:
            n = c.window if L.swa else self.max_ctx
            k = torch.zeros(L.n_kv_l, n, L.hd, dtype=self.kv_dtype, device=dev)
            self.cache.append((k, torch.zeros_like(k)))

    def kv_bytes(self) -> int:
        return sum(k.numel() * k.element_size() * 2 for k, _ in self.cache)

    # ---- prompt (T > 1) path: torch ops ----
    def _attention(self, il: int, q, k, v, pos: torch.Tensor, pos0: int) -> torch.Tensor:
        """q [T, Hq, D]; k, v [T, Hkv, D] (already normed/roped). Returns [T, Hq*D]."""
        L = self.layers[il]
        kc, vc = self.cache[il]
        T = q.shape[0]
        mask = None
        if L.swa:
            W = self.cfg.window
            # slot j holds the latest position < pos0 that is == j (mod W), if any
            j = torch.arange(W, device=q.device)
            kp = (pos0 - 1) - torch.remainder(pos0 - 1 - j, W)
            K = torch.cat([kc, k.transpose(0, 1).to(kc.dtype)], dim=1)
            V = torch.cat([vc, v.transpose(0, 1).to(vc.dtype)], dim=1)
            kp = torch.cat([kp, pos])
            d = pos[:, None] - kp[None, :]
            mask = (kp[None, :] >= 0) & (d >= 0) & (d < W)
            n = min(T, W)
            slots = pos[T - n:] % W
            kc[:, slots] = k[T - n:].transpose(0, 1).to(kc.dtype)
            vc[:, slots] = v[T - n:].transpose(0, 1).to(vc.dtype)
        else:
            kc[:, pos0:pos0 + T] = k.transpose(0, 1)
            vc[:, pos0:pos0 + T] = v.transpose(0, 1)
            K, V = kc[:, :pos0 + T], vc[:, :pos0 + T]
            mask = torch.arange(pos0 + T, device=q.device)[None, :] <= pos[:, None]
        Hq, D = q.shape[1], q.shape[2]
        Hkv = K.shape[0]
        G = Hq // Hkv
        qh = q.view(T, Hkv, G, D).permute(1, 2, 0, 3).reshape(Hkv, G * T, D)
        s = torch.matmul(qh, K.float().transpose(1, 2)).view(Hkv, G, T, -1)   # scale 1.0
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1).view(Hkv, G * T, -1)
        o = torch.matmul(p, V.float())
        return o.view(Hkv, G, T, D).permute(2, 0, 1, 3).reshape(T, Hq * D)

    def _layer(self, il: int, x: torch.Tensor, pos: torch.Tensor, pos0: int) -> torch.Tensor:
        c, L, tp = self.cfg, self.layers[il], self.tp
        T = x.shape[0]
        Hq, Hkv, D = self.n_head_l, L.n_kv_l, L.hd
        h = rms_norm(x, L.attn_norm, c.eps)
        qkv = L.qkv(h)
        q = qkv[:, :Hq * D].view(T, Hq, D)
        k = qkv[:, Hq * D:(Hq + Hkv) * D].view(T, Hkv, D)
        v = qkv[:, (Hq + Hkv) * D:].view(T, Hkv, D) if L.has_v else k
        cos, sin = self.rope_swa if L.swa else self.rope
        cos, sin = cos[pos], sin[pos]
        q = rope_neox(rms_norm(q, L.q_norm, c.eps), cos, sin)
        v = rms_norm(v, None, c.eps)
        k = rope_neox(rms_norm(k, L.k_norm, c.eps), cos, sin)
        a = self._attention(il, q, k, v, pos, pos0)
        a = tp.all_reduce(L.o(a))
        x = x + rms_norm(a, L.post_attn_norm, c.eps)

        h = rms_norm(x, L.ffn_norm, c.eps)
        gu = L.gate_up(h)
        f = F.gelu(gu[:, :L.ff_l], approximate="tanh") * gu[:, L.ff_l:]
        f = tp.all_reduce(L.down(f))
        x = x + rms_norm(f, L.post_ffn_norm, c.eps)
        if L.out_scale is not None:
            x = x * L.out_scale
        return x

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        local = (ids >= self.v0) & (ids < self.v1)
        rows = self.embed.rows_dequant((ids - self.v0).clamp(0, self.v1 - self.v0 - 1))
        x = self.tp.all_reduce(rows * local[:, None])
        return x * math.sqrt(self.cfg.n_embd)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Final norm + output head + soft-cap for the rows of x: [M, n_vocab] float32 on every rank."""
        x = rms_norm(x, self.output_norm, self.cfg.eps)
        out = self.tp.all_gather(self.lm_head(x))
        if self.cfg.softcap:
            out = torch.tanh(out / self.cfg.softcap) * self.cfg.softcap
        return out

    @torch.inference_mode()
    def forward(self, tokens: list[int], chunk: int = 512, all_logits: bool = False) -> torch.Tensor:
        """Append tokens to the context; return the logits of the last token
        ([n_vocab]) or of every token ([len(tokens), n_vocab])."""
        dev = self.tp.device
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
            for il in range(self.cfg.n_layer):
                x = self._layer(il, x, pos, pos0)
            self.n_past += len(part)
            if all_logits:
                outs.append(self.logits(x))
        if all_logits:
            return torch.cat(outs)
        return self.logits(x[-1:])[0]

    # ---- decode (T == 1) path: fused Triton kernels, optional XPUGraph ----
    def _init_decode(self):
        c, dev = self.cfg, self.tp.device
        # tok, pos, sliding-window slot, keys visible in sliding / full layers
        self._host = torch.zeros(5, dtype=torch.long).pin_memory()
        self._dbuf = torch.zeros(5, dtype=torch.long, device=dev)
        self._split_swa = 128
        self._nsp_swa = triton.cdiv(c.window, self._split_swa)
        self._split_full = max(256, triton.cdiv(triton.cdiv(self.max_ctx, 64), 16) * 16)
        self._nsp_full = triton.cdiv(self.max_ctx, self._split_full)
        self._graph = None
        self._graph_out = None

    def _decode_body(self) -> torch.Tensor:
        c, tp, eps = self.cfg, self.tp, self.cfg.eps
        b = self._dbuf
        tok, pos, slot, n_swa, n_full = b[0:1], b[1:2], b[2:3], b[3:4], b[4:5]
        x = self.embed_tokens(tok)
        h = K.rms_norm(x, self.layers[0].attn_norm, eps)
        Hq = self.n_head_l
        for il, L in enumerate(self.layers):
            kc, vc = self.cache[il]
            Hkv, D = L.n_kv_l, L.hd
            qkv = L.qkv(h)
            cos, sin = self.rope_swa if L.swa else self.rope
            q = K.norm_rope(qkv[:, :Hq * D].view(1, Hq, D), L.q_norm, eps, cos, sin, pos)
            kv_slot = slot if L.swa else pos
            k_view = qkv[:, Hq * D:(Hq + Hkv) * D].view(1, Hkv, D)
            v_view = qkv[:, (Hq + Hkv) * D:].view(1, Hkv, D) if L.has_v else k_view
            K.norm_rope(k_view, L.k_norm, eps, cos, sin, pos, out=kc, slot=kv_slot)
            K.norm_rope(v_view, None, eps, out=vc, slot=kv_slot)
            if L.swa:
                a = K.attn_decode(q.view(Hq, D), kc, vc, n_swa, self._nsp_swa, self._split_swa)
            else:
                a = K.attn_decode(q.view(Hq, D), kc, vc, n_full, self._nsp_full, self._split_full)
            o = tp.all_reduce(L.o(a.view(1, Hq * D)))
            x, h = K.add_norm(x, o, L.post_attn_norm, eps, w2=L.ffn_norm)
            f = K.gelu_mul(L.gate_up(h))
            d = tp.all_reduce(L.down(f))
            w_next = self.layers[il + 1].attn_norm if il + 1 < c.n_layer else self.output_norm
            x, h = K.add_norm(x, d, L.post_ffn_norm, eps, scale=L.out_scale, w2=w_next)
        out = tp.all_gather(self.lm_head(h))
        if c.softcap:
            out = torch.tanh(out / c.softcap) * c.softcap
        return out

    def decode(self, token: int) -> torch.Tensor:
        """One generated token: logits [n_vocab]. With use_graph the step is captured
        on the first call and replayed afterwards (the result is overwritten by the next call)."""
        W, pos = self.cfg.window, self.n_past
        h = self._host
        h[0], h[1], h[2], h[3], h[4] = token, pos, pos % W, min(pos + 1, W), pos + 1
        self._dbuf.copy_(h)
        if not self.use_graph:
            out = self._decode_body()
        else:
            if self._graph is None:
                self._capture()
            self._graph.replay()
            out = self._graph_out
        self.n_past += 1
        return out[0]

    def _capture(self):
        dev = self.tp.device
        s = torch.xpu.Stream(dev)
        s.wait_stream(torch.xpu.current_stream(dev))
        with torch.xpu.stream(s):          # warm-up: Triton compile/autotune, allocator pools
            for _ in range(2):
                self._decode_body()
        torch.xpu.current_stream(dev).wait_stream(s)
        torch.xpu.synchronize(dev)
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            self._graph_out = self._decode_body()
        self._graph = g
