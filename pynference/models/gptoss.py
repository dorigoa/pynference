"""GPT-OSS dense models (GGUF arch "gpt-oss", e.g. gpt-oss-120b), tensor-parallel.

Follows llama.cpp src/models/openai-moe.cpp (S16 map: reports/S16-map.md):
- attention Q/K/V with biases added after the projection, no q/k norms;
- attention with sinks (one logit per Q head in the softmax denominator),
  scale 1/sqrt(head_dim), sliding window 128 on even layers (pattern 2);
- RoPE NEOX with YaRN: freq_scale 1/32 over ctx_orig 4096, beta 32/1,
  ext_factor 1, net mscale 1 + 0.1*ln(32) baked into the cos/sin table,
  applied identically on windowed and full layers (openai-moe.cpp:12);
- output projection with bias (added by rank 0 before the all-reduce);
- MoE (router + 128 fused MXFP4 experts) arrives in S17.

Tensor split over `tp.world` GPUs: attention heads split (Q rows [q_r; k_r; v_r],
sinks sliced by Q head), the output projection split by input columns with
all-reduce, the vocabulary split for the embedding lookup and the output head;
norms replicated.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..gguf import GGUFFile
from ..parallel import TP
# QLinear (Triton) is imported inside the loading helpers: importing Triton at
# module load would break the TRITON_INTERPRET contract of tests/cpu/test_quant_decode_interp.py
# (see the S15 report).


@dataclass
class GPTOSSConfig:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    window: int
    is_swa: list[bool]
    rope_base: float
    yarn_factor: float
    yarn_orig_ctx: int
    yarn_beta_fast: float
    yarn_beta_slow: float
    yarn_ext_factor: float
    yarn_attn_factor: float
    eps: float
    n_vocab: int
    n_ctx_train: int

    @classmethod
    def from_gguf(cls, g: GGUFFile) -> "GPTOSSConfig":
        n_layer = int(g.get("block_count"))
        rope_type = str(g.get("rope.scaling.type", "none")).lower()
        if rope_type != "yarn":
            raise NotImplementedError(f"gpt-oss: rope scaling {rope_type!r} not supported (yarn only)")
        factor = float(g.get("rope.scaling.factor"))
        if factor <= 1.0:
            raise ValueError(f"gpt-oss: rope scaling factor {factor} <= 1")
        kv = g.get("attention.head_count_kv")
        kv = [int(x) for x in kv] if hasattr(kv, "__len__") else [int(kv)] * n_layer
        if any(h != kv[0] for h in kv):
            raise NotImplementedError("gpt-oss: per-layer head_count_kv not supported")
        # llama.cpp load_swa_pattern(ml, 2) with dense_first=false: is_swa[il] = il % 2 < 1
        return cls(
            n_layer=n_layer,
            n_embd=int(g.get("embedding_length")),
            n_head=int(g.get("attention.head_count")),
            n_head_kv=kv[0],
            head_dim=int(g.get("attention.key_length")),
            n_rot=int(g.get("rope.dimension_count", g.get("attention.key_length"))),
            window=int(g.get("attention.sliding_window")),
            is_swa=[il % 2 == 0 for il in range(n_layer)],
            rope_base=float(g.get("rope.freq_base")),
            yarn_factor=factor,
            yarn_orig_ctx=int(g.get("rope.scaling.original_context_length")),
            yarn_beta_fast=float(g.get("rope.scaling.yarn_beta_fast", 32.0)),
            yarn_beta_slow=float(g.get("rope.scaling.yarn_beta_slow", 1.0)),
            # llama-context.cpp:265-267: ext_factor unset -> 1.0 for YARN; the attn_factor
            # chain (295/303/306) nets to the GGUF value, absent here -> 1.0 (S16 map §5).
            yarn_ext_factor=1.0,
            yarn_attn_factor=float(g.get("rope.scaling.attn_factor", 1.0) or 1.0),
            eps=float(g.get("attention.layer_norm_rms_epsilon")),
            n_vocab=int(g.tensors["token_embd.weight"].shape[1]),
            n_ctx_train=int(g.get("context_length")),
        )


def rope_yarn_dims(n_rot: int, base: float, orig_ctx: int, beta_fast: float, beta_slow: float) -> tuple[int, int]:
    """ggml_rope_yarn_corr_dims (ggml.c:4373-4389) with truncate = true (the rope_yarn call
    in ggml-cpu/ops.cpp passes the truncated dims): floor of the beta_fast dim, ceil of the
    beta_slow dim, clamped to [0, n_rot - 1]. For gpt-oss: (8, 18)."""
    def corr_dim(beta: float) -> float:
        return n_rot * math.log(orig_ctx / (beta * 2 * math.pi)) / (2 * math.log(base))
    return max(0, math.floor(corr_dim(beta_fast))), min(n_rot - 1, math.ceil(corr_dim(beta_slow)))


def rope_yarn_table(n_rot: int, base: float, factor: float, orig_ctx: int,
                    beta_fast: float, beta_slow: float, n_pos: int, device,
                    ext_factor: float = 1.0, attn_factor: float = 1.0):
    """cos/sin tables [n_pos, n_rot/2] for NEOX RoPE with YaRN, as computed by
    ggml_rope_cache_init -> rope_yarn (ggml-cpu/ops.cpp:6076-6114, 6268-6271):
    freq_scale = 1/factor; for pair k (dims k, k + n_rot/2) with corr_dims [c0, c1]:
    ramp = clamp01(1 - (k - c0)/(c1 - c0)), theta = theta_raw*((1 - ramp*ext)/factor + ramp*ext);
    mscale = attn_factor*(1 + 0.1*ln(factor)) multiplies cos/sin (ops.cpp:6095-6096)."""
    c0, c1 = rope_yarn_dims(n_rot, base, orig_ctx, beta_fast, beta_slow)
    freq_scale = 1.0 / factor
    k = np.arange(n_rot // 2, dtype=np.float64)
    inv = base ** (-2.0 * k / n_rot)                      # theta_scale^k, ops.cpp:6268
    ramp = 1.0 - np.clip((k - c0) / max(0.001, c1 - c0), 0.0, 1.0)
    mix = ramp * ext_factor
    scale = freq_scale * (1.0 - mix) + mix                # theta = theta_raw * scale
    mscale = attn_factor * (1.0 + 0.1 * math.log(1.0 / freq_scale))
    ang = np.arange(n_pos, dtype=np.float64)[:, None] * (inv * scale)[None, :]
    return (torch.from_numpy((mscale * np.cos(ang)).astype(np.float32)).to(device),
            torch.from_numpy((mscale * np.sin(ang)).astype(np.float32)).to(device))


class _Layer:
    pass


class GPTOss:
    def __init__(self, g: GGUFFile, tp: TP, max_ctx: int = 8192, kv_dtype=torch.float16):
        self.g, self.tp, self.cfg = g, tp, GPTOSSConfig.from_gguf(g)
        self.max_ctx, self.kv_dtype = max_ctx, kv_dtype
        c, dev = self.cfg, tp.device
        if c.n_head % tp.world or c.n_head_kv % tp.world:
            raise ValueError(f"heads not divisible by {tp.world} GPUs")
        self.n_head_l = c.n_head // tp.world
        self.n_kv_l = c.n_head_kv // tp.world
        t0 = time.time()

        self.v0, self.v1 = tp.split(c.n_vocab)
        self.embed = self._qlinear("token_embd.weight", rows=(self.v0, self.v1))
        self.lm_head = self._qlinear("output.weight", rows=(self.v0, self.v1))
        self.output_norm = self._f32("output_norm.weight")
        self.cos, self.sin = rope_yarn_table(
            c.n_rot, c.rope_base, c.yarn_factor, c.yarn_orig_ctx,
            c.yarn_beta_fast, c.yarn_beta_slow, max_ctx, dev,
            ext_factor=c.yarn_ext_factor, attn_factor=c.yarn_attn_factor)

        self.layers = []
        for il in range(c.n_layer):
            self.layers.append(self._load_layer(il))
            if tp.is_main and (il + 1) % 10 == 0:
                print(f"  loaded {il + 1}/{c.n_layer} layers ({time.time() - t0:.0f}s)", flush=True)
        torch.xpu.synchronize(dev)
        self.load_seconds = time.time() - t0

    # ---- loading ----
    def _f32(self, name: str) -> torch.Tensor:
        t = self.g.tensors[name]
        assert t.type == "F32", (name, t.type)
        return torch.from_numpy(np.array(self.g.raw(name).view(np.float32))).to(self.tp.device)

    def _check_q8(self, name: str):
        if self.g.tensors[name].type != "Q8_0":
            raise NotImplementedError(f"{name}: {self.g.tensors[name].type} weights not supported yet (Q8_0 only)")

    def _qlinear(self, name: str, rows=None, cols=None):
        """Q8_0 [K, N] weight; rows=(r0, r1) output rows, cols=(c0, c1) input columns."""
        from ..quant import QLinear
        self._check_q8(name)
        K, N = self.g.tensors[name].shape
        r0, r1 = rows if rows else (0, N)
        raw = self.g.rows(name, r0, r1)
        if cols:
            raw = raw[:, cols[0] // 32 * 34: cols[1] // 32 * 34]
        return QLinear.from_q8_0(raw, raw.shape[0], raw.shape[1] // 34 * 32, self.tp.device)

    def _stack_rows(self, names, row_ranges):
        from ..quant import QLinear
        parts = []
        for name, (r0, r1) in zip(names, row_ranges):
            self._check_q8(name)
            parts.append(self.g.rows(name, r0, r1))
        raw = np.concatenate(parts, axis=0)
        return QLinear.from_q8_0(raw, raw.shape[0], raw.shape[1] // 34 * 32, self.tp.device)

    def _load_layer(self, il: int) -> _Layer:
        c, tp = self.cfg, self.tp
        p = f"blk.{il}."
        L = _Layer()
        L.swa = c.is_swa[il]
        qr = tp.split(c.n_head * c.head_dim)
        kr = tp.split(c.n_head_kv * c.head_dim)
        hr = tp.split(c.n_head)
        L.qkv = self._stack_rows([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"],
                                 [qr, kr, kr])
        # biases are added after the projection (llama-graph.cpp:1924-1953), split like the rows
        L.qkv_bias = torch.cat([self._f32(p + "attn_q.bias")[qr[0]:qr[1]],
                                self._f32(p + "attn_k.bias")[kr[0]:kr[1]],
                                self._f32(p + "attn_v.bias")[kr[0]:kr[1]]])
        L.o = self._qlinear(p + "attn_output.weight", cols=qr)
        # column-split bias: rank 0 adds the full [n_embd] bias before the all-reduce (plan 3.4)
        L.o_bias = self._f32(p + "attn_output.bias") if tp.is_main else None
        # sinks are per Q head (S16 map §7): keep this rank's slice of heads
        L.sinks = self._f32(p + "attn_sinks.weight")[hr[0]:hr[1]]
        L.attn_norm = self._f32(p + "attn_norm.weight")
        L.post_attn_norm = self._f32(p + "post_attention_norm.weight")
        return L
