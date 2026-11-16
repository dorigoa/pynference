"""CPU test: rope_yarn_table vs direct transcriptions of llama.cpp rope_yarn.

The transcriptions mirror rope_yarn and ggml_rope_cache_init (ggml-cpu/ops.cpp:
6076-6114) and ggml_rope_yarn_corr_dims (ggml.c:4373-4389, truncate=true), the
one float32 in the C loop style (theta accumulates as theta *= theta_scale),
the other in float64 (pure formula, to pin down the table exactly). Parameters
are the gpt-oss facts (S13 facts.json): NEOX, base 150000, YaRN factor 32 over
ctx 4096, beta 32/1, ext_factor 1 (llama-context.cpp:265-267), attn_factor 1
(net, S16 map §5).
"""
import numpy as np
import torch

from pynference.models.gptoss import rope_yarn_dims, rope_yarn_table

BASE, FACTOR, ORIG = 150000.0, 32.0, 4096
BETA_FAST, BETA_SLOW = 32.0, 1.0
N_ROT = 64
MS = 1.0 + 0.1 * np.log(32.0)          # net mscale on cos/sin (1.346574)


def _mix(k, start, end, ext):
    y = (k - start) / max(0.001, end - start)
    ramp = 1.0 - min(1.0, max(0.0, y))
    return ramp * ext


def rope_yarn_ref64(n_pos, n_rot=N_ROT, base=BASE, factor=FACTOR, orig=ORIG,
                    beta_fast=BETA_FAST, beta_slow=BETA_SLOW, ext=1.0, attn=1.0):
    """Formula transcription in float64: per pair k, theta_raw = p*base^(-2k/n),
    theta = fs*theta_raw*(1 - mix) + theta_raw*mix, cos/sin multiplied by mscale."""
    corr = lambda b: n_rot * np.log(orig / (b * 2 * np.pi)) / (2 * np.log(base))
    start, end = max(0.0, np.floor(corr(beta_fast))), min(n_rot - 1, np.ceil(corr(beta_slow)))
    freq_scale = 1.0 / factor
    mscale = attn * (1.0 + 0.1 * np.log(1.0 / freq_scale))
    k = np.arange(n_rot // 2, dtype=np.float64)
    inv = base ** (-2.0 * k / n_rot)
    mix = np.array([_mix(kk, start, end, ext) for kk in k])
    scale = freq_scale * (1.0 - mix) + mix
    ang = np.arange(n_pos, dtype=np.float64)[:, None] * (inv * scale)[None, :]
    return mscale * np.cos(ang), mscale * np.sin(ang)


def rope_yarn_ref32(n_pos, **kw):
    """The C loop in float32: theta starts at pos and multiplies by theta_scale per pair."""
    f32 = np.float32
    corr = lambda b: f32(N_ROT) * np.log(f32(ORIG) / (f32(b) * f32(2) * f32(np.pi))) / (f32(2) * np.log(f32(BASE)))
    start, end = max(0.0, float(np.floor(corr(BETA_FAST)))), float(min(N_ROT - 1, np.ceil(corr(BETA_SLOW))))
    theta_scale = f32(BASE) ** f32(-2.0 / N_ROT)          # ops.cpp:6268
    freq_scale = f32(1.0 / FACTOR)
    mscale = f32(1.0) * (f32(1.0) + f32(0.1) * np.log(f32(1.0) / freq_scale))   # ops.cpp:6095
    cos = np.zeros((n_pos, N_ROT // 2), dtype=np.float32)
    sin = np.zeros_like(cos)
    for p in range(n_pos):
        theta = f32(p)                                     # ggml_rope_cache_init(theta_base=p)
        for k in range(N_ROT // 2):
            # rope_yarn_ramp(corr_start, corr_end, i0) with i0 = 2k (ops.cpp:6076-6079)
            mix = f32(_mix(float(k), start, end, 1.0))
            theta_interp = freq_scale * theta              # rope_yarn
            th = f32(f32(theta_interp * f32(1.0) - theta_interp * mix) + f32(theta * mix))
            cos[p, k] = f32(np.cos(float(th))) * mscale
            sin[p, k] = f32(np.sin(float(th))) * mscale
            theta = f32(theta * theta_scale)
    return cos, sin


def test_corr_dims_gpt_oss():
    assert rope_yarn_dims(N_ROT, BASE, ORIG, BETA_FAST, BETA_SLOW) == (8, 18)


def test_yarn_table_matches_formula():
    n_pos = 4097
    cos, sin = rope_yarn_table(N_ROT, BASE, FACTOR, ORIG, BETA_FAST, BETA_SLOW, n_pos, "cpu")
    assert cos.shape == (n_pos, N_ROT // 2) and cos.dtype == torch.float32
    rcos, rsin = rope_yarn_ref64(n_pos)
    np.testing.assert_allclose(cos.numpy(), rcos, atol=2e-6)
    np.testing.assert_allclose(sin.numpy(), rsin, atol=2e-6)


def test_yarn_table_vs_float32_loop():
    # moderate positions: the float32 accumulation is the only difference
    n_pos = 129
    cos, sin = rope_yarn_table(N_ROT, BASE, FACTOR, ORIG, BETA_FAST, BETA_SLOW, n_pos, "cpu")
    rcos, rsin = rope_yarn_ref32(n_pos)
    np.testing.assert_allclose(cos.numpy(), rcos, atol=1e-4)
    np.testing.assert_allclose(sin.numpy(), rsin, atol=1e-4)


def test_yarn_table_structure():
    # net mscale on cos/sin; pair k <= 8 un-scaled angle, k >= 18 angle /32
    cos, sin = rope_yarn_table(N_ROT, BASE, FACTOR, ORIG, BETA_FAST, BETA_SLOW, 8, "cpu")
    c, s = cos.numpy(), sin.numpy()
    assert abs(c[1, 0] - MS * np.cos(1.0)) < 1e-6          # k=0: un-scaled angle, mscale applied
    inv18 = BASE ** (-2.0 * 18 / N_ROT)
    assert abs(c[1, 18] - MS * np.cos(inv18 / 32.0)) < 1e-6  # k=18: angle /32
    r2 = c ** 2 + s ** 2
    np.testing.assert_allclose(r2, MS ** 2, rtol=1e-6)     # rotation scaled by the net mscale
