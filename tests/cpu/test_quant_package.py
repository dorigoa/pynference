"""The pynference.quant package exposes the old Q8_0 API (S9 step 2).

Behavior must not change: QLinear and q8_0_to_device are re-exported from
quant/q8_0, and the Q8_0 repack is checked on CPU torch (no Triton launch).
"""
import numpy as np
import torch

from pynference.quant import QLinear, q8_0_to_device


def test_reexports():
    from pynference.quant.q8_0 import QLinear as A, q8_0_to_device as B
    assert QLinear is A and q8_0_to_device is B


def test_gemma4_import_path():
    import pynference.models.gemma4 as g4
    assert g4.QLinear is QLinear


def test_q8_0_to_device_repack():
    n, k = 2, 64
    rng = np.random.default_rng(0)
    blocks = np.zeros((n, k // 32, 34), np.uint8)
    d = rng.uniform(0.01, 0.1, (n, k // 32)).astype(np.float16)
    blocks[:, :, :2] = d.view(np.uint8).reshape(n, k // 32, 2)
    qs = rng.integers(-127, 128, (n, k // 32, 32), dtype=np.int8)
    blocks[:, :, 2:] = qs.view(np.uint8)
    raw = blocks.reshape(-1)
    qs_t, d_t = q8_0_to_device(raw, n, k, torch.device("cpu"))
    assert qs_t.shape == (n, k) and qs_t.dtype == torch.int8
    assert d_t.shape == (n, k // 32) and d_t.dtype == torch.float16
    assert torch.equal(qs_t, torch.from_numpy(qs.reshape(n, k)))
    assert torch.equal(d_t, torch.from_numpy(d))
    lin = QLinear(qs_t, d_t)
    assert (lin.N, lin.K) == (n, k)
