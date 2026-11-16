"""S16 step 4: the layer-0 attention block of the real gpt-oss-120b, on 2 GPUs
(tensor-parallel, worker under torchrun), against a CPU reference built from
gguf-py-dequantized Q8_0 attention weights (cos >= 0.9999), and — when
GPTOSS_DUMP points to a `llama_ref.sh dump` file — against the attn_out-0
values printed by llama.cpp's eval-callback (loose: bf16 kq accumulation,
4-decimal print).

Run only inside a GPU job (2.4); skipped when the GPUs or the GGUF are missing.
"""
import os
import subprocess
import sys

import pytest
import torch

if not hasattr(torch, "xpu") or torch.xpu.device_count() == 0:
    pytest.skip("no XPU device available", allow_module_level=True)

MODEL = os.environ.get("GPTOSS_GGUF", "/nvme/gguf/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(ROOT, "tests", "gpu", "_gptoss_attn0_worker.py")


def test_gptoss_attn0_layer0():
    if not os.path.exists(MODEL):
        pytest.skip(f"{MODEL} not present")
    r = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", WORKER],
        cwd=ROOT, capture_output=True, text=True, timeout=3000,
        env={**os.environ, "GPTOSS_GGUF": MODEL})
    out = r.stdout + "\n" + r.stderr
    assert "RES cos_attn0=" in out, out[-4000:]
    assert "ALL OK" in out and r.returncode == 0, out[-4000:]
