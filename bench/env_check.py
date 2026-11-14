"""Print the XPU software stack and device properties; run a sanity matmul per device."""
import importlib
import platform

import torch


def version(mod):
    try:
        return getattr(importlib.import_module(mod), "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return f"n/a ({type(e).__name__})"


print(f"python {platform.python_version()} | torch {torch.__version__} | triton {version('triton')}"
      f" | numpy {version('numpy')}")
print(f"xpu available: {torch.xpu.is_available()} | device count: {torch.xpu.device_count()}")
dist = torch.distributed
print(f"distributed: {dist.is_available()} | xccl: {getattr(dist, 'is_xccl_available', lambda: 'n/a')()}"
      f" | gloo: {dist.is_gloo_available()}")
print(f"xpu graph API: {[n for n in ('XPUGraph', 'graph', 'graphs') if hasattr(torch.xpu, n)] or 'none'}")

for i in range(torch.xpu.device_count()):
    p = torch.xpu.get_device_properties(i)
    print(f"\n[xpu:{i}] {p.name}")
    for k in ("total_memory", "max_compute_units", "gpu_eu_count", "gpu_subslice_count", "max_work_group_size",
              "sub_group_sizes", "has_fp16", "has_fp64", "has_atomic64", "has_bfloat16_conversions",
              "has_subgroup_matrix_multiply_accumulate", "has_subgroup_2d_block_io", "driver_version", "version",
              "platform_name", "device_id", "architecture"):
        if hasattr(p, k):
            v = getattr(p, k)
            if k == "total_memory":
                v = f"{v / 2**30:.2f} GiB"
            print(f"  {k}: {v}")
    free, total = torch.xpu.mem_get_info(i)
    print(f"  mem_get_info: free {free / 2**30:.2f} GiB / total {total / 2**30:.2f} GiB")

    d = torch.device(f"xpu:{i}")
    a = torch.randn(1024, 1024, device=d, dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device=d, dtype=torch.bfloat16)
    ref = a.float().cpu() @ b.float().cpu()
    err = ((a @ b).float().cpu() - ref).abs().max().item()
    print(f"  bf16 matmul 1024^3 max abs err vs CPU fp32: {err:.3f}")
