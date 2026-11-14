"""Tensor-parallel context: one process per GPU (torchrun), xccl collectives.

Without torchrun (no RANK in the environment) it runs as a single process on
xpu:0 and the collectives are no-ops.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


class TP:
    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        local = int(os.environ.get("LOCAL_RANK", self.rank))
        self.device = torch.device("xpu", local)
        torch.xpu.set_device(self.device)
        if self.world > 1:
            dist.init_process_group("xccl", device_id=self.device)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def split(self, n: int) -> tuple[int, int]:
        """This rank's [start, end) of a dimension of size n split evenly."""
        assert n % self.world == 0, (n, self.world)
        s = n // self.world
        return self.rank * s, (self.rank + 1) * s

    # oneCCL registers every new buffer address for GPU-to-GPU access; with large
    # messages on ever-changing torch allocations this ends in
    # UR_RESULT_ERROR_OUT_OF_RESOURCES. All collectives therefore go through one
    # persistent buffer per (purpose, dtype), which also keeps addresses fixed
    # for graph capture.
    _bufs: dict

    def _buf(self, key, numel: int, dtype) -> torch.Tensor:
        if not hasattr(self, "_bufs"):
            self._bufs = {}
        b = self._bufs.get((key, dtype))
        if b is None or b.numel() < numel:
            b = torch.empty(max(numel, 1 << 20), dtype=dtype, device=self.device)
            self._bufs[(key, dtype)] = b
        return b[:numel]

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """Sum t over ranks, in place (t contiguous)."""
        if self.world > 1:
            b = self._buf("ar", t.numel(), t.dtype)
            b.copy_(t.view(-1))
            dist.all_reduce(b)
            t.view(-1).copy_(b)
        return t

    def all_gather(self, t: torch.Tensor) -> torch.Tensor:
        """Concatenate the last dimension of t across ranks."""
        if self.world == 1:
            return t
        n = t.numel()
        src = self._buf("ag_in", n, t.dtype)
        src.copy_(t.reshape(-1))
        out = self._buf("ag_out", n * self.world, t.dtype)
        dist.all_gather_into_tensor(out, src)
        parts = out.view(self.world, *t.shape).unbind(0)
        return torch.cat(parts, dim=-1)

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def close(self):
        if self.world > 1:
            dist.destroy_process_group()
