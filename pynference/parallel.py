"""Tensor-parallel context: one process per GPU (torchrun), xccl collectives.

Without torchrun (no RANK in the environment) it runs as a single process on
xpu:0 and the collectives are no-ops.
"""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def _ctrl_group(rank: int, world: int):
    """CPU-only gloo group for the control channel (contract C4).

    `dist.new_group(backend="gloo")` is unusable once the default group is
    bound to a GPU (job S7-1): torch inherits `bound_device_id` (xpu:0 here)
    into every subgroup and the call dies in `pg._get_backend(xpu:0)` on the
    gloo group — "No backend type associated with device type xpu" (passing
    `device_id=cpu` is rejected with "Mismatched bound device"). So we call
    the same private creator torch itself uses for new_group,
    `_new_process_group_helper`, with `device_id=None`: exactly the group
    new_group would build, minus the device binding. Private APIs on purpose,
    tied to the torch installed on the PC (2.15.0+xpu)."""
    from torch.distributed import distributed_c10d as c10d

    ranks = list(range(world))
    pg, _ = c10d._new_process_group_helper(
        world,
        rank,
        ranks,
        c10d.Backend.GLOO,
        c10d._get_default_store(),
        "pynference_ctrl",
        # idle ranks sit in the first broadcast between chat turns
        timeout=timedelta(days=1),
        group_desc="pynference control channel",
    )
    c10d._world.pg_group_ranks[pg] = {g: g for g in ranks}
    dist.barrier()  # like _new_group_with_tag's post-init barrier
    return pg


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

    def broadcast_obj(self, obj):
        """Send a pickled Python object from rank 0 to every rank (contract C4).

        Runs on a CPU-only gloo group created on first use (see `_ctrl_group`),
        so it never interferes with the xccl tensor collectives. With one
        process it just returns `obj`."""
        if self.world == 1:
            return obj
        if not hasattr(self, "_ctrl_group"):
            self._ctrl_group = _ctrl_group(self.rank, self.world)
        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self._ctrl_group,
                                   device=torch.device("cpu"))
        return box[0]

    def close(self):
        if self.world > 1:
            dist.destroy_process_group()
