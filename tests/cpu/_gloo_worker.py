"""Worker for test_tp_gloo: run under torchrun with two CPU (gloo) ranks.

Exercises the real TP.broadcast_obj path — control-group creation and the
object round-trip — with a default group bound to a device, like the GPU
run has (xpu:0 there); here the runner hides the GPUs, so the bound device
is cpu:0."""
import os
import sys
from datetime import timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.distributed as dist

from pynference.parallel import TP


def main():
    dist.init_process_group("gloo", device_id=torch.device("cpu", 0),
                            timeout=timedelta(seconds=120))
    tp = TP.__new__(TP)  # skip __init__: no GPU and no xccl here
    tp.world = int(os.environ["WORLD_SIZE"])
    tp.rank = int(os.environ["RANK"])

    cmd = ({"op": "generate", "ids": [5, 6, 7], "max_tokens": 32,
            "sampling": {"temp": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 7},
            "stop_ids": [1, 106]} if tp.rank == 0 else None)
    got = tp.broadcast_obj(cmd)
    assert got["op"] == "generate" and got["ids"] == [5, 6, 7]
    assert got["sampling"]["seed"] == 7 and got["stop_ids"] == [1, 106]
    stop = tp.broadcast_obj({"op": "stop"} if tp.rank == 0 else None)
    assert stop["op"] == "stop"
    if tp.rank == 1:
        print("gloo ctrl ok", flush=True)
    dist.destroy_process_group()


main()
