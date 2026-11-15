"""Worker for test_server_two_rank: two gloo ranks under torchrun.

Rank 0 runs the real `PynferenceServer` over an `Engine` with `tp` (so every
command goes through `broadcast_obj`); rank 1 just `follow()`s. The test
kills rank 0 with SIGTERM: `run_server` must finish cleanly (broadcast stop,
close the group), and rank 1 must then return from `follow()` — both write a
marker file with what they did. Same TP/broadcast trick as `_gloo_worker.py`
(the runner hides the GPUs, so `TP.__init__` would fail on `torch.xpu`)."""
import os
import sys
from datetime import timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "cpu"))

import torch
import torch.distributed as dist

from fake_model import GREEDY, FakeModel, FakeTok

from pynference.engine import Engine
from pynference.parallel import TP
from pynference.server import PynferenceServer, run_server


def main():
    outdir = sys.argv[1]
    dist.init_process_group("gloo", device_id=torch.device("cpu", 0),
                            timeout=timedelta(seconds=120))
    tp = TP.__new__(TP)  # skip __init__: no GPU and no xccl here
    tp.world = int(os.environ["WORLD_SIZE"])
    tp.rank = int(os.environ["RANK"])

    engine = Engine(FakeModel(), FakeTok(), tp)
    if tp.rank != 0:
        engine.follow()
        with open(os.path.join(outdir, "rank1.txt"), "w") as f:
            f.write(f"follow returned; forwards={engine.model.forward_sizes}\n")
        tp.close()
        return

    server = PynferenceServer(
        engine, FakeTok(), model_id="fake", defaults=dict(GREEDY, max_tokens=512),
        prompt_ids=lambda msgs: [int(ch) for m in msgs for ch in m["content"] if ch.isdigit()],
        host="127.0.0.1", port=0)
    with open(os.path.join(outdir, "port"), "w") as f:
        f.write(str(server.port))
    with open(os.path.join(outdir, "rank0.pid"), "w") as f:
        f.write(str(os.getpid()))
    run_server(server, engine, tp)      # SIGTERM => clean shutdown
    with open(os.path.join(outdir, "rank0.txt"), "w") as f:
        f.write("rank0 clean exit\n")


main()
