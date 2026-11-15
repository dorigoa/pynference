"""Multi-rank CPU test of TP.broadcast_obj (contract C4): two gloo ranks
under torchrun exchange a command dict on the control group. The GPU run
(job S7-1) hit torch 2.15's device binding, which kills
`dist.new_group(backend="gloo")` once the default group is bound to xpu;
this worker runs the fixed path end to end (with GPUs hidden, so the
binding is cpu:0 and nothing touches a device)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(ROOT, "tests", "cpu", "_gloo_worker.py")


def test_broadcast_obj_two_ranks():
    r = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", WORKER],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    assert "gloo ctrl ok" in r.stdout
