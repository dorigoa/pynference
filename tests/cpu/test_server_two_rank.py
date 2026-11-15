"""Two-rank CPU test for pynference.server (step 3): under torchrun (gloo,
GPUs hidden), rank 1 replays every `Engine` command (C4) while rank 0 serves
HTTP; SIGTERM on rank 0 must shut both down cleanly (broadcast `stop`, then
`follow()` returns on rank 1). See `_server_worker.py`."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(ROOT, "tests", "cpu", "_server_worker.py")


def _read_when_ready(path, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return open(path).read()
        time.sleep(0.25)
    raise TimeoutError(f"{path} not ready in time")


def test_two_ranks_replay_and_clean_sigterm():
    tmp = tempfile.mkdtemp(prefix="pynference-s8-")
    proc = subprocess.Popen(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", WORKER, tmp],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        port = None
        end = time.time() + 240
        while port is None:
            p = os.path.join(tmp, "port")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                port = int(open(p).read())
            elif proc.poll() is not None:
                out, err = proc.communicate(timeout=30)
                raise RuntimeError("worker died: " + out[-3000:] + err[-3000:])
            elif time.time() > end:
                raise TimeoutError("port file never appeared")
            else:
                time.sleep(0.25)
        pid = int(_read_when_ready(os.path.join(tmp, "rank0.pid"), 30))

        body = json.dumps({"messages": [{"role": "user", "content": "3"}],
                           "temperature": 0, "max_tokens": 4}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        assert out["choices"][0]["message"]["content"] == "4 8 4 8"
        assert out["usage"]["completion_tokens"] == 4

        os.kill(pid, signal.SIGTERM)          # clean shutdown on rank 0
        proc.communicate(timeout=120)

        rank0 = open(os.path.join(tmp, "rank0.txt")).read()
        rank1 = open(os.path.join(tmp, "rank1.txt")).read()
        assert "rank0 clean exit" in rank0, rank0
        assert "follow returned" in rank1, rank1
        assert "forwards=[1, 1, 1, 1, 1]" in rank1, rank1   # prefill + 4 tokens, on rank 1 too
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
