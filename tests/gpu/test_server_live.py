"""Live test of pynference.server on the real Gemma (job S8-1, contract D3).

Skipped unless the job exported PYI_SERVER_URL (running `pynference.server`),
PYI_REF_FILE (stdout of `pynference.cli generate --chat -p PROMPT -n 64
--temp 0`) and PYI_REF_PROMPT. Both paths build the prompt with
`tokenizer.chat_prompt` + `encode`, so a greedy request must reproduce the
reference text, streaming must reproduce the non-streaming answer byte for
byte, and `usage` must be consistent."""
import json
import os
import urllib.request
from http.client import HTTPConnection

import pytest

URL = os.environ.get("PYI_SERVER_URL", "").rstrip("/")
REF_FILE = os.environ.get("PYI_REF_FILE", "")
PROMPT = os.environ.get("PYI_REF_PROMPT", "")
N = 64

if not URL or not PROMPT or not os.path.exists(REF_FILE):
    pytest.skip("live server env not set (run inside job S8-1)",
                allow_module_level=True)

REF = open(REF_FILE).read().strip()
# the job strips loader progress ("  loaded N/M layers ...", models/gemma4.py)
# and oneAPI CCL warnings that land on stdout; guard against regressions:
assert "CCL_WARN" not in REF and "loaded " not in REF.split("\n")[0], \
    "REF_FILE still contains loader/CCL noise"
REQ = {"messages": [{"role": "user", "content": PROMPT}],
       "temperature": 0, "max_tokens": N}


def get(path):
    with urllib.request.urlopen(URL + path, timeout=60) as r:
        return json.loads(r.read())


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def stream(body):
    """POST with stream: true; returns (content, finish_reason, n_content_chunks)."""
    conn = HTTPConnection(URL.split("//", 1)[1], timeout=600)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    raw = r.read().decode()
    conn.close()
    events = [e[len("data: "):] for e in raw.split("\n\n") if e.startswith("data: ")]
    assert events[-1] == "[DONE]", raw[-200:]
    content, finish, nchunks = "", None, 0
    for e in events[:-1]:
        c = json.loads(e)
        assert c["object"] == "chat.completion.chunk"
        d = c["choices"][0]["delta"]
        content += d.get("content", "")
        if d.get("content"):
            nchunks += 1
        if c["choices"][0]["finish_reason"]:
            finish = c["choices"][0]["finish_reason"]
    return content, finish, nchunks


def test_health_and_models():
    health = get("/health")
    assert health["status"] == "ok"
    ids = [d["id"] for d in get("/v1/models")["data"]]
    assert health["model"] in ids and len(ids) == 1


def test_greedy_matches_cli_generate():
    body = post(REQ)
    content = body["choices"][0]["message"]["content"]
    assert content.strip() == REF                 # same prompt, same greedy path
    # measured (job S8-1): greedy on this prompt closes the turn (eog) at 54
    # tokens, well before max_tokens=64 -> finish is "stop", not "length"
    assert body["choices"][0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert 0 < usage["completion_tokens"] < N and usage["prompt_tokens"] > 10
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_stream_equals_completion():
    content, finish, nchunks = stream(dict(REQ, stream=True))
    body = post(REQ)
    assert content == body["choices"][0]["message"]["content"]   # byte for byte
    assert finish == body["choices"][0]["finish_reason"] == "stop"
    assert nchunks == body["usage"]["completion_tokens"]


def test_usage_prompt_tokens_consistent():
    short = {"messages": [{"role": "user", "content": PROMPT}],
             "temperature": 0, "max_tokens": 8}
    long = {"messages": [{"role": "user", "content": PROMPT + " Aggiungi dettagli."}],
            "temperature": 0, "max_tokens": 8}
    p1, p2, p3 = (post(b)["usage"]["prompt_tokens"] for b in (short, short, long))
    assert p1 == p2 and p3 > p1


def test_stop_string_truncates():
    word = REF.split()[3]        # with a clean REF this is a word of the answer
    assert len(word) > 2
    full = post(REQ)["choices"][0]["message"]["content"]
    body = post(dict(REQ, stop=[word]))
    content = body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "stop"
    assert len(content) < len(full) and full.startswith(content)
