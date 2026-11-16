"""CPU tests for pynference.server (contract D3) over the fake C3 model.

A real HTTP server (ThreadingHTTPServer) runs in a background thread on a
random loopback port; an `urllib` client exercises every endpoint. Prompt
building is injected (`prompt_ids`): the digit contents of the messages
become the prompt ids, so the greedy stream is predictable. Streaming and
non-streaming must produce the same text, `stop`/`max_tokens` must behave,
and bad requests must answer 400/404 with a JSON body."""
import json
import threading
import urllib.error
import urllib.request
from http.client import HTTPConnection

import pytest
import torch
from fake_model import GREEDY, FakeModel, FakeTok

from pynference.engine import Engine
from pynference.server import PynferenceServer

PROMPT_IDS = lambda msgs: [int(ch) for m in msgs for ch in m["content"] if ch.isdigit()]
DEFAULTS = dict(GREEDY, max_tokens=512)


def make_client():
    engine = Engine(FakeModel(), FakeTok())
    server = PynferenceServer(engine, FakeTok(), model_id="fake", defaults=dict(DEFAULTS),
                              prompt_ids=PROMPT_IDS, host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return engine, server, f"http://127.0.0.1:{server.port}"


def get(url, path):
    try:
        with urllib.request.urlopen(url + path, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(url, body):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def stream(url, body):
    """POST with stream: true; returns (content, finish_reason, n_content_chunks, raw)."""
    hostport = url.split("//", 1)[1]
    conn = HTTPConnection(hostport, timeout=60)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    assert r.status == 200, r.status
    assert r.getheader("Content-Type") == "text/event-stream"
    raw = r.read().decode()
    conn.close()
    events = [e[len("data: "):] for e in raw.split("\n\n") if e.startswith("data: ")]
    assert events[-1] == "[DONE]", raw[-200:]
    content, finish, nchunks = "", None, 0
    for e in events[:-1]:
        c = json.loads(e)
        assert c["object"] == "chat.completion.chunk" and c["model"] == "fake"
        d = c["choices"][0]["delta"]
        content += d.get("content", "")
        if d.get("content"):
            nchunks += 1
        if c["choices"][0]["finish_reason"]:
            finish = c["choices"][0]["finish_reason"]
    return content, finish, nchunks, events


@pytest.fixture()
def srv():
    engine, server, url = make_client()
    yield engine, url
    server.stopping = True
    server.httpd.shutdown()


def test_health_and_models(srv):
    _, url = srv
    assert get(url, "/health") == (200, {"status": "ok", "model": "fake"})
    status, body = get(url, "/v1/models")
    assert status == 200 and body["object"] == "list"
    assert [d["id"] for d in body["data"]] == ["fake"]


def test_completion_matches_engine(srv):
    engine, url = srv
    expected = engine.run({"op": "generate", "ids": [3], "max_tokens": 6,
                           "sampling": GREEDY, "stop_ids": [0]})
    status, body = post(url, {"messages": [{"role": "user", "content": "3"}],
                              "temperature": 0, "max_tokens": 6})
    assert status == 200
    assert body["object"] == "chat.completion" and body["model"] == "fake"
    assert body["id"].startswith("chatcmpl-") and isinstance(body["created"], int)
    ch = body["choices"]
    assert len(ch) == 1 and ch[0]["index"] == 0
    assert ch[0]["message"] == {"role": "assistant", "content": FakeTok().decode(expected)}
    assert ch[0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 1, "completion_tokens": len(expected),
                             "total_tokens": 1 + len(expected)}


def test_stream_equals_completion(srv):
    _, url = srv
    req = {"messages": [{"role": "user", "content": "3579"}], "temperature": 0,
           "max_tokens": 9}
    _, body = post(url, req)
    content, finish, nchunks, events = stream(url, dict(req, stream=True))
    assert content == body["choices"][0]["message"]["content"]
    assert finish == body["choices"][0]["finish_reason"]
    assert nchunks == body["usage"]["completion_tokens"]     # one chunk per token
    first = json.loads(events[0])
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}


def test_max_tokens_finish_length(srv):
    _, url = srv
    status, body = post(url, {"messages": [{"role": "user", "content": "1"}],
                              "temperature": 0, "max_tokens": 3,
                              "max_completion_tokens": 4})
    assert status == 200
    assert body["usage"]["completion_tokens"] == 4           # max_completion_tokens wins
    assert body["choices"][0]["finish_reason"] == "length"


def test_stop_string(srv):
    _, url = srv
    # ids [5]: greedy 6, 12, ... — "12" starts at char 2 of "6 12", and the
    # token-level cut of the engine keeps only "6" (decode(acc[:2]) is too long)
    req = {"messages": [{"role": "user", "content": "5"}], "temperature": 0,
           "max_tokens": 12, "stop": ["12"]}
    status, body = post(url, req)
    assert status == 200 and body["choices"][0]["finish_reason"] == "stop"
    content = body["choices"][0]["message"]["content"]
    assert content == "6" and body["usage"]["completion_tokens"] == 1
    _, _, _, events = stream(url, dict(req, stop="12", stream=True))
    # the streamed text is byte-identical to the non-streaming one (no "12" leak)
    got = "".join(json.loads(e)["choices"][0]["delta"].get("content", "")
                  for e in events[:-1])
    assert got == content


def test_greedy_is_deterministic_with_seed(srv):
    _, url = srv
    body = {"messages": [{"role": "user", "content": "2"}], "temperature": 0.7,
            "top_k": 16, "top_p": 1.0, "seed": 42, "max_tokens": 5}
    _, a = post(url, body)
    _, b = post(url, dict(body, seed=42))
    assert a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"]


def test_one_at_a_time(srv):
    """Two concurrent requests both complete (the second waits on the lock)."""
    import concurrent.futures
    _, url = srv
    req = {"messages": [{"role": "user", "content": "35"}], "temperature": 0, "max_tokens": 8}
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        outs = list(ex.map(lambda _: post(url, req), range(2)))
    assert [o[0] for o in outs] == [200, 200]
    assert outs[0][1]["choices"][0]["message"]["content"] == \
        outs[1][1]["choices"][0]["message"]["content"]


def test_errors(srv):
    _, url = srv
    bad_json = post(url, b"{not json")
    assert bad_json[0] == 400 and bad_json[1]["error"]["type"] == "invalid_request_error"
    for body in [{"messages": []},
                 {"messages": "hi"},
                 [{"role": "user", "content": "x"}],
                 {"messages": [{"role": "user"}]},
                 {"messages": [{"role": "user", "content": 5}], "temperature": 0},
                 {"messages": [{"role": "user", "content": "1"}], "temperature": -1},
                 {"messages": [{"role": "user", "content": "1"}], "max_tokens": 0},
                 {"messages": [{"role": "user", "content": "1"}], "top_p": 2},
                 {"messages": [{"role": "user", "content": "1"}], "stream": "yes"},
                 {"messages": [{"role": "user", "content": "1"}], "stop": ["", "x"]},
                 {"messages": [{"role": "user", "content": "1"}], "n": 2}]:
        status, err = post(url, body)
        assert status == 400, (body, err)
        assert err["error"]["message"], err
    status, body = get(url, "/nope")
    assert status == 404 and body["error"]["message"].count("nope") == 1


def test_404_on_wrong_method(srv):
    _, url = srv
    assert get(url, "/v1/chat/completions")[0] == 404
    status, _ = post(url, {"messages": [{"role": "user", "content": "3"}]})
    assert status == 200          # sanity: POST works


# ---- reasoning separation (S14 step 6) ----
#
# A `pre`-carrying tokenizer makes the server split the answer per family:
# reasoning goes to `reasoning_content` (delta while streaming, message key
# otherwise), `content` only ever carries the visible text. Every answer
# token is one id (`token_bytes` below), so tags arrive in one piece.

LT = "\x3c"                    # first char escaped: no chat marker in sources
REASON = "deep"
ANSWER = "Rome."


class SplitTok:
    """FakeTok with a `pre`, so `ReasoningSplitter.from_tokenizer` fires;
    TEXT maps ids to their (tag-included) text, others render as digits."""
    bos_id, eos_id = 2, 0
    eog_ids = {0}
    pre = "qwen35"
    TEXT = {}

    def token_bytes(self, tid: int, special: bool = True) -> bytes:
        return self.TEXT.get(tid, str(tid)).encode()

    def decode(self, ids, special: bool = False) -> str:
        return "".join(self.token_bytes(i, special).decode() for i in ids)


class QwenTok(SplitTok):
    TEXT = {1: REASON, 2: LT + "/think>", 3: "\n\n", 4: ANSWER}


class GptOssTok(SplitTok):
    pre = "gpt-4o"
    TEXT = {1: LT + "|channel|>", 2: "analysis", 3: LT + "|message|>",
            4: REASON, 5: LT + "channel|>", 6: LT + "|start|>assistant",
            7: LT + "|channel|>", 8: "final", 9: LT + "|message|>",
            10: ANSWER, 11: LT + "|return|>"}


def make_split_client(text_tok, stream_ids):
    """Server over a fake model whose next token walks `stream_ids` in order
    (first forward = the prompt logits, then one id per decode step)."""
    class FixedModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.stream, self.n_fwd = list(stream_ids), 0

        def forward(self, tokens, chunk: int = 512, all_logits: bool = False):
            self.forward_sizes.append(len(tokens))
            self.ctx.extend(tokens)
            self.n_past = len(self.ctx)
            nxt = self.stream[min(self.n_fwd, len(self.stream) - 1)]
            self.n_fwd += 1
            logits = torch.zeros(FakeModel.n_vocab, dtype=torch.float32)
            logits[nxt] = 2.0
            return logits

    engine = Engine(FixedModel(), text_tok)
    server = PynferenceServer(engine, text_tok, model_id="fake",
                              defaults=dict(GREEDY, max_tokens=64),
                              prompt_ids=lambda msgs: [7], host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.port}"


def stream_parts(url, body):
    """POST stream: returns (content, reasoning_content, finish)."""
    hostport = url.split("//", 1)[1]
    conn = HTTPConnection(hostport, timeout=60)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    assert r.status == 200, r.status
    raw = r.read().decode()
    conn.close()
    events = [e[len("data: "):] for e in raw.split("\n\n") if e.startswith("data: ")]
    assert events[-1] == "[DONE]", raw[-200:]
    content, reasoning, finish = "", "", None
    for e in events[:-1]:
        c = json.loads(e)
        d = c["choices"][0]["delta"]
        content += d.get("content", "")
        reasoning += d.get("reasoning_content", "")
        if c["choices"][0]["finish_reason"]:
            finish = c["choices"][0]["finish_reason"]
    return content, reasoning, finish


@pytest.fixture()
def qwen_srv():
    server, url = make_split_client(QwenTok(), [1, 2, 3, 4, 0])
    yield url
    server.stopping = True
    server.httpd.shutdown()


@pytest.fixture()
def oss_srv():
    server, url = make_split_client(GptOssTok(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0])
    yield url
    server.stopping = True
    server.httpd.shutdown()


def test_stream_reasoning_qwen(qwen_srv):
    content, reasoning, finish = stream_parts(
        qwen_srv, {"messages": [{"role": "user", "content": "x"}], "stream": True})
    assert (content, reasoning) == ("\n\n" + ANSWER, REASON)
    assert finish == "stop"


def test_completion_reasoning_qwen(qwen_srv):
    status, body = post(qwen_srv, {"messages": [{"role": "user", "content": "x"}]})
    msg = body["choices"][0]["message"]
    assert status == 200 and msg["content"] == "\n\n" + ANSWER
    assert msg["reasoning_content"] == REASON


def test_stream_reasoning_gpt_oss(oss_srv):
    content, reasoning, finish = stream_parts(
        oss_srv, {"messages": [{"role": "user", "content": "x"}], "stream": True})
    assert (content, reasoning) == (ANSWER, REASON)
    assert finish == "stop"


def test_completion_reasoning_gpt_oss(oss_srv):
    status, body = post(oss_srv, {"messages": [{"role": "user", "content": "x"}]})
    msg = body["choices"][0]["message"]
    assert status == 200 and msg["content"] == ANSWER
    assert msg["reasoning_content"] == REASON


def test_stop_strings_cut_only_visible(qwen_srv):
    content, reasoning, finish = stream_parts(
        qwen_srv, {"messages": [{"role": "user", "content": "x"}],
                   "stream": True, "stop": [ANSWER[:2]]})
    assert reasoning == REASON            # reasoning streams before the stop cut
    assert ANSWER[:2] not in content and content.startswith("\n\n")
    assert finish == "stop"


def test_completion_without_splitter_unchanged(srv):
    # FakeTok has no `pre`: no reasoning_content key anywhere (regression lock)
    _, url = srv
    status, body = post(url, {"messages": [{"role": "user", "content": "3"}],
                              "max_tokens": 4})
    assert "reasoning_content" not in body["choices"][0]["message"]
    content, reasoning, _ = stream_parts(url, {"messages": [{"role": "user", "content": "3"}],
                                               "stream": True, "max_tokens": 4})
    assert reasoning == "" and content
