"""#164: the answer cap bounds the visible answer, not the private reasoning of a model that streams reasoning.

A real local HTTP server stands in for the model server. No model endpoint is contacted. Synthetic text only."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

AUX_MARKERS = ("CLASSIFY", "SUMMARIZE", "GATE", "EDIT")
ANSWER = "A synthetic short answer."
CAP = 16


class ReasoningUpstream:
    """Streams `reasoning` reasoning pieces, then the `answer` content pieces, then finish_reason "stop".
    With `honour_limit` it stops with finish_reason "length" once max_tokens pieces were sent, as a model
    server does. Keeps every request body."""

    def __init__(self, reasoning, answer, honour_limit=True, inline=False):
        self.requests = []
        up = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                up.requests.append(body)
                last = (body.get("messages") or [{}])[-1].get("content", "")
                if any(last.startswith(m) for m in AUX_MARKERS):
                    raw = json.dumps({"choices": [{"message": {"content": "LOG"}, "finish_reason": "stop"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                limit = body.get("max_tokens") or 10**9
                key = "content" if inline else "reasoning_content"
                pieces = (([("content", "<think>")] if inline else []) + [(key, "thought ")] * reasoning
                          + ([("content", "</think>")] if inline else []) + [("content", p) for p in answer])
                finish = "stop"
                try:
                    for n, (k, text) in enumerate(pieces):
                        if honour_limit and n >= limit:
                            finish = "length"
                            break
                        self._event({"choices": [{"delta": {k: text}}]})
                    self._event({"choices": [{"delta": {}, "finish_reason": finish}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass  # the client closed the stream

            def _event(self, data):
                self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = "http://127.0.0.1:%d/v1" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstreams():
    made = []

    def make(**kw):
        made.append(ReasoningUpstream(**kw))
        return made[-1]
    yield make
    for u in made:
        u.stop()


class _Store:
    index_enabled = False

    def __init__(self):
        self.saved = []

    def get_standing_sections_text(self, max_chars=0):
        return ""

    def update_standing_sections(self, *a, **k):
        return None

    def log_exchange(self, **kwargs):
        self.saved.append(kwargs)
        return "xid-synthetic"


TEMPLATES = {
    "skip_classifier": "CLASSIFY {user_message} {assistant_message}",
    "summarizer": "SUMMARIZE {assistant_message}",
    "index_maintenance": "GATE {user_message} {assistant_summary} {current_sections}",
    "index_edit": "EDIT {user_message} {assistant_summary} {current_sections} {today}",
}


def _state(upstream, allowance):
    from agent.llm import LLMClient
    from agent.pipeline import LoggingPipeline
    client = LLMClient(upstream.url, chat_model="synthetic-model", max_retries=1)
    client.reasoning_tokens = allowance
    store = _Store()
    st = SimpleNamespace(
        assembler=SimpleNamespace(build=lambda day, message, session_turns=None: [{"role": "user", "content": message}]),
        llm_main=client, pipeline=LoggingPipeline(store, client, client, TEMPLATES), store=store,
        max_answer_tokens=CAP)
    return st, store, client


def _answer_requests(upstream):
    return [r for r in upstream.requests
            if not any((r.get("messages") or [{}])[-1].get("content", "").startswith(m) for m in AUX_MARKERS)]


def _v1(upstream, monkeypatch, allowance=4096):
    """POST /v1/chat/completions, running the real exchange against the fake upstream: (response, store)."""
    import agent.app as appmod
    from fastapi.testclient import TestClient
    st, store, client = _state(upstream, allowance)
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda s: None)
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.setattr(appmod, "_tenant_state", lambda r, **k: st)
    real = appmod._run_exchange
    monkeypatch.setattr(appmod, "_run_exchange", lambda _st, *a, **k: real(st, *a, **k))
    sessions = dict(appmod.SESSIONS)
    try:
        r = TestClient(appmod.app, raise_server_exceptions=False).post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Synthetic question?"}]})
    finally:
        appmod.SESSIONS.clear()
        appmod.SESSIONS.update(sessions)
        client.close()
    return r, store


def _stream(upstream, monkeypatch, allowance=4096):
    """The streamed exchange against the fake upstream: (events, store)."""
    import asyncio
    import agent.app as appmod
    from agent.streaming import exchange_stream
    st, store, client = _state(upstream, allowance)
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda s: None)

    def work(emit):
        try:
            return appmod._run_exchange(st, "Synthetic question", "cap-164", "legacy", background=False, emit=emit)
        finally:
            appmod.SESSIONS.pop("legacy:cap-164", None)

    async def go():
        return "".join([c async for c in exchange_stream(work).body_iterator])
    try:
        text = asyncio.run(go())
    finally:
        client.close()
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")], store


def _answer_pieces(text):
    words = text.split(" ")
    return [w + " " for w in words[:-1]] + [words[-1], "\n[LOG: ok]"]


def test_long_reasoning_then_a_short_answer_is_200_on_v1(upstreams, monkeypatch):
    up = upstreams(reasoning=CAP * 5, answer=_answer_pieces(ANSWER))
    r, store = _v1(up, monkeypatch)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert body["choices"][0]["finish_reason"] == "stop"
    assert _answer_requests(up)[0]["max_tokens"] == CAP + 4096
    assert store.saved and "thought" not in str(store.saved[0])


def test_long_reasoning_then_a_short_answer_is_answered_on_the_stream(upstreams, monkeypatch):
    up = upstreams(reasoning=CAP * 5, answer=_answer_pieces(ANSWER))
    events, store = _stream(up, monkeypatch)
    assert not [e for e in events if e.get("type") == "error"], events
    answer = [e for e in events if e.get("type") == "answer"]
    assert answer and answer[-1]["text"] == ANSWER
    assert store.saved
    assert not any("thought" in (e.get("text") or "") for e in events if e.get("type") in ("delta", "answer"))


def test_inline_reasoning_is_not_counted_against_the_cap(upstreams, monkeypatch):
    up = upstreams(reasoning=CAP * 5, answer=_answer_pieces(ANSWER), inline=True)
    events, _ = _stream(up, monkeypatch)
    answer = [e for e in events if e.get("type") == "answer"]
    assert answer and answer[-1]["text"] == ANSWER, events


def test_a_stream_cut_at_the_larger_limit_with_no_answer_is_still_length_cap(upstreams, monkeypatch):
    up = upstreams(reasoning=CAP + 64 + 10, answer=_answer_pieces(ANSWER))
    r, store = _v1(up, monkeypatch, allowance=64)
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "length_cap"
    assert _answer_requests(up)[0]["max_tokens"] == CAP + 64
    assert not store.saved


def test_a_stream_cut_at_the_larger_limit_with_no_answer_is_still_length_cap_on_the_stream(upstreams, monkeypatch):
    up2 = upstreams(reasoning=CAP + 64 + 10, answer=_answer_pieces(ANSWER))
    events, store2 = _stream(up2, monkeypatch, allowance=64)
    error = next(e for e in events if e.get("type") == "error")
    assert error["reason"] == "length_cap" and error["saved"] is False
    assert not store2.saved


@pytest.mark.parametrize("stream", [False, True])
def test_the_visible_answer_is_cut_at_the_answer_cap(upstreams, monkeypatch, stream):
    from agent.llm import SHORTENED_NOTE
    words = ["w%d " % i for i in range(CAP * 3)]
    up = upstreams(reasoning=10, answer=words, honour_limit=False)
    if stream:
        events, store = _stream(up, monkeypatch)
        text = [e for e in events if e.get("type") == "answer"][-1]["text"]
    else:
        r, store = _v1(up, monkeypatch)
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["finish_reason"] == "length"
        text = r.json()["choices"][0]["message"]["content"]
    assert text.endswith(SHORTENED_NOTE)
    assert text[:-len(SHORTENED_NOTE)].split() == [w.strip() for w in words[:CAP]]
    assert store.saved, "an answer cut at the cap is still saved"


def test_zero_allowance_sends_the_answer_cap_alone(upstreams, monkeypatch):
    up = upstreams(reasoning=0, answer=_answer_pieces(ANSWER))
    events, _ = _stream(up, monkeypatch, allowance=0)
    assert _answer_requests(up)[0]["max_tokens"] == CAP
    assert [e for e in events if e.get("type") == "answer"][-1]["text"] == ANSWER


def test_the_reasoning_allowance_setting_is_validated():
    from agent.app import reasoning_token_allowance

    class Cfg:
        def __init__(self, value):
            self.value = value

        def get(self, key, default=None):
            assert key == "llm.max_reasoning_tokens"
            return default if self.value is None else self.value

    assert reasoning_token_allowance(Cfg(None)) == 4096
    assert reasoning_token_allowance(Cfg("8192")) == 8192
    assert reasoning_token_allowance(Cfg(" 0 ")) == 0
    for bad in ("-1", "many", "", "1.5"):
        assert reasoning_token_allowance(Cfg(bad)) == 4096


def test_the_environment_sets_the_reasoning_allowance(monkeypatch):
    from agent.config import load_config
    monkeypatch.setenv("LLM_MAX_REASONING_TOKENS", "2048")
    assert str(load_config().get("llm.max_reasoning_tokens")) == "2048"
