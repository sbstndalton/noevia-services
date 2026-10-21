"""#164: the answer cap bounds the visible answer, not the private reasoning of a model that streams reasoning.
The visible cap is approximate tokens enforced as characters, independent of how the stream is chunked.

A real local HTTP server stands in for the model server. No model endpoint is contacted. Synthetic text only."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

AUX_MARKERS = ("CLASSIFY", "SUMMARIZE", "GATE", "EDIT")
ANSWER = "A synthetic short answer."
MARKER = "\n[LOG: ok]"
CAP = 16
CAP_CHARS = CAP * 4  # the documented visible cap: tokens times 4 characters
SERVER_CHARS_PER_TOKEN = 4  # the fake server budget, measured on characters, never on pieces


def letters(n):
    """Synthetic answer text without whitespace, so a cut boundary is never trimmed away."""
    return "".join(chr(97 + i % 26) for i in range(n))


def chunk(text, how):
    if not text:
        return []
    if how == "merged":
        return [text]
    if how == "chars":
        return list(text)
    return [text[i:i + 5] for i in range(0, len(text), 5)]  # "small": many short pieces


class Upstream:
    """A fake model server. It produces `reasoning` (separate reasoning_content, or inline when `inline`) and
    then `answer`, and enforces the request max_tokens as a budget of max_tokens * SERVER_CHARS_PER_TOKEN
    characters over everything it generates (reasoning included), cutting mid-piece with finish "length" -
    independent of how the output is chunked. `how` chooses the chunking; `json` answers with one JSON body
    even when a stream was asked for; `usage` adds a terminal usage-only chunk."""

    def __init__(self, reasoning="", answer="", how="small", inline=False, json_reply=False, usage=False,
                 unclosed=False):
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
                    return self._json({"content": "LOG"}, "stop")
                budget = (body.get("max_tokens") or 10**9) * SERVER_CHARS_PER_TOKEN
                if inline:
                    parts = [("content", "<think>" + reasoning + ("" if unclosed else "</think>") + answer)]
                else:
                    parts = [("reasoning_content", reasoning), ("content", answer)]
                out, finish = [], "stop"
                for key, text in parts:
                    take = text[:budget]
                    budget -= len(take)
                    out.append((key, take))
                    if len(take) < len(text):
                        finish = "length"
                        break
                if json_reply or not body.get("stream"):
                    message = {"content": "".join(t for k, t in out if k == "content")}
                    r = "".join(t for k, t in out if k == "reasoning_content")
                    if r:
                        message["reasoning_content"] = r
                    return self._json(message, finish)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for key, text in out:
                        for piece in chunk(text, how):
                            self._event({"choices": [{"delta": {key: piece}}]})
                    self._event({"choices": [{"delta": {}, "finish_reason": finish}]})
                    if usage:
                        self._event({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                                                               "total_tokens": 8}})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass  # the client closed the stream

            def _json(self, message, finish):
                raw = json.dumps({"choices": [{"message": message, "finish_reason": finish}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

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
        made.append(Upstream(**kw))
        return made[-1]
    yield make
    for u in made:
        u.stop()


class _Store:
    index_enabled = False

    def __init__(self):
        self.saved = []
        self.answers = []

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
    pipeline = LoggingPipeline(store, client, client, TEMPLATES)
    save = pipeline.log_exchange

    def spy(**kwargs):  # the answer text handed to the save step
        store.answers.append(kwargs["assistant_message"])
        return save(**kwargs)
    pipeline.log_exchange = spy
    st = SimpleNamespace(
        assembler=SimpleNamespace(build=lambda day, message, session_turns=None: [{"role": "user", "content": message}]),
        llm_main=client, pipeline=pipeline, store=store, max_answer_tokens=CAP)
    return st, store, client


def _answer_requests(upstream):
    return [r for r in upstream.requests
            if not any((r.get("messages") or [{}])[-1].get("content", "").startswith(m) for m in AUX_MARKERS)]


def _exchange(upstream, monkeypatch, stream, allowance=4096):
    """One real exchange (streamed or not) against the fake: (result, events, store)."""
    import agent.app as appmod
    st, store, client = _state(upstream, allowance)
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda s: None)
    events = []
    try:
        result = appmod._run_exchange(st, "Synthetic question", "cap-164", "legacy", background=False,
                                      emit=events.append if stream else None)
    finally:
        appmod.SESSIONS.pop("legacy:cap-164", None)
        client.close()
    return result, events, store


def _v1(upstream, monkeypatch, allowance=4096):
    """POST /v1/chat/completions, running the real exchange against the fake: (response, store)."""
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


def _stream(upstream, allowance=4096):
    """The streamed exchange through the stream endpoint wrapper: (events, store)."""
    import asyncio
    import agent.app as appmod
    from agent.streaming import exchange_stream
    st, store, client = _state(upstream, allowance)
    orig = appmod._reindex_dirty
    appmod._reindex_dirty = lambda s: None

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
        appmod._reindex_dirty = orig
        client.close()
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")], store


# ---- the visible cap does not depend on chunking ----------------------------------------------------------

DELIVERIES = [("merged", False), ("small", False), ("chars", False), ("json", False),
              ("merged", True), ("small", True), ("chars", True), ("json", True)]


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_the_same_output_is_cut_identically_however_it_is_chunked(upstreams, monkeypatch, offset):
    from agent.llm import SHORTENED_NOTE
    answer = letters(CAP_CHARS + offset)
    outcomes = []
    for how, stream in DELIVERIES:
        up = upstreams(reasoning="thought " * 40, answer=answer + MARKER,
                       how="small" if how == "json" else how, json_reply=how == "json")
        result, events, store = _exchange(up, monkeypatch, stream)
        if stream:
            final = [e for e in events if e.get("type") == "answer"]
            assert final and final[-1]["text"] == result["reply"]
            assert len("".join(e["text"] for e in events if e.get("type") == "delta")) <= CAP_CHARS
        outcomes.append((result["reply"], bool(result.get("truncated")), store.answers))
    expected_text = answer if offset <= 0 else answer[:CAP_CHARS]
    expected_reply = expected_text if offset <= 0 else expected_text + "\n\n" + SHORTENED_NOTE
    for (how, stream), outcome in zip(DELIVERIES, outcomes):
        assert outcome == (expected_reply, offset > 0, [expected_text]), (how, stream)


def test_the_non_streamed_chat_call_has_the_same_visible_cap(upstreams):
    from agent.llm import LLMClient
    for offset in (0, 1):
        up = upstreams(answer=letters(CAP_CHARS + offset) + MARKER)
        client = LLMClient(up.url, chat_model="synthetic-model", max_retries=1)
        try:
            reply = client.chat([{"role": "user", "content": "Synthetic?"}], max_tokens=CAP)
        finally:
            client.close()
        assert up.requests[0]["stream"] is False and up.requests[0]["max_tokens"] == CAP + 4096
        if offset:
            assert str(reply) == letters(CAP_CHARS) and reply.truncated is True
        else:
            assert str(reply) == letters(CAP_CHARS) + MARKER and reply.truncated is False


def test_a_normal_non_streamed_answer_is_unchanged(upstreams, monkeypatch):
    up = upstreams(answer=ANSWER + MARKER, json_reply=True)
    result, _, store = _exchange(up, monkeypatch, stream=False)
    assert result["reply"] == ANSWER and not result.get("truncated") and store.answers == [ANSWER]


@pytest.mark.parametrize("stream", [False, True])
def test_a_terminal_usage_chunk_does_not_change_the_outcome(upstreams, monkeypatch, stream):
    outcomes = []
    for usage in (False, True):
        up = upstreams(reasoning="thought " * 40, answer=letters(CAP_CHARS + 9) + MARKER, usage=usage)
        result, _, store = _exchange(up, monkeypatch, stream)
        outcomes.append((result["reply"], result.get("truncated"), store.answers))
    assert outcomes[0] == outcomes[1] and outcomes[0][1] is True


# ---- reasoning does not use up the answer ----------------------------------------------------------------


def test_long_reasoning_then_a_short_answer_is_200_on_v1(upstreams, monkeypatch):
    up = upstreams(reasoning="thought " * (CAP * 10), answer=ANSWER + MARKER)
    r, store = _v1(up, monkeypatch)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert body["choices"][0]["finish_reason"] == "stop"
    assert _answer_requests(up)[0]["max_tokens"] == CAP + 4096
    assert store.answers == [ANSWER]


def test_long_reasoning_then_a_short_answer_is_answered_on_the_stream(upstreams):
    up = upstreams(reasoning="thought " * (CAP * 10), answer=ANSWER + MARKER)
    events, store = _stream(up)
    assert not [e for e in events if e.get("type") == "error"], events
    answer = [e for e in events if e.get("type") == "answer"]
    assert answer and answer[-1]["text"] == ANSWER
    assert store.answers == [ANSWER]
    assert not any("thought" in (e.get("text") or "") for e in events if e.get("type") in ("delta", "answer"))


@pytest.mark.parametrize("how", ["merged", "chars"])
def test_inline_reasoning_is_not_counted_against_the_cap(upstreams, monkeypatch, how):
    up = upstreams(reasoning="thought " * (CAP * 10), answer=ANSWER + MARKER, inline=True, how=how)
    result, _, store = _exchange(up, monkeypatch, stream=True)
    assert result["reply"] == ANSWER and store.answers == [ANSWER]


# ---- a genuinely cut answer with no text is still length_cap --------------------------------------------


def test_reasoning_cut_by_the_server_with_no_answer_is_length_cap_on_v1(upstreams, monkeypatch):
    up = upstreams(reasoning="thought " * 200, answer=ANSWER + MARKER)
    r, store = _v1(up, monkeypatch, allowance=64)
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "length_cap"
    assert _answer_requests(up)[0]["max_tokens"] == CAP + 64
    assert not store.saved


@pytest.mark.parametrize("inline", [False, True], ids=["reasoning_content", "unclosed_inline"])
@pytest.mark.parametrize("json_reply", [False, True], ids=["stream", "json"])
def test_reasoning_only_cut_by_the_server_is_length_cap_on_the_stream(upstreams, inline, json_reply):
    up = upstreams(reasoning="thought " * 200, answer="", inline=inline, unclosed=inline, json_reply=json_reply)
    events, store = _stream(up, allowance=64)
    error = next(e for e in events if e.get("type") == "error")
    assert error["code"] == "modelFailed" and error["reason"] == "length_cap" and error["saved"] is False
    assert not store.saved and not [e for e in events if e.get("type") == "delta"]


# ---- the setting ------------------------------------------------------------------------------------------


def test_zero_allowance_sends_the_answer_cap_alone(upstreams, monkeypatch):
    up = upstreams(answer=ANSWER + MARKER)
    result, _, _ = _exchange(up, monkeypatch, stream=True, allowance=0)
    assert _answer_requests(up)[0]["max_tokens"] == CAP
    assert result["reply"] == ANSWER


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
