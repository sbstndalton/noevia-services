"""#127: every companion answer is capped, a cut-off answer is kept with a note, and when the exchange
deadline gives an answer up the model connection is closed so the server stops generating.

A real local HTTP server stands in for the model server, so a client disconnect can be observed. No model
endpoint is contacted. Synthetic text only."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

AUX_MARKERS = ("CLASSIFY", "SUMMARIZE", "GATE", "EDIT")
ANSWER = "A synthetic answer about a quiet day."


class FakeUpstream:
    """mode: "answer" (a full answer), "stall_mid" (headers and one token, then nothing), "stall_head"
    (nothing at all). Every request body is kept; a client disconnect during a stall is recorded."""

    def __init__(self, mode="answer", finish="stop"):
        self.mode, self.finish = mode, finish
        self.requests = []
        self.release = threading.Event()
        self.disconnected = threading.Event()
        self.disconnected_at = None
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                upstream.requests.append(body)
                last = (body.get("messages") or [{}])[-1].get("content", "")
                if any(last.startswith(m) for m in AUX_MARKERS):
                    return self._json("LOG", "stop")
                if upstream.mode == "stall_head":
                    return self._wait_for_disconnect()
                if not body.get("stream"):
                    if upstream.mode == "stall_mid":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.flush()
                        return self._wait_for_disconnect()
                    return self._json(ANSWER + "\n[LOG: ok]", upstream.finish)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self._event({"choices": [{"delta": {"content": ANSWER}}]})
                if upstream.mode == "stall_mid":
                    return self._wait_for_disconnect()
                if upstream.finish == "stop":
                    self._event({"choices": [{"delta": {"content": "\n[LOG: ok]"}}]})
                self._event({"choices": [{"delta": {}, "finish_reason": upstream.finish}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def _event(self, data):
                self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                self.wfile.flush()

            def _json(self, content, finish):
                raw = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": finish}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _wait_for_disconnect(self):
                self.connection.settimeout(0.02)
                end = time.monotonic() + 10
                while not upstream.release.is_set() and time.monotonic() < end:
                    try:
                        if self.connection.recv(1) == b"":
                            break
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                else:
                    return
                upstream.disconnected_at = time.monotonic()
                upstream.disconnected.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = "http://127.0.0.1:%d/v1" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream_factory():
    made = []

    def make(**kw):
        made.append(FakeUpstream(**kw))
        return made[-1]
    yield make
    for u in made:
        u.stop()


@pytest.fixture
def fast(monkeypatch):
    from agent import deadline
    monkeypatch.setattr(deadline, "EXCHANGE_DEADLINE_S", 0.6)
    monkeypatch.setattr(deadline, "FIRST_TOKEN_DEADLINE_S", 0.4)
    monkeypatch.setattr(deadline, "AUX_STEP_DEADLINE_S", 0.5)
    return deadline


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


def _exchange(upstream, stream, monkeypatch, **st_extra):
    """One _run_exchange against the fake upstream: (result, events, store, client)."""
    import agent.app as appmod
    from agent.llm import LLMClient
    from agent.pipeline import LoggingPipeline
    client = LLMClient(upstream.url, chat_model="synthetic-model", max_retries=1)
    store = _Store()
    pipeline = LoggingPipeline(store, client, client, TEMPLATES)
    save = pipeline.log_exchange

    def spy(**kwargs):  # the answer text handed to the save step
        store.answers.append(kwargs["assistant_message"])
        return save(**kwargs)
    pipeline.log_exchange = spy
    st = SimpleNamespace(
        assembler=SimpleNamespace(build=lambda day, message, session_turns=None: [{"role": "user", "content": message}]),
        llm_main=client, pipeline=pipeline, store=store, **st_extra)
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda s: None)
    events = []
    try:
        result = appmod._run_exchange(st, "Synthetic question", "cap-127", "legacy", background=False,
                                      emit=events.append if stream else None)
    finally:
        appmod.SESSIONS.pop("legacy:cap-127", None)
    return result, events, store, client


def _answer_requests(upstream):
    return [r for r in upstream.requests
            if not any((r.get("messages") or [{}])[-1].get("content", "").startswith(m) for m in AUX_MARKERS)]


# ---- the cap ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_every_chat_call_of_an_exchange_sends_a_token_cap(upstream_factory, monkeypatch, stream):
    upstream = upstream_factory()
    result, _, _, client = _exchange(upstream, stream, monkeypatch)
    client.close()
    assert result["reply"] == ANSWER
    answers = _answer_requests(upstream)
    assert len(answers) == 1 and answers[0]["max_tokens"] == 1024
    assert len(upstream.requests) > 1, "the aux steps also ran"
    assert all(isinstance(r.get("max_tokens"), int) and r["max_tokens"] > 0 for r in upstream.requests)


@pytest.mark.parametrize("stream", [False, True])
def test_the_configured_cap_is_the_one_sent(upstream_factory, monkeypatch, stream):
    upstream = upstream_factory()
    _, _, _, client = _exchange(upstream, stream, monkeypatch, max_answer_tokens=300)
    client.close()
    assert _answer_requests(upstream)[0]["max_tokens"] == 300


def test_the_cap_setting_falls_back_to_the_default_when_unusable():
    from agent.app import answer_token_cap

    class Cfg:
        def __init__(self, value):
            self.value = value

        def get(self, key, default=None):
            assert key == "llm.max_answer_tokens"
            return default if self.value is None else self.value

    assert answer_token_cap(Cfg(None)) == 1024
    assert answer_token_cap(Cfg("2048")) == 2048
    for bad in ("0", "-5", "many", ""):
        assert answer_token_cap(Cfg(bad)) == 1024


def test_the_environment_sets_the_cap(monkeypatch, tmp_path):
    from agent.config import load_config
    monkeypatch.setenv("LLM_MAX_ANSWER_TOKENS", "512")
    cfg = load_config()
    assert str(cfg.get("llm.max_answer_tokens")) == "512"


# ---- a cut-off answer ------------------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_a_length_cut_answer_is_returned_with_the_shortened_note(upstream_factory, monkeypatch, stream):
    from agent.llm import SHORTENED_NOTE
    upstream = upstream_factory(finish="length")
    result, events, store, client = _exchange(upstream, stream, monkeypatch)
    client.close()
    assert result["reply"].startswith(ANSWER) and result["reply"].endswith(SHORTENED_NOTE)
    assert result["truncated"] is True
    assert result["decision"] == "logged", "a shortened answer is still an answer"
    assert store.answers and SHORTENED_NOTE not in store.answers[0], "the note is not diary text"
    if stream:
        answer = [e for e in events if e.get("type") == "answer"]
        assert answer and answer[-1]["text"].endswith(SHORTENED_NOTE)


@pytest.mark.parametrize("stream", [False, True])
def test_a_normal_answer_is_unchanged(upstream_factory, monkeypatch, stream):
    upstream = upstream_factory()
    result, events, store, client = _exchange(upstream, stream, monkeypatch)
    client.close()
    assert result["reply"] == ANSWER and "truncated" not in result
    assert result["decision"] == "logged" and store.answers == [ANSWER] and store.saved
    assert all("shortened" not in str(e) for e in events)
    if stream:
        assert [e["text"] for e in events if e.get("type") == "answer"] == [ANSWER]


def _v1(monkeypatch, run):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.setattr(appmod, "_tenant_state", lambda r, **k: SimpleNamespace())
    monkeypatch.setattr(appmod, "_run_exchange", run)
    return TestClient(appmod.app).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Synthetic?"}]})


def test_v1_marks_a_shortened_answer_and_keeps_a_normal_one_as_before(monkeypatch):
    shortened = {"reply": "Partial. (note)", "reasoning": "", "decision": "logged", "xid": "x1", "reason": "", "truncated": True}
    body = _v1(monkeypatch, lambda *a, **k: shortened).json()
    assert body["choices"][0]["finish_reason"] == "length" and body["diary"]["truncated"] is True
    normal = {"reply": "Whole.", "reasoning": "", "decision": "logged", "xid": "x2", "reason": ""}
    body = _v1(monkeypatch, lambda *a, **k: normal).json()
    assert body["choices"][0]["finish_reason"] == "stop" and body["diary"] == {"decision": "logged", "xid": "x2"}


# ---- the deadline closes the model connection ------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["stall_mid", "stall_head"])
def test_the_deadline_closes_the_upstream_connection(upstream_factory, monkeypatch, fast, stream, mode):
    upstream = upstream_factory(mode=mode)
    with pytest.raises(fast.ExchangeTimeout) as raised:
        _exchange(upstream, stream, monkeypatch)
    gave_up = time.monotonic()
    assert raised.value.nothing_saved is True
    assert upstream.disconnected.wait(3), "the model server never saw the diary close the connection"
    assert upstream.disconnected_at - gave_up < 2.0
    assert len(_answer_requests(upstream)) == 1, "a given-up answer is never sent again"


def test_an_abandon_closer_runs_once_and_late_ones_run_at_once():
    from agent.deadline import ExchangeBudget, ExchangeTimeout
    calls, seen = [], {}
    release = threading.Event()

    def call(gate, alive):
        seen["alive"] = alive
        alive.on_abandon(lambda: calls.append("closer"))
        unregister = alive.on_abandon(lambda: calls.append("unregistered"))
        unregister()
        release.wait(5)

    with pytest.raises(ExchangeTimeout):
        ExchangeBudget(total_s=0.2).run(call)
    assert calls == ["closer"] and seen["alive"].abandoned
    seen["alive"].on_abandon(lambda: calls.append("late"))
    assert calls == ["closer", "late"]
    release.set()


# ---- the timeout error -----------------------------------------------------------------------------------


def test_v1_timeout_is_a_coded_502(monkeypatch):
    from agent.deadline import ExchangeTimeout

    def run(*a, **k):
        exc = ExchangeTimeout("the companion did not finish before its deadline")
        exc.nothing_saved = True
        raise exc

    r = _v1(monkeypatch, run)
    assert r.status_code == 502
    error = r.json()["error"]
    assert error["code"] == "chat_timeout" and error["type"] == "api_error"
    assert "nothing was saved" in error["message"]


def test_v1_other_failures_keep_the_plain_model_error(monkeypatch):
    def run(*a, **k):
        raise RuntimeError("synthetic failure")

    r = _v1(monkeypatch, run)
    assert r.status_code == 502 and r.json() == {"error": {"message": "model error", "type": "api_error"}}


def test_the_streamed_timeout_event_names_the_deadline():
    import asyncio
    from agent.deadline import ExchangeTimeout
    from agent.streaming import exchange_stream

    def work(emit):
        exc = ExchangeTimeout("synthetic deadline")
        exc.nothing_saved = True
        raise exc

    async def run():
        return "".join([c async for c in exchange_stream(work).body_iterator])

    events = [json.loads(line[6:]) for line in asyncio.run(run()).splitlines() if line.startswith("data: ")]
    error = next(e for e in events if e["type"] == "error")
    assert error["reason"] == "chat_timeout" and error["code"] == "modelFailed" and error["saved"] is False
