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

    def __init__(self, mode="answer", finish="stop", text=ANSWER, marker=None):
        # text: the visible answer ("" = the model spent its budget on reasoning only). marker: True or False
        # forces the trailing log marker on or off; None keeps the original behaviour (always sent unstreamed,
        # sent streamed only when the answer was not cut).
        self.mode, self.finish, self.text, self.marker = mode, finish, text, marker
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
                    return self._json(upstream.text + ("\n[LOG: ok]" if upstream.marker is not False else ""), upstream.finish)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if upstream.text:
                    self._event({"choices": [{"delta": {"content": upstream.text}}]})
                if upstream.mode == "stall_mid":
                    return self._wait_for_disconnect()
                if upstream.marker if upstream.marker is not None else upstream.finish == "stop":
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


def _state(upstream, **st_extra):
    """An app state wired to the fake upstream: (st, store, client)."""
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
    return st, store, client


def _exchange(upstream, stream, monkeypatch, sink=None, **st_extra):
    """One _run_exchange against the fake upstream: (result, events, store, client). `sink`, when given, is
    filled with the store, the events and the client before the exchange runs, so a failed exchange (which
    raises instead of returning) can still be inspected."""
    import agent.app as appmod
    st, store, client = _state(upstream, **st_extra)
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda s: None)
    events = []
    if sink is not None:
        sink.update(store=store, events=events, client=client)
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


# ---- #135: a length cut that leaves no answer, and a cut answer without the log marker ------------------


def _stream_events(work):
    import asyncio
    from agent.streaming import exchange_stream

    async def run():
        return "".join([c async for c in exchange_stream(work).body_iterator])

    return [json.loads(line[6:]) for line in asyncio.run(run()).splitlines() if line.startswith("data: ")]


@pytest.mark.parametrize("stream", [False, True])
def test_an_empty_answer_after_a_length_cut_raises_the_distinct_error(upstream_factory, monkeypatch, stream):
    from agent.llm import LengthCapError
    upstream = upstream_factory(finish="length", text="", marker=False)
    with pytest.raises(LengthCapError) as raised:
        _exchange(upstream, stream, monkeypatch)
    assert "LLM_MAX_ANSWER_TOKENS" in str(raised.value)


@pytest.mark.parametrize("stream", [False, True])
def test_an_empty_answer_after_a_length_cut_saves_nothing(upstream_factory, monkeypatch, stream):
    from agent.llm import LengthCapError
    import agent.app as appmod
    upstream = upstream_factory(finish="length", text="", marker=False)
    store = {}
    original = appmod._session

    def spy(session_id, tenant_id):
        store["sess"] = original(session_id, tenant_id)
        return store["sess"]
    monkeypatch.setattr(appmod, "_session", spy)
    with pytest.raises(LengthCapError):
        _exchange(upstream, stream, monkeypatch)
    assert store["sess"]["turns"] == [], "no session history for the failed turn"
    assert all(not r.get("messages", [{}])[-1].get("content", "").startswith(("SUMMARIZE", "GATE", "EDIT"))
               for r in upstream.requests), "no save step ran"


def test_v1_length_cap_is_a_coded_502(monkeypatch):
    from agent.llm import LengthCapError

    def run(*a, **k):
        raise LengthCapError()

    r = _v1(monkeypatch, run)
    assert r.status_code == 502
    error = r.json()["error"]
    assert error == {"message": error["message"], "type": "api_error", "code": "length_cap"}
    assert "LLM_MAX_ANSWER_TOKENS" in error["message"] and "before any answer text" in error["message"]
    assert "nothing was saved" in error["message"]


def test_the_streamed_length_cap_event_names_the_cap():
    from agent.llm import LengthCapError

    def work(emit):
        raise LengthCapError()

    events = _stream_events(work)
    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "modelFailed" and error["reason"] == "length_cap" and error["saved"] is False
    assert "LLM_MAX_ANSWER_TOKENS" in error["text"]
    assert not any(e["type"] == "diary" for e in events)


@pytest.mark.parametrize("stream", [False, True])
def test_a_cut_answer_without_the_marker_is_logged_and_recorded(upstream_factory, monkeypatch, caplog, stream):
    import logging
    from agent.llm import SHORTENED_NOTE
    upstream = upstream_factory(finish="length", marker=False)
    with caplog.at_level(logging.INFO, logger="diary"):
        result, _, store, client = _exchange(upstream, stream, monkeypatch)
    client.close()
    assert result["decision"] == "logged" and store.saved, "a cut answer with no marker defaults to logging"
    assert store.answers == [ANSWER] and SHORTENED_NOTE not in store.answers[0]
    assert result["reply"].endswith(SHORTENED_NOTE)
    lines = [r for r in caplog.records if r.levelno == logging.INFO and "log marker" in r.getMessage()]
    assert len(lines) == 1 and "truncat" in lines[0].getMessage()
    assert ANSWER not in lines[0].getMessage(), "the answer text is never logged"


def test_a_complete_answer_without_the_marker_logs_no_truncation_line(upstream_factory, monkeypatch, caplog):
    import logging
    upstream = upstream_factory(marker=False)
    with caplog.at_level(logging.INFO, logger="diary"):
        result, _, _, client = _exchange(upstream, False, monkeypatch)
    client.close()
    assert result["decision"] == "logged"
    assert not [r for r in caplog.records if "log marker" in r.getMessage()]


# ---- #140: the local workspace-files exchange reports a cut-to-nothing answer with its reason ------------

LOCAL_TENANT = "11111111-1111-4111-8111-111111111111"


def _local_exchange(monkeypatch, run, stream):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.setattr(appmod, "_tenant_state", lambda r: pytest.fail("local mode must not resolve remote storage"))
    monkeypatch.setattr(appmod, "_run_exchange", run)
    client = TestClient(appmod.app, raise_server_exceptions=False)
    return client.post("/api/local-exchange", headers={"X-Cowork-User-ID": LOCAL_TENANT},
                       json={"stream": stream, "files": {}, "message": "A quiet day.", "history": []})


def test_an_unstreamed_local_exchange_cut_to_nothing_is_a_coded_length_cap_503(monkeypatch):
    from agent.llm import LENGTH_CAP_TEXT, LengthCapError

    def run(*a, **k):
        raise LengthCapError()

    r = _local_exchange(monkeypatch, run, stream=False)
    assert r.status_code == 503, r.text
    assert r.json() == {"error": LENGTH_CAP_TEXT, "detail": LENGTH_CAP_TEXT, "code": "modelFailed",
                        "reason": "length_cap", "saved": False}
    assert "LLM_MAX_ANSWER_TOKENS" in r.json()["error"]


def test_a_streamed_local_exchange_cut_to_nothing_reports_length_cap(monkeypatch):
    from agent.llm import LengthCapError

    def run(*a, **k):
        raise LengthCapError()

    r = _local_exchange(monkeypatch, run, stream=True)
    assert r.status_code == 200
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "modelFailed" and error["reason"] == "length_cap" and error["saved"] is False


def test_an_ordinary_model_failure_on_the_unstreamed_local_exchange_is_unchanged(monkeypatch):
    from agent.llm import LLMError

    def run(*a, **k):
        raise LLMError("synthetic provider fault")

    r = _local_exchange(monkeypatch, run, stream=False)
    assert r.status_code == 500
    assert "length_cap" not in r.text and "modelFailed" not in r.text and "empty_answer" not in r.text


# ---- #140 / #146: an empty answer that stopped normally is its own coded error, never the length-cap one ---


@pytest.mark.parametrize("marker", [False, True], ids=["empty", "marker_only"])
@pytest.mark.parametrize("stream", [False, True])
def test_an_empty_answer_after_a_normal_stop_is_an_empty_answer_error(upstream_factory, monkeypatch, stream, marker):
    from agent.llm import EMPTY_ANSWER_TEXT, EmptyAnswerError, LengthCapError, LLMError
    upstream = upstream_factory(finish="stop", text="", marker=marker)
    sink = {}
    with pytest.raises(LLMError) as raised:
        _exchange(upstream, stream, monkeypatch, sink=sink)
    sink["client"].close()
    assert isinstance(raised.value, EmptyAnswerError) and not isinstance(raised.value, LengthCapError)
    assert str(raised.value) == EMPTY_ANSWER_TEXT and "LLM_MAX_ANSWER_TOKENS" not in str(raised.value)
    # #146: nothing is logged or saved, and no diary event is emitted.
    assert sink["store"].answers == [] and sink["store"].saved == [], "log_exchange must not run"
    assert not any(e.get("type") == "diary" for e in sink["events"])
    assert len(upstream.requests) == 1, "one answer request: no retry and no save step (summary, gate, edit)"


def test_v1_empty_answer_is_a_coded_502(monkeypatch):
    from agent.llm import EMPTY_ANSWER_TEXT, EmptyAnswerError

    def run(*a, **k):
        raise EmptyAnswerError()

    r = _v1(monkeypatch, run)
    assert r.status_code == 502
    assert r.json() == {"error": {"message": EMPTY_ANSWER_TEXT, "type": "api_error", "code": "empty_answer"}}
    assert "nothing was saved" in r.json()["error"]["message"]


def test_v1_empty_answer_from_a_real_exchange_is_a_coded_502_and_saves_nothing(upstream_factory, monkeypatch):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    upstream = upstream_factory(finish="stop", text="", marker=True)
    st, store, client = _state(upstream)
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.setattr(appmod, "_tenant_state", lambda r, **k: st)
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda s: None)
    r = TestClient(appmod.app).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Synthetic?"}]})
    client.close()
    assert r.status_code == 502 and r.json()["error"]["code"] == "empty_answer"
    assert store.saved == [] and store.answers == []
    assert len(upstream.requests) == 1


def test_the_streamed_empty_answer_event_names_the_reason():
    from agent.llm import EMPTY_ANSWER_TEXT, EmptyAnswerError

    def work(emit):
        raise EmptyAnswerError()

    events = _stream_events(work)
    error = next(e for e in events if e["type"] == "error")
    assert error == {"type": "error", "code": "modelFailed", "reason": "empty_answer", "saved": False,
                     "text": EMPTY_ANSWER_TEXT}
    assert not any(e["type"] in ("diary", "done") for e in events)


def test_a_streamed_real_exchange_with_an_empty_answer_ends_in_empty_answer_and_no_diary_event(upstream_factory, monkeypatch):
    upstream = upstream_factory(finish="stop", text="", marker=True)
    sink = {}

    def work(emit):
        return _exchange(upstream, True, monkeypatch, sink=sink)[0]

    events = _stream_events(work)
    sink["client"].close()
    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "modelFailed" and error["reason"] == "empty_answer" and error["saved"] is False
    assert not any(e["type"] == "diary" for e in events)
    assert sink["store"].saved == [] and sink["store"].answers == []


def test_an_unstreamed_local_exchange_with_an_empty_answer_is_a_coded_503(monkeypatch):
    from agent.llm import EMPTY_ANSWER_TEXT, EmptyAnswerError

    def run(*a, **k):
        raise EmptyAnswerError()

    r = _local_exchange(monkeypatch, run, stream=False)
    assert r.status_code == 503, r.text
    assert r.json() == {"error": EMPTY_ANSWER_TEXT, "detail": EMPTY_ANSWER_TEXT, "code": "modelFailed",
                        "reason": "empty_answer", "saved": False}


def test_a_streamed_local_exchange_with_an_empty_answer_reports_empty_answer(monkeypatch):
    from agent.llm import EmptyAnswerError

    def run(*a, **k):
        raise EmptyAnswerError()

    r = _local_exchange(monkeypatch, run, stream=True)
    assert r.status_code == 200
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "modelFailed" and error["reason"] == "empty_answer" and error["saved"] is False
    assert not any(e["type"] == "diary" for e in events)


def test_the_gate_records_the_empty_answer_code_on_every_path():
    from gate.harness import error_code_of
    assert error_code_of({"error": {"message": "m", "type": "api_error", "code": "empty_answer"}}) == "empty_answer"
    assert error_code_of({"error": "m", "detail": "m", "code": "modelFailed", "reason": "empty_answer", "saved": False}) == "empty_answer"
    assert error_code_of({"type": "error", "code": "modelFailed", "reason": "empty_answer", "saved": False}) == "empty_answer"


# ---- #154: the unstreamed local-exchange timeout 503 has the same shape as its siblings -------------------


def _local_timeout(*a, **k):
    from agent.deadline import ExchangeTimeout
    exc = ExchangeTimeout("synthetic deadline")
    exc.nothing_saved = True
    raise exc


def test_an_unstreamed_local_exchange_timeout_is_a_coded_503_with_reason_and_saved_false(monkeypatch):
    from agent.streaming import MODEL_TIMEOUT_TEXT

    r = _local_exchange(monkeypatch, _local_timeout, stream=False)
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["reason"] == "chat_timeout" and body["saved"] is False
    assert body == {"error": MODEL_TIMEOUT_TEXT, "detail": MODEL_TIMEOUT_TEXT, "code": "modelFailed",
                    "reason": "chat_timeout", "saved": False}


def test_the_gate_records_chat_timeout_for_the_unstreamed_local_exchange_timeout_body(monkeypatch):
    from gate.harness import error_code_of

    r = _local_exchange(monkeypatch, _local_timeout, stream=False)
    assert error_code_of(r.json()) == "chat_timeout"
