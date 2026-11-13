"""R9 (D9): DIARY_RUST_ANSWER=on against a stand-in for the Rust process's POST /internal/answer
(mode "answer"). The Rust side builds the context and answers; Python still saves. Anything the
Rust side does not take is answered by Python as before, its failures become the exceptions
Python's own model call raises, the exchange deadline bounds the call, and no content reaches a
log. Synthetic data only."""
import functools
import json
import logging
import time
from datetime import date
from types import SimpleNamespace

import httpx
import pytest

import agent.app as appmod
from agent import rust_answer, rust_forward
from agent.config import Config
from agent.deadline import ExchangeBudget, ExchangeTimeout
from agent.llm import ChatReply, EmptyAnswerError, EngineUnreachable, LengthCapError, LLMError, ModelNotServed
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_files import MemoryBackend

TENANT = "5b0c7d1e-2f3a-4b5c-9d6e-7f8091a2b3c4"
SECRET_WORD = "QA-synthetic-marigold"  # must never appear in a log line
PY_MESSAGES = [
    {"role": "system", "content": "SYSTEM"},
    {"role": "user", "content": f"Recent: {SECRET_WORD} walk"},
    {"role": "user", "content": "Synthetic question"},
]
RUST_REPLY = f"The synthetic diary mentions the {SECRET_WORD} walk."


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(rust_answer, "STATS", {k: 0 for k in rust_answer.OUTCOMES})
    monkeypatch.setattr(rust_answer, "ANSWER_STATS", {k: 0 for k in rust_answer.ANSWER_OUTCOMES})
    monkeypatch.setenv("DIARY_RUST_ANSWER", "on")
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda st: None)
    monkeypatch.setattr(rust_forward, "_server", FakeServer())


class FakeServer:
    base_url = "http://127.0.0.1:1"

    def __init__(self, ready=True, token="synthetic-control-token"):
        self._ready, self._token = ready, token

    def control(self):
        return self._token, self._ready


class FakeEmbeddingConfig:
    def get(self):
        return {"active": {"model": "qa-embed-a", "base_url": None, "profile": "plain"}, "next": None,
                "previous": None, "source": "seeded", "version": 1}


class FakeManaged(ManagedCorpusBackend):
    def __init__(self, tenant):
        self.tenant = tenant


def state(order, local_reference="", backend=None, model_reply="Python reply.\n[LOG: ok]"):
    cfg = Config({"context": {"max_today_tokens": 4500}, "retrieval": {"top_k": 8, "min_score": 0.3},
                  "prompts": {"system_path": ""}})

    def build(day, message, session_turns=None, trace=None):
        order.append("python-context")
        return [dict(m) for m in PY_MESSAGES]

    class Main:
        setting_key = "LLM_CHAT_MODEL"

        def chat(self, messages, **kwargs):
            order.append("python-model")
            return ChatReply(model_reply)

        def chat_stream(self, messages, emit, **kwargs):
            order.append("python-model")
            emit({"type": "answer", "text": "Python reply."})
            return ChatReply(model_reply)

    class Pipeline:
        def __init__(self):
            self.saved = []

        def log_exchange(self, **kwargs):
            order.append("save")
            self.saved.append(kwargs["assistant_message"])
            return SimpleNamespace(decision="logged", xid="xid-synthetic", reason="")

    assembler = SimpleNamespace(cfg=cfg, local_reference=local_reference, _system_prompt=lambda: "SYSTEM", build=build)
    return SimpleNamespace(
        assembler=assembler, backend=FakeManaged(TENANT) if backend is None else backend,
        retrieval=SimpleNamespace(embedding=SimpleNamespace(config=FakeEmbeddingConfig())),
        store=SimpleNamespace(require_write_transform=lambda: None), llm_main=Main(), pipeline=Pipeline())


def sse(*events):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          content="".join(f"data: {json.dumps(e)}\n\n" for e in events).encode())


def answered(**extra):
    return {"visible": RUST_REPLY, "marker": "ok", "truncated": False, "trace": None, **extra}


def exchange(monkeypatch, handler, st=None, emit=None, trace=False, sid="r9-on", order=None):
    order = [] if order is None else order
    st = st or state(order)
    seen = []

    def recording(request):
        order.append("rust")
        seen.append({"tenant": request.url.params.get("tenant"),
                     "control": request.headers.get(rust_forward.CONTROL_HEADER),
                     "body": json.loads(request.content)})
        return handler(request)
    monkeypatch.setattr(appmod.rust_answer, "answer",
                        functools.partial(rust_answer.answer, transport=httpx.MockTransport(recording)))
    try:
        result = appmod._run_exchange(st, "Synthetic question", sid, TENANT, entry_day=date(2026, 9, 1),
                                      background=False, emit=emit, retrieval_trace=trace)
    finally:
        appmod.SESSIONS.pop(f"{TENANT}:{sid}", None)
    return result, order, seen, st


# ---- the Rust side answers, Python saves ----

def test_a_plain_answer_comes_from_the_rust_side_and_python_saves_it(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    result, order, seen, st = exchange(monkeypatch, lambda r: httpx.Response(200, json=answered()))
    assert order == ["rust", "save"]  # no Python context, no Python model call
    assert result == {"reply": RUST_REPLY, "reasoning": "", "decision": "logged", "xid": "xid-synthetic", "reason": ""}
    assert st.pipeline.saved == [RUST_REPLY]
    (req,) = seen
    assert req["tenant"] == TENANT and req["control"] == "synthetic-control-token"
    body = req["body"]
    assert body["mode"] == "answer" and body["stream"] is False and body["trace"] is False
    assert body["message"] == "Synthetic question" and body["day"] == "2026-09-01" and body["turns"] == []
    assert body["system_prompt"] == "SYSTEM" and body["embedding"]["active"]["model"] == "qa-embed-a"
    assert body["cfg"] == {"context": {"max_today_tokens": 4500}, "retrieval": {"top_k": 8, "min_score": 0.3}}
    assert rust_answer.answer_stats()["rust"] == 1
    assert SECRET_WORD not in caplog.text


def test_the_session_turns_carry_the_rust_answer_to_the_next_exchange(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=answered())
    monkeypatch.setattr(appmod.rust_answer, "answer",
                        functools.partial(rust_answer.answer, transport=httpx.MockTransport(handler)))
    st = state([])
    try:
        for _ in range(2):
            appmod._run_exchange(st, "Synthetic question", "r9-on-turns", TENANT, entry_day=date(2026, 9, 1),
                                 background=False)
    finally:
        appmod.SESSIONS.pop(f"{TENANT}:r9-on-turns", None)
    assert bodies[0]["turns"] == []
    assert bodies[1]["turns"] == [{"role": "user", "content": "Synthetic question"},
                                  {"role": "companion", "content": RUST_REPLY}]


def test_a_streamed_answer_relays_the_rust_events_in_order(monkeypatch):
    events = []
    handler = lambda r: sse({"type": "delta", "text": "The synthetic "},
                            {"type": "mtp", "model": "qa-chat", "timings": {"draft_n": 2, "draft_n_accepted": 1}},
                            {"type": "answer", "text": RUST_REPLY},
                            {"type": "answered", **answered()})
    result, order, seen, _ = exchange(monkeypatch, handler, emit=events.append)
    assert seen[0]["body"]["stream"] is True
    assert [e["type"] for e in events] == ["status", "status", "delta", "mtp", "answer"]
    assert [e["text"] for e in events[:2]] == ["Reading diary entries and memory…", "Generating companion response…"]
    assert result["reply"] == RUST_REPLY and order == ["rust", "save"]


def test_a_shortened_answer_keeps_its_note_and_defaults_to_logging(monkeypatch):
    result, order, _, st = exchange(monkeypatch, lambda r: httpx.Response(
        200, json=answered(visible="A synthetic answer cut", marker=None, truncated=True)))
    assert result["truncated"] is True and result["reply"].startswith("A synthetic answer cut")
    assert result["reply"] != "A synthetic answer cut"  # the shortened note
    assert st.pipeline.saved == ["A synthetic answer cut"] and order == ["rust", "save"]


def test_a_traced_exchange_returns_the_rust_trace(monkeypatch):
    rust_trace = {"status": "ok", "candidates": []}
    result, _, seen, _ = exchange(monkeypatch, lambda r: httpx.Response(200, json=answered(trace=rust_trace)),
                                  trace=True)
    assert seen[0]["body"]["trace"] is True
    assert result["retrieval_trace"] == rust_trace
    result, _, _, _ = exchange(monkeypatch, lambda r: httpx.Response(200, json=answered(trace=None)), trace=True)
    assert "retrieval_trace" not in result


# ---- failures: the exceptions Python's own model call raises, nothing saved ----

@pytest.mark.parametrize("failure,expected", [
    ({"kind": "model", "class": "LengthCapError", "message": "x"}, LengthCapError),
    ({"kind": "model", "class": "EmptyAnswerError", "message": "x"}, EmptyAnswerError),
    ({"kind": "model", "class": "EngineUnreachable", "message": "x"}, EngineUnreachable),
    ({"kind": "model", "class": "ModelNotServed", "message": "x"}, ModelNotServed),
    ({"kind": "model", "class": "HTTPStatusError", "message": "x"}, LLMError),
    ({"kind": "timeout", "nothing_saved": True, "message": "x"}, ExchangeTimeout),
    ({"kind": "other"}, LLMError),
])
@pytest.mark.parametrize("streamed", [False, True])
def test_a_rust_failure_raises_what_python_raises(monkeypatch, failure, expected, streamed):
    if streamed:
        handler = lambda r: sse({"type": "failed", "failure": failure, "event": {"type": "error"}})
    else:
        handler = lambda r: httpx.Response(200, json={"failed": failure})
    order = []
    with pytest.raises(expected) as info:
        exchange(monkeypatch, handler, emit=[].append if streamed else None, order=order)
    assert order == ["rust"]  # nothing saved, no Python retry of the model
    if expected is ExchangeTimeout:
        assert info.value.nothing_saved is True
    if expected is ModelNotServed:
        assert info.value.setting == "LLM_CHAT_MODEL"
    assert rust_answer.answer_stats()["rust"] == 1


@pytest.mark.parametrize("reply", [
    lambda r: sse({"type": "delta", "text": "cut off"}),  # the stream ends without an outcome
    lambda r: httpx.Response(200, content=b"not json"),
    lambda r: httpx.Response(200, json={"visible": 3, "marker": "ok"}),
    lambda r: httpx.Response(200, json={"visible": "x", "marker": "maybe"}),
])
def test_an_unreadable_rust_reply_is_a_model_failure_not_a_second_answer(monkeypatch, reply):
    order = []
    streamed = reply(None).headers.get("content-type") == "text/event-stream"
    with pytest.raises(LLMError):
        exchange(monkeypatch, reply, emit=[].append if streamed else None, order=order)
    assert order == ["rust"]


def test_an_empty_rust_answer_is_refused_before_the_save(monkeypatch):
    order = []
    with pytest.raises(EmptyAnswerError):
        exchange(monkeypatch, lambda r: httpx.Response(200, json=answered(visible="")), order=order)
    with pytest.raises(LengthCapError):
        exchange(monkeypatch, lambda r: httpx.Response(200, json=answered(visible="", truncated=True)), order=order)
    assert "save" not in order


def test_the_exchange_deadline_bounds_the_rust_call(monkeypatch):
    monkeypatch.setattr(appmod, "ExchangeBudget", functools.partial(ExchangeBudget, total_s=0.3))

    def slow(request):
        time.sleep(1.0)
        return httpx.Response(200, json=answered())
    order = []
    started = time.monotonic()
    with pytest.raises(ExchangeTimeout) as info:
        exchange(monkeypatch, slow, order=order)
    assert time.monotonic() - started < 0.9
    assert info.value.nothing_saved is True and "save" not in order


def test_a_given_up_call_shuts_its_socket_before_any_reply(monkeypatch):
    # A Rust process that has taken the request but not replied (the plain answer replies only when
    # the model has finished): the deadline must close the connection at once, so the Rust side drops
    # its model call, as Python's own model call does (#127).
    import socket
    import threading
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    closed = threading.Event()

    def serve():
        conn, _ = listener.accept()
        conn.settimeout(10)
        try:
            while conn.recv(65536):
                pass  # the request, then nothing: wait for the caller to go away
            closed.set()
        except OSError:
            closed.set()
        finally:
            conn.close()
    threading.Thread(target=serve, daemon=True).start()
    server = FakeServer()
    server.base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    st = state([])
    snap = rust_answer.snapshot(st, [])
    started = time.monotonic()
    try:
        with pytest.raises(ExchangeTimeout):
            rust_answer.answer(st, TENANT, "Synthetic question", snap, date(2026, 9, 1), "", 1024,
                               ExchangeBudget(total_s=0.3), server_lookup=lambda: server)
        assert closed.wait(3), "the connection was left open"
        assert time.monotonic() - started < 2.5
    finally:
        listener.close()
    assert rust_answer.answer_stats()["rust"] == 1


@pytest.mark.parametrize("streamed", [False, True])
def test_a_rust_server_error_is_a_failure_never_a_second_answer(monkeypatch, streamed):
    order = []
    with pytest.raises(LLMError):
        exchange(monkeypatch, lambda r: httpx.Response(500), emit=[].append if streamed else None, order=order)
    assert order == ["rust"]


# ---- Python answers what the Rust side does not take ----

@pytest.mark.parametrize("reply,outcome", [
    (lambda r: httpx.Response(421, headers={rust_forward.DECLINE_HEADER.decode(): "1"}), "declined"),
    (lambda r: httpx.Response(404), "unavailable"),
    (lambda r: (_ for _ in ()).throw(httpx.ConnectError("synthetic")), "unavailable"),
])
def test_python_answers_when_the_rust_side_does_not(monkeypatch, reply, outcome, caplog):
    caplog.set_level(logging.DEBUG)
    events = []
    result, order, _, st = exchange(monkeypatch, reply, emit=events.append)
    assert order == ["rust", "python-context", "python-model", "save"]
    assert result["reply"] == "Python reply." and st.pipeline.saved == ["Python reply."]
    # One status per step, even though the Rust side was asked first.
    assert [e["text"] for e in events if e["type"] == "status"] == [
        "Reading diary entries and memory…", "Generating companion response…"]
    assert rust_answer.answer_stats()[outcome] == 1 and rust_answer.answer_stats()["rust"] == 0
    assert SECRET_WORD not in caplog.text


@pytest.mark.parametrize("setup,outcome", [
    (lambda st, mp: setattr(st.assembler, "local_reference", "a synthetic reference"), "skipped"),
    (lambda st, mp: setattr(st, "backend", MemoryBackend({})), "skipped"),
    (lambda st, mp: setattr(st.retrieval.embedding, "config", None), "skipped"),
    (lambda st, mp: mp.setattr(rust_forward, "_server", FakeServer(ready=False)), "unavailable"),
    (lambda st, mp: mp.setattr(rust_forward, "_server", None), "unavailable"),
])
def test_cases_the_rust_side_does_not_take_never_call_it(monkeypatch, setup, outcome):
    order = []
    st = state(order)
    setup(st, monkeypatch)
    result, order, seen, _ = exchange(monkeypatch, lambda r: httpx.Response(200, json=answered()), st=st, order=order)
    assert seen == [] and order == ["python-context", "python-model", "save"]
    assert result["reply"] == "Python reply."
    assert rust_answer.answer_stats()[outcome] == 1


def test_the_legacy_tenant_and_the_other_modes_never_call_it(monkeypatch):
    for mode in ("", "shadow", "bogus"):
        monkeypatch.setenv("DIARY_RUST_ANSWER", mode)
        assert not rust_answer.answering()
    monkeypatch.setenv("DIARY_RUST_ANSWER", "on")
    called = []
    monkeypatch.setattr(appmod.rust_answer, "answer", functools.partial(
        rust_answer.answer, transport=httpx.MockTransport(lambda r: called.append(1) or httpx.Response(500))))
    order = []
    try:
        appmod._run_exchange(state(order), "Synthetic question", "r9-on-legacy", "legacy",
                             entry_day=date(2026, 9, 1), background=False)
    finally:
        appmod.SESSIONS.pop("legacy:r9-on-legacy", None)
    assert called == [] and order == ["python-context", "python-model", "save"]


def test_on_sends_the_answer_config_and_runs_no_shadow(monkeypatch):
    cfg = Config({"llm": {"embed_model": "qa-embed-a", "timeout_s": 30, "max_retries": 2}})
    assert rust_answer.answer_config(cfg, 64)["max_retries"] == 2
    st = state([])
    snap = rust_answer.snapshot(st, [])
    assert snap is not None
    assert rust_answer.begin(st, TENANT, "q", snap, date(2026, 9, 1), "", PY_MESSAGES, 1024,
                             server_lookup=lambda: FakeServer()) is None
    assert sum(rust_answer.stats().values()) == 0
