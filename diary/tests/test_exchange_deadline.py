"""#1505: one exchange cannot hang. The companion's answer and the steps before its save are bounded
by a deadline; a stalled answer ends the exchange as a failure before any save began, and a stalled
aux step falls back the way its existing failure does. Synthetic fixtures only."""
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest


def _dl():
    from agent import deadline
    return deadline


@pytest.fixture
def fast(monkeypatch):
    dl = _dl()
    monkeypatch.setattr(dl, "EXCHANGE_DEADLINE_S", 0.6)
    monkeypatch.setattr(dl, "FIRST_TOKEN_DEADLINE_S", 0.2)
    monkeypatch.setattr(dl, "AUX_STEP_DEADLINE_S", 0.3)
    return dl


def test_the_named_deadlines_are_bounded_and_ordered():
    dl = _dl()
    assert 0 < dl.FIRST_TOKEN_DEADLINE_S < dl.EXCHANGE_DEADLINE_S <= 600
    assert 0 < dl.AUX_STEP_DEADLINE_S <= dl.EXCHANGE_DEADLINE_S


def test_a_call_that_never_returns_is_abandoned_at_the_total_deadline():
    dl = _dl()
    release = threading.Event()
    budget = dl.ExchangeBudget(total_s=0.3)
    start = time.monotonic()
    try:
        with pytest.raises(dl.ExchangeTimeout, match="did not finish"):
            budget.run(lambda gate, alive: release.wait(5) and "late")
        assert time.monotonic() - start < 2
    finally:
        release.set()


def test_no_first_token_is_reported_before_the_total_deadline():
    dl = _dl()
    release = threading.Event()
    budget = dl.ExchangeBudget(total_s=5, first_token_s=0.2)
    start = time.monotonic()
    try:
        with pytest.raises(dl.ExchangeTimeout, match="first token"):
            budget.run(lambda gate, alive: release.wait(5) and "late", first_token=True)
        assert time.monotonic() - start < 2
    finally:
        release.set()


def test_provider_progress_keeps_a_slow_but_live_call_past_the_first_token_deadline():
    dl = _dl()
    budget = dl.ExchangeBudget(total_s=5, first_token_s=0.2)

    def call(gate, alive):
        for _ in range(10):
            time.sleep(0.05)
            alive()
        return "done"

    assert budget.run(call, first_token=True) == "done"


def test_call_errors_propagate_unchanged():
    dl = _dl()
    budget = dl.ExchangeBudget(total_s=5)

    def call(gate, alive):
        raise ValueError("synthetic failure")

    with pytest.raises(ValueError, match="synthetic failure"):
        budget.run(call)


def test_an_abandoned_call_cannot_emit_after_its_deadline():
    dl = _dl()
    seen = []
    release = threading.Event()

    def call(gate, alive):
        gate({"type": "delta", "text": "early"})
        release.wait(5)
        gate({"type": "delta", "text": "LATE"})
        return "x"

    budget = dl.ExchangeBudget(total_s=0.2)
    try:
        with pytest.raises(dl.ExchangeTimeout):
            budget.run(call, emit=seen.append)
    finally:
        release.set()
    time.sleep(0.1)
    assert seen == [{"type": "delta", "text": "early"}]


def test_an_exhausted_budget_refuses_to_start_another_call():
    dl = _dl()
    budget = dl.ExchangeBudget(total_s=0)
    with pytest.raises(dl.ExchangeTimeout):
        budget.run(lambda gate, alive: pytest.fail("the call must not start"))


class _Store:
    def __init__(self, index_enabled=False):
        self.index_enabled = index_enabled
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


def test_a_stalled_classifier_and_summarizer_fail_open_and_the_exchange_still_saves(fast):
    from agent.deadline import ExchangeBudget
    from agent.pipeline import LoggingPipeline
    release = threading.Event()

    class Aux:
        def chat(self, messages, **kwargs):
            prompt = messages[-1]["content"]
            if prompt.startswith("CLASSIFY") or prompt.startswith("SUMMARIZE"):
                release.wait(5)
            return "unused"

    store = _Store()
    pipe = LoggingPipeline(store, None, Aux(), TEMPLATES)
    start = time.monotonic()
    try:
        outcome = pipe.log_exchange("Synthetic question", "Synthetic reply.", now=None,
                                    budget=ExchangeBudget(total_s=fast.EXCHANGE_DEADLINE_S))
    finally:
        release.set()
    assert time.monotonic() - start < 3
    assert outcome.decision == "logged"  # classifier fails open to LOG
    assert store.saved and store.saved[0]["claude_text"] == "Synthetic reply."  # summarizer falls back to verbatim


def test_a_stalled_index_step_after_the_save_cannot_hold_the_exchange_open(fast):
    from agent.pipeline import LoggingPipeline
    release = threading.Event()

    class Aux:
        def chat(self, messages, **kwargs):
            prompt = messages[-1]["content"]
            if prompt.startswith("GATE"):
                release.wait(5)
            return "Synthetic summary."

    store = _Store(index_enabled=True)
    pipe = LoggingPipeline(store, None, Aux(), TEMPLATES)
    start = time.monotonic()
    try:
        outcome = pipe.log_exchange("Synthetic question", "Synthetic reply.", now=None)
    finally:
        release.set()
    assert time.monotonic() - start < 3
    assert outcome.decision == "logged" and len(store.saved) == 1


def _stalling_provider(chunks_before_stall, release):
    """A synthetic streaming provider that sends some tokens and then never finishes until released."""
    def body():
        for i in range(chunks_before_stall):
            yield ('data: ' + json.dumps({"choices": [{"delta": {"content": "Partial reply %d " % i}}]}) + "\n\n").encode()
        release.wait(5)

    def handler(request):
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body())

    return handler


@pytest.mark.parametrize("chunks_before_stall,expected", [(0, "first token"), (1, "did not finish")])
def test_a_stalled_companion_answer_ends_before_any_save_and_says_so_to_the_stream(fast, chunks_before_stall, expected):
    import agent.app as appmod
    from agent.llm import LLMClient
    from agent.pipeline import LoggingPipeline
    release = threading.Event()
    client = LLMClient("http://synthetic.invalid/v1", chat_model="synthetic-model")
    client._client.close()
    client._client = httpx.Client(base_url="http://synthetic.invalid/v1",
                                  transport=httpx.MockTransport(_stalling_provider(chunks_before_stall, release)))
    store = _Store()
    st = SimpleNamespace(
        assembler=SimpleNamespace(build=lambda day, message, session_turns=None: [{"role": "user", "content": message}]),
        llm_main=client,
        pipeline=LoggingPipeline(store, client, client, TEMPLATES),
        store=store,
    )
    events = []
    import unittest.mock as mock
    start = time.monotonic()
    try:
        with mock.patch.object(appmod, "_reindex_dirty", lambda s: None):
            with pytest.raises(fast.ExchangeTimeout, match=expected):
                appmod._run_exchange(st, "Synthetic question", "deadline-stall", "legacy", background=False, emit=events.append)
    finally:
        release.set()
        appmod.SESSIONS.pop("legacy:deadline-stall", None)
        client.close()
    assert time.monotonic() - start < 3
    assert store.saved == [], "a stalled answer must never reach the save"
    statuses = [e["text"] for e in events if e.get("type") == "status"]
    assert statuses[-1] == "Generating companion response…", "the failure must come before the saving-began status"
    assert not any("Checking whether to save" in s for s in statuses)


# ---- review follow-up (#1505) ---------------------------------------------------------------------------


def test_an_abandoned_call_stops_at_its_next_sign_of_life():
    dl = _dl()
    release, ended = threading.Event(), threading.Event()
    progress = []

    def call(gate, alive):
        try:
            alive()
            release.wait(5)
            for i in range(50):  # the provider keeps sending lines after we gave up
                alive()
                progress.append(i)
            return "late"
        finally:
            ended.set()

    budget = dl.ExchangeBudget(total_s=0.3, first_token_s=0.2)
    with pytest.raises(dl.ExchangeTimeout):
        budget.run(call)
    release.set()
    assert ended.wait(2), "the abandoned call must end"
    assert progress == [], "alive() raised before the call read another line"


def _stalled_local_exchange(monkeypatch, fast, stream):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    release = threading.Event()
    strip_marker = appmod.LLMClient.strip_log_marker

    class LLM:
        strip_log_marker = staticmethod(strip_marker)

        def __init__(self, **kwargs):
            pass

        def chat(self, *args, **kwargs):
            release.wait(5)
            return "late"

        def chat_stream(self, messages, emit, **kwargs):
            release.wait(5)
            return "late"

        def embed(self, *args, **kwargs):
            return []

        def close(self):
            pass

    monkeypatch.setattr(appmod, "LLMClient", LLM)
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.setattr(appmod, "_tenant_state", lambda r: pytest.fail("local mode must not resolve remote storage"))
    client = TestClient(appmod.app)
    try:
        return client.post("/api/local-exchange", headers={"X-Cowork-User-ID": "11111111-1111-4111-8111-111111111111"},
                           json={"stream": stream, "files": {}, "message": "A quiet day.", "history": []})
    finally:
        release.set()


def test_a_stalled_non_streamed_local_exchange_is_a_coded_not_saved_failure_not_a_500(monkeypatch, fast):
    r = _stalled_local_exchange(monkeypatch, fast, stream=False)
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["code"] == "modelFailed" and body["saved"] is False
    assert "nothing was saved" in body["error"]
    assert "Saving may be incomplete" not in r.text


def test_a_stalled_streamed_local_exchange_says_nothing_was_saved(monkeypatch, fast):
    r = _stalled_local_exchange(monkeypatch, fast, stream=True)
    assert r.status_code == 200
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "modelFailed" and error["saved"] is False
    assert "nothing was saved" in error["text"] and "incomplete" not in error["text"].lower()


def test_a_failure_after_a_save_began_keeps_the_may_be_incomplete_wording():
    import asyncio
    from agent.deadline import ExchangeTimeout
    from agent.streaming import exchange_stream

    def wire(exc):
        def work(emit):
            raise exc

        async def run():
            return "".join([c async for c in exchange_stream(work).body_iterator])
        return asyncio.run(run())

    for exc in (RuntimeError("synthetic failure while saving"), ExchangeTimeout("a timeout nobody marked as pre-save")):
        out = wire(exc)
        assert "Saving may be incomplete" in out
        assert '"saved"' not in out
