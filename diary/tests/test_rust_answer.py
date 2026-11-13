"""R9 (D9): the answer shadow (DIARY_RUST_ANSWER=shadow) against a stand-in for the Rust process's
POST /internal/answer. Off by default and then inert; when on, it compares the context only, never
logs content, never changes the exchange, and never reads the corpus after the save. Synthetic data
only."""
import functools
import json
import logging
import threading
import time
from datetime import date
from types import SimpleNamespace

import httpx
import pytest

import agent.app as appmod
from agent import rust_answer, rust_forward
from agent.config import Config
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_files import MemoryBackend

TENANT = "3e8190a1-6d5f-4fb1-8c43-4d5e6f708192"
SECRET_WORD = "QA-synthetic-heliotrope"  # must never appear in a log line
PY_MESSAGES = [
    {"role": "system", "content": "SYSTEM"},
    {"role": "user", "content": f"Recent: {SECRET_WORD} walk"},
    {"role": "user", "content": "Synthetic question"},
]


@pytest.fixture(autouse=True)
def fresh_stats(monkeypatch):
    monkeypatch.setattr(rust_answer, "STATS", {k: 0 for k in rust_answer.OUTCOMES})
    monkeypatch.delenv("DIARY_RUST_ANSWER", raising=False)


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


def server_state():
    client = SimpleNamespace(base_url="http://engine.invalid", chat_model="qa-chat", fallback_model="",
                             setting_key="LLM_CHAT_MODEL", fallback_setting_key="", reasoning_tokens=64)
    return SimpleNamespace(auth_token="", llm_main=client, llm_aux=client)


class FakeManaged(ManagedCorpusBackend):
    """Stands for the tenant's managed storage (no database behind it)."""

    def __init__(self, tenant):
        self.tenant = tenant


def fake_state(local_reference="", backend=None):
    cfg = Config({"context": {"max_today_tokens": 4500}, "retrieval": {"top_k": 8, "min_score": 0.3},
                  "prompts": {"system_path": ""}})
    assembler = SimpleNamespace(cfg=cfg, local_reference=local_reference, _system_prompt=lambda: "SYSTEM",
                                build=lambda day, message, session_turns=None, trace=None: [dict(m) for m in PY_MESSAGES])
    return SimpleNamespace(assembler=assembler, backend=FakeManaged(TENANT) if backend is None else backend,
                           retrieval=SimpleNamespace(embedding=SimpleNamespace(config=FakeEmbeddingConfig())))


def replying(messages=None, status=200, headers=None, delay=0.0, seen=None):
    def handler(request):
        if seen is not None:
            seen.append({"path": request.url.path, "tenant": request.url.params.get("tenant"),
                         "control": request.headers.get(rust_forward.CONTROL_HEADER),
                         "body": json.loads(request.content), "at": time.monotonic()})
        if delay:
            time.sleep(delay)
        if status != 200:
            return httpx.Response(status, headers=headers or {}, json={"detail": "synthetic"})
        return httpx.Response(200, json={"messages": PY_MESSAGES if messages is None else messages, "trace": None})
    return httpx.MockTransport(handler)


def run(transport, st=None, tenant=TENANT, server=None, turns=()):
    st = st or fake_state()
    snap = rust_answer.snapshot(st, list(turns))
    shadow = rust_answer.begin(st, tenant, "Synthetic question", snap, date(2026, 9, 1),
                               "", PY_MESSAGES, 1024, server_lookup=lambda: server or FakeServer(), transport=transport)
    return shadow, (shadow.finish() if shadow is not None else None)


# ---- off by default ----

def test_off_by_default_nothing_runs_and_no_config_is_sent(monkeypatch, caplog):
    assert rust_answer.mode() == "" and not rust_answer.wanted()
    called = []
    shadow, _ = run(httpx.MockTransport(lambda r: called.append(r) or httpx.Response(200)))
    assert shadow is None and called == []
    assert rust_answer.answer_config(Config({"llm": {"embed_model": "qa"}}), 0) is None
    cfg = rust_forward.server_config(Config({}), server_state(), 8011, "t", "f")
    assert "answer" not in cfg
    assert rust_forward.start(lambda *a: {}, {}, frozenset(), answering=False) is None


def test_unknown_mode_is_off_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(rust_answer, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="diary.rust.answer"):
        assert rust_answer.mode("full") == "" and rust_answer.mode("full") == ""
    assert sum("DIARY_RUST_ANSWER" in r.getMessage() for r in caplog.records) == 1
    assert rust_answer.mode(" ON ") == "on"
    assert rust_answer.snapshot(fake_state(), []) is None  # off: Python builds from the live turns
    assert rust_answer.mode(" Shadow ") == "shadow"


def test_shadow_mode_sends_the_answer_config(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    base = Config({"llm": {"embed_model": "qa-embed-a", "embed_base_url": "http://embed.invalid", "chat_model": "qa-chat",
                           "timeout_s": 42, "max_retries": 2}})
    assert rust_answer.answer_config(base, 128) == {
        "embedding_seed": {"llm": {"embed_model": "qa-embed-a", "embed_base_url": "http://embed.invalid"}},
        "timeout_s": 42.0, "max_retries": 2, "reasoning_tokens": 128}
    cfg = rust_forward.server_config(Config({}), server_state(), 8011, "t", "f")
    assert set(cfg["answer"]) == {"embedding_seed", "timeout_s", "max_retries", "reasoning_tokens"}


# ---- the comparison ----

def test_compare_names_the_first_difference_without_content():
    assert rust_answer.compare(PY_MESSAGES, [dict(m) for m in PY_MESSAGES]) is None
    changed = [dict(m) for m in PY_MESSAGES]
    changed[1]["content"] = changed[1]["content"].replace("walk", "wa1k")
    d = rust_answer.compare(PY_MESSAGES, changed)
    assert d == f"message 1 (user): content differs at character {len('Recent: ' + SECRET_WORD) + 3} (lengths 36 vs 36)"
    assert SECRET_WORD not in d
    assert rust_answer.compare(PY_MESSAGES, PY_MESSAGES[:2]) == "message counts differ (3 vs 2)"
    assert rust_answer.compare(PY_MESSAGES, [{"role": "user", "content": "SYSTEM"}] + PY_MESSAGES[1:]).startswith("message 0: role")
    assert rust_answer.compare(PY_MESSAGES, {"messages": []}) == "the Rust reply has no message list"
    # A prefix: the shorter content differs at its end.
    cut = [dict(m) for m in PY_MESSAGES]
    cut[2]["content"] = "Synthetic"
    assert rust_answer.compare(PY_MESSAGES, cut) == "message 2 (user): content differs at character 9 (lengths 18 vs 9)"


@pytest.mark.parametrize("reply,outcome", [
    (dict(), "match"),
    (dict(messages=PY_MESSAGES[:2]), "mismatch"),
    (dict(status=421, headers={"x-noevia-diary-decline": "1"}), "declined"),
    (dict(status=421), "refused"),
    (dict(status=404), "refused"),
    (dict(status=500), "refused"),
])
def test_each_outcome_is_counted(monkeypatch, caplog, reply, outcome):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    with caplog.at_level(logging.INFO, logger="diary.rust.answer"):
        _, got = run(replying(**reply))
    assert got == outcome
    assert rust_answer.stats()[outcome] == 1 and sum(rust_answer.stats().values()) == 1
    assert all(SECRET_WORD not in r.getMessage() for r in caplog.records)


def test_a_transport_failure_is_counted_and_never_raises(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")

    def refuse(request):
        raise httpx.ConnectError("synthetic refusal")
    _, got = run(httpx.MockTransport(refuse))
    assert got == "failed" and rust_answer.stats()["failed"] == 1


def test_the_request_carries_python_inputs(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    seen = []
    turns = [{"role": "user", "content": "Earlier question"}, {"role": "companion", "content": "Earlier answer"}]
    run(replying(seen=seen), turns=turns)
    (req,) = seen
    assert req["path"] == "/internal/answer" and req["tenant"] == TENANT and req["control"] == "synthetic-control-token"
    body = req["body"]
    assert body["mode"] == "context" and body["stream"] is False and body["trace"] is False
    assert body["message"] == "Synthetic question" and body["day"] == "2026-09-01" and body["turns"] == turns
    assert body["system_prompt"] == "SYSTEM" and body["max_answer_tokens"] == 1024 and body["extra_context"] == ""
    assert body["cfg"] == {"context": {"max_today_tokens": 4500}, "retrieval": {"top_k": 8, "min_score": 0.3}}
    assert body["embedding"]["active"]["model"] == "qa-embed-a"


def test_both_sides_get_one_snapshot_of_the_turns(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    seen, gate = [], threading.Event()
    turns = [{"role": "user", "content": "Earlier question"}]
    st = fake_state()
    snap = rust_answer.snapshot(st, turns)
    turns.append({"role": "companion", "content": "added after the snapshot"})
    turns[0]["content"] = "changed after the snapshot"

    def handler(request):
        gate.wait(2)
        seen.append(json.loads(request.content)["turns"])
        return httpx.Response(200, json={"messages": PY_MESSAGES})
    shadow = rust_answer.begin(st, TENANT, "Synthetic question", snap, date(2026, 9, 1), "", PY_MESSAGES,
                               1024, server_lookup=FakeServer, transport=httpx.MockTransport(handler))
    gate.set()
    assert snap.turns == [{"role": "user", "content": "Earlier question"}]
    assert shadow.finish() == "match" and seen == [snap.turns]


# ---- what does not apply ----

def test_legacy_tenant_no_server_or_not_ready_does_nothing(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    called = []
    t = httpx.MockTransport(lambda r: called.append(r) or httpx.Response(200))
    assert run(t, tenant="legacy")[0] is None
    st = fake_state()
    assert rust_answer.begin(st, TENANT, "q", rust_answer.snapshot(st, []), date(2026, 9, 1), "", PY_MESSAGES, 1024,
                             server_lookup=lambda: None, transport=t) is None
    assert run(t, server=FakeServer(ready=False))[0] is None
    assert run(t, server=FakeServer(token=""))[0] is None
    assert called == [] and sum(rust_answer.stats().values()) == 0


def test_local_reference_and_unserialisable_config_are_skipped(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    called = []
    t = httpx.MockTransport(lambda r: called.append(r) or httpx.Response(200))
    assert run(t, st=fake_state(local_reference="file reference"))[0] is None
    st = fake_state()
    st.assembler.cfg = Config({"context": {"max_today_tokens": date(2026, 1, 1)}, "retrieval": {}})
    assert run(t, st=st)[0] is None
    st = fake_state()
    st.retrieval = SimpleNamespace(embedding=SimpleNamespace())  # no shared configuration
    assert run(t, st=st)[0] is None
    assert called == [] and rust_answer.stats()["skipped"] == 3


@pytest.mark.parametrize("backend", [MemoryBackend({}), FakeManaged("6b14c3d4-9a82-4ce4-bf76-708192a3b4c5"),
                                     SimpleNamespace(tenant=TENANT)])
def test_only_the_tenants_managed_storage_is_compared(monkeypatch, backend):
    # POST /api/local-exchange builds from an in-memory corpus: the Rust side would read the online one.
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    called = []
    assert run(httpx.MockTransport(lambda r: called.append(r) or httpx.Response(200)), st=fake_state(backend=backend))[0] is None
    assert called == [] and rust_answer.stats()["skipped"] == 1


def test_an_unexpected_failure_in_the_comparison_is_counted(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")

    def broken(request):
        raise RuntimeError("synthetic")
    _, got = run(httpx.MockTransport(broken))
    assert got == "failed" and rust_answer.stats()["failed"] == 1 and rust_answer.stats()["late"] == 0


def test_a_broken_state_never_fails_the_exchange(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    st = fake_state()
    st.assembler._system_prompt = lambda: (_ for _ in ()).throw(RuntimeError("synthetic"))
    assert run(httpx.MockTransport(lambda r: httpx.Response(200)), st=st)[0] is None
    assert rust_answer.stats()["failed"] == 1


# ---- never after the save ----

def test_a_shadow_still_running_at_the_save_is_late_and_ignored(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    monkeypatch.setattr(rust_answer, "SHADOW_DEADLINE_S", 0.1)
    done = threading.Event()

    def slow(request):
        time.sleep(0.4)
        done.set()
        return httpx.Response(200, json={"messages": PY_MESSAGES[:1]})
    shadow, got = run(httpx.MockTransport(slow))
    assert got == "late"
    assert done.wait(3)
    time.sleep(0.05)
    assert shadow.outcome is None
    assert rust_answer.stats() == {**{k: 0 for k in rust_answer.OUTCOMES}, "late": 1}


def _exchange_state(order, sent=None, fail=False):
    from agent.llm import ChatReply

    class Main:
        def chat(self, messages, **kwargs):
            order.append("model")
            if sent is not None:
                sent.append([dict(m) for m in messages])
            if fail:
                raise RuntimeError("synthetic model failure")
            return ChatReply("Synthetic reply.\n[LOG: ok]")

    class Pipeline:
        def log_exchange(self, **kwargs):
            order.append("save")
            return SimpleNamespace(decision="logged", xid="xid-synthetic", reason="")

    st = fake_state()
    st.store = SimpleNamespace(require_write_transform=lambda: None)
    st.llm_main = Main()
    st.pipeline = Pipeline()
    return st


@pytest.mark.parametrize("on", [False, True])
def test_the_exchange_is_unchanged_and_the_shadow_ends_before_the_save(monkeypatch, on):
    if on:
        monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    order, sent = [], []

    def handler(request):
        time.sleep(0.05)  # finishes after the model, before the save
        order.append("shadow")
        return httpx.Response(200, json={"messages": PY_MESSAGES})
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda st: None)
    monkeypatch.setattr(rust_forward, "_server", FakeServer())
    monkeypatch.setattr(appmod.rust_answer, "begin",
                        functools.partial(rust_answer.begin, transport=httpx.MockTransport(handler)))
    sid = f"r9-shadow-{on}"
    try:
        result = appmod._run_exchange(_exchange_state(order, sent), "Synthetic question", sid, TENANT,
                                      entry_day=date(2026, 9, 1), background=False)
    finally:
        appmod.SESSIONS.pop(f"{TENANT}:{sid}", None)
    assert sent == [PY_MESSAGES]  # the model gets Python's own context either way
    assert result == {"reply": "Synthetic reply.", "reasoning": "", "decision": "logged", "xid": "xid-synthetic", "reason": ""}
    if on:
        assert order == ["model", "shadow", "save"] and rust_answer.stats()["match"] == 1
    else:
        assert order == ["model", "save"] and sum(rust_answer.stats().values()) == 0


def test_an_exchange_that_fails_after_the_start_closes_the_shadow(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    release, done = threading.Event(), threading.Event()

    def handler(request):
        release.wait(3)
        done.set()
        return httpx.Response(200, json={"messages": PY_MESSAGES})
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda st: None)
    monkeypatch.setattr(rust_forward, "_server", FakeServer())
    started = []
    real_begin = rust_answer.begin

    def begin(*a, **k):
        shadow = real_begin(*a, transport=httpx.MockTransport(handler), **k)
        started.append(shadow)
        return shadow
    monkeypatch.setattr(appmod.rust_answer, "begin", begin)
    order = []
    try:
        with pytest.raises(RuntimeError):
            appmod._run_exchange(_exchange_state(order, fail=True), "Synthetic question", "r9-shadow-fail", TENANT,
                                 entry_day=date(2026, 9, 1), background=False)
    finally:
        appmod.SESSIONS.pop(f"{TENANT}:r9-shadow-fail", None)
    release.set()
    assert done.wait(3)
    time.sleep(0.05)
    (shadow,) = started
    assert shadow.outcome is None and order == ["model"]
    assert sum(rust_answer.stats().values()) == 0  # closed: its later result is neither counted nor late
