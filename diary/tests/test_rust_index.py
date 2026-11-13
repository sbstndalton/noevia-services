"""R9 (D9): DIARY_RUST_INDEX hands a writer tenant's dirty-document re-index to the Rust process
(a stand-in for POST /internal/reindex here). Off by default and then inert; anything short of a
complete reply re-indexes in Python under the same locks. Synthetic data only."""
import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest

import agent.app as appmod
from agent import rust_answer, rust_forward, rust_index, tenant_writer
from agent.config import Config

TENANT = "4f92a1b2-7e60-4ac2-9d54-5e6f708192a3"


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    monkeypatch.delenv("DIARY_RUST_INDEX", raising=False)
    monkeypatch.delenv("DIARY_RUST_ANSWER", raising=False)
    monkeypatch.setattr(tenant_writer, "is_writer", lambda uid: (uid or "").lower() == TENANT)


class FakeServer:
    base_url = "http://127.0.0.1:1"

    def __init__(self, ready=True):
        self._ready, self.lost_calls = ready, 0

    def control(self):
        return "synthetic-control-token", self._ready

    def lost(self):
        self.lost_calls += 1


DIRTY = ["Entries/2026/September/September 1, 2026.md", "AI Memory/qa.md"]


def state(dirty=DIRTY):
    config = SimpleNamespace(get=lambda: {"active": {"model": "qa-embed-a", "base_url": None, "profile": "plain"}})
    return SimpleNamespace(retrieval=SimpleNamespace(embedding=SimpleNamespace(config=config)),
                           journal=SimpleNamespace(dirty_documents=lambda: list(dirty)),
                           cfg=Config({"llm": {"timeout_s": 7, "max_retries": 2}}))


def call(transport, tenant=TENANT, server=None, st=None, held=True):
    server = server or FakeServer()
    with (tenant_writer.hold(TENANT) if held else nullcontext()):
        return rust_index.reindex_dirty(st or state(), tenant, server_lookup=lambda: server, transport=transport), server


def recording(seen, status=200, body=None, headers=None):
    def handler(request):
        seen.append(request)
        return httpx.Response(status, headers=headers or {},
                              content=json.dumps({"documents": []} if body is None else body).encode())
    return httpx.MockTransport(handler)


def test_off_by_default_nothing_is_sent():
    seen = []
    assert call(recording(seen))[0] is False and seen == []
    assert rust_index.mode() == "" and rust_index.mode("yes") == ""
    assert rust_answer.answer_config(Config({}), 0) is None


def test_on_sends_the_start_up_config(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_INDEX", "on")
    assert rust_answer.answer_config(Config({}), 0) is not None


def test_a_non_writer_legacy_or_unready_tenant_stays_in_python(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_INDEX", "on")
    seen = []
    assert call(recording(seen), tenant="5a03b2c3-8f71-4bd3-ae65-6f708192a3b4")[0] is False
    assert call(recording(seen), tenant="legacy")[0] is False
    assert call(recording(seen), tenant=None)[0] is False
    assert call(recording(seen), server=FakeServer(ready=False))[0] is False
    with tenant_writer.hold(TENANT):
        assert rust_index.reindex_dirty(state(), TENANT, server_lookup=lambda: None, transport=recording(seen)) is False
    assert call(recording(seen), held=False)[0] is False  # never claims a lock this context does not hold
    assert seen == []


def test_nothing_dirty_makes_no_call(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_INDEX", "on")
    seen = []
    assert call(recording(seen), st=state(dirty=[]))[0] is True and seen == []


def test_the_deadline_grows_with_the_dirty_documents(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_INDEX", "on")
    seen = []
    call(recording(seen))
    (req,) = seen
    # 60 s, plus per document llm.timeout_s for each of 1 + llm.max_retries attempts.
    assert req.extensions["timeout"]["read"] == rust_index.REINDEX_BASE_S + len(DIRTY) * 7 * 3


def test_a_complete_reply_is_the_reindex(monkeypatch, caplog):
    monkeypatch.setenv("DIARY_RUST_INDEX", "on")
    seen = []
    body = {"documents": [{"document": "Entries/2026/September/September 1, 2026.md", "embedded": 2, "pending": False, "error": None},
                          {"document": "AI Memory/qa.md", "embedded": 0, "pending": True, "error": "read"}]}
    with caplog.at_level(logging.WARNING, logger="diary.rust.index"):
        done, _ = call(recording(seen, body=body), tenant=TENANT.upper())
    assert done is True
    (req,) = seen
    assert req.url.path == "/internal/reindex" and req.url.params["tenant"] == TENANT
    assert req.headers[rust_forward.CONTROL_HEADER] == "synthetic-control-token"
    assert req.headers[rust_forward.WRITER_HEADER.decode()] == "1"
    assert json.loads(req.content) == {"embedding": {"active": {"model": "qa-embed-a", "base_url": None, "profile": "plain"}}}
    assert [r.getMessage() for r in caplog.records] == ["edit reindex pending: read"]


@pytest.mark.parametrize("status,headers,body", [
    (421, {"x-noevia-diary-decline": "1"}, None),
    (503, {}, None),
    (409, {}, None),
    (404, {}, None),
    (200, {}, {"documents": "not a list"}),
    (200, {}, {"other": []}),
])
def test_anything_short_of_a_complete_reply_reindexes_in_python(monkeypatch, status, headers, body):
    monkeypatch.setenv("DIARY_RUST_INDEX", "on")
    done, server = call(recording([], status=status, body=body, headers=headers))
    assert done is False and server.lost_calls == 0


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
def test_a_call_never_made_falls_back_without_losing_the_child(monkeypatch, error):
    monkeypatch.setenv("DIARY_RUST_INDEX", "on")

    def refuse(request):
        raise error("synthetic refusal")
    done, server = call(httpx.MockTransport(refuse))
    assert done is False and server.lost_calls == 0


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.WriteTimeout])
def test_a_call_that_may_have_reached_the_child_loses_it_and_leaves_the_documents_dirty(monkeypatch, error):
    # The lost child may still be writing: no Python re-index now; the next pass does it after the reap.
    monkeypatch.setenv("DIARY_RUST_INDEX", "on")

    def stall(request):
        raise error("synthetic stall")
    done, server = call(httpx.MockTransport(stall))
    assert done is True and server.lost_calls == 1


class Journal:
    def __init__(self):
        self.listed = 0

    def dirty_documents(self):
        self.listed += 1
        return []


@pytest.mark.parametrize("rust_done", [True, False])
def test_reindex_dirty_asks_rust_under_the_locks_and_skips_the_python_loop_only_after_it(monkeypatch, rust_done):
    import agent.workspace_import as wi
    order = []
    monkeypatch.setattr(wi, "drain_index_outbox", lambda backend, journal: order.append("drain"))
    lock = tenant_writer.TenantWriteLock(TENANT)

    def rust(st, uid):
        order.append(("rust", uid, tenant_writer.held(uid), lock._is_owned()))
        return rust_done
    monkeypatch.setattr(appmod.rust_index, "reindex_dirty", rust)
    journal = Journal()
    st = SimpleNamespace(tenant_id=TENANT, journal=journal, store=SimpleNamespace(_write_lock=lock, backend=None))
    appmod._reindex_dirty(st)
    assert order == ["drain", ("rust", TENANT, True, True)]
    assert journal.listed == (0 if rust_done else 1)
