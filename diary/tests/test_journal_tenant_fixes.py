"""Regressions: journal poison entries, tenant delete vs in-flight write,
recovery outside the global tenant lock, import error hygiene (synthetic data only)."""
from __future__ import annotations

import threading
import time
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.app as appmod
from agent.config import Config
from agent.corpus_store import MAX_APPLY_ATTEMPTS, CorpusStore, StoreClosed
from agent.journal import Journal
from agent.local_storage import LocalCorpusBackend
from tests.test_diary_edit import DAY, _store, client  # noqa: F401 (fixture)

U1 = "55555555-5555-4555-8555-555555555555"
U2 = "66666666-6666-4666-8666-666666666666"


def _row(journal, jid):
    return journal._conn.execute("SELECT applied, attempts, last_error FROM journal WHERE id=?", (jid,)).fetchone()


# ---- 1: an edit whose target vanished must not block the journal ----

def test_edit_of_deleted_document_is_quarantined_and_later_exchange_applies(tmp_path):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "synthetic words", "reply", now=datetime(2026, 9, 1, 9))
    jid = st.journal.enqueue("exchange_edit", {"xid": xid, "new_me": "x", "new_claude": "y", "month": "2026-09"})
    st.backend.files.clear()  # user deleted the day file after a transient failure
    later = st.journal.enqueue("exchange", {
        "day": "2026-09-02", "xid": "77777777-7777-4777-8777-777777777777",
        "sub_header": "later", "body": "later synthetic body",
    })
    st.apply_pending()
    assert st.journal.is_applied(later)
    assert "later synthetic body" in st.read_month(date(2026, 9, 2))[0]
    applied, _, err = _row(st.journal, jid)
    assert applied == 1 and err.startswith("quarantined:")  # retired but inspectable


def test_edit_with_malformed_block_is_quarantined(tmp_path):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "synthetic words", "reply", now=datetime(2026, 9, 1, 9))
    path = st.document_path(DAY)
    body, etag = st.backend.files[path]
    st.backend.files[path] = (body.replace("**Me:**", "Me?"), etag)  # marker kept, block broken
    jid = st.journal.enqueue("exchange_edit", {"xid": xid, "new_me": "x", "new_claude": "y"})
    st.apply_pending()
    assert _row(st.journal, jid)[2].startswith("quarantined:")


def test_transient_failures_are_capped(tmp_path):
    st = _store(tmp_path)
    jid = st.journal.enqueue("exchange", {"day": "2026-09-01", "xid": "x1", "sub_header": "s", "body": "b"})
    st.backend.put = MagicMock(side_effect=OSError("synthetic outage"))
    for _ in range(MAX_APPLY_ATTEMPTS - 1):
        st.apply_pending()
        assert not st.journal.is_applied(jid)
    st.apply_pending()
    applied, attempts, err = _row(st.journal, jid)
    assert applied == 1 and attempts == MAX_APPLY_ATTEMPTS and "synthetic outage" in err


# ---- 2: tenant delete must not race an in-flight write ----

def test_delete_tenant_waits_for_write_and_leaves_no_files(client, tmp_path):  # noqa: F811
    root = Path(appmod._base_cfg.get("retrieval.db_path")).parent / "users" / U1
    cfg = Config({"corpus": {"webdav": {"remote_root": ""}, "monthly_prefix": "", "index_file": "INDEX.md"}})
    backend = LocalCorpusBackend(str(root / "corpus"))
    store = CorpusStore(cfg, backend, Journal(root / "index.db"))
    state = SimpleNamespace(store=store, backend=backend, journal=store.journal,
                            llm_main=MagicMock(), llm_aux=MagicMock(), retrieval=MagicMock())
    appmod._tenant_states[f"{U1}:test"] = state

    entered, release = threading.Event(), threading.Event()

    def mutate(current):
        entered.set()
        release.wait(5)
        return "synthetic text\n", True

    def write():  # as every serialized store method does
        with store._write_lock:
            store._guarded_write("2026-09.md", mutate)

    writer = threading.Thread(target=write)
    writer.start()
    assert entered.wait(5)
    deleter = threading.Thread(target=lambda: client.delete("/api/internal/tenant", headers={"X-Cowork-User-ID": U1}))
    deleter.start()
    time.sleep(0.2)
    assert deleter.is_alive()  # delete waits for the in-flight write
    release.set()
    writer.join(5)
    deleter.join(5)
    assert not root.exists()
    with pytest.raises(StoreClosed):
        store.log_exchange(DAY, "t", "late", "late")
    with pytest.raises(StoreClosed):
        backend.put("late.md", b"late")
    assert not root.exists()


# ---- 3: recovery runs outside the global tenant lock, once per state ----

def _req(user):
    return SimpleNamespace(headers={"X-Cowork-User-ID": user})


def test_slow_recovery_does_not_block_other_tenants(client):  # noqa: F811
    slow = appmod._tenant_state(_req(U1), recover=False)
    fast = appmod._tenant_state(_req(U2), recover=False)
    assert slow is not fast
    gate, started = threading.Event(), threading.Event()
    calls = []

    def slow_apply(*a, **k):
        calls.append(1)
        started.set()
        gate.wait(5)
        return 0

    slow.recovered = False
    slow.store.apply_pending = slow_apply
    results = []
    threads = [threading.Thread(target=lambda: results.append(appmod._tenant_state(_req(U1)))) for _ in range(2)]
    for t in threads:
        t.start()
    assert started.wait(5)
    t0 = time.monotonic()
    assert appmod._tenant_state(_req(U2)) is fast
    assert time.monotonic() - t0 < 1.0
    assert not appmod._tenant_lock.locked()
    gate.set()
    for t in threads:
        t.join(5)
    assert results == [slow, slow]
    assert calls == [1] and slow.recovered


# ---- 5: import errors do not echo exception text ----

def test_import_failure_hides_exception_text():
    import inspect
    src = inspect.getsource(appmod)
    assert 'detail=f"import failed: {exc}"' not in src
    assert 'detail=f"file unreadable: {exc}"' not in src


def test_local_listing_does_not_read_files(tmp_path, monkeypatch):
    backend = LocalCorpusBackend(str(tmp_path))
    backend.put("a.md", b"synthetic")
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("read")))
    (entry,) = backend.list_dir("")
    assert entry["etag"] and entry["name"] == "a.md"


# ---- malformed index_update ops must never poison the journal ----

@pytest.mark.parametrize("payload", [
    {"open_questions": ["x"], "timeline": [], "today": "2026-09-01"},
    {"open_questions": [], "timeline": {"a": 1}, "today": "2026-09-01"},
    {"open_questions": [{"action": "add", "text": 5}], "timeline": [], "today": "2026-09-01"},
])
def test_malformed_index_update_is_quarantined_and_exchange_applies(tmp_path, payload):
    st = _store(tmp_path)
    jid = st.journal.enqueue("index_update", payload)
    xid = st.log_exchange(DAY, "topic", "synthetic words", "reply", now=datetime(2026, 9, 1, 9))
    assert xid in st.read_month(DAY)[0]
    assert _row(st.journal, jid)[2].startswith("quarantined:")
    assert st.journal.pending_count() == 0


def test_applier_shape_error_is_permanent_but_store_closed_is_not(tmp_path):
    st = _store(tmp_path)
    entry = SimpleNamespace(kind="index_update", attempts=0, payload={"open_questions": [], "timeline": []})
    assert st._is_permanent_failure(entry, AttributeError("x"))
    assert st._is_permanent_failure(entry, TypeError("x"))
    assert not st._is_permanent_failure(entry, StoreClosed("closed"))
    assert not st._is_permanent_failure(entry, OSError("transient"))


# ---- tenant delete vs in-flight ManagedCorpusBackend construction race ----

def test_construct_racing_delete_leaves_no_directory(client, tmp_path, monkeypatch):  # noqa: F811
    """A request whose ManagedCorpusBackend construction is in flight when
    DELETE /api/internal/tenant runs (and finishes) must not leave behind a
    recreated, empty tenant directory: it should observe the tombstone and
    fail closed instead."""
    from agent.managed_storage import ManagedCorpusBackend

    U3 = "77777777-7777-4777-8777-777777777779"
    root = Path(appmod._base_cfg.get("retrieval.db_path")).parent / "users" / U3
    # Seed a pre-existing tenant so the delete has something real to remove.
    seed = ManagedCorpusBackend(root, U3)
    seed.close()
    assert root.exists()

    real_init = ManagedCorpusBackend.__init__
    constructing, deleted = threading.Event(), threading.Event()

    def delayed_init(self, *a, **k):
        real_init(self, *a, **k)
        if getattr(self, "tenant", None) == U3 or a and a[-1] == U3:
            constructing.set()
            assert deleted.wait(5)  # block until the delete has completed

    monkeypatch.setattr(ManagedCorpusBackend, "__init__", delayed_init)

    result = {}

    def racer():
        req = SimpleNamespace(headers={"X-Cowork-User-ID": U3})
        try:
            result["state"] = appmod._tenant_state(req)
        except Exception as exc:  # noqa: BLE001
            result["exc"] = exc

    racer_thread = threading.Thread(target=racer)
    racer_thread.start()
    assert constructing.wait(5)

    resp = client.delete("/api/internal/tenant", headers={"X-Cowork-User-ID": U3})
    assert resp.status_code == 200
    deleted.set()
    racer_thread.join(5)

    assert "exc" in result and getattr(result["exc"], "status_code", None) == 404
    assert not root.exists()


def test_tenant_state_rejects_request_racing_delete(client, tmp_path):  # noqa: F811
    """_tenant_state itself must fail closed (404) and leave no directory
    when a tombstone is set concurrently with its construction."""
    U4 = "88888888-8888-4888-8888-888888888884"
    root = Path(appmod._base_cfg.get("retrieval.db_path")).parent / "users" / U4
    with appmod._tenant_lock:
        appmod._mark_tenant_deleted_locked(U4)

    req = SimpleNamespace(headers={"X-Cowork-User-ID": U4})
    with pytest.raises(Exception) as excinfo:
        appmod._tenant_state(req)
    # FastAPI's HTTPException carries status_code 404.
    assert getattr(excinfo.value, "status_code", None) == 404
    assert not root.exists()


def test_update_standing_sections_drops_malformed_ops(tmp_path):
    st = _store(tmp_path)
    assert st.update_standing_sections(["x", {"text": 1}], {"a": 1}, "2026-09-01") is None
    assert st.journal.pending_count() == 0
    jid = st.update_standing_sections(
        [{"action": "add", "text": "synthetic question?"}, {"action": "edit", "text": "a", "replacement": 3}],
        [], "2026-09-01")
    assert jid is not None and st.journal.is_applied(jid)
    row = st.journal._conn.execute("SELECT payload FROM journal WHERE id=?", (jid,)).fetchone()
    assert "replacement" not in row[0]
