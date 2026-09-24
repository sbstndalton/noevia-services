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
