"""Regression tests for review findings D1, D2, D3, D-P3, D-P4 (synthetic data only)."""
from __future__ import annotations

from datetime import datetime

import pytest

import agent.app as appmod
from agent.corpus import exchange_hash, render_exchange
from agent.corpus_store import CorpusError, EditConflict
from tests.test_diary_edit import DAY, _store, client  # noqa: F401 (fixture)
from tests.test_dedicated_storage import volume, B  # noqa: F401 (fixture)

U1 = "33333333-3333-4333-8333-333333333333"
U2 = "44444444-4444-4444-8444-444444444444"


# ---- D1: evicted tenant states are closed, outside the tenant lock ----

def test_lru_evicted_tenant_state_is_closed_outside_lock(client, monkeypatch):
    monkeypatch.setattr(appmod, "_TENANT_STATE_CAP", 1)
    closed = []

    def record(state):
        assert not appmod._tenant_lock.locked(), "state closed while holding _tenant_lock"
        closed.append(state)

    monkeypatch.setattr(appmod, "_close_state", record)
    client.get("/api/storage-status", headers={"X-Cowork-User-ID": U1})
    first = list(appmod._tenant_states.values())
    assert len(first) == 1 and not closed
    client.get("/api/storage-status", headers={"X-Cowork-User-ID": U2})
    assert closed == first
    assert first[0] not in appmod._tenant_states.values()


def test_ttl_expired_tenant_state_is_closed(client, monkeypatch):
    closed = []
    monkeypatch.setattr(appmod, "_close_state", closed.append)
    client.get("/api/storage-status", headers={"X-Cowork-User-ID": U1})
    (stale,) = appmod._tenant_states.values()
    stale.last_used -= appmod._TENANT_STATE_TTL_S + 1
    client.get("/api/storage-status", headers={"X-Cowork-User-ID": U2})
    assert closed == [stale]


# ---- D2: base-hash optimistic concurrency on edits ----

def test_store_edit_rejects_stale_base_hash(tmp_path):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "original words", "reply")
    base = exchange_hash(st.read_month(DAY)[0], xid)
    # Tab A saves with the right base.
    st.edit_exchange(xid, "tab A words", "reply", month="2026-09", base_hash=base)
    # Tab B still holds the old base: conflict, nothing enqueued or written.
    pending_before = st.journal.pending_count()
    with pytest.raises(EditConflict) as err:
        st.edit_exchange(xid, "tab B words", "reply", month="2026-09", base_hash=base)
    text = st.read_month(DAY)[0]
    assert "tab A words" in text and "tab B words" not in text
    assert err.value.current_hash == exchange_hash(text, xid)
    assert "tab A words" in err.value.current_text
    assert st.journal.pending_count() == pending_before
    # Retrying with the fresh hash succeeds.
    st.edit_exchange(xid, "tab B words", "reply", month="2026-09", base_hash=err.value.current_hash)
    assert "tab B words" in st.read_month(DAY)[0]


def test_store_edit_without_base_hash_stays_last_write_wins(tmp_path):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "original", "reply")
    st.edit_exchange(xid, "first", "reply", month="2026-09")
    st.edit_exchange(xid, "second", "reply", month="2026-09")
    assert "**Me:** second" in st.read_month(DAY)[0]


def test_exchange_hash_ignores_sibling_exchanges():
    a, b = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
    one = "## Day\n\n" + render_exchange("x", "y", a)
    two = one + "\n" + render_exchange("z", "w", b)
    assert exchange_hash(one, a) == exchange_hash(two, a)
    assert exchange_hash(one, b) is None


def test_edit_endpoint_returns_409_with_current_text(client):
    xid = client.post("/api/chat", json={"message": "Synthetic entry one."}).json()["xid"]
    month = datetime.now().strftime("%Y-%m")
    st = appmod.get_state()
    base = exchange_hash(st.store.read_month(datetime.now().date())[0], xid)
    ok = client.post("/api/entries/edit", json={"xid": xid, "me": "Tab A.", "month": month, "base_hash": base})
    assert ok.status_code == 200
    new_hash = ok.json()["hash"]
    assert new_hash and new_hash != base
    assert new_hash == exchange_hash(st.store.read_month(datetime.now().date())[0], xid)
    stale = client.post("/api/entries/edit", json={"xid": xid, "me": "Tab B.", "month": month, "base_hash": base})
    assert stale.status_code == 409
    body = stale.json()
    assert body["conflict"] is True and body["current_hash"] == new_hash
    assert "Tab A." in body["current_text"]
    assert client.post("/api/entries/edit", json={"xid": xid, "me": "Tab B.", "month": month, "base_hash": new_hash}).status_code == 200
    assert client.post("/api/entries/edit", json={"xid": xid, "me": "x", "base_hash": "nothex"}).status_code == 400


# ---- D3: month validation ----

@pytest.mark.parametrize("month", ["2026-9", "abcd-ef", "2026-13", "2026-00", "../x", "2026-09-01"])
def test_edit_endpoint_rejects_bad_month_with_400(client, month):
    r = client.post("/api/entries/edit", json={"xid": "11111111-1111-4111-8111-111111111111", "me": "x", "month": month})
    assert r.status_code == 400
    assert r.json()["error"] == "month must be YYYY-MM"


def test_store_rejects_bad_month(tmp_path):
    with pytest.raises(ValueError):
        _store(tmp_path).edit_exchange("11111111-1111-4111-8111-111111111111", "x", "", month="2026-1x")


# ---- D-P3 / D-P4: tenant id case normalization ----

def test_storage_backup_accepts_uppercase_tenant_id(volume):  # noqa: F811
    from fastapi.testclient import TestClient
    r = TestClient(appmod.app).post("/api/storage-backup", headers={"X-Cowork-User-ID": B.upper()})
    assert r.status_code == 200 and r.json()["mode"] == "legacy"


def test_chat_session_keyed_by_normalized_tenant():
    appmod.SESSIONS.clear()
    assert appmod._session("s1", U1.upper()) is appmod._session("s1", U1)
    assert appmod._session("s1", U1) is not appmod._session("s1", U2)


def test_store_returns_hash_of_applied_document(tmp_path):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "original", "reply")
    _, _, new_hash = st.edit_exchange(xid, "edited", "reply", month="2026-09")
    assert new_hash == exchange_hash(st.read_month(DAY)[0], xid)


def test_edit_endpoint_hides_exception_text(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    st = appmod.get_state()
    monkeypatch.setattr(type(st.store), "edit_exchange", boom)
    r = client.post("/api/entries/edit", json={"xid": "11111111-1111-4111-8111-111111111111", "me": "x"})
    assert r.status_code == 502 and r.json() == {"error": "edit failed"}
