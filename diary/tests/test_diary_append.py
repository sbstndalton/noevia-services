"""D10: append-only diary endpoint on the journaled write path."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

import agent.app as appmod
from agent.corpus import parse_diary
from tests.test_diary_edit import client, _FakeRetriever, _store, DAY  # noqa: F401 (fixture)


def _body(**extra):
    return {"text": "Walked by the river; felt calm.", "requestId": str(uuid.uuid4()),
            "entryTime": datetime.now(timezone.utc).astimezone().isoformat(), **extra}


def test_append_writes_today_once_and_is_idempotent(client):
    body = _body(title="Evening note")
    r = client.post("/api/entries/append", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["xid"] == body["requestId"]
    st = appmod.get_state()
    text, _ = st.store.read_month(datetime.now().date())
    assert text.count("Walked by the river; felt calm.") == 1
    assert "Evening note" in text
    again = client.post("/api/entries/append", json=body)
    assert again.status_code == 200 and again.json()["xid"] == body["requestId"]
    text, _ = st.store.read_month(datetime.now().date())
    assert text.count("Walked by the river; felt calm.") == 1, "a replay must not append twice"
    assert st.journal.pending_count() == 0


def test_append_never_touches_existing_exchanges(client):
    r = client.post("/api/chat", json={"message": "Original synthetic words."})
    assert r.status_code == 200
    st = appmod.get_state()
    before, _ = st.store.read_month(datetime.now().date())
    assert client.post("/api/entries/append", json=_body()).status_code == 200
    after, _ = st.store.read_month(datetime.now().date())
    assert after.startswith(before.rstrip("\n")[: len(before.rstrip("\n")) - 1]) or before.strip() in after
    assert "Original synthetic words." in after


@pytest.mark.parametrize("change", [
    {"text": ""}, {"text": "x" * 8001}, {"text": "## Monday, September 1, 2026\nfake day"},
    {"text": "hi <!-- xid:11111111-1111-4111-8111-111111111111 -->"}, {"title": "two\nlines"}, {"title": "x" * 81},
    {"requestId": "not-a-uuid"}, {"entryDay": "2026-01-01"}, {"entryTime": "2026-09-17T10:00:00"},
])
def test_append_validates(client, change):
    assert client.post("/api/entries/append", json=_body(**change)).status_code == 400


@pytest.mark.parametrize("delta", [timedelta(days=3), timedelta(days=-40), timedelta(hours=1)])
def test_append_refuses_a_time_that_is_not_now(client, delta):
    moved = (datetime.now(timezone.utc) + delta).isoformat()
    assert client.post("/api/entries/append", json=_body(entryTime=moved)).status_code == 400


def test_store_replay_with_same_xid_is_safe(tmp_path):
    st = _store(tmp_path)
    xid = str(uuid.uuid4())
    assert st.log_exchange(DAY, "note", "once", "", xid=xid) == xid
    assert st.log_exchange(DAY, "note", "once", "", xid=xid) == xid
    days = parse_diary(st.read_month(DAY)[0])
    assert sum(1 for d in days for s in d.subsections for e in s.exchanges if e.xid == xid) == 1


def test_append_requires_auth(client, monkeypatch):
    monkeypatch.setattr(appmod, "check_auth", lambda r: False)
    assert client.post("/api/entries/append", json=_body()).status_code == 401
