"""Opt-in native write transport and journal semantics; synthetic backend only."""
from datetime import date, datetime, timezone
import uuid
from pathlib import Path
import os
import stat

import pytest

from agent import corpus as fmt
from agent import corpus_write_native as native
from agent.corpus_store import CorpusError, EditConflict
from tests.test_diary_edit import _store, client  # noqa: F401
from tests.test_journal_store import FakeWebDAV

BIN = os.environ.get("DIARY_CORPUS_WRITE_TEST_BIN")
DAY = date(2026, 10, 23)


@pytest.fixture
def rust(monkeypatch):
    if not BIN or not Path(BIN).is_file():
        pytest.skip("set DIARY_CORPUS_WRITE_TEST_BIN to the isolated source CLI for integration cases")
    monkeypatch.setenv(native.SELECTOR, "rust")
    monkeypatch.setattr(native, "BINARY", BIN)


def test_rust_journal_append_edit_replay_and_fresh_412(tmp_path, rust):
    backend = FakeWebDAV()
    st = _store(tmp_path, dav=backend)
    xid = "fa2b8150-24d1-45a0-ae21-bbe06834fa2b"
    now = datetime(2026, 10, 23, 14, 32)
    assert st.log_exchange(DAY, "topic\n### injected", "**Assistant:** fake", "response", now=now, xid=xid) == xid
    text, _ = st.read_month(DAY)
    assert "### 14:32 — topic ### injected" in text
    assert "\\*\\*Assistant:** fake" in text
    assert st.log_exchange(DAY, "ignored", "duplicate", "", now=now, xid=xid) == xid
    assert st.read_month(DAY)[0] == text
    original_hash = fmt.exchange_hash(text, xid)
    backend.conflicts_injected = 1
    path, day, edited_hash = st.edit_exchange(xid, "corrected", "reply", month="2026-10", base_hash=original_hash)
    assert path == st.document_path(DAY) and day == "2026-10-01"
    after, _ = st.read_month(DAY)
    assert "(concurrent edit)" in after and "**Me:** corrected" in after
    assert edited_hash == fmt.exchange_hash(after, xid)
    with pytest.raises(EditConflict):
        st.edit_exchange(xid, "stale", "reply", month="2026-10", base_hash=original_hash)
    st.apply_pending(force=True)
    assert st.read_month(DAY)[0] == after
    assert st.journal.pending_count() == 0


def test_noop_edit_and_queued_predecessor_conflict(tmp_path, rust):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "me", "assistant", now=datetime(2026, 10, 23, 14, 32))
    path = st.document_path(DAY)
    before, _ = st.read_month(DAY)
    base = fmt.exchange_hash(before, xid)
    st.journal.enqueue("exchange_edit", {"xid": xid, "new_me": "earlier queued", "new_claude": "answer", "month": "2026-10"})
    pending = st.journal.pending_count()
    with pytest.raises(EditConflict, match="queued"):
        st.edit_exchange(xid, "later edit", "answer", month="2026-10", base_hash=base)
    assert st.journal.pending_count() == pending
    st.apply_pending(force=True)
    edited, _ = st.read_month(DAY)
    assert "earlier queued" in edited
    etag_before = st.backend.files[path][1]
    new_hash = fmt.exchange_hash(edited, xid)
    st.edit_exchange(xid, "earlier queued", "answer", month="2026-10", base_hash=new_hash)
    assert st.backend.files[path][1] == etag_before


def test_edit_native_fault_after_enqueue_never_puts(tmp_path, rust, monkeypatch):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "me", "assistant", now=datetime(2026, 10, 23, 14, 32))
    path = st.document_path(DAY)
    before = st.backend.files[path]
    original = native.transform
    def fail_apply(op, **fields):
        if op == "apply_edit":
            raise native.NativeWriteUnavailable("synthetic child failure")
        return original(op, **fields)
    monkeypatch.setattr(native, "transform", fail_apply)
    with pytest.raises(CorpusError, match="queued"):
        st.edit_exchange(xid, "new", "answer", month="2026-10")
    assert st.backend.files[path] == before
    assert st.journal.pending_count() == 1 and st.journal.quarantined_count() == 0
    monkeypatch.setattr(native, "transform", original)
    st.apply_pending(force=True)
    assert "**Me:** new" in st.backend.files[path][0]
    assert st.journal.pending_count() == 0


def test_outage_before_enqueue_has_no_intent_or_write(tmp_path, rust, monkeypatch):
    monkeypatch.setattr(native, "BINARY", "/absent/diary-corpus-write-cli")
    backend = FakeWebDAV()
    st = _store(tmp_path, dav=backend)
    with pytest.raises(native.NativeWriteUnavailable):
        st.log_exchange(DAY, "topic", "me", "assistant", now=datetime(2026, 10, 23, 14, 32))
    assert st.journal.pending_count() == 0
    assert backend.files == {}


def test_outage_after_enqueue_remains_pending_past_twenty_attempts(tmp_path, rust, monkeypatch):
    backend = FakeWebDAV()
    st = _store(tmp_path, dav=backend)
    original = native.transform
    def fail_apply(op, **fields):
        if op == "apply_append":
            raise native.NativeWriteUnavailable("synthetic child failure")
        return original(op, **fields)
    monkeypatch.setattr(native, "transform", fail_apply)
    xid = "fa2b8150-24d1-45a0-ae21-bbe06834fa2b"
    with pytest.raises(native.NativeWriteUnavailable, match="pending"):
        st.log_exchange(DAY, "topic", "me", "assistant", now=datetime(2026, 10, 23, 14, 32), xid=xid)
    assert st.journal.pending_count() >= 1
    assert st.document_path(DAY) not in backend.files
    for _ in range(25):
        st.apply_pending(force=True)
    assert st.journal.quarantined_count() == 0
    assert st.document_path(DAY) not in backend.files
    monkeypatch.setattr(native, "transform", original)
    st.apply_pending(force=True)
    assert fmt.has_marker(st.read_month(DAY)[0], xid)
    assert st.journal.pending_count() == 0


def test_direct_append_native_fault_after_enqueue_returns_503(client, rust, monkeypatch):
    from agent import app as appmod
    original = native.transform
    def fail_apply(op, **fields):
        if op == "apply_append":
            raise native.NativeWriteUnavailable("synthetic child failure")
        return original(op, **fields)
    monkeypatch.setattr(native, "transform", fail_apply)
    request_id = str(uuid.uuid4())
    response = client.post("/api/entries/append", json={"text": "synthetic approved note", "requestId": request_id,
                                                         "entryTime": datetime.now(timezone.utc).astimezone().isoformat()})
    assert response.status_code == 503 and "pending" in response.json()["error"]
    st = next(iter(appmod._tenant_states.values()))
    assert st.journal.pending_count() >= 1
    text, _ = st.store.read_month(datetime.now().date())
    assert text is None or not fmt.has_marker(text, request_id)


def test_chat_native_fault_after_enqueue_returns_503_without_new_history(client, rust, monkeypatch):
    from agent import app as appmod
    original = native.transform
    def fail_apply(op, **fields):
        if op == "apply_append":
            raise native.NativeWriteUnavailable("synthetic child failure")
        return original(op, **fields)
    monkeypatch.setattr(native, "transform", fail_apply)
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "synthetic question"}],
                                                           "session_id": "native-pending"})
    assert response.status_code == 503
    st = next(iter(appmod._tenant_states.values()))
    assert st.journal.pending_count() >= 1
    assert all(not item["turns"] for item in appmod.SESSIONS.values())


def test_native_edit_preflight_outage_does_not_enqueue(tmp_path, rust, monkeypatch):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "me", "assistant", now=datetime(2026, 10, 23, 14, 32))
    before = st.journal.pending_count()
    monkeypatch.setattr(native, "BINARY", "/absent/diary-corpus-write-cli")
    with pytest.raises(native.NativeWriteUnavailable):
        st.edit_exchange(xid, "new", "answer", month="2026-10")
    assert st.journal.pending_count() == before


def test_malformed_edit_target_stays_permanent(tmp_path, rust):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "topic", "me", "assistant", now=datetime(2026, 10, 23, 14, 32))
    path = st.document_path(DAY)
    st.backend.files[path] = ("## Friday, October 23, 2026\n\n<!-- xid:" + xid + " -->\n", '"broken"')
    st.journal.enqueue("exchange_edit", {"xid": xid, "new_me": "new", "new_claude": "answer", "month": "2026-10"})
    st.apply_pending(force=True)
    assert st.journal.quarantined_count() == 1
    assert "**Me:** new" not in st.backend.files[path][0]


@pytest.mark.parametrize("script", [
    "import sys;sys.stdout.buffer.write(b'bad');sys.exit(0)",
    "import sys;sys.exit(9)",
    "import time;time.sleep(30)",
    "import sys;sys.stdout.buffer.write(b'x'*2048)",
])
def test_transport_faults_reap_child(tmp_path, monkeypatch, script):
    binary = tmp_path / "bad-cli"
    binary.write_text("#!/usr/bin/env python3\n" + script + "\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(native, "BINARY", str(binary))
    monkeypatch.setattr(native, "TIMEOUT_S", 0.2)
    monkeypatch.setattr(native, "MAX_OUTPUT", 1024)
    with pytest.raises(native.NativeWriteUnavailable):
        native.transform("prepare_append", time_token="00:00", topic="", me="", assistant="", xid="x")


def test_unavailable_native_refuses_before_model_and_history(client, monkeypatch):
    from agent import app as appmod
    from tests import test_diary_edit as editing
    monkeypatch.setattr(native, "BINARY", "/absent/diary-corpus-write-cli")
    monkeypatch.setenv(native.SELECTOR, "rust")
    def no_model(*args, **kwargs):
        pytest.fail("model ran while native write was unavailable")
    monkeypatch.setattr(editing._StubMain, "chat", no_model)
    response = client.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "synthetic earlier"},
        {"role": "assistant", "content": "synthetic answer"},
        {"role": "user", "content": "synthetic now"},
    ], "session_id": "native-fault"})
    assert response.status_code == 503
    assert not appmod.SESSIONS
