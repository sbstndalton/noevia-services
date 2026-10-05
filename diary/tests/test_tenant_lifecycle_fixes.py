"""#802 streamed credential retry, #804 delete without the global lock across I/O,
#805 legacy create_directory, backup vs deletion, chat recovery off the event loop.

Synthetic tenants, descriptors and text only; no real Diary, model or storage."""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import agent.app as appmod
from agent.diary_migration import LegacyWriteGuard
from agent.workspace_files import directory_create
from tests.test_dedicated_storage import volume, A, B  # noqa: F401 (fixture)

C = "44444444-4444-4444-8444-444444444441"
REMOTE = {"kind": "webdav", "baseUrl": "http://offline.invalid/dav", "username": "synthetic", "corpusRoot": "Diary"}
STREAM_BODY = {"stream": True, "diary_events": True, "session_id": "lifecycle",
               "messages": [{"role": "user", "content": "Synthetic message"}]}


def b64(obj):
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


@pytest.fixture
def app_client(volume, monkeypatch):  # noqa: F811
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda st: None)
    monkeypatch.delenv("DIARY_TENANT_KEY", raising=False)
    appmod._deleted_tenants.pop(A, None)
    appmod._deleted_tenants.pop(B, None)
    appmod._deleted_tenants.pop(C, None)
    yield TestClient(appmod.app)
    for user in (A, B, C):
        appmod._deleted_tenants.pop(user, None)
        appmod._backup_backends.pop(user, None)
        for key in [k for k in appmod.SESSIONS if k.startswith(user + ":")]:
            appmod.SESSIONS.pop(key, None)


def _no_exchange(monkeypatch):
    def exchange(*args, **kwargs):
        raise AssertionError("no exchange may start before the tenant resolves")
    monkeypatch.setattr(appmod, "_run_exchange", exchange)


# ---------------- #802: real statuses before the stream starts ----------------


def test_streamed_chat_without_the_secret_gets_a_real_428_then_streams_with_it(app_client, monkeypatch):
    _no_exchange(monkeypatch)
    ref_only = {"X-Cowork-User-ID": B, "X-Cowork-Storage": b64({**REMOTE, "secretRef": "a" * 32})}
    r = app_client.post("/v1/chat/completions", headers=ref_only, json=STREAM_BODY)
    assert r.status_code == 428
    assert r.json()["detail"] == {"code": "storage_credential_required"}
    assert "text/event-stream" not in r.headers.get("content-type", "")

    # Web's retry: the same request with the secret builds the state and streams.
    calls, recovered = [], []
    real_recover = appmod._recover_state
    monkeypatch.setattr(appmod, "_recover_state", lambda st: (recovered.append(st), real_recover(st)))

    def exchange(st, message, sid, owner, now, day, background, extra, emit):
        assert recovered == [st]  # recovery ran inside the stream, before the exchange
        calls.append(message)
        emit({"type": "answer", "text": "Synthetic answer"})
        return {"reply": "Synthetic answer", "decision": "skipped", "xid": None}

    monkeypatch.setattr(appmod, "_run_exchange", exchange)
    with_secret = {**ref_only, "X-Cowork-Storage": b64({**REMOTE, "secretRef": "a" * 32, "secret": "synthetic-secret"})}
    ok = app_client.post("/v1/chat/completions", headers=with_secret, json=STREAM_BODY)
    assert ok.status_code == 200 and "text/event-stream" in ok.headers["content-type"]
    events = [json.loads(line[6:]) for line in ok.text.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events][-2:] == ["diary", "done"] and calls == ["Synthetic message"]
    # The cached state now serves the ref-only form without asking again.
    _no_exchange(monkeypatch)
    monkeypatch.setattr(appmod, "_run_exchange", exchange)
    recovered.clear()
    assert app_client.post("/v1/chat/completions", headers=ref_only, json=STREAM_BODY).status_code == 200


def test_streamed_chat_for_a_deleted_or_blocked_tenant_gets_a_real_status(app_client, monkeypatch):
    _no_exchange(monkeypatch)
    with appmod._tenant_lock:
        appmod._mark_tenant_deleted_locked(C)
    gone = app_client.post("/v1/chat/completions", headers={"X-Cowork-User-ID": C}, json=STREAM_BODY)
    assert gone.status_code == 410
    blocked = {"X-Cowork-User-ID": B, "X-Cowork-Storage-Blocked": "1", "X-Cowork-Storage": b64({"kind": "blocked"})}
    assert app_client.post("/v1/chat/completions", headers=blocked, json=STREAM_BODY).status_code == 403


@pytest.mark.parametrize("stream", [True, False])
def test_chat_resolves_the_tenant_off_the_event_loop(app_client, monkeypatch, stream):
    seen = []

    def state(request, recover=True):
        try:
            asyncio.get_running_loop()
            seen.append("event loop")
        except RuntimeError:
            seen.append("worker thread")
        return SimpleNamespace()

    monkeypatch.setattr(appmod, "_tenant_state", state)
    monkeypatch.setattr(appmod, "_run_exchange", lambda *a, **k: {"reply": "Synthetic", "decision": "skipped", "xid": None})
    body = STREAM_BODY if stream else {**STREAM_BODY, "stream": False}
    assert app_client.post("/v1/chat/completions", headers={"X-Cowork-User-ID": B}, json=body).status_code == 200
    assert seen == ["worker thread"]


# ---------------- #804: deletion never holds _tenant_lock across I/O ----------------


def test_deleting_a_tenant_mid_write_does_not_stall_other_tenants(app_client):
    held = appmod._tenant_state(SimpleNamespace(headers={"X-Cowork-User-ID": A}))
    assert app_client.get("/api/months", headers={"X-Cowork-User-ID": B}).status_code == 200
    locked, release = threading.Event(), threading.Event()

    def writer():  # an in-flight guarded write doing slow storage I/O
        with held.store._write_lock:
            locked.set()
            release.wait(10)

    results = {}
    threads = [threading.Thread(target=writer),
               threading.Thread(target=lambda: results.setdefault("delete", TestClient(appmod.app).delete(
                   "/api/internal/tenant", headers={"X-Cowork-User-ID": A})))]
    threads[0].start()
    try:
        assert locked.wait(5)
        threads[1].start()
        deadline = time.monotonic() + 5
        while A not in appmod._deleted_tenants and time.monotonic() < deadline:
            time.sleep(0.01)
        assert A in appmod._deleted_tenants
        time.sleep(0.1)
        assert threads[1].is_alive()  # the delete still waits for A's write to finish

        other = threading.Thread(target=lambda: results.setdefault("months", TestClient(appmod.app).get(
            "/api/months", headers={"X-Cowork-User-ID": B})))
        other.start()
        other.join(5)
        assert not other.is_alive(), "another tenant's request blocked behind the deletion"
        assert results["months"].status_code == 200
        gone = TestClient(appmod.app).get("/api/months", headers={"X-Cowork-User-ID": A})
        assert gone.status_code == 410  # refused at once, not queued behind the delete
    finally:
        release.set()
        for t in threads:
            t.join(10)
    assert results["delete"].status_code == 200
    assert held.store.closed
    assert not any(k.startswith(A + ":") for k in appmod._tenant_states)
    assert not (Path(appmod._base_cfg.get("retrieval.db_path")).parent / "users" / A).exists()


# ---------------- #805 ----------------


@contextmanager
def _unlocked():
    yield


class _NoDirectories:
    """Like the WebDAV/S3 backends: no create_directory."""


class _Directories:
    def __init__(self):
        self.created = []

    def create_directory(self, path):
        self.created.append(path)


def test_legacy_remote_storage_refuses_directory_creation_with_405():
    managed = SimpleNamespace(active=lambda: False, migration_lock=_unlocked)
    guard = LegacyWriteGuard(_NoDirectories(), managed)
    assert getattr(guard, "create_directory", None) is None
    store = SimpleNamespace(backend=guard, _write_lock=threading.RLock(), _join=lambda p: p)
    with pytest.raises(HTTPException) as exc:
        directory_create(store, "New folder")
    assert exc.value.status_code == 405


def test_legacy_local_storage_still_creates_directories_behind_the_guard():
    active = [False]
    managed = SimpleNamespace(active=lambda: active[0], migration_lock=_unlocked)
    inner = _Directories()
    guard = LegacyWriteGuard(inner, managed)
    guard.create_directory("Notes")
    assert inner.created == ["Notes"]
    active[0] = True  # the diary moved into the app: the retired corpus is read-only
    with pytest.raises(RuntimeError, match="moved into the app"):
        guard.create_directory("Late")
    assert inner.created == ["Notes"]


def test_directory_endpoint_on_legacy_webdav_storage_is_405(app_client):
    storage = b64({**REMOTE, "secretRef": "", "secret": "synthetic-secret"})
    r = app_client.post("/api/directory", headers={"X-Cowork-User-ID": B, "X-Cowork-Storage": storage},
                        json={"path": "New folder"})
    assert r.status_code == 405


def _managed_tenant(client, user):
    assert client.get("/api/storage-status", headers={"X-Cowork-User-ID": user}).json()["mode"] == "managed"
    return Path(appmod._base_cfg.get("retrieval.db_path")).parent / "users" / user


def test_backup_after_deletion_is_refused_and_recreates_nothing(app_client):
    root = _managed_tenant(app_client, C)
    assert app_client.delete("/api/internal/tenant", headers={"X-Cowork-User-ID": C}).status_code == 200
    assert not root.exists()
    r = app_client.post("/api/storage-backup", headers={"X-Cowork-User-ID": C})
    assert r.status_code == 410
    assert not root.exists()


def test_backup_racing_a_deletion_leaves_no_database(app_client, monkeypatch):
    root = _managed_tenant(app_client, C)
    appmod._backup_backends.pop(C, None)
    real = appmod._get_backup_backend

    def deleted_meanwhile(user_id, path):
        # The deletion lands between the existence check and opening the
        # backend: tombstone, then the directory is removed.
        with appmod._tenant_lock:
            appmod._mark_tenant_deleted_locked(user_id)
        appmod.shutil.rmtree(path, ignore_errors=True)
        return real(user_id, path)  # opening recreates folder + database

    monkeypatch.setattr(appmod, "_get_backup_backend", deleted_meanwhile)
    r = app_client.post("/api/storage-backup", headers={"X-Cowork-User-ID": C})
    assert r.status_code == 410
    assert not root.exists()
    assert C not in appmod._backup_backends


def test_backup_for_a_live_tenant_still_reports_status(app_client):
    _managed_tenant(app_client, C)
    r = app_client.post("/api/storage-backup", headers={"X-Cowork-User-ID": C})
    assert r.status_code == 200 and r.json()["mode"] == "managed"
