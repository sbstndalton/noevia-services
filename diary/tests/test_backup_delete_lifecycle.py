"""Deletion proves lifecycle absence or fencing before tombstoning synthetic state."""
import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import agent.app as appmod
from agent import rust_forward, tenant_writer
from agent.config import Config
from agent.managed_storage import ManagedCorpusBackend

UID = '12345678-1234-4234-8234-123456789abc'


@pytest.fixture
def deleting(tmp_path, monkeypatch):
    cfg = Config({'retrieval': {'db_path': str(tmp_path / 'index.db')}})
    monkeypatch.setattr(appmod, '_base_cfg', cfg)
    monkeypatch.setattr(appmod, '_deleted_tenants', {})
    monkeypatch.setattr(appmod, '_tenant_states', {})
    monkeypatch.setattr(appmod, '_backup_backends', {})
    monkeypatch.setattr(appmod, 'SESSIONS', {})
    monkeypatch.setattr(tenant_writer, '_writers', frozenset())
    tenant_writer.configure_data_root(tmp_path)
    root = tmp_path / 'users' / UID
    backend = ManagedCorpusBackend(root, UID)
    server = rust_forward.RustServer(lambda *args: {}, restart_delays=())
    monkeypatch.setattr(rust_forward, '_server', server)
    yield root, backend, server
    tenant_writer.configure_data_root(None)


@pytest.mark.parametrize('state', ['never_started', 'fully_reaped'])
def test_confirmed_absence_allows_deletion(deleting, monkeypatch, state):
    root, _, server = deleting
    if state == 'never_started':
        monkeypatch.setattr(rust_forward, '_binary', lambda: None)
        assert server._spawn() is None
    else:
        server._token = 'synthetic-old-incarnation'
        server._proc = SimpleNamespace(poll=lambda: 0)
        server._mark_lost(server._proc)
    assert appmod._delete_tenant(UID).status_code == 200
    assert not root.exists()
    assert appmod._tenant_deleted(UID)


@pytest.mark.parametrize('state', ['pending', 'unreaped', 'live_unconfirmed'])
def test_uncertainty_refuses_before_tombstone(deleting, monkeypatch, state):
    root, backend, server = deleting
    if state == 'pending':
        with backend.db() as db:
            db.execute("INSERT INTO meta VALUES ('backup_effect_pending','synthetic')")
    elif state == 'unreaped':
        server.reaping = 1
    else:
        server._proc = SimpleNamespace(poll=lambda: None)
        server.ready, server._token = True, 'synthetic'
        monkeypatch.setattr(server, 'fence', lambda *args, **kwargs: False)
    try:
        with pytest.raises(HTTPException) as failure:
            appmod._delete_tenant(UID)
        assert failure.value.status_code == 503
        assert root.exists()
        assert not appmod._tenant_deleted(UID)
    finally:
        server.reaping = 0


def test_spawn_in_progress_is_not_confirmed_absence(deleting, monkeypatch):
    root, _, server = deleting
    entered, release = threading.Event(), threading.Event()
    def config(*args):
        entered.set()
        assert release.wait(5)
        raise OSError('synthetic no process was created')
    server._config_factory = config
    server._binary = '/synthetic/never-executed'
    thread = threading.Thread(target=server._spawn)
    thread.start()
    assert entered.wait(2)
    try:
        with pytest.raises(HTTPException):
            appmod._delete_tenant(UID)
        assert root.exists() and not appmod._tenant_deleted(UID)
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert appmod._delete_tenant(UID).status_code == 200


def test_deleted_reservation_blocks_a_later_incarnation(deleting):
    _, _, server = deleting
    assert appmod._delete_tenant(UID).status_code == 200
    server.ready, server.forward_secret, server._token = True, 'new-secret', 'new-token'
    calls = []
    async def check():
        async def respond(request):
            calls.append(request)
            return httpx.Response(200, content=b'{}')
        mw = rust_forward.ForwardMiddleware(None, routes={})
        mw._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        mw._client_loop = asyncio.get_running_loop()
        try:
            result = await mw._forward(server, {'method':'POST','path':'/api/storage-backup',
                    'headers':[(b'x-cowork-user-id',UID.encode())]}, __import__('time').monotonic()+2, writer=True)
            assert result is None
            assert calls == []
        finally:
            await mw._client.aclose()
    asyncio.run(check())


def test_failed_config_feed_reaps_owned_child_before_absence(deleting, monkeypatch):
    _, _, server = deleting
    lifecycle = {'exit':None, 'killed':False}
    def broken(_):
        raise BrokenPipeError('synthetic config channel loss')
    def kill():
        lifecycle['killed'], lifecycle['exit'] = True, -9
    child = SimpleNamespace(stdin=SimpleNamespace(write=broken, close=lambda: None),
                            poll=lambda: lifecycle['exit'], kill=kill, wait=lambda timeout=None: lifecycle['exit'])
    server._binary = '/synthetic/not-executed'
    monkeypatch.setattr(rust_forward.subprocess, 'Popen', lambda *a, **k: child)
    assert server._spawn() is None
    assert lifecycle['killed'] and server.reaping == 0 and server._proc is None
    assert appmod._delete_tenant(UID).status_code == 200
