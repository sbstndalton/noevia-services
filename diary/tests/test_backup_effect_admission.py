"""Durable uncertainty is checked before admission, independently of a child or selectors."""
import asyncio
import json
import sqlite3
import pytest
from agent import tenant_writer, rust_forward, backup_effect
from agent.managed_storage import ManagedCorpusBackend
from tests.test_rust_forward_d5 import SpyApp, asgi

UID = '12345678-1234-4234-8234-123456789abc'


@pytest.mark.parametrize('writers', [frozenset(), frozenset({UID})])
@pytest.mark.parametrize('path,method', [('/api/file','PUT'),('/api/workspace-ops','POST'),('/api/workspace-import','POST'),('/api/internal/tenant','DELETE')])
@pytest.mark.parametrize('headerless', [False, True])
def test_all_fallback_admission_is_fenced(tmp_path, monkeypatch, writers, path, method, headerless):
    root = tmp_path / 'users' / UID
    backend = ManagedCorpusBackend(root, UID)
    tenant_writer.configure_data_root(tmp_path)
    monkeypatch.setattr(tenant_writer, '_writers', writers)
    monkeypatch.setenv('DIARY_LEGACY_USER_ID', UID)
    try:
        with sqlite3.connect(backend.db_path) as db:
            db.execute("INSERT INTO meta VALUES ('backup_effect_pending',?)", (json.dumps({'id':'synthetic','method':'PUT','path':'synthetic'}),))
        spy = SpyApp()
        middleware = rust_forward.ForwardMiddleware(spy, routes={}, writers=writers)
        headers = [] if headerless else [(b'x-cowork-user-id',UID.encode())]
        reply = asyncio.run(asgi(middleware, headers, [b'{}'], method=method, path=path))
        assert reply[0] == 503
        assert spy.calls == []
        with pytest.raises(tenant_writer.WriterBusy):
            tenant_writer.acquire(UID, deadline_s=.01)
        assert backup_effect.pending(root)
    finally:
        tenant_writer.configure_data_root(None)
