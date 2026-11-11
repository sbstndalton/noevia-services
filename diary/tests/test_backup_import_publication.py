"""Actual import child paused at EOF, after admission but before BEGIN IMMEDIATE.

The fake DAV gate is placed before the first intent or after a terminal object
reply. Caller cancellation frees Python admission while the native task survives.
No fake import executor or pre-launch check stands in for its real transaction.
"""
import asyncio
import base64
import json
import os
import subprocess
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

import agent.app as appmod
from agent import rust_forward, tenant_writer, workspace_import, workspace_import_native
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from tests.test_rust_storage_backup import lab, UID, BIN  # noqa: F401
from tests.test_rust_forward_d5 import asgi

HELPER = os.environ.get('WORKSPACE_IMPORT_APPLY_TEST_BIN', '')
pytestmark = pytest.mark.skipif(not (BIN and HELPER), reason='actual server/import helper required')


@pytest.mark.parametrize('gap', [0, 1], ids=['initial-get', 'terminal-next-intent'])
@pytest.mark.parametrize('child_loss', [False, True])
def test_admitted_import_cannot_publish_after_late_uncertain_effect(lab, tmp_path, monkeypatch, gap, child_loss):
    backend, state, storage, _, base = lab
    source = ManagedCorpusBackend(tmp_path / 'source', '00000000-0000-0000-0000-000000000001')
    source.activate({'synthetic.md': b'import must remain uncommitted'}, {})
    body = archive(source, '', {})
    report = workspace_import.prepare(backend, body, 'Copy')[2]
    monkeypatch.setattr(tenant_writer, '_writers', frozenset({UID}))
    monkeypatch.setenv('WORKSPACE_IMPORT_APPLY_IMPL', 'rust')
    monkeypatch.setattr(workspace_import_native, 'DEFAULT_BIN', HELPER)
    fake_state = SimpleNamespace(backend=backend, managed=backend, recovered=True,
                   store=SimpleNamespace(_write_lock=tenant_writer.TenantWriteLock(UID), backend=backend),
                   journal=SimpleNamespace(pending_count=lambda: 0, mark_dirty=lambda _: None))
    monkeypatch.setattr(appmod, '_tenant_state', lambda request, recover=False: fake_state)
    server = rust_forward.RustServer(lambda *args: {}, port=httpx.URL(base).port, restart_delays=())
    server.ready, server.forward_secret, server._token, server._proc = True, 'forward', 'ctl', state['proc']
    state['hold_get_objects'] = gap
    admitted, resume = threading.Event(), threading.Event()
    children = []
    popen = subprocess.Popen

    class HeldInput:
        def __init__(self, stream):
            self.stream, self.paused = stream, False
        def __getattr__(self, key):
            return getattr(self.stream, key)
        def close(self):
            if not self.paused:
                self.paused = True
                admitted.set()
                assert resume.wait(10)
            self.stream.close()

    def launch(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        child.stdin = HeldInput(child.stdin)
        return child
    monkeypatch.setattr(workspace_import_native.subprocess, 'Popen', launch)

    async def run():
        mw = rust_forward.ForwardMiddleware(None, server=server, routes={}, writers=frozenset({UID}))
        request = asyncio.create_task(asgi(mw, [(b'x-cowork-user-id', UID.encode()), (b'content-length', b'0'),
                    (b'x-cowork-storage', base64.urlsafe_b64encode(json.dumps(storage).encode()).rstrip(b'='))],
                                           [b''], method='POST', path='/api/storage-backup'))
        importing = None
        try:
            assert await asyncio.to_thread(state['get_entered'].wait, 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            with backend.db() as db:
                assert not db.execute("SELECT 1 FROM meta WHERE key='backup_effect_pending'").fetchone()
            sent = False
            async def receive():
                nonlocal sent
                if sent:
                    return {'type':'http.disconnect'}
                sent = True
                return {'type':'http.request', 'body':body, 'more_body':False}
            scope = {'type':'http', 'method':'POST', 'path':'/api/workspace-import',
                     'query_string':('action=apply&name=Copy&fingerprint=' + report['fingerprint']).encode(),
                     'headers':[(b'x-cowork-user-id', UID.encode())]}
            importing = asyncio.create_task(appmod.api_workspace_import(Request(scope, receive)))
            assert await asyncio.to_thread(admitted.wait, 5)
            assert children and children[0].poll() is None
            # Helper has the real verified frame but cannot open its transaction until EOF.
            state['block'] = True
            state['get_release'].set()
            assert await asyncio.to_thread(state['entered'].wait, 5)
            with backend.db() as db:
                assert db.execute("SELECT 1 FROM meta WHERE key='backup_effect_pending'").fetchone()
            if child_loss:
                async def fail(_):
                    raise httpx.ReadTimeout('synthetic unrelated header-less read failure')
                other = rust_forward.ForwardMiddleware(None, routes={})
                other._client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
                other._client_loop = asyncio.get_running_loop()
                try:
                    await other._forward(server, {'type':'http','method':'GET','path':'/api/months','headers':[]},
                                         __import__('time').monotonic() + 2)
                finally:
                    await other._client.aclose()
                await asyncio.to_thread(state['proc'].wait, timeout=5)
            resume.set()
            with pytest.raises(HTTPException) as refusal:
                await asyncio.wait_for(importing, 6)
            assert refusal.value.status_code == 503
            assert children[0].poll() == 3
            with backend.db() as db:
                assert db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0] == 0
                assert db.execute('SELECT count(*) FROM workspace_index_outbox').fetchone()[0] == 0
                assert not db.execute("SELECT 1 FROM files WHERE path LIKE 'Imports/%'").fetchone()
                assert backend.meta(db, 'generation') == '1'
        finally:
            resume.set()
            state['get_release'].set()
            state['release'].set()
            if not request.done():
                request.cancel()
                try:
                    await request
                except asyncio.CancelledError:
                    pass
            if importing is not None and not importing.done():
                try:
                    await asyncio.wait_for(importing, 7)
                except HTTPException:
                    pass
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
    asyncio.run(run())
