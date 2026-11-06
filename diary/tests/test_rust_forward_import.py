"""Real-handler byte differential and lifecycle tests, synthetic archives only."""
import asyncio
import hashlib
import io
import json
import sqlite3
import zipfile
from urllib.parse import urlencode

import pytest
from agent import rust_forward, tenant_writer
import agent.app as appmod
from tests.tenant_binary import needs_binary
from tests.test_rust_forward import DAILY, MONTHLY, world, needs_server  # noqa: F401
from tests.test_rust_forward_d4b import sign
from tests.test_rust_forward_d5 import (writers, _writer_registry, Recorder, wire, snapshot, restore, storage,
                                        FakeChild, forwarder, asgi, put)  # noqa: F401


def archive(entries=None):
    data = b'synthetic imported note\r\n'
    manifest = {'format': 'noevia-workspace-v1', 'files': [{'path': 'note.md', 'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest()}], 'directories': ['empty'], 'fileCount': 1, 'bytes': len(data), 'settings': {}}
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for name, value in (entries if entries is not None else [('manifest.json', json.dumps(manifest)),
                ('workspace/', b''), ('workspace/empty/', b''), ('workspace/note.md', data)]):
            z.writestr(name, value)
    return out.getvalue()


def request(client, body, action='preview', name='Copy', fingerprint=None):
    query = urlencode({'action': action, 'name': name, **({'fingerprint': fingerprint} if fingerprint else {})}).encode()
    path = '/api/workspace-import'
    return client.post(path + '?' + query.decode(), content=body,
        headers=sign(DAILY, 'POST', path, body, query=query, content_type='application/zip'))


@needs_binary
@needs_server
@pytest.mark.parametrize('body,action,name,status', [
    (b'bad zip', 'preview', 'Copy', 400),
    (b'bad zip', 'apply', 'Copy', 400),
    (archive([('../escape', b'x')]), 'preview', 'Copy', 400),
    (archive([('same', b'x'), ('same', b'y')]), 'preview', 'Copy', 400),
    (archive(), 'preview', '../bad', 400),
    (b'x' * (32*1024*1024+1), 'preview', 'Copy', 413),
    (b'x' * (65*1024*1024), 'preview', 'Copy', 413),
    (archive(), 'preview', 'Copy', 200),
])
def test_import_real_python_byte_parity(writers, body, action, name, status):
    client, start, state = writers
    modes = start()
    snap = snapshot(state, DAILY)
    modes.python()
    p = request(client, body, action, name)
    before = storage(state, DAILY)
    restore(state, DAILY, snap)
    modes.rust(); Recorder.served = []
    r = request(client, body, action, name)
    assert wire(p) == wire(r), (p.content, r.content)
    assert r.status_code == status
    assert storage(state, DAILY) == before
    assert Recorder.served == [status == 200 or len(body) > 32*1024*1024 or action == "apply" or name == "../bad"]


@needs_binary
@needs_server
def test_preview_apply_receipt_outbox_and_retry(writers):
    client, start, state = writers
    modes = start(); body = archive()
    snap = snapshot(state, DAILY)
    def run(native):
        (modes.rust if native else modes.python)()
        Recorder.served = []
        p = request(client, body)
        r = request(client, body, 'apply', fingerprint=p.json()['fingerprint'])
        assert r.status_code == 200, r.text
        with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
            receipt = db.execute('SELECT * FROM workspace_imports').fetchall()
            outbox = db.execute('SELECT * FROM workspace_index_outbox').fetchall()
        with sqlite3.connect(state / 'users' / DAILY / 'managed-index.db') as db:
            dirty = db.execute('SELECT * FROM dirty_documents').fetchall()
        return wire(p), wire(r), storage(state, DAILY), receipt, outbox, dirty
    p = run(False); restore(state, DAILY, snap)
    # Python apply marks the cached state unrecovered; restore the fixture's initial state too.
    for st in appmod._tenant_states.values(): st.recovered = True
    r = run(True)
    assert Recorder.served == [True, True]
    assert p == r
    assert not appmod._rust_recovered(DAILY)
    # Recover via Python then repeat the reviewed request: the durable receipt avoids a second apply.
    modes.python()
    client.get('/api/months', headers=sign(DAILY, 'GET', '/api/months'))
    modes.rust()
    retry = request(client, body, 'apply', fingerprint=json.loads(r[0][2])['fingerprint'])
    assert retry.status_code == 200 and retry.json()['alreadyApplied']


def test_import_is_a_writer_and_accepts_zip(monkeypatch):
    child = FakeChild()
    try:
        _, spy, mw = forwarder(child.port, monkeypatch)
        h, _ = put(body=b'archive')
        h = [(k, b'application/zip' if k == b'content-type' else v) for k,v in h]
        status, _ = asyncio.run(asgi(mw, h, [b'archive'], method='POST', path='/api/workspace-import'))
        assert status == 200 and not spy.calls and child.requests
    finally: child.close()


def test_import_lost_reply_is_uncertain_and_never_falls_back(monkeypatch):
    child = FakeChild('hang'); uncertain = []
    try:
        _, spy, mw = forwarder(child.port, monkeypatch, deadline=.15, uncertain=uncertain)
        h, _ = put(body=b'archive')
        status, body = asyncio.run(asgi(mw, h, [b'archive'], method='POST', path='/api/workspace-import'))
        assert status == 503 and not spy.calls
        assert json.loads(body) == {'detail': 'Workspace import outcome is uncertain. Retry the same archive and destination with the same reviewed fingerprint.'}
        assert uncertain
    finally:
        child.close()

@needs_binary
@needs_server
@pytest.mark.parametrize('case', ['fence', 'non-managed'])
def test_import_declines_before_effect_for_fenced_or_nonmanaged(writers, case):
    client, start, state = writers
    modes = start()
    body = archive()
    if case == 'fence':
        assert modes.server.fence(DAILY)
    else:
        # Deactivate the synthetic fixture and leave a real local legacy corpus, so Python
        # constructs its legacy backend and returns the route's explicit 409.
        root = state / 'users' / DAILY
        (root / 'managed-diary.active').unlink()
        with sqlite3.connect(root / 'managed-diary.db') as db:
            db.execute("UPDATE meta SET value='0' WHERE key='active'")
        (root / 'corpus').mkdir(exist_ok=True)
        (root / 'corpus' / 'legacy.md').write_text('synthetic legacy note')
        for st in appmod._tenant_states.values(): appmod._close_state(st)
        appmod._tenant_states.clear()
    modes.python(); p = request(client, body)
    modes.rust(); Recorder.served = []; r = request(client, body)
    assert wire(p) == wire(r)
    if case == 'non-managed': assert r.status_code == 409
    assert Recorder.served == [False]


@needs_binary
@needs_server
def test_simultaneous_imports_commit_one_receipt(writers):
    import concurrent.futures
    client, start, state = writers
    start(); body = archive()
    preview = request(client, body).json()
    def apply(_): return request(client, body, 'apply', fingerprint=preview['fingerprint'])
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(apply, range(2)))
    assert all(r.status_code == 200 for r in results), [r.text for r in results]
    assert sorted(r.json()['alreadyApplied'] for r in results) == [False, True]
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        assert db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0] == 1


@needs_binary
@needs_server
def test_uncertain_reply_matches_real_python_handler(writers, monkeypatch):
    from agent import workspace_import, workspace_import_native
    client, start, state = writers
    modes = start(); body = archive(); fp = request(client, body).json()['fingerprint']
    def uncertain(*a, **kw):
        raise workspace_import_native.ImportUncertain('Workspace import outcome is uncertain. Retry the same archive and destination with the same reviewed fingerprint.')
    monkeypatch.setattr(workspace_import, 'apply', uncertain)
    modes.python(); p = request(client, body, 'apply', fingerprint=fp)
    assert p.status_code == 503
    assert not appmod._rust_recovered(DAILY)
    assert p.content == b'{"detail":"Workspace import outcome is uncertain. Retry the same archive and destination with the same reviewed fingerprint."}'


def test_cancelled_import_never_falls_back_and_marks_uncertain(monkeypatch):
    child = FakeChild('hang'); uncertain = []
    try:
        _, spy, mw = forwarder(child.port, monkeypatch, deadline=5, uncertain=uncertain)
        h, body = put()
        async def cancel():
            task = asyncio.create_task(asgi(mw, h, [body], method='POST', path='/api/workspace-import'))
            for _ in range(100):
                if child.requests: break
                await asyncio.sleep(.01)
            assert child.requests
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        asyncio.run(cancel())
        assert not spy.calls and uncertain == [DAILY]
    finally: child.close()

@needs_binary
@needs_server
def test_real_child_crash_while_apply_waits_reports_uncertain(writers):
    import concurrent.futures
    import time
    from pathlib import Path
    client, start, state = writers
    modes = start(); body = archive(); fp = request(client, body).json()['fingerprint']
    db_path = state / 'users' / DAILY / 'managed-diary.db'
    db = sqlite3.connect(db_path)
    db.execute('BEGIN IMMEDIATE')
    try:
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(request, client, body, 'apply', 'Copy', fp)
            deadline = time.monotonic() + 5
            observed = False
            while time.monotonic() < deadline:
                # Native database handles are ephemeral. With the source write transaction
                # blocked, an open handle proves the child has entered the guarded operation.
                fds = Path(f'/proc/{modes.server._proc.pid}/fd')
                for fd in fds.iterdir():
                    try: observed |= fd.resolve() == db_path
                    except OSError: pass
                if observed: break
                time.sleep(.01)
            try:
                assert observed
                modes.server._proc.kill()
                r = future.result(timeout=5)
                assert r.status_code == 503 and 'outcome is uncertain' in r.text
            finally:
                db.rollback()
            assert not appmod._rust_recovered(DAILY)
    finally: db.close()
    with sqlite3.connect(db_path) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0] == 0
    modes.python()
    retry = request(client, body, 'apply', fingerprint=fp)
    assert retry.status_code == 200 and not retry.json()['alreadyApplied']

@needs_binary
@needs_server
@pytest.mark.parametrize('native_errors', [False, True])
def test_transaction_fault_classification_matches_configured_python(writers, monkeypatch, native_errors):
    import os
    from fastapi.testclient import TestClient
    from agent import workspace_import_native
    client, start, state = writers
    monkeypatch.setenv('WORKSPACE_IMPORT_APPLY_IMPL', 'rust' if native_errors else 'python')
    if native_errors:
        monkeypatch.setattr(workspace_import_native, 'DEFAULT_BIN', os.environ['WORKSPACE_IMPORT_APPLY_TEST_BIN'])
    modes = start(); body = archive(); fp = request(client, body).json()['fingerprint']
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        db.execute("CREATE TRIGGER synthetic_import_failure BEFORE INSERT ON directories BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
    client = TestClient(appmod.app, raise_server_exceptions=False)
    modes.python(); p = request(client, body, 'apply', fingerprint=fp)
    for st in appmod._tenant_states.values(): st.recovered = True
    modes.rust(); Recorder.served = []; r = request(client, body, 'apply', fingerprint=fp)
    assert wire(p) == wire(r), (p.content, r.content)
    assert r.status_code == (503 if native_errors else 500)
    assert Recorder.served == [True]
    assert not appmod._rust_recovered(DAILY)
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        assert db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0] == 0
