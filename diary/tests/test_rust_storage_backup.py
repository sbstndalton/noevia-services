"""Real Python handler vs native HTTP, synthetic tenants and loopback DAV only."""
import base64
import fcntl
import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

import agent.app as appmod
from agent.config import Config
from agent.managed_storage import ManagedCorpusBackend

UID = '12345678-1234-4234-8234-123456789abc'
BIN = os.environ.get('DIARY_RUST_SERVER_BIN', '')
pytestmark = pytest.mark.skipif(not BIN, reason='native server required')


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.delenv('DIARY_TENANT_KEY', raising=False)
    monkeypatch.setattr(appmod, '_base_cfg', Config({'retrieval': {'db_path': str(tmp_path / 'index.db')}}))
    monkeypatch.setattr(appmod, 'check_auth', lambda _: True)
    monkeypatch.setattr(appmod, '_deleted_tenants', {})
    appmod._backup_backends.clear()
    from agent import storage_backoff, tenant_writer
    if hasattr(storage_backoff, 'configure_shared'):
        storage_backoff.configure_shared(tmp_path / 'storage-gates.db')
    if hasattr(tenant_writer, 'configure_data_root'):
        tenant_writer.configure_data_root(tmp_path)
    root = tmp_path / 'users' / UID
    backend = ManagedCorpusBackend(root, UID)
    backend.activate({'a ü!@.md': b'synthetic one', 'b.md': b'synthetic two'}, {}, ['empty'])
    with backend.db() as db:
        db.execute("UPDATE meta SET value='0' WHERE key='due'")
    state = {'objects': {}, 'status': 0, 'partial': False, 'calls': 0, 'entered': threading.Event(), 'release': threading.Event(), 'block': False, 'mutations': 0, 'delay_manifest': False, 'drop_reply': False, 'committed': threading.Event(), 'slow_read': None, 'slow_used': False, 'hold_get_objects': None, 'get_entered': threading.Event(), 'get_release': threading.Event(), 'get_held': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, code, body=b''):
            self.send_response(code)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_MKCOL(self):
            state['mutations'] += 1
            self.answer(201)

        def do_GET(self):
            state['calls'] += 1
            if state['hold_get_objects'] is not None and len(state['objects']) == state['hold_get_objects'] and not state['get_held']:
                state['get_held'] = True
                state['get_entered'].set()
                assert state['get_release'].wait(12)
            if state['slow_read'] and not state['slow_used']:
                state['slow_used'] = True
                body = b'synthetic one'
                mode = state['slow_read']
                state['entered'].set()
                if mode == 'headers':
                    assert state['release'].wait(25)
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    if mode == 'body':
                        self.wfile.write(body[:1])
                        self.wfile.flush()
                        assert state['release'].wait(25)
                        self.wfile.write(body[1:])
                    elif mode == 'trickle':
                        for start in (0, 4, 8, 12):
                            if start:
                                time.sleep(6)
                            self.wfile.write(body[start:start + 4])
                            self.wfile.flush()
                        state['objects'][self.path] = body
                    else:
                        self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if state['status'] or (state['partial'] and len(state['objects']) >= 1 and self.path not in state['objects']):
                self.answer(state['status'] or 503, b'synthetic reflected secret')
            elif self.path in state['objects']:
                self.answer(200, state['objects'][self.path])
            else:
                self.answer(404)

        def do_PUT(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            state['mutations'] += 1
            if state['drop_reply'] or (state['delay_manifest'] and '/manifests/' in self.path):
                state['entered'].set()
                assert state['release'].wait(25)
                state['objects'][self.path] = body
                state['committed'].set()
                if state['drop_reply']:
                    self.close_connection = True
                    return
            if state['block']:
                state['entered'].set()
                assert state['release'].wait(25)
            state['objects'][self.path] = body
            self.answer(state.get('put_status', 201))

    dav = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=dav.serve_forever, daemon=True)
    thread.start()
    storage = {'kind': 'webdav', 'baseUrl': f'http://127.0.0.1:{dav.server_port}/dav/',
               'username': 'synthetic', 'secret': 'synthetic', 'corpusRoot': 'Diary'}
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.close()
    cfg = {'port': port, 'control_token': 'ctl', 'forward_secret': 'forward', 'data_root': str(tmp_path),
           'corpus': {}, 'llm': {'main': {'base_url': 'http://127.0.0.1:9', 'setting_key': 'MAIN', 'fallback_setting_key': 'FALLBACK'},
                                'aux': {'base_url': 'http://127.0.0.1:9', 'setting_key': 'AUX', 'fallback_setting_key': 'FALLBACK'}}}
    proc = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    proc.stdin.write(json.dumps(cfg).encode() + b'\n')
    proc.stdin.flush()
    base = f'http://127.0.0.1:{port}'
    for _ in range(200):
        try:
            if httpx.get(base + '/internal/ready', headers={'x-noevia-diary-control': 'ctl'}, timeout=1).status_code == 200:
                break
        except httpx.TransportError:
            pass
        if proc.poll() is not None:
            raise AssertionError(proc.stderr.read().decode())
        time.sleep(.02)
    else:
        raise AssertionError('native server unavailable')

    def call(native):
        descriptor = base64.urlsafe_b64encode(json.dumps(storage).encode()).decode().rstrip('=')
        headers = {'x-cowork-user-id': UID, 'x-cowork-storage': descriptor,
                   'x-noevia-diary-forward': 'forward', 'x-noevia-diary-writer': '1'}
        if native:
            reply = httpx.post(base + '/api/storage-backup', headers=headers, timeout=30)
            return reply.status_code, reply.json()
        req = Request({'type': 'http', 'headers': [(k.encode(), v.encode()) for k, v in headers.items()]})
        try:
            return 200, appmod.api_storage_backup(req)
        except HTTPException as exc:
            return exc.status_code, {'detail': exc.detail}

    state['proc'] = proc
    state['cfg'] = cfg
    state['data_root'] = tmp_path
    yield backend, state, storage, call, base
    state['get_release'].set()
    state['release'].set()
    proc.terminate()
    proc.wait(timeout=10)
    dav.shutdown()
    thread.join(timeout=5)
    appmod._backup_backends.clear()
    if hasattr(storage_backoff, 'configure_shared'):
        storage_backoff.configure_shared(None)
    if hasattr(tenant_writer, 'configure_data_root'):
        tenant_writer.configure_data_root(None)


def reset(backend, state):
    with backend.db() as db:
        db.execute('DELETE FROM backups')
    state['objects'].clear()
    from agent import storage_backoff
    storage_backoff._gates.clear()
    shared = backend.root.parent.parent / 'storage-gates.db'
    if shared.exists():
        import sqlite3
        with sqlite3.connect(shared) as db:
            db.execute('DELETE FROM storage_gates')


@pytest.mark.parametrize('mode', ['success', '401', '403', '500', '503', '429', '302', 'partial', 'locked', 'missing'])
def test_handler_parity(lab, mode):
    backend, state, storage, call, _ = lab
    state['status'] = int(mode) if mode.isdigit() else 0
    state['partial'] = mode == 'partial'
    if mode == 'missing':
        storage.pop('secret')
        storage['secretRef'] = 'a' * 32
    with (backend.root / 'managed-backup.lock').open('a') as held:
        if mode == 'locked':
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results, objects = [], []
        for native in (False, True):
            reset(backend, state)
            status, body = call(native)
            if body.get('lastBackedUp') is not None:
                body['lastBackedUp'] = 'clock'
            if body.get('retryAt') is not None:
                body['retryAt'] = 'clock'
            results.append((status, body))
            objects.append(dict(state['objects']))
        assert results[0] == results[1]
        assert objects[0] == objects[1]
        if mode == 'success':
            assert results[1][1]['storageContacted'] is True
            assert len(objects[1]) == 3
        if mode in ('401', '403'):
            assert results[1][1]['loginRefused'] is True
        assert 'synthetic reflected secret' not in json.dumps(results)


def test_fenced_tenant(lab):
    _, _, _, call, base = lab
    appmod._deleted_tenants[UID] = time.monotonic()
    assert httpx.post(base + '/internal/fence?tenant=' + UID, headers={'x-noevia-diary-control': 'ctl'}).status_code == 200
    assert call(False) == call(True) == (410, {'detail': 'tenant deleted'})


def test_disconnect_retains_flock_until_put_settles(lab):
    backend, state, storage, _, base = lab
    state['block'] = True
    descriptor = base64.urlsafe_b64encode(json.dumps(storage).encode()).decode().rstrip('=')
    url = httpx.URL(base)
    conn = socket.create_connection((url.host, url.port), timeout=3)
    conn.sendall(('POST /api/storage-backup HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n'
                  'X-Noevia-Diary-Forward: forward\r\nX-Noevia-Diary-Writer: 1\r\n'
                  f'X-Cowork-User-ID: {UID}\r\nX-Cowork-Storage: {descriptor}\r\n\r\n').encode())
    assert state['entered'].wait(5)
    conn.close()
    with (backend.root / 'managed-backup.lock').open('a') as held:
        with pytest.raises(BlockingIOError):
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state['release'].set()
        for _ in range(300):
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.01)
        else:
            raise AssertionError('backup never settled')
        assert len(state['objects']) == 3


def test_login_cooldown_and_changed_secret(lab):
    backend, state, storage, call, _ = lab
    state['status'] = 401
    assert call(True)[1]['loginRefused'] is True
    calls = state['calls']
    with backend.db() as db:
        db.execute('DELETE FROM backups')
    assert 'loginRefused' not in call(True)[1]
    assert state['calls'] == calls
    storage['secret'] = 'synthetic changed'
    with backend.db() as db:
        db.execute('DELETE FROM backups')
    assert call(True)[1]['loginRefused'] is True
    assert state['calls'] > calls


@pytest.mark.parametrize('reference', [True, 1, -1, [1], {'x': 1}, False, 0, [], {}, None, ''])
def test_secret_ref_python_truthiness(lab, reference):
    backend, state, storage, call, _ = lab
    storage.pop('secret')
    storage['secretRef'] = reference
    replies = []
    for native in (False, True):
        reset(backend, state)
        before = state['calls'] + state['mutations']
        status, body = call(native)
        replies.append((status, body.get('detail'), body.get('storageContacted')))
        if reference:
            assert state['calls'] + state['mutations'] == before
    assert replies[0] == replies[1]
    assert replies[0][0] == (428 if reference else 200)


@pytest.mark.parametrize('status', [401, 429])
@pytest.mark.parametrize('native_first', [False, True])
def test_shared_credential_cooldown(lab, status, native_first):
    from agent.webdav import WebDAVCorpusBackend
    backend, state, storage, call, _ = lab
    state['status'] = status
    assert call(native_first)[1]['backup'] == 'failed'
    before = state['calls']
    with backend.db() as db:
        db.execute('DELETE FROM backups')
    # One credential, even when spelling the collection with no trailing slash.
    storage['baseUrl'] = storage['baseUrl'].rstrip('/')
    assert call(not native_first)[1]['backup'] == 'failed'
    assert state['calls'] == before
    remote = WebDAVCorpusBackend(storage['baseUrl'], storage['username'], storage['secret'])
    try:
        assert remote.storage_gate.reset_login() is (status == 401)
        with backend.db() as db:
            db.execute('DELETE FROM backups')
        call(True)
        assert (state['calls'] > before) is (status == 401)
        if status == 401:
            assert remote.storage_gate.reset_login() is False  # shared minimum-reset interval
        gate_bytes = (backend.root.parent.parent / 'storage-gates.db').read_bytes()
        assert storage['secret'].encode() not in gate_bytes
        assert storage['baseUrl'].encode() not in gate_bytes
    finally:
        remote.close()


def assert_durable_fence(lab):
    backend, _, _, call, base = lab
    assert call(False)[0] == 503
    assert backend.db_path.exists()
    with pytest.raises(HTTPException) as failure:
        appmod._delete_tenant(UID)
    assert failure.value.status_code == 503
    assert backend.db_path.exists()


def test_manifest_commits_after_native_timeout_stays_fenced(lab):
    import concurrent.futures
    backend, state, _, call, _ = lab
    state['delay_manifest'] = True
    with concurrent.futures.ThreadPoolExecutor() as pool:
        request = pool.submit(call, True)
        assert state['entered'].wait(5)
        # Real native 15-second transport timeout, not a shortened mock timeout.
        result = request.result(timeout=20)
        try:
            assert result[0] == 503
            with (backend.root / 'managed-backup.lock').open('a') as held:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_durable_fence(lab)
        finally:
            state['release'].set()
        assert state['committed'].wait(5)
        # Remote commit/readback is not a receipt for the original uncertain operation.
        assert_durable_fence(lab)
        state['proc'].kill()
        state['proc'].wait(timeout=5)
        assert_durable_fence(lab)


def test_unrelated_forward_failure_kills_child_but_retains_effect_fence(lab, monkeypatch):
    import asyncio
    import concurrent.futures
    from agent import rust_forward
    backend, state, _, call, base = lab
    state['block'] = True
    with concurrent.futures.ThreadPoolExecutor() as pool:
        request = pool.submit(call, True)
        assert state['entered'].wait(5)
        server = rust_forward.RustServer(lambda *args: {}, port=httpx.URL(base).port, restart_delays=())
        server.ready, server.forward_secret, server._proc = True, 'forward', state['proc']
        async def fail_unrelated_headerless_read():
            async def fail(req):
                raise httpx.ReadTimeout('synthetic timeout')
            middleware = rust_forward.ForwardMiddleware(None, routes={})
            middleware._client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
            middleware._client_loop = asyncio.get_running_loop()
            await middleware._forward(server, {'type': 'http', 'path': '/api/months', 'method': 'GET', 'headers': []}, time.monotonic() + 2)
            await middleware._client.aclose()
        try:
            asyncio.run(fail_unrelated_headerless_read())
            state['proc'].wait(timeout=5)
            assert_durable_fence(lab)
            # Even after switching off writer selectors, fallback cannot replay/delete.
            from agent import tenant_writer
            tenant_writer.configure(frozenset())
            assert_durable_fence(lab)
        finally:
            state['release'].set()
            try:
                request.result(timeout=5)
            except httpx.TransportError:
                pass
        assert_durable_fence(lab)


@pytest.mark.parametrize('phase', ['headers', 'body', 'trickle'])
def test_read_timeout_is_inactivity_not_total_response_time(lab, phase):
    backend, state, _, call, _ = lab
    state['slow_read'] = phase
    started = time.monotonic()
    status, body = call(True)
    elapsed = time.monotonic() - started
    assert status == 200
    if phase == 'trickle':
        assert elapsed >= 17
        assert body['storageContacted'] is True
    else:
        assert 14 <= elapsed < 23
        assert body['backup'] == 'failed'
        from agent.backup_effect import pending
        assert not pending(backend.root)  # no mutation was dispatched by a stalled GET
    state['release'].set()


@pytest.mark.parametrize('reply', ['disconnect', 202, 500])
def test_nonterminal_mutation_reply_never_retires_intent(lab, reply):
    _, state, _, call, _ = lab
    if reply == 'disconnect':
        state['drop_reply'] = True
        state['release'].set()
    else:
        state['put_status'] = reply
    assert call(True)[0] == 503
    assert_durable_fence(lab)
    state['proc'].kill()
    state['proc'].wait(timeout=5)
    assert_durable_fence(lab)
    restart_native(lab)



def restart_native(lab):
    _, state, _, call, base = lab
    proc = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    proc.stdin.write(json.dumps(state['cfg']).encode() + b'\n')
    proc.stdin.flush()
    state['proc'] = proc
    for _ in range(200):
        try:
            if httpx.get(base + '/internal/ready', headers={'x-noevia-diary-control': 'ctl'}, timeout=1).status_code == 200:
                break
        except httpx.TransportError:
            pass
        time.sleep(.02)
    else:
        raise AssertionError('restarted native server unavailable')
    assert call(True)[0] == 503
    reply = httpx.put(base + '/api/file', headers={'x-noevia-diary-forward': 'forward',
                     'x-noevia-diary-writer': '1', 'x-cowork-user-id': UID}, json={'path': 'later.md', 'content': 'synthetic', 'version': None})
    assert reply.status_code == 503
    assert httpx.post(base + '/internal/fence?tenant=' + UID, headers={'x-noevia-diary-control': 'ctl'}).status_code == 503
