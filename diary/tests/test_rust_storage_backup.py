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
    root = tmp_path / 'users' / UID
    backend = ManagedCorpusBackend(root, UID)
    backend.activate({'a ü!@.md': b'synthetic one', 'b.md': b'synthetic two'}, {}, ['empty'])
    with backend.db() as db:
        db.execute("UPDATE meta SET value='0' WHERE key='due'")
    state = {'objects': {}, 'status': 0, 'partial': False, 'calls': 0, 'entered': threading.Event(), 'release': threading.Event(), 'block': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, code, body=b''):
            self.send_response(code)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_MKCOL(self):
            self.answer(201)

        def do_GET(self):
            state['calls'] += 1
            if state['status'] or (state['partial'] and len(state['objects']) >= 1 and self.path not in state['objects']):
                self.answer(state['status'] or 503, b'synthetic reflected secret')
            elif self.path in state['objects']:
                self.answer(200, state['objects'][self.path])
            else:
                self.answer(404)

        def do_PUT(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            if state['block']:
                state['entered'].set()
                assert state['release'].wait(10)
            state['objects'][self.path] = body
            self.answer(201)

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
            reply = httpx.post(base + '/api/storage-backup', headers=headers, timeout=20)
            return reply.status_code, reply.json()
        req = Request({'type': 'http', 'headers': [(k.encode(), v.encode()) for k, v in headers.items()]})
        try:
            return 200, appmod.api_storage_backup(req)
        except HTTPException as exc:
            return exc.status_code, {'detail': exc.detail}

    yield backend, state, storage, call, base
    state['release'].set()
    proc.terminate()
    proc.wait(timeout=10)
    dav.shutdown()
    thread.join(timeout=5)
    appmod._backup_backends.clear()


def reset(backend, state):
    with backend.db() as db:
        db.execute('DELETE FROM backups')
    state['objects'].clear()
    from agent import storage_backoff
    storage_backoff._gates.clear()


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
