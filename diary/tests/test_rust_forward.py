"""D4: the Rust Diary server (noevia-rs bins/noevia-diary) behind DIARY_RUST_ROUTES.

The parity cases run every request twice on the same synthetic state: once served by Python alone
(DIARY_RUST_ROUTES empty, today's behaviour) and once through the forwarder with the Rust process,
and compare status, content type and body byte for byte. Each side has its own nonce cache, so each
side gets its own freshly signed request. They need the real binaries (DIARY_RUST_SERVER_BIN and the
tenant-assertion leaf); the routing and fallback cases without a server binary always run.
"""
import base64
import datetime
import json
import os
import secrets
import socket
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agent.app as appmod
from agent import rust_forward
from agent import tenant_assertion as ta
from agent.config import Config
from agent.managed_storage import ManagedCorpusBackend
from tests.tenant_binary import needs_binary

KEY = 'synthetic-tenant-key-d4'
TOKEN = 'synthetic-service-token-d4'
DAILY = '0b5f6c1e-3a2d-4c8e-9f10-1a2b3c4d5e6f'
MONTHLY = '1c6f7d2f-4b3e-4d9f-8a21-2b3c4d5e6f70'
OTHER = '2d708e30-5c4f-4ea0-9b32-3c4d5e6f7081'
BIN = os.environ.get('DIARY_RUST_SERVER_BIN', '')
needs_server = pytest.mark.skipif(not (BIN and os.path.isfile(BIN)) and os.environ.get('DIARY_RUST_SERVER_REQUIRED') != '1',
                                  reason='noevia-diary not built (set DIARY_RUST_SERVER_BIN)')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def b64(obj):
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip('=')


def entry(day, text, xid):
    return (f"## {day.strftime('%A')}, {day.strftime('%B')} {day.day}, {day.year}\n\n### 09:15 — synthetic\n\n"
            f"**Me:** {text}\n\n**Assistant:** Noted.\n\n<!-- xid:{xid} -->\n")


def daily_path(day):
    return f"Entries/{day.year}/{day.strftime('%B')}/{day.strftime('%B')} {day.day}, {day.year}.md"


@pytest.fixture
def world(tmp_path, monkeypatch):
    cfg = Config({
        'corpus': {'backend': 'local', 'root': '', 'monthly_prefix': '', 'entry_layout': 'daily',
                   'entries_prefix': 'Entries', 'index_enabled': False, 'month_file_template': '{year}-{month02}.md',
                   'index_file': 'INDEX.md', 'local': {'root': str(tmp_path / 'default')}},
        'retrieval': {'db_path': str(tmp_path / 'state' / 'index.db')},
        'llm': {'base_url': 'http://127.0.0.1:9/v1', 'chat_model': 'chat-a', 'embed_model': 'e',
                'aux': {'base_url': 'http://127.0.0.1:9/v1', 'model': 'aux-a'}},
        'ui': {'auth_token': TOKEN},
    })
    (tmp_path / 'state').mkdir()
    monkeypatch.setattr(appmod, '_base_cfg', cfg)
    monkeypatch.setattr(appmod, '_state', appmod.AppState(cfg))
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    monkeypatch.delenv('DIARY_LEGACY_USER_ID', raising=False)
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    monkeypatch.setattr(appmod, '_deleted_tenants', {})
    ta._reset_for_tests()
    appmod._tenant_states.clear()
    yield tmp_path / 'state'
    for state in appmod._tenant_states.values():
        appmod._close_state(state)
    appmod._tenant_states.clear()
    ta._reset_for_tests()


def signed(user, method, path, extra=None, query=b''):
    headers = {'Authorization': f'Bearer {TOKEN}', 'X-Cowork-User-ID': user, **(extra or {})}
    headers[ta.HEADER] = ta.sign(KEY, user, method, path, headers, int(time.time()), secrets.token_hex(16), query)
    return headers


def settle(state_dir, uid):
    """Dirty documents only feed the index; they must not stop native reads (left in place)."""
    with sqlite3.connect(state_dir / 'users' / uid / 'managed-index.db') as db:
        assert db.execute('SELECT COUNT(*) FROM journal WHERE applied = 0').fetchone()[0] == 0


def seed(state_dir, python):
    today = datetime.datetime.now().date()
    jan = datetime.date(2030, 1, 6)
    # A fresh account starts managed (Python activates it); a monthly one is activated directly.
    assert python.get('/api/storage-status', headers=signed(DAILY, 'GET', '/api/storage-status')).status_code == 200
    managed = ManagedCorpusBackend(state_dir / 'users' / DAILY, DAILY)
    for day, text in ((today, 'Synthetic walk by the river, “quoted” ünïcode.'), (jan, 'Synthetic January note.')):
        assert managed.put(daily_path(day), entry(day, text, '00000000-0000-4000-8000-%012d' % day.day).encode())[0]
    managed.put('Entries/2030/February/notes.txt', b'not a day file')
    monthly = ManagedCorpusBackend(state_dir / 'users' / MONTHLY, MONTHLY)
    monthly.activate({'2030-01.md': entry(jan, 'Synthetic monthly note.', '00000000-0000-4000-8000-000000000099').encode(),
                      '2029-12.md': b'## Monday, December 31, 2029\n', 'INDEX.md': b'# index'},
                     {'entry_layout': 'monthly', 'month_file_template': '{year}-{month02}.md', 'index_enabled': False})
    assert python.get('/api/storage-status', headers=signed(MONTHLY, 'GET', '/api/storage-status')).status_code == 200
    settle(state_dir, DAILY)
    settle(state_dir, MONTHLY)


class Recording(rust_forward.ForwardMiddleware):
    """Records whether the Rust side served (True) or Python did (False) each forwarded request."""
    served = []

    async def _forward(self, server, scope, body):
        reply = await super()._forward(server, scope, body)
        Recording.served.append(reply is not None)
        return reply


@pytest.fixture
def both(world):
    python = TestClient(appmod.app)
    seed(world, python)
    port = free_port()
    server = rust_forward.RustServer(lambda p, t: rust_forward.server_config(appmod._base_cfg, appmod.get_state(), p, t),
                                     port=port, binary=BIN, restart_delays=())
    server.start()
    deadline = time.monotonic() + 15
    while not server.ready and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.ready, 'noevia-diary did not become ready'
    # Assertions signed in the switch second are refused there by design (its floor); start after it.
    time.sleep(1.05 - (time.time() % 1))
    Recording.served = []
    routes = {path: 'GET' for path in rust_forward.SUPPORTED}
    rust = TestClient(Recording(appmod.app, server=server, routes=routes, is_deleted=appmod._tenant_deleted))
    yield python, rust, server
    server.stop()


def wire(resp):
    return resp.status_code, resp.headers.get('content-type'), resp.content


LONG = 'A' * 4097
BAD_REF = b64({'kind': 'webdav', 'baseUrl': 'http://offline.invalid', 'secretRef': 'NOT-HEX'})
WRONG_SECRET = b64({'kind': 'webdav', 'baseUrl': 'http://offline.invalid', 'secretRef': 'a' * 32, 'secret': 'synthetic'})
REF_ONLY = b64({'kind': 'webdav', 'baseUrl': 'http://offline.invalid', 'corpusRoot': 'D', 'secretRef': 'b' * 32})

# (name, user, path, query, extra headers, expected status, served natively by Rust?)
CASES = [
    ('months daily', DAILY, '/api/months', b'', {}, 200, True),
    ('months monthly', MONTHLY, '/api/months', b'', {}, 200, True),
    ('day today', DAILY, '/api/day', b'', {}, 200, True),
    ('day month', DAILY, '/api/day', b'month=2030-01', {}, 200, True),
    ('day month monthly', MONTHLY, '/api/day', b'month=2030-01', {}, 200, True),
    ('day month empty', DAILY, '/api/day', b'month=2031-05', {}, 200, True),
    ('day month unicode digits', DAILY, '/api/day', b'month=%D9%A2%D9%A0%D9%A3%D9%A0-%D9%A0%D9%A1', {}, 200, True),
    ('day month repeated (last wins)', DAILY, '/api/day', b'month=bad&month=2030-01', {}, 200, True),
    ('day bad month', DAILY, '/api/day', b'month=2030-1', {}, 400, True),
    ('day month out of range', DAILY, '/api/day', b'month=2030-13', {}, 400, True),
    ('non-uuid tenant', 'not-a-uuid', '/api/months', b'', {}, 400, True),
    ('storage header too large', DAILY, '/api/months', b'', {'X-Cowork-Storage': LONG}, 413, True),
    ('bad secretRef', DAILY, '/api/months', b'', {'X-Cowork-Storage': BAD_REF}, 400, True),
    ('secret does not match its ref', DAILY, '/api/months', b'', {'X-Cowork-Storage': WRONG_SECRET}, 400, True),
    ('remote ref without secret, unmanaged', OTHER, '/api/months', b'', {'X-Cowork-Storage': REF_ONLY}, 428, False),
    ('health process part', None, '/api/health', b'', {}, 200, True),
    ('health with tenant detail', DAILY, '/api/health', b'', {'X-Cowork-Storage': b64({'kind': 'local'})}, 200, False),
]


def request(client, case, sign=True):
    name, user, path, query, extra, _, _ = case
    if user is None:
        headers = {'Authorization': f'Bearer {TOKEN}', **extra}
    else:
        headers = signed(user, 'GET', path, extra, query) if sign else {'Authorization': f'Bearer {TOKEN}', 'X-Cowork-User-ID': user, **extra}
    target = path + ('?' + query.decode() if query else '')
    return client.get(target, headers=headers)


@needs_binary
@needs_server
def test_parity_table_python_vs_rust(both):
    python, rust, _ = both
    rows = []
    for case in CASES:
        Recording.served = []
        p, r = request(python, case), request(rust, case)
        rows.append((case[0], p.status_code, r.status_code, Recording.served))
        assert wire(p) == wire(r), case[0]
        assert p.status_code == case[5], case[0]
        assert Recording.served == [case[6]], (case[0], Recording.served)
    print('\n'.join(f'{n:45} python={a} rust={b} native={s}' for n, a, b, s in rows))


@needs_binary
@needs_server
def test_auth_and_assertion_refusals_match(both):
    python, rust, _ = both
    for client in (python, rust):
        Recording.served = []
        bad_token = signed(DAILY, 'GET', '/api/months', {'Authorization': 'Bearer wrong'})
        bad_token['Authorization'] = 'Bearer wrong'
        resp = client.get('/api/months', headers=bad_token)
        assert (resp.status_code, resp.json()) == (401, {'error': 'unauthorized'})
        tampered = signed(DAILY, 'GET', '/api/day')
        assert client.get('/api/months', headers=tampered).json() == {'detail': 'invalid tenant assertion'}
        once = signed(DAILY, 'GET', '/api/months')
        assert client.get('/api/months', headers=once).status_code == 200
        replay = client.get('/api/months', headers=once)
        assert (replay.status_code, replay.json()) == (401, {'detail': 'invalid tenant assertion'})
        no_tenant = client.get('/api/months', headers={'Authorization': f'Bearer {TOKEN}'})
        assert (no_tenant.status_code, no_tenant.json()) == (400, {'detail': 'missing or invalid X-Cowork-User-ID'})
        health = client.get('/api/health')
        assert (health.status_code, health.json()) == (401, {'error': 'unauthorized'})
        if client is rust:
            assert all(Recording.served), Recording.served


@needs_binary
@needs_server
def test_deleted_pending_and_broken_tenants_stay_with_python(both, world):
    python, rust, _ = both
    # 410: deleted in this process (tombstone) -> answered by Python on both paths.
    victim = OTHER
    assert python.get('/api/storage-status', headers=signed(victim, 'GET', '/api/storage-status')).status_code == 200
    settle(world, victim)
    Recording.served = []
    assert rust.get('/api/months', headers=signed(victim, 'GET', '/api/months')).status_code == 200
    assert Recording.served == [True]
    deleted = python.delete('/api/internal/tenant', headers=signed(victim, 'DELETE', '/api/internal/tenant'))
    assert deleted.status_code == 200
    for client in (python, rust):
        resp = client.get('/api/months', headers=signed(victim, 'GET', '/api/months'))
        assert (resp.status_code, resp.json()) == (410, {'detail': 'tenant deleted'})
    # A pending journal entry: Python replays it, so Rust declines.
    with sqlite3.connect(world / 'users' / DAILY / 'managed-index.db') as db:
        db.execute("INSERT INTO journal(id,created_at,kind,payload) VALUES ('synthetic','t','index_month','{}')")
    Recording.served = []
    rust.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))
    assert Recording.served == [False]
    # 503: an activation marker with the wrong identity; Python refuses it, Rust never serves it.
    (world / 'users' / MONTHLY / 'managed-diary.active').write_text(OTHER)
    appmod._tenant_states.clear()
    p = python.get('/api/months', headers=signed(MONTHLY, 'GET', '/api/months'))
    r = rust.get('/api/months', headers=signed(MONTHLY, 'GET', '/api/months'))
    assert p.status_code == 503 and wire(p) == wire(r)


@needs_binary
@needs_server
def test_process_loss_falls_back_to_python_and_raises_its_floor(both):
    _, rust, server = both
    assert rust.get('/api/months', headers=signed(DAILY, 'GET', '/api/months')).status_code == 200
    server._proc.kill()
    server._proc.wait()
    # Signed before the loss was noticed: Python refuses it (Rust may have accepted it).
    early = signed(DAILY, 'GET', '/api/months')
    deadline = time.monotonic() + 5
    while server.ready and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not server.ready
    assert rust.get('/api/months', headers=early).status_code == 401
    time.sleep(1.1)
    later = rust.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))
    assert later.status_code == 200 and later.json()['months']


def test_routes_default_to_none_and_add_no_middleware(monkeypatch):
    monkeypatch.delenv('DIARY_RUST_ROUTES', raising=False)
    assert rust_forward.configured_routes() == {}
    assert rust_forward.start(lambda p, t: {}, {}) is None
    assert not any(m.cls is rust_forward.ForwardMiddleware for m in appmod.app.user_middleware)
    assert rust_forward.configured_routes(' /api/months, /api/nope ,/api/day') == {'/api/months': 'GET', '/api/day': 'GET'}


def test_a_binary_that_cannot_start_leaves_python_serving_without_a_crash_loop(world, tmp_path):
    failing = tmp_path / 'noevia-diary'
    failing.write_text('#!/bin/sh\nexit 3\n')
    failing.chmod(0o755)
    server = rust_forward.RustServer(lambda p, t: {}, port=free_port(), binary=str(failing), restart_delays=(0.01, 0.01))
    server.start()
    server._thread.join(timeout=30)
    assert not server._thread.is_alive() and not server.ready and server.restarts == 2
    client = TestClient(rust_forward.ForwardMiddleware(appmod.app, server=server, routes={'/api/months': 'GET'}))
    resp = client.get('/api/months', headers={'Authorization': f'Bearer {TOKEN}'})
    assert (resp.status_code, resp.json()) == (400, {'detail': 'missing or invalid X-Cowork-User-ID'})
    missing = rust_forward.RustServer(lambda p, t: {}, port=free_port(), binary=str(tmp_path / 'absent'), restart_delays=())
    missing.start()
    missing._thread.join(timeout=10)
    assert not missing.ready


def test_raise_floor_refuses_assertions_signed_at_or_before_it(world):
    ta._reset_for_tests()
    now = int(time.time())
    ta.raise_floor(now)
    with ta._seen_lock:
        assert ta._floor == now
    assert ta._remember_nonce('a' * 32, now, now + 0.5) == 'replayed or older than the replay cache'
    assert ta._remember_nonce('b' * 32, now + 1, now + 1.5) is None
