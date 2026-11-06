"""D4: the Rust Diary server (noevia-rs bins/noevia-diary) behind DIARY_RUST_ROUTES.

The parity cases run every request twice on the same synthetic state: once served by Python alone
(DIARY_RUST_ROUTES empty, today's behaviour) and once through the forwarder with the Rust process,
and compare status, content type and body byte for byte. Python's TenantAssertionMiddleware is the
only verifier and nonce cache in both modes, so each mode gets its own freshly signed request and
every replay is refused by that one cache, whichever process would have answered. They need the real binaries (DIARY_RUST_SERVER_BIN and the
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
from starlette.middleware import Middleware

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
    # A reindex that succeeds: what a completed Python recovery leaves (no dirty documents).
    monkeypatch.setattr(appmod, '_reindex_dirty', clear_dirty)
    monkeypatch.setattr(appmod, '_deleted_tenants', {})
    ta._reset_for_tests()
    appmod._tenant_states.clear()
    yield tmp_path / 'state'
    for state in appmod._tenant_states.values():
        appmod._close_state(state)
    appmod._tenant_states.clear()
    ta._reset_for_tests()


def clear_dirty(st):
    for document in st.journal.dirty_documents():
        st.journal.clear_dirty(document)


def signed(user, method, path, extra=None, query=b'', ts=None):
    headers = {'Authorization': f'Bearer {TOKEN}', 'X-Cowork-User-ID': user, **(extra or {})}
    ts = int(time.time()) if ts is None else ts
    headers[ta.HEADER] = ta.sign(KEY, user, method, path, headers, ts, secrets.token_hex(16), query)
    return headers


def index_db(state_dir, uid):
    return sqlite3.connect(state_dir / 'users' / uid / 'managed-index.db')


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


class Recording(rust_forward.ForwardMiddleware):
    """Records whether the Rust side served (True) or Python did (False) each forwarded request."""
    served = []

    async def _forward(self, server, scope, deadline, **kwargs):
        reply = await super()._forward(server, scope, deadline, **kwargs)
        Recording.served.append(reply is not None)
        return reply


def install_forwarder():
    """The production order: the forwarder INSIDE TenantAssertionMiddleware (app.py adds it first)."""
    routes = dict(rust_forward.SUPPORTED)
    saved = list(appmod.app.user_middleware)
    position = next(i for i, m in enumerate(saved) if m.cls is appmod.TenantAssertionMiddleware)
    appmod.app.user_middleware.insert(position + 1, Middleware(Recording, routes=routes, is_deleted=appmod._tenant_deleted,
                                                               is_recovered=appmod._rust_recovered,
                                                               retrieval_facts=appmod._rust_retrieval_facts))
    appmod.app.middleware_stack = None
    return saved


def start_server():
    server = rust_forward.RustServer(
        lambda p, t, f: rust_forward.server_config(appmod._base_cfg, appmod.get_state(), p, t, f),
        port=free_port(), binary=BIN, restart_delays=())
    server.start()
    deadline = time.monotonic() + 15
    while not server.ready and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.ready, 'noevia-diary did not become ready'
    return server


class Modes:
    def __init__(self, monkeypatch, server):
        self.monkeypatch, self.server = monkeypatch, server

    def python(self):
        self.monkeypatch.setattr(rust_forward, '_server', None)

    def rust(self, server=None):
        if server is not None:
            self.server = server
        self.monkeypatch.setattr(rust_forward, '_server', self.server)


@pytest.fixture
def both(world, monkeypatch):
    saved = install_forwarder()
    client = TestClient(appmod.app)
    modes = Modes(monkeypatch, None)
    modes.python()
    seed(world, client)
    modes.rust(start_server())
    Recording.served = []
    yield client, modes
    modes.server.stop()
    appmod.app.user_middleware[:] = saved
    appmod.app.middleware_stack = None


def wire(resp):
    return resp.status_code, resp.headers.get('content-type'), resp.content


LONG = 'A' * 4097
BAD_REF = b64({'kind': 'webdav', 'baseUrl': 'http://offline.invalid', 'secretRef': 'NOT-HEX'})
WRONG_SECRET = b64({'kind': 'webdav', 'baseUrl': 'http://offline.invalid', 'secretRef': 'a' * 32, 'secret': 'synthetic'})
REF_ONLY = b64({'kind': 'webdav', 'baseUrl': 'http://offline.invalid', 'corpusRoot': 'D', 'secretRef': 'b' * 32})
UNVERIFIED = b'{"detail":"invalid tenant assertion"}'

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
    ('client-supplied internal headers are dropped', DAILY, '/api/months', b'',
     {'X-Noevia-Diary-Forward': 'forged', 'X-Noevia-Diary-Asserted': '0'}, 200, True),
    ('health process part', None, '/api/health', b'', {}, 200, True),
    ('health with tenant detail', DAILY, '/api/health', b'', {'X-Cowork-Storage': b64({'kind': 'local'})}, 200, True),
]


def request(client, case):
    name, user, path, query, extra, _, _ = case
    headers = {'Authorization': f'Bearer {TOKEN}', **extra} if user is None else signed(user, 'GET', path, extra, query)
    return client.get(path + ('?' + query.decode() if query else ''), headers=headers)


@needs_binary
@needs_server
def test_parity_table_python_vs_rust(both):
    client, modes = both
    rows = []
    for case in CASES:
        modes.python()
        p = request(client, case)
        modes.rust()
        Recording.served = []
        r = request(client, case)
        rows.append((case[0], p.status_code, r.status_code, Recording.served))
        assert wire(p) == wire(r), case[0]
        assert p.status_code == case[5], case[0]
        assert Recording.served == [case[6]], (case[0], Recording.served)
    print('\n'.join(f'{n:48} python={a} rust={b} native={s}' for n, a, b, s in rows))


@needs_binary
@needs_server
def test_auth_and_assertion_refusals_match(both):
    client, modes = both
    answers = {}
    for mode in (modes.python, modes.rust):
        mode()
        bad_token = signed(DAILY, 'GET', '/api/months')
        bad_token['Authorization'] = 'Bearer wrong'
        once = signed(DAILY, 'GET', '/api/months')
        answers[mode.__name__] = [wire(client.get('/api/months', headers=h)) for h in (
            bad_token, signed(DAILY, 'GET', '/api/day'), once, once,
            {'Authorization': f'Bearer {TOKEN}'})] + [wire(client.get('/api/health'))]
    assert answers['python'] == answers['rust']
    statuses = [a[0] for a in answers['rust']]
    assert statuses == [401, 401, 200, 401, 400, 401]
    assert answers['rust'][1][2] == UNVERIFIED and answers['rust'][3][2] == UNVERIFIED


@needs_binary
@needs_server
def test_deleted_pending_and_broken_tenants_stay_with_python(both, world):
    client, modes = both
    victim = OTHER
    modes.python()
    assert client.get('/api/months', headers=signed(victim, 'GET', '/api/months')).status_code == 200
    modes.rust()
    Recording.served = []
    assert client.get('/api/months', headers=signed(victim, 'GET', '/api/months')).status_code == 200
    assert Recording.served == [True]
    assert client.delete('/api/internal/tenant', headers=signed(victim, 'DELETE', '/api/internal/tenant')).status_code == 200
    for mode in (modes.python, modes.rust):
        mode()
        resp = client.get('/api/months', headers=signed(victim, 'GET', '/api/months'))
        assert (resp.status_code, resp.json()) == (410, {'detail': 'tenant deleted'})
    with index_db(world, DAILY) as db:
        db.execute("INSERT INTO journal(id,created_at,kind,payload) VALUES ('synthetic','t','index_month','{}')")
    Recording.served = []
    client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))
    assert Recording.served == [False]
    (world / 'users' / MONTHLY / 'managed-diary.active').write_text(OTHER)
    appmod._tenant_states.clear()
    modes.python()
    p = client.get('/api/months', headers=signed(MONTHLY, 'GET', '/api/months'))
    modes.rust()
    r = client.get('/api/months', headers=signed(MONTHLY, 'GET', '/api/months'))
    assert p.status_code == 503 and wire(p) == wire(r)


@needs_binary
@needs_server
def test_future_stamped_replays_are_refused_across_rust_death_and_restart(both):
    client, modes = both
    client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))  # Python recovers the tenant
    future = signed(DAILY, 'GET', '/api/months', ts=int(time.time()) + 30)
    Recording.served = []
    assert client.get('/api/months', headers=future).status_code == 200
    assert Recording.served == [True]
    assert client.get('/api/months', headers=future).content == UNVERIFIED
    modes.server.lost()  # Rust dies: Python serves every route
    assert not modes.server.ready
    resp = client.get('/api/months', headers=future)
    assert (resp.status_code, resp.content) == (401, UNVERIFIED)
    modes.server.stop()
    modes.rust(start_server())  # a restarted Rust process
    resp = client.get('/api/months', headers=future)
    assert (resp.status_code, resp.content) == (401, UNVERIFIED)
    Recording.served = []
    assert client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months', ts=int(time.time()) + 30)).status_code == 200
    assert Recording.served == [True]


@needs_binary
@needs_server
def test_decline_then_replay_is_refused_in_both_directions(both, world):
    client, modes = both
    client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))
    # Served by Rust, then pending journal work appears (Rust would decline): the replay is refused.
    first = signed(DAILY, 'GET', '/api/months')
    Recording.served = []
    assert client.get('/api/months', headers=first).status_code == 200 and Recording.served == [True]
    with index_db(world, DAILY) as db:
        db.execute("INSERT INTO journal(id,created_at,kind,payload) VALUES ('pending','t','index_month','{}')")
    resp = client.get('/api/months', headers=first)
    assert (resp.status_code, resp.content) == (401, UNVERIFIED)
    # Declined (pending work) and served by Python, then the work clears (Rust would serve): refused.
    second = signed(DAILY, 'GET', '/api/months')
    Recording.served = []
    assert client.get('/api/months', headers=second).status_code == 200 and Recording.served == [False]
    with index_db(world, DAILY) as db:
        db.execute("UPDATE journal SET applied = 1")
    Recording.served = []
    assert client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months')).status_code == 200
    assert Recording.served == [True]
    resp = client.get('/api/months', headers=second)
    assert (resp.status_code, resp.content) == (401, UNVERIFIED)


@needs_binary
@needs_server
def test_concurrent_in_flight_requests_during_a_switch_are_each_served_once(both):
    import asyncio
    import httpx
    client, modes = both
    client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))
    requests = [signed(DAILY, 'GET', '/api/months') for _ in range(24)]
    Recording.served = []

    async def run():
        # At most 3 at once: the tenant-assertion leaf allows 4 concurrent checks (busy = 401).
        gate = asyncio.Semaphore(3)
        done = []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=appmod.app), base_url='http://diary') as http:
            async def one(headers):
                async with gate:
                    resp = await http.get('/api/months', headers=headers)
                    done.append(resp)
                    return resp

            async def killer():
                while len(done) < 8:
                    await asyncio.sleep(0.005)
                modes.server._proc.kill()  # mid-flight: the next forwards fail and fall back

            first = await asyncio.gather(killer(), *(one(h) for h in requests))
            again = await asyncio.gather(*(http.get('/api/months', headers=h) for h in requests[:6]))
            return first[1:], again

    first, again = asyncio.run(run())
    assert [r.status_code for r in first] == [200] * len(requests)
    assert len({r.content for r in first}) == 1
    # Some were answered natively, the rest by Python after the loss.
    assert True in Recording.served and (False in Recording.served or len(Recording.served) < len(requests))
    # Every replay, whichever process answered the original, is refused by the one cache.
    for h in requests:
        resp = client.get('/api/months', headers=h)
        assert (resp.status_code, resp.content) == (401, UNVERIFIED)
    assert all(r.status_code == 401 for r in again)


@needs_binary
@needs_server
def test_a_missing_or_forged_forward_secret_gets_pythons_401(both):
    import httpx
    _, modes = both
    server = modes.server
    for headers in ({}, {'x-noevia-diary-forward': 'forged'}, {'x-noevia-diary-forward': server.forward_secret + '0'},
                    {'x-noevia-diary-forward': ''}):
        for path in ('/api/months', '/api/day', '/api/health'):
            resp = httpx.get(server.base_url + path, headers={'Authorization': f'Bearer {TOKEN}', 'X-Cowork-User-ID': DAILY,
                                                              'x-noevia-diary-asserted': '1', 'x-noevia-diary-recovered': '1',
                                                              **headers}, trust_env=False)
            assert (resp.status_code, resp.headers['content-type'], resp.content) == (401, 'application/json', UNVERIFIED)
    ok = httpx.get(server.base_url + '/api/health', headers={'Authorization': f'Bearer {TOKEN}',
                                                             'x-noevia-diary-forward': server.forward_secret}, trust_env=False)
    assert ok.status_code == 200


@needs_binary
@needs_server
def test_dirty_state_first_read_is_served_and_recovered_by_python(both, world, monkeypatch):
    client, modes = both
    calls = []
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: (calls.append(1), clear_dirty(st)))
    appmod._tenant_states.clear()  # a fresh Python process: nothing recovered yet
    with index_db(world, DAILY) as db:
        db.execute("INSERT OR IGNORE INTO dirty_documents VALUES (?)", (daily_path(datetime.date(2030, 1, 6)),))
    Recording.served = []
    # A forged "recovered" header from the client is dropped: Python serves and recovers first.
    first = client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months', {'X-Noevia-Diary-Recovered': '1'}))
    assert first.status_code == 200 and Recording.served == [False] and calls
    with index_db(world, DAILY) as db:
        assert db.execute('SELECT COUNT(*) FROM dirty_documents').fetchone()[0] == 0
    second = client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))
    assert Recording.served == [False, True] and second.content == first.content
    # Dirty documents appearing later (a write elsewhere) send reads back to Python.
    with index_db(world, DAILY) as db:
        db.execute("INSERT INTO dirty_documents VALUES ('Entries/x.md')")
    client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))
    assert Recording.served[-1] is False


def test_routes_default_to_none_and_add_no_middleware(monkeypatch):
    monkeypatch.delenv('DIARY_RUST_ROUTES', raising=False)
    assert rust_forward.configured_routes() == {}
    assert rust_forward.start(lambda p, t, f: {}, {}) is None
    assert not any(m.cls is rust_forward.ForwardMiddleware for m in appmod.app.user_middleware)
    assert rust_forward.configured_routes(' /api/months, /api/nope ,/api/day') == {'/api/months': 'GET', '/api/day': 'GET'}
    assert not hasattr(ta, 'raise_floor')


def test_a_binary_that_cannot_start_leaves_python_serving_without_a_crash_loop(world, tmp_path):
    failing = tmp_path / 'noevia-diary'
    failing.write_text('#!/bin/sh\nexit 3\n')
    failing.chmod(0o755)
    server = rust_forward.RustServer(lambda p, t, f: {}, port=free_port(), binary=str(failing), restart_delays=(0.01, 0.01))
    server.start()
    server._thread.join(timeout=30)
    assert not server._thread.is_alive() and not server.ready and server.restarts == 2
    client = TestClient(rust_forward.ForwardMiddleware(appmod.app, server=server, routes={'/api/months': 'GET'}))
    resp = client.get('/api/months', headers={'Authorization': f'Bearer {TOKEN}'})
    assert (resp.status_code, resp.json()) == (400, {'detail': 'missing or invalid X-Cowork-User-ID'})
    missing = rust_forward.RustServer(lambda p, t, f: {}, port=free_port(), binary=str(tmp_path / 'absent'), restart_delays=())
    missing.start()
    missing._thread.join(timeout=10)
    assert not missing.ready


class _HungChild:
    """Accepts one forward, then trickles reply bytes forever (never completes)."""

    def __init__(self):
        import threading
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen()
        self.port = self.listener.getsockname()[1]
        self.stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        try:
            conn, _ = self.listener.accept()
        except OSError:
            return
        conn.recv(65536)
        conn.sendall(b'HTTP/1.1 200 OK\r\ncontent-length: 100000\r\n\r\n')
        while not self.stop.is_set():
            try:
                conn.sendall(b'x')
            except OSError:
                return
            time.sleep(0.02)

    def close(self):
        self.stop.set()
        self.listener.close()


class _SlowReapProc:
    """A child whose kill/reap takes seconds (the pre-fix code waited for it on the request path)."""

    def __init__(self, reap_s):
        import threading
        self.reap_s, self.killed, self.reaped = reap_s, threading.Event(), threading.Event()
        self.returncode = None

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed.set()

    def wait(self, timeout=None):
        time.sleep(min(self.reap_s, timeout or self.reap_s))
        self.returncode = -9
        self.reaped.set()
        return self.returncode


def _hung_forwarder(port, deadline_s, monkeypatch):
    monkeypatch.setattr(rust_forward, 'FORWARD_DEADLINE_S', deadline_s)
    server = rust_forward.RustServer(lambda p, t, f: {}, port=port, restart_delays=())
    server.ready, server.forward_secret = True, 'synthetic'
    return server, rust_forward.ForwardMiddleware(appmod.app, server=server, routes={'/api/months': 'GET'})


def test_a_hung_child_with_slow_reaping_does_not_delay_the_fallback(world, monkeypatch):
    child = _HungChild()
    try:
        server, app = _hung_forwarder(child.port, 0.5, monkeypatch)
        proc = _SlowReapProc(reap_s=4.0)
        server._proc = proc
        client = TestClient(app)
        started = time.monotonic()
        resp = client.get('/api/months', headers={'Authorization': f'Bearer {TOKEN}'})
        elapsed = time.monotonic() - started
        # Python's own answer, within the deadline plus a small margin, while reaping is still pending.
        assert (resp.status_code, resp.json()) == (400, {'detail': 'missing or invalid X-Cowork-User-ID'})
        assert elapsed < 0.5 + 0.5, elapsed
        assert not server.ready
        assert proc.killed.wait(1) and not proc.reaped.is_set()
        assert proc.reaped.wait(6)
    finally:
        child.close()


def test_total_forward_time_is_the_deadline_plus_a_small_margin(world, monkeypatch):
    for deadline in (0.3, 1.0):
        child = _HungChild()
        try:
            server, app = _hung_forwarder(child.port, deadline, monkeypatch)
            started = time.monotonic()
            resp = TestClient(app).get('/api/months', headers={'Authorization': f'Bearer {TOKEN}'})
            elapsed = time.monotonic() - started
            assert resp.status_code == 400
            assert deadline <= elapsed < deadline + 0.4, (deadline, elapsed)
        finally:
            child.close()


def test_a_slow_request_body_is_never_read_or_forwarded_and_times_like_python(world, monkeypatch):
    """A request that declares a body is handed to the app untouched: the forwarder never waits for
    it, and the answer and its timing are the Python-only ones."""
    import asyncio
    forwarded = []

    class Spy(rust_forward.ForwardMiddleware):
        async def _forward(self, server, scope, deadline):
            forwarded.append(scope['path'])
            return None

    server = rust_forward.RustServer(lambda p, t, f: {}, port=free_port(), restart_delays=())
    server.ready, server.forward_secret = True, 'synthetic'
    spy = Spy(appmod.app, server=server, routes={'/api/months': 'GET'})

    async def call(app, headers):
        sent, reads = [], []

        async def receive():
            reads.append(time.monotonic())
            await asyncio.sleep(3600)  # the body never arrives

        async def send(message):
            sent.append(message)

        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'GET',
                 'scheme': 'http', 'path': '/api/months', 'raw_path': b'/api/months', 'query_string': b'',
                 'root_path': '', 'headers': headers, 'client': ('127.0.0.1', 1), 'server': ('127.0.0.1', 8010)}
        started = time.monotonic()
        await asyncio.wait_for(app(scope, receive, send), 5)
        return time.monotonic() - started, sent[0]['status'], b''.join(m.get('body', b'') for m in sent[1:]), reads

    base = [(b'authorization', f'Bearer {TOKEN}'.encode()), (b'content-type', b'text/plain')]
    for extra in ([(b'content-length', b'10')], [(b'transfer-encoding', b'chunked')]):
        py_time, py_status, py_body, _ = asyncio.run(call(appmod.app, base + extra))
        fw_time, fw_status, fw_body, reads = asyncio.run(call(spy, base + extra))
        assert (fw_status, fw_body) == (py_status, py_body) == (400, b'{"detail":"missing or invalid X-Cowork-User-ID"}')
        assert forwarded == [] and fw_time < 1 and py_time < 1
    # Without a body the same request is forwarded.
    asyncio.run(call(spy, base))
    assert forwarded == ['/api/months']
