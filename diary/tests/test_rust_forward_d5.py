"""D5: tenant writes on the Rust Diary server (PUT /api/file, POST /api/directory,
POST /api/entries/append, POST /api/entries/edit) behind DIARY_RUST_WRITES.

Three groups:
- the switch and the forwarder's write path with a scripted fake child (always run): tenant parsing,
  bounded bodies, a decline or a
  refused connection served by Python with the exact bytes, an uncertain forward never re-run, the
  per-tenant serialization across both processes and the wait for a lost child to be reaped;
- differential cases against the real binary: each write runs on the same synthetic state twice
  (Python alone, then forwarded), the state is restored between runs, and the replies (status, every
  header, body) and the resulting storage rows are compared;
- journal replay order, a killed child mid-write and concurrent writers.
Synthetic data only. The real-binary cases need DIARY_RUST_SERVER_BIN, the tenant-assertion leaf and
DIARY_CORPUS_WRITE_TEST_BIN (entries are written natively only with DIARY_CORPUS_WRITE_IMPL=rust).
"""
import asyncio
import datetime
import json
import os
import socket
import sqlite3
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.middleware import Middleware

import agent.app as appmod
from agent import corpus_write_native, rust_forward, tenant_writer
from agent.managed_storage import ManagedCorpusBackend
from tests.tenant_binary import needs_binary
from tests.test_rust_forward import DAILY, MONTHLY, TOKEN, Modes, free_port, needs_server, seed, start_server, world  # noqa: F401
from tests.test_rust_forward_d4b import sign

CW_BIN = os.environ.get('DIARY_CORPUS_WRITE_TEST_BIN', '')
needs_cw = pytest.mark.skipif(not (CW_BIN and os.path.isfile(CW_BIN)) and os.environ.get('DIARY_RUST_SERVER_REQUIRED') != '1',
                              reason='diary-corpus-write-cli not built (set DIARY_CORPUS_WRITE_TEST_BIN)')


# ------------------------------------------------------------------ the switch

@pytest.fixture(autouse=True)
def _writer_registry(monkeypatch):
    """tenant_writer's writer set is process-wide; every test starts and ends with none."""
    monkeypatch.setattr(tenant_writer, '_writers', frozenset())
    monkeypatch.setattr(tenant_writer, 'max_concurrent', {})
    monkeypatch.setattr(tenant_writer, 'denials', [])
    yield


def test_writes_switch_names_tenants_and_is_empty_by_default(monkeypatch):
    monkeypatch.delenv('DIARY_RUST_WRITES', raising=False)
    assert rust_forward.configured_writers() == frozenset()
    assert rust_forward.configured_writers('') == frozenset()
    assert rust_forward.start(lambda p, t, f: {}, {}, frozenset()) is None
    assert rust_forward.configured_writers(f' {DAILY.upper()} , not-a-uuid,{MONTHLY}') == frozenset({DAILY, MONTHLY})
    assert rust_forward.configured_writers(f'{DAILY},*') == rust_forward.ALL_TENANTS
    assert not hasattr(rust_forward, 'PYTHON_ONLY_WRITES'), 'the D5 no-op gate is gone'
    # Writes are never chosen per route.
    assert rust_forward.configured_routes('/api/file,/api/entries/append,/api/months') == {'/api/months': 'GET'}
    # The app's process default selects no tenant: every tenant_writer call is a no-op.
    assert appmod._RUST_WRITERS == frozenset() and not tenant_writer.is_writer(DAILY)


# ------------------------------------------------------------------ scripted child

MEET_CAP_S = 8.0  # the two-tenant rendezvous must give up before the forwarder's deadline
FORWARD_DEADLINE_S = 20.0


class FakeChild:
    """A loopback HTTP/1.1 server standing in for noevia-diary: records each request (headers and
    body) and answers per `mode`: 'native', 'decline', 'hang' or 'slow' (native after 0.3 s)."""

    def __init__(self, mode='native'):
        self.mode, self.requests, self.active, self.max_active = mode, [], 0, 0
        self.delay = 0.3
        # Time the serving thread is slow to run after the reply was published (scheduling lag).
        # It must not matter: a write is over when its reply is published, never later.
        self.cleanup_lag = 0.0
        # meet=N: each request waits (at most MEET_CAP_S, below the forwarder's deadline) until N
        # requests have ARRIVED, a latched count that never goes down, so concurrency is observed
        # deterministically whatever the load.
        self.meet = 0
        self.arrived = 0
        self.all_arrived = threading.Event()
        self.lock = threading.Lock()
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(16)
        self.port = self.listener.getsockname()[1]
        self.stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        data = b''
        while b'\r\n\r\n' not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return
            data += chunk
        head, body = data.split(b'\r\n\r\n', 1)
        headers = {k.lower(): v for k, v in (line.split(b': ', 1) for line in head.split(b'\r\n')[1:])}
        length = int(headers.get(b'content-length', b'0'))
        while len(body) < length:
            body += conn.recv(65536)
        with self.lock:
            self.requests.append((headers, body))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.arrived += 1
            if self.meet and self.arrived >= self.meet:
                self.all_arrived.set()
        if self.meet:
            # Latched: once set it stays set, so no arrival can miss it however the others run.
            self.all_arrived.wait(timeout=MEET_CAP_S)
        finished = False

        def finish():
            # The simulated write is over HERE, before its reply exists: the client cannot see a
            # reply (and send the next request) while this write is still counted as active.
            nonlocal finished
            if not finished:
                finished = True
                with self.lock:
                    self.active -= 1

        try:
            if self.mode == 'hang':
                self.stop.wait(30)
                return
            if self.mode == 'slow':
                time.sleep(self.delay)
            if self.mode == 'decline':
                reply = (b'HTTP/1.1 421 Misdirected Request\r\nx-noevia-diary-decline: 1\r\nconnection: close\r\n'
                         b'content-length: 0\r\n\r\n')
            else:
                body = b'{"native":true}'
                reply = (b'HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\n'
                         b'content-length: %d\r\n\r\n' % len(body) + body)
            finish()
            # One request per connection, said so on the wire: the forwarder's pool must never reuse
            # a socket this fake is closing (a race that read as a lost child under load).
            conn.sendall(reply)
            conn.close()
            time.sleep(self.cleanup_lag)
        finally:
            finish()
            conn.close()

    def phase(self, meet=0):
        """Start a measurement phase: forget earlier maxima and arrivals, atomically. No request is
        in flight between phases (each reply is published after its write is recorded as over)."""
        with self.lock:
            assert self.active == 0, 'a write is still counted as running between phases'
            self.max_active, self.arrived, self.meet = 0, 0, meet
            self.all_arrived = threading.Event()

    def close(self):
        self.stop.set()
        self.listener.close()


class SpyApp:
    """Stands in for the Python app behind the forwarder: records the body it was given."""

    def __init__(self):
        self.calls = []

    async def __call__(self, scope, receive, send):
        body, more = b'', True
        while more:
            message = await receive()
            body += message.get('body', b'')
            more = message.get('more_body', False)
        self.calls.append((scope['path'], body))
        payload = b'{"python":true}'
        await send({'type': 'http.response.start', 'status': 200,
                    'headers': [(b'content-length', str(len(payload)).encode()), (b'content-type', b'application/json')]})
        await send({'type': 'http.response.body', 'body': payload})


def forwarder(child_port, monkeypatch, writers=frozenset({DAILY}), deadline=5.0, uncertain=None):
    monkeypatch.setattr(rust_forward, 'FORWARD_DEADLINE_S', deadline)
    server = rust_forward.RustServer(lambda p, t, f: {}, port=child_port, restart_delays=())
    server.ready, server.forward_secret = True, 'synthetic'
    spy = SpyApp()
    mw = rust_forward.ForwardMiddleware(spy, server=server, routes={}, writers=writers,
                                        on_uncertain=(uncertain.append if uncertain is not None else None))
    return server, spy, mw


def put(user=DAILY, body=b'{"path":"a.md","content":"x","version":null}', extra=()):
    return [(b'authorization', f'Bearer {TOKEN}'.encode()), (b'x-cowork-user-id', user.encode()),
            (b'content-type', b'application/json'), (b'content-length', str(len(body)).encode()), *extra], body


async def asgi(app, headers, chunks, method='PUT', path='/api/file', gap=0.0):
    """One request through `app`; `chunks` are delivered as separate receive() messages."""
    sent, queue = [], list(chunks)

    async def receive():
        if not queue:
            await asyncio.sleep(3600)
        if gap:
            await asyncio.sleep(gap)
        chunk = queue.pop(0)
        return {'type': 'http.request', 'body': chunk, 'more_body': bool(queue)}

    async def send(message):
        sent.append(message)

    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method, 'scheme': 'http',
             'path': path, 'raw_path': path.encode(), 'query_string': b'', 'root_path': '', 'headers': headers,
             'client': ('127.0.0.1', 1), 'server': ('127.0.0.1', 8010), 'state': {'tenant_asserted': True}}
    await asyncio.wait_for(app(scope, receive, send), 20)
    return sent[0]['status'], b''.join(m.get('body', b'') for m in sent[1:])


def test_a_writer_tenants_write_is_forwarded_with_its_body_and_writer_role(monkeypatch):
    child = FakeChild('native')
    try:
        _, spy, mw = forwarder(child.port, monkeypatch)
        headers, body = put()
        status, reply = asyncio.run(asgi(mw, headers, [body[:10], body[10:]]))
        assert (status, reply, spy.calls) == (200, b'{"native":true}', [])
        (sent_headers, sent_body), = child.requests
        assert sent_body == body
        assert sent_headers[b'x-noevia-diary-writer'] == b'1'
        assert sent_headers[b'x-noevia-diary-asserted'] == b'1'
        # Another tenant, or a read route, is not a write of a writer tenant.
        headers, body = put(user=MONTHLY)
        assert asyncio.run(asgi(mw, headers, [body]))[1] == b'{"python":true}'
        assert len(child.requests) == 1
    finally:
        child.close()


def test_a_decline_or_a_refused_connection_is_served_by_python_with_the_same_bytes(monkeypatch):
    child = FakeChild('decline')
    try:
        _, spy, mw = forwarder(child.port, monkeypatch)
        headers, body = put()
        assert asyncio.run(asgi(mw, headers, [body[:7], body[7:]])) == (200, b'{"python":true}')
        assert spy.calls == [('/api/file', body)] and len(child.requests) == 1
    finally:
        child.close()
    server, spy, mw = forwarder(free_port(), monkeypatch)  # nothing listens: never reached
    headers, body = put()
    assert asyncio.run(asgi(mw, headers, [body])) == (200, b'{"python":true}')
    assert spy.calls == [('/api/file', body)] and not server.ready


def test_an_uncertain_write_is_never_run_again_and_the_next_waits_for_the_reap(monkeypatch):
    child, uncertain = FakeChild('hang'), []
    try:
        server, spy, mw = forwarder(child.port, monkeypatch, deadline=0.4, uncertain=uncertain)
        headers, body = put()
        started = time.monotonic()
        status, reply = asyncio.run(asgi(mw, headers, [body]))
        assert (status, reply) == (503, rust_forward.UNCERTAIN)
        assert time.monotonic() - started < 1.5
        assert spy.calls == [] and uncertain == [DAILY] and not server.ready
        # A lost child still being reaped: the next write of the tenant waits (the admission
        # barrier, at most the lock deadline), then refuses.
        monkeypatch.setattr(tenant_writer, 'DEADLINE_S', 0.2)
        server.reaping = 1
        assert asyncio.run(asgi(mw, headers, [body])) == (503, rust_forward.RESTARTING)
        server.reaping = 0
        # Once nothing of the child can run, Python is the writer.
        assert asyncio.run(asgi(mw, headers, [body])) == (200, b'{"python":true}')
        assert spy.calls == [('/api/file', body)]
    finally:
        child.close()


def test_write_bodies_are_bounded_and_a_slow_body_is_never_forwarded(monkeypatch):
    child = FakeChild('native')
    try:
        monkeypatch.setattr(rust_forward, 'MAX_WRITE_BODY', 64)
        _, spy, mw = forwarder(child.port, monkeypatch, deadline=0.3)
        big = b'{"path":"a.md","content":"' + b'x' * 80 + b'","version":null}'
        # Declared too large: refused with Python's own 413, nothing read or forwarded.
        assert asyncio.run(asgi(mw, put(body=big)[0], [big])) == (413, rust_forward.TOO_LARGE)
        # Undeclared (chunked) and too large: refused as soon as the bound is passed.
        chunked = [h for h in put(body=big)[0] if h[0] != b'content-length'] + [(b'transfer-encoding', b'chunked')]
        assert asyncio.run(asgi(mw, chunked, [big[:40], big[40:]])) == (413, rust_forward.TOO_LARGE)
        assert spy.calls == [] and child.requests == []
        # A body still arriving at the deadline: nothing sent, and never handed to Python
        # unfinished (a stalled sender would hold the tenant's lock): 408.
        headers, body = put()
        status, reply = asyncio.run(asgi(mw, headers, [body[:5], body[5:]], gap=0.25))
        assert (status, reply) == (408, rust_forward.BODY_TIMEOUT)
        assert spy.calls == [] and child.requests == []
        # Not JSON: FastAPI's 422 path, on Python, with the bounded body.
        text = [(k, b'text/plain' if k == b'content-type' else v) for k, v in headers]
        assert asyncio.run(asgi(mw, text, [body]))[1] == b'{"python":true}'
        assert spy.calls == [('/api/file', body)] and child.requests == []
    finally:
        child.close()


def test_writes_of_one_tenant_are_serialized_across_both_processes(monkeypatch):
    child = FakeChild('slow')
    try:
        assert MEET_CAP_S < FORWARD_DEADLINE_S
        _, spy, mw = forwarder(child.port, monkeypatch, writers=frozenset({DAILY, MONTHLY}),
                               deadline=FORWARD_DEADLINE_S)

        async def run():
            headers, body = put()
            child.cleanup_lag = 0.2  # the serving thread lags after each reply: must not count
            one = await asyncio.gather(*(asgi(mw, headers, [body]) for _ in range(4)))
            serial = child.max_active
            child.phase(meet=2)  # two tenants' writes must be inside the child at once
            other = put(user=MONTHLY)
            both = await asyncio.gather(asgi(mw, headers, [body]), asgi(mw, other[0], [other[1]]))
            return one, serial, both

        one, serial, both = asyncio.run(run())
        assert [s for s, _ in one] == [200] * 4
        assert serial == 1, 'two writes of one tenant overlapped'
        assert child.max_active == 2, 'different tenants are not serialized against each other'
        assert all(s == 200 for s, _ in both)
        assert tenant_writer._slots == {}, 'a writer lock was left behind'
        assert tenant_writer.max_concurrent == {DAILY: 1, MONTHLY: 1}
    finally:
        child.close()


def test_the_fake_childs_measurement_does_not_count_its_own_cleanup(monkeypatch):
    """Regression (review P3-1): two consecutive writes of one tenant, with the serving thread slow
    to run after each reply, are never counted as overlapping, and a phase starts from zero."""
    child = FakeChild('slow')
    try:
        child.delay, child.cleanup_lag = 0.02, 0.4
        _, _, mw = forwarder(child.port, monkeypatch, deadline=FORWARD_DEADLINE_S)
        headers, body = put()

        async def two():
            return [await asgi(mw, headers, [body]) for _ in range(3)]

        assert [s for s, _ in asyncio.run(two())] == [200] * 3
        assert child.max_active == 1, 'the previous write was still counted after its reply'
        child.phase(meet=0)
        assert (child.max_active, child.arrived, child.active) == (0, 0, 0)
    finally:
        child.close()


# ------------------------------------------------------------------ real binary: differential

class Recorder(rust_forward.ForwardMiddleware):
    served = []

    async def _forward(self, server, scope, deadline, **kwargs):
        reply = await super()._forward(server, scope, deadline, **kwargs)
        Recorder.served.append(reply is not None and reply != 'uncertain')
        return reply


def install(writers):
    saved = list(appmod.app.user_middleware)
    position = next(i for i, m in enumerate(saved) if m.cls is appmod.TenantAssertionMiddleware)
    appmod.app.user_middleware.insert(position + 1, Middleware(
        Recorder, routes={p: 'GET' for p in rust_forward.SUPPORTED}, writers=writers, is_deleted=appmod._tenant_deleted,
        is_recovered=appmod._rust_recovered, on_uncertain=appmod._rust_write_uncertain))
    appmod.app.middleware_stack = None
    return saved


def snapshot(state, uid):
    """Copy of a tenant's two databases (sqlite backup API: safe with Python's open connections)."""
    out = {}
    for name in ('managed-diary.db', 'managed-index.db'):
        src = sqlite3.connect(state / 'users' / uid / name)
        mem = sqlite3.connect(':memory:')
        src.backup(mem)
        src.close()
        out[name] = mem
    return out


def restore(state, uid, snap):
    for name, mem in snap.items():
        dst = sqlite3.connect(state / 'users' / uid / name, timeout=30)
        mem.backup(dst)
        dst.close()


def storage(state, uid):
    """The tenant's stored effect of a write, minus clocks, journal ids and the index lag."""
    with sqlite3.connect(state / 'users' / uid / 'managed-diary.db') as db:
        files = db.execute('SELECT path, data, version FROM files ORDER BY path').fetchall()
        dirs = db.execute('SELECT path FROM directories ORDER BY path').fetchall()
        meta = dict(db.execute("SELECT key, value FROM meta WHERE key IN ('generation','tenant','active','settings')"))
        clocks = dict(db.execute("SELECT key, value FROM meta WHERE key IN ('due','snapshot_time')"))
    for key in clocks:
        float(clocks[key])  # Python's str(float) form either way
    with sqlite3.connect(state / 'users' / uid / 'managed-index.db') as idx:
        journal = idx.execute('SELECT kind, payload, applied, attempts, last_error FROM journal ORDER BY rowid').fetchall()
    return files, dirs, meta, journal


def wire(resp):
    return resp.status_code, [(k, v) for k, v in resp.headers.raw if k.lower() not in (b'date', b'server')], resp.content


@pytest.fixture
def writers(world, monkeypatch):
    if CW_BIN:
        monkeypatch.setattr(corpus_write_native, 'BINARY', CW_BIN)
    monkeypatch.setenv('DIARY_CORPUS_WRITE_IMPL', 'rust')
    saved = install(frozenset({DAILY, MONTHLY}))
    client = TestClient(appmod.app)
    modes = Modes(monkeypatch, None)
    modes.python()
    # Seeding writes the tenants directly, as Python's earlier releases left them: the writer set
    # takes effect afterwards (the forwarder configures it when the middleware stack is built).
    real_configure = tenant_writer.configure
    monkeypatch.setattr(tenant_writer, 'configure', lambda _w: None)
    seed(world, client)
    monkeypatch.setattr(tenant_writer, 'configure', real_configure)
    tenant_writer.configure(frozenset({DAILY, MONTHLY}))
    for uid in (DAILY, MONTHLY):
        assert client.get('/api/months', headers=sign(uid, 'GET', '/api/months')).status_code == 200
    # Python's appends need fresh AppStates that read DIARY_CORPUS_WRITE_IMPL=rust.
    started = []

    def start():
        started.append(start_server())
        modes.rust(started[-1])
        return modes

    yield client, start, world
    for server in started:
        server.stop()
    appmod.app.user_middleware[:] = saved
    appmod.app.middleware_stack = None


def request(client, user, method, path, body):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    headers = sign(user, method, path, raw, None, b'', 'application/json')
    return client.request(method, path, headers=headers, content=raw)


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def differential(client, modes, state, cases):
    rows = []
    for name, user, method, path, body, status, native in cases:
        snap = snapshot(state, user)
        modes.python()
        p = request(client, user, method, path, body)
        after_python = storage(state, user)
        restore(state, user, snap)
        modes.rust()
        Recorder.served = []
        r = request(client, user, method, path, body)
        after_rust = storage(state, user)
        rows.append(f'{name:50} python={p.status_code} rust={r.status_code} native={Recorder.served}')
        assert wire(p) == wire(r), (name, p.content[:300], r.content[:300])
        assert p.status_code == status, (name, p.status_code, p.content[:300])
        assert after_python == after_rust, name
        assert Recorder.served == [native], (name, Recorder.served)
    print('\n'.join(rows))


@needs_binary
@needs_server
@needs_cw
def test_write_routes_are_wire_and_storage_identical(writers):
    client, start, state = writers
    modes = start()
    today = datetime.datetime.now().date()
    xid_today = '00000000-0000-4000-8000-%012d' % today.day
    doc = f"Entries/{today.year}/{today.strftime('%B')}/{today.strftime('%B')} {today.day}, {today.year}.md"
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        doc_text = db.execute('SELECT data FROM files WHERE path=?', (doc,)).fetchone()[0].decode()
    memory = 'Prefers tea ✓'
    rid = str(uuid.uuid4())
    cases = [
        ('put create', DAILY, 'PUT', '/api/file', {'path': 'AI Memory/memory.md', 'content': memory, 'version': None}, 200, True),
        ('put stale version', DAILY, 'PUT', '/api/file', {'path': doc, 'content': 'x', 'version': '0' * 64}, 409, True),
        ('put replace', DAILY, 'PUT', '/api/file',
         {'path': 'AI Memory/memory.md', 'content': 'Prefers green tea\n', 'version': hashlib_sha(memory.encode())}, 200, True),
        ('put unchanged', DAILY, 'PUT', '/api/file', {'path': doc, 'content': doc_text, 'version': hashlib_sha(doc_text.encode())}, 200, True),
        ('put content null', DAILY, 'PUT', '/api/file', {'path': doc, 'content': None, 'version': None}, 400, True),
        ('put under a file', DAILY, 'PUT', '/api/file', {'path': f'{doc}/x.md', 'content': 'x', 'version': None}, 409, True),
        ('put not markdown', DAILY, 'PUT', '/api/file', {'path': 'a.txt', 'content': 'x', 'version': None}, 400, True),
        ('put no version', DAILY, 'PUT', '/api/file', {'path': 'a.md', 'content': 'x'}, 400, True),
        ('put too large', DAILY, 'PUT', '/api/file', {'path': 'a.md', 'content': 'x' * (512 * 1024 + 1), 'version': None}, 400, True),
        ('put not an object (422)', DAILY, 'PUT', '/api/file', [1], 422, False),
        ('put monthly tenant', MONTHLY, 'PUT', '/api/file', {'path': 'notes.md', 'content': 'm', 'version': None}, 200, True),
        ('mkdir', DAILY, 'POST', '/api/directory', {'path': 'Projects/Sub'}, 409, True),
        ('mkdir top', DAILY, 'POST', '/api/directory', {'path': 'Projects'}, 200, True),
        ('mkdir root', DAILY, 'POST', '/api/directory', {}, 405, True),
        ('mkdir exists', DAILY, 'POST', '/api/directory', {'path': 'Entries'}, 405, True),
        ('mkdir bad path', DAILY, 'POST', '/api/directory', {'path': 7}, 400, True),
        ('append', DAILY, 'POST', '/api/entries/append',
         {'text': 'Synthetic tea with “Ana” ü', 'title': 'tea', 'requestId': rid, 'entryTime': now_iso()}, 200, True),
        ('append no title', DAILY, 'POST', '/api/entries/append',
         {'text': 'Synthetic note', 'requestId': str(uuid.uuid4()), 'entryTime': now_iso()}, 200, True),
        ('append repeat of a stored xid', DAILY, 'POST', '/api/entries/append',
         {'text': 'again', 'requestId': xid_today, 'entryTime': now_iso()}, 200, True),
        ('append heading', DAILY, 'POST', '/api/entries/append', {'text': '# no', 'requestId': rid}, 400, True),
        ('append marker', DAILY, 'POST', '/api/entries/append', {'text': '<!-- xid:ab -->', 'requestId': rid}, 400, True),
        ('append entryDay', DAILY, 'POST', '/api/entries/append', {'text': 'x', 'requestId': rid, 'entryDay': '2030-01-01'}, 400, True),
        ('append naive time', DAILY, 'POST', '/api/entries/append', {'text': 'x', 'requestId': rid, 'entryTime': '2030-01-06T09:00:00'}, 400, True),
        ('append stale time', DAILY, 'POST', '/api/entries/append', {'text': 'x', 'requestId': rid, 'entryTime': '2020-01-06T09:00:00Z'}, 400, True),
        ('append compact iso time (Python decides)', DAILY, 'POST', '/api/entries/append',
         {'text': 'x', 'requestId': rid, 'entryTime': '20200106T0900+0000'}, 400, False),
        ('append monthly tenant', MONTHLY, 'POST', '/api/entries/append',
         {'text': 'Synthetic monthly capture', 'requestId': str(uuid.uuid4()), 'entryTime': now_iso()}, 200, True),
        ('edit', DAILY, 'POST', '/api/entries/edit', {'xid': xid_today, 'me': 'Edited synthetic walk.', 'assistant': 'Ok.'}, 200, True),
        ('edit with month', DAILY, 'POST', '/api/entries/edit',
         {'xid': '00000000-0000-4000-8000-000000000006', 'me': 'Edited January.', 'month': '2030-01'}, 200, True),
        ('edit stale base', DAILY, 'POST', '/api/entries/edit', {'xid': xid_today, 'me': 'x', 'base_hash': 'a' * 64}, 409, True),
        ('edit missing', DAILY, 'POST', '/api/entries/edit', {'xid': 'ffffffff-0000-4000-8000-000000000000', 'me': 'x'}, 404, True),
        ('edit bad xid', DAILY, 'POST', '/api/entries/edit', {'xid': 'nope', 'me': 'x'}, 400, True),
        ('edit bad month', DAILY, 'POST', '/api/entries/edit', {'xid': xid_today, 'me': 'x', 'month': '2030-13'}, 400, True),
        ('edit pydantic 422', DAILY, 'POST', '/api/entries/edit', {'xid': xid_today, 'me': 5}, 422, False),
        ('edit monthly tenant', MONTHLY, 'POST', '/api/entries/edit',
         {'xid': '00000000-0000-4000-8000-000000000099', 'me': 'Edited monthly.'}, 200, True),
    ]
    differential(client, modes, state, cases)
    # Unauthorized writes: each route's own 401 bytes, from both sides.
    for path, method, body in (('/api/file', 'PUT', {'path': 'a.md'}), ('/api/entries/append', 'POST', {'text': 'x'})):
        replies = []
        for mode in (modes.python, modes.rust):
            mode()
            raw = json.dumps(body).encode()
            headers = sign(DAILY, method, path, raw, None, b'', 'application/json')
            headers['Authorization'] = 'Bearer wrong'
            replies.append(wire(client.request(method, path, headers=headers, content=raw)))
        assert replies[0] == replies[1] and replies[0][0] == 401


@needs_binary
@needs_server
@needs_cw
def test_a_body_fastapi_refuses_gets_its_422_before_auth_on_every_write_route(writers):
    """FastAPI validates the body parameter (422) before the handler's check_auth (401): with a
    wrong or missing token, Python and Rust must give the same 422 for every invalid-body kind,
    and the same 401 for a valid shape."""
    client, start, state = writers
    modes = start()
    shape = [b'{"path":', b'[1]', b'"text"', b'']
    edit_model = [b'{"me":"x"}', b'{"xid":5,"me":"x"}', b'{"xid":"0123abcd","me":"x","month":3}',
                  b'{"xid":"0123abcd","me":"x","assistant":null}']
    routes = [('PUT', '/api/file', b'{"path":"a.md","content":"x","version":null}'),
              ('POST', '/api/directory', b'{"path":"Projects"}'),
              ('POST', '/api/entries/append', b'{"text":"x"}'),
              ('POST', '/api/entries/edit', b'{"xid":"zz","me":"x"}')]
    rows = []
    for token in ('wrong', None):
        for method, path, valid in routes:
            bodies = shape + (edit_model if path.endswith('/edit') else [])
            # NaN: Python's json reads it (a dict, so auth answers 401); Rust cannot read it alike and
            # declines. The valid shape is the only one Rust answers itself. (A NaN body the edit model
            # refuses makes FastAPI's own 422 fail to encode: pre-existing, Python-only.)
            nan = valid[:-1] + b',"n":NaN}'  # a model-valid body Rust cannot read alike
            for raw, want, native in [(b, 422, False) for b in bodies] + [(nan, 401, False), (valid, 401, True)]:
                snap = snapshot(state, DAILY)
                replies = []
                for mode in (modes.python, modes.rust):
                    mode()
                    Recorder.served = []
                    headers = sign(DAILY, method, path, raw, None, b'', 'application/json')
                    if token is None:
                        headers.pop('Authorization', None)
                    else:
                        headers['Authorization'] = f'Bearer {token}'
                    replies.append((wire(client.request(method, path, headers=headers, content=raw)), Recorder.served))
                    restore(state, DAILY, snap)
                (p, _), (r, served) = replies
                rows.append(f'{token!s:6} {method} {path:22} {raw[:40]!r:44} python={p[0]} rust={r[0]} native={served}')
                assert p == r, (token, path, raw, p[2][:200], r[2][:200])
                assert p[0] == want, (token, path, raw, p[0])
                # A non-JSON body never reaches Rust; a JSON one Rust must decline, not answer.
                assert served == [True] if native else served in ([], [False]), (token, path, raw, served)
    print('\n'.join(rows))


def hashlib_sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


@needs_binary
@needs_server
@needs_cw
def test_decline_paths_outbox_pending_journal_and_recovery(writers):
    client, start, state = writers
    modes = start()
    body = {'path': 'decline.md', 'content': 'x', 'version': None}
    # Index outbox: Python's write drains it, so Python serves.
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        db.execute("INSERT INTO workspace_index_outbox VALUES ('Notes/x.md')")
    Recorder.served = []
    assert request(client, DAILY, 'PUT', '/api/file', body).status_code == 200
    assert Recorder.served == [True]
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        db.execute('DELETE FROM workspace_index_outbox')
    # A pending journal entry: Python replays it first.
    with sqlite3.connect(state / 'users' / DAILY / 'managed-index.db') as idx:
        idx.execute("INSERT INTO journal (id, created_at, kind, payload) VALUES ('synthetic-pending', 't', 'index_month', '{}')")
    Recorder.served = []
    assert request(client, DAILY, 'PUT', '/api/file', {**body, 'path': 'decline2.md'}).status_code == 200
    assert Recorder.served == [True]
    with sqlite3.connect(state / 'users' / DAILY / 'managed-index.db') as idx:
        idx.execute('UPDATE journal SET applied=1')
    # Dirty documents do not stop a write (the index lags; Python reindexes later).
    with sqlite3.connect(state / 'users' / DAILY / 'managed-index.db') as idx:
        idx.execute("INSERT OR IGNORE INTO dirty_documents VALUES ('memory.md')")
    Recorder.served = []
    assert request(client, DAILY, 'PUT', '/api/file', {**body, 'path': 'native.md'}).status_code == 200
    assert Recorder.served == [True]
    # Recovery pending (a fresh Python process): Python serves and recovers first.
    appmod._tenant_states.clear()
    Recorder.served = []
    assert request(client, DAILY, 'PUT', '/api/file', {**body, 'path': 'after-restart.md'}).status_code == 200
    assert Recorder.served == [True]
    Recorder.served = []
    assert request(client, DAILY, 'PUT', '/api/file', {**body, 'path': 'after-recovery.md'}).status_code == 200
    assert Recorder.served == [True]


@needs_binary
@needs_server
@needs_cw
def test_journal_rows_written_natively_replay_in_order_on_python(writers):
    """Python and Rust alternate on one tenant across a rollback: entries the Rust side journaled but
    never applied (a crash after the enqueue) are replayed by Python in rowid order, and the result
    is the text the native run produced."""
    client, start, state = writers
    modes = start()
    rid_a, rid_b = str(uuid.uuid4()), str(uuid.uuid4())
    steps = [('/api/entries/append', {'text': 'Synthetic first note', 'requestId': rid_a, 'entryTime': now_iso()}),
             ('/api/entries/edit', {'xid': rid_a, 'me': 'Synthetic first note, corrected'}),
             ('/api/entries/append', {'text': 'Synthetic second note', 'requestId': rid_b, 'entryTime': now_iso()}),
             ('/api/entries/edit', {'xid': rid_a, 'me': 'Synthetic first note, corrected twice'})]
    before = snapshot(state, DAILY)
    for path, body in steps:
        assert request(client, DAILY, 'POST', path, body).status_code == 200
    native_files, _, _, journal = storage(state, DAILY)
    assert [row[0] for row in journal[-4:]] == ['exchange', 'exchange_edit', 'exchange', 'exchange_edit']
    with sqlite3.connect(state / 'users' / DAILY / 'managed-index.db') as idx:
        rows = idx.execute('SELECT id, created_at, kind, payload FROM journal ORDER BY rowid').fetchall()[-4:]
    # Roll the corpus back to before the run and leave only the journal rows, unapplied.
    restore(state, DAILY, before)
    with sqlite3.connect(state / 'users' / DAILY / 'managed-index.db') as idx:
        idx.executemany('INSERT INTO journal (id, created_at, kind, payload) VALUES (?,?,?,?)', rows)
    modes.python()
    appmod._tenant_states.clear()  # a restarted Python: recovery replays the journal
    assert client.get('/api/months', headers=sign(DAILY, 'GET', '/api/months')).status_code == 200
    replayed_files, _, _, journal = storage(state, DAILY)
    assert all(row[2] == 1 and row[4] is None for row in journal)
    assert replayed_files == native_files


@needs_binary
@needs_server
@needs_cw
def test_a_child_killed_mid_write_leaves_no_torn_state(writers):
    client, start, state = writers
    modes = start()
    server = modes.server
    ids = [str(uuid.uuid4()) for _ in range(30)]
    results = {}
    Recorder.served = []
    # Armed once the child has served a write natively, so the kill always lands mid-run.
    killer = threading.Timer(0.15, lambda: server._proc and server._proc.kill())
    for rid in ids:
        r = request(client, DAILY, 'POST', '/api/entries/append', {'text': f'Synthetic {rid}', 'requestId': rid,
                                                                    'entryTime': now_iso()})
        results[rid] = r.status_code
        if True in Recorder.served and not killer.is_alive() and not killer.finished.is_set():
            killer.start()
    killer.cancel()
    assert set(results.values()) <= {200, 503}, results
    assert True in Recorder.served and not server.ready, 'the child served some writes and was then lost'
    print('kill mid-write:', sorted(results.values()).count(503), 'uncertain of', len(results))
    # Recovery (Python here: the child is gone) settles every journaled entry.
    modes.python()
    appmod._tenant_states.clear()
    assert client.get('/api/months', headers=sign(DAILY, 'GET', '/api/months')).status_code == 200
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        text = '\n'.join(row[0].decode() for row in db.execute("SELECT data FROM files WHERE path LIKE 'Entries/%'"))
    with sqlite3.connect(state / 'users' / DAILY / 'managed-index.db') as idx:
        assert idx.execute('SELECT COUNT(*) FROM journal WHERE applied=0').fetchone()[0] == 0
    for rid, status in results.items():
        count = text.count(f'<!-- xid:{rid} -->')
        assert count == (1 if status == 200 else count) and count <= 1, (rid, status, count)
    # A retry of each uncertain save with its requestId lands it exactly once.
    for rid, status in results.items():
        if status == 503:
            r = request(client, DAILY, 'POST', '/api/entries/append', {'text': f'Synthetic {rid}', 'requestId': rid,
                                                                        'entryTime': now_iso()})
            assert r.status_code == 200
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        text = '\n'.join(row[0].decode() for row in db.execute("SELECT data FROM files WHERE path LIKE 'Entries/%'"))
    assert all(text.count(f'<!-- xid:{rid} -->') == 1 for rid in ids)


@needs_binary
@needs_server
@needs_cw
def test_two_writers_on_one_tenant_keep_their_order(writers):
    import httpx
    client, start, state = writers
    modes = start()
    del modes
    ids = [str(uuid.uuid4()) for _ in range(10)]
    bodies = [json.dumps({'text': f'Synthetic {i}', 'requestId': rid, 'entryTime': now_iso()}).encode()
              for i, rid in enumerate(ids)]

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=appmod.app), base_url='http://diary') as http:
            gate = asyncio.Semaphore(3)  # the assertion leaf allows 4 concurrent checks

            async def one(raw):
                async with gate:
                    headers = sign(DAILY, 'POST', '/api/entries/append', raw, None, b'', 'application/json')
                    return await http.post('/api/entries/append', headers=headers, content=raw)
            return await asyncio.gather(*(one(b) for b in bodies))

    Recorder.served = []
    replies = asyncio.run(run())
    assert [r.status_code for r in replies] == [200] * len(ids)
    assert all(Recorder.served)
    files, _, _, journal = storage(state, DAILY)
    order = [json.loads(p)['xid'] for kind, p, *_ in journal if kind == 'exchange' and json.loads(p)['xid'] in ids]
    text = next(data.decode() for path, data, _ in files if path.startswith('Entries/') and ids[0] in data.decode())
    positions = [text.index(f'<!-- xid:{rid} -->') for rid in order]
    assert sorted(order) == sorted(ids) and positions == sorted(positions)


@needs_binary
@needs_server
def test_tenant_deletion_fences_the_rust_writer(writers, monkeypatch):
    client, start, state = writers
    modes = start()
    server = modes.server
    assert server.fence(DAILY) is True
    Recorder.served = []
    # Python has not tombstoned it (only fenced here): the Rust side declines; Python serves.
    assert request(client, DAILY, 'PUT', '/api/file', {'path': 'fenced.md', 'content': 'x', 'version': None}).status_code == 200
    assert Recorder.served == [False]
    resp = client.request('DELETE', '/api/internal/tenant', headers=sign(MONTHLY, 'DELETE', '/api/internal/tenant'))
    assert resp.status_code == 200 and not (state / 'users' / MONTHLY).exists()
    assert request(client, MONTHLY, 'PUT', '/api/file', {'path': 'x.md', 'content': 'x', 'version': None}).status_code == 410
    assert not (state / 'users' / MONTHLY).exists()
