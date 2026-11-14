"""D5: one writer per tenant across the Python and the Rust Diary processes (agent/tenant_writer.py).

- Coverage: every route and every background thread target is listed in app's inventories (a new
  one fails here until it is classified), and the write paths run for writer tenants with the SQLite
  guard on: an unlocked write to a writer tenant's managed databases is refused and recorded, and
  none is recorded.
- Concurrency: Python-only writes and forwarded writes of one tenant never overlap in time (a
  scripted child always; the real binary when built).
- Lock order: the writer lock is never taken under the registry locks or with a second tenant's,
  waits at most its deadline (503), and a tenant that is not recovered still declines to Rust.
Synthetic data only.
"""
import ast
import asyncio
import base64
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.middleware import Middleware

import agent.app as appmod
from agent import corpus_write_native, rust_forward, tenant_writer
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from agent.workspace_files import file_write
from tests.test_dedicated_storage import A, B, request as plain_request, volume  # noqa: F401
from tests.tenant_binary import needs_binary
from tests.test_rust_forward import DAILY, MONTHLY, needs_server, world  # noqa: F401
from tests.test_rust_forward_d5 import CW_BIN, FakeChild, Recorder, asgi, forwarder, put, writers  # noqa: F401
from tests.test_rust_forward_d5 import request as signed

AGENT = Path(appmod.__file__).resolve().parent
C = '33333333-3333-4333-8333-333333333333'
REAL_REINDEX_DIRTY = appmod._reindex_dirty


@pytest.fixture(autouse=True)
def _writer_registry(monkeypatch):
    monkeypatch.setattr(tenant_writer, '_writers', frozenset())
    monkeypatch.setattr(tenant_writer, 'max_concurrent', {})
    monkeypatch.setattr(tenant_writer, 'denials', [])
    yield
    assert tenant_writer._slots == {}, 'a writer lock was left behind'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ inventories

def test_every_route_is_classified_as_a_write_or_a_read():
    routes = set()
    for route in appmod.app.routes:
        for method in getattr(route, 'methods', None) or ():
            routes.add((method, route.path))
    writes, reads = set(appmod.WRITE_ENTRY_POINTS), set(appmod.READ_ENTRY_POINTS)
    assert not writes & reads
    unlisted = routes - writes - reads
    assert not unlisted, f'classify these routes in app.WRITE_ENTRY_POINTS or READ_ENTRY_POINTS: {sorted(unlisted)}'
    stale = (writes | reads) - routes
    assert not stale, f'listed but not routes: {sorted(stale)}'
    assert {(m, p) for p, m in rust_forward.WRITE_ROUTES.items()} <= writes
    assert all(isinstance(r, APIRoute) or r.path == '/openapi.json' for r in appmod.app.routes)


def _thread_targets():
    found = set()
    for path in sorted(AGENT.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name.endswith('Thread'):
                target = next((k.value for k in node.keywords if k.arg == 'target'), None)
            elif name.endswith('Timer'):
                target = node.args[1] if len(node.args) > 1 else next(
                    (k.value for k in node.keywords if k.arg == 'function'), None)
            elif name in ('asyncio.create_task', 'asyncio.to_thread', 'loop.run_in_executor', 'create_task'):
                target = node.args[0] if node.args else None
                if name == 'asyncio.to_thread' and ast.unparse(target) == 'events.get':
                    continue  # streaming: reads the exchange's event queue
            else:
                continue
            assert target is not None, f'{path.name}:{node.lineno}: a thread without a target'
            found.add(f'{path.stem}:{ast.unparse(target)}')
    return found


def test_every_background_job_is_classified():
    found = _thread_targets()
    unlisted = found - set(appmod.BACKGROUND_ENTRY_POINTS)
    assert not unlisted, f'classify these thread targets in app.BACKGROUND_ENTRY_POINTS: {sorted(unlisted)}'
    assert not set(appmod.BACKGROUND_ENTRY_POINTS) - found, 'listed but no longer started anywhere'


# ------------------------------------------------------------------ the guard

def test_an_unlocked_write_to_a_writer_tenants_database_is_refused(tmp_path):
    import sqlite3
    root = tmp_path / 'users' / B
    tenant_writer.configure(frozenset({B}))
    backend = ManagedCorpusBackend(root, B)  # construction takes the lock itself
    with tenant_writer.hold(B):
        backend.activate({'a.md': b'synthetic'}, {})
    errors = []

    def unlocked():
        try:
            backend.put('b.md', b'synthetic')
        except sqlite3.DatabaseError as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=unlocked)
    worker.start()
    worker.join(10)
    assert errors and 'not authorized' in errors[0]
    assert tenant_writer.denials and tenant_writer.denials[0][0] == B
    assert backend.get('b.md')[0] is None
    with tenant_writer.hold(B):
        assert backend.put('b.md', b'synthetic')[0]
    # Another tenant is never guarded.
    other = ManagedCorpusBackend(tmp_path / 'users' / C, C)
    other.activate({}, {})
    assert other.put('c.md', b'synthetic')[0]
    # Reads never need the lock.
    tenant_writer.denials.clear()
    assert backend.get('a.md')[0] == b'synthetic' and backend.list_dir('')
    assert tenant_writer.denials == []


# ------------------------------------------------------------------ every write path, guarded

def install_forwarder(writers, cls=rust_forward.ForwardMiddleware):
    saved = list(appmod.app.user_middleware)
    position = next(i for i, m in enumerate(saved) if m.cls is appmod.TenantAssertionMiddleware)
    appmod.app.user_middleware.insert(position + 1, Middleware(
        cls, routes={}, writers=writers, is_deleted=appmod._tenant_deleted,
        is_recovered=appmod._rust_recovered, on_uncertain=appmod._rust_write_uncertain))
    appmod.app.middleware_stack = None
    return saved


@pytest.fixture
def guarded(volume, monkeypatch, tmp_path):
    """Tenants A (a legacy volume) and B (fresh, managed) are writer tenants; the forwarder is
    installed with no Rust process, so every write is Python's, under the forwarder's lock or its
    own."""
    monkeypatch.setattr(rust_forward, '_server', None)
    monkeypatch.setattr(appmod, '_reindex_dirty', REAL_REINDEX_DIRTY)
    monkeypatch.setattr(appmod, '_deleted_tenants', {})
    if CW_BIN:
        monkeypatch.setattr(corpus_write_native, 'BINARY', CW_BIN)
    saved = install_forwarder(frozenset({A, B}))
    client = TestClient(appmod.app)
    yield client, volume
    appmod.app.user_middleware[:] = saved
    appmod.app.middleware_stack = None


def wait_threads(prefix=''):
    deadline = time.monotonic() + 20
    for t in list(threading.enumerate()):
        if t is not threading.current_thread() and t.daemon and t.name.startswith(prefix) and t.is_alive():
            t.join(max(0.0, deadline - time.monotonic()))


def test_every_write_path_of_a_writer_tenant_takes_the_writer_lock(guarded, monkeypatch, tmp_path):
    client, vol = guarded
    hb = {'X-Cowork-User-ID': B}
    assert client.get('/api/storage-status', headers=hb).json()['mode'] == 'managed'  # fresh activation
    assert tenant_writer.is_writer(B) and tenant_writer.is_writer(A)
    st = appmod._tenant_state(plain_request(B), recover=False)
    called = []
    # PUT /api/file, POST /api/directory (forwarded routes, served here under the forwarder's lock).
    assert client.put('/api/file', headers=hb, json={'path': 'notes.md', 'content': 'synthetic', 'version': None}).status_code == 200
    assert client.put('/api/file', headers=hb, json={'path': 'gone.md', 'content': 'synthetic', 'version': None}).status_code == 200
    assert client.post('/api/directory', headers=hb, json={'path': 'Folder'}).status_code == 200
    called += ['PUT /api/file', 'POST /api/directory']
    # Workspace ops and trash (Python only).
    copy = client.post('/api/workspace-ops', headers=hb, json={'op': 'copy', 'path': 'notes.md', 'destination': 'Folder/notes.md'})
    assert copy.status_code == 200, copy.text
    assert client.post('/api/workspace-ops', headers=hb, json={'op': 'delete', 'path': 'gone.md', 'version': sha(b'synthetic')}).status_code == 200
    trash = {'action': 'trash', 'id': str(uuid.uuid4()), 'path': 'Folder/notes.md', 'version': sha(b'synthetic')}
    assert client.post('/api/workspace-trash', headers=hb, json=trash).status_code == 200
    called += ['POST /api/workspace-ops', 'POST /api/workspace-trash']
    # Workspace import (preview, apply) from a non-writer tenant's export.
    source = ManagedCorpusBackend(tmp_path / 'source', C)
    source.activate({'raw/note.md': b'synthetic import\n'}, {'entry_layout': 'daily'})
    body = archive(source, '', source.settings())
    url = '/api/workspace-import?name=Synthetic&action='
    zh = {**hb, 'Content-Type': 'application/zip'}
    report = client.post(url + 'preview', content=body, headers=zh).json()
    applied = client.post(url + 'apply&fingerprint=' + report['fingerprint'], content=body, headers=zh)
    assert applied.status_code == 200, applied.text
    called.append('POST /api/workspace-import')
    # Recovery: journal replay, outbox drain, the moved dirty marks and the reindex (embeddings
    # unreachable: the reindex fails and stays pending, which is what it does today).
    st.recovered = False
    st.dirty_on_recover = ['notes.md']
    assert client.get('/api/files', headers=hb).status_code == 200
    assert st.recovered
    # Entries (with the write transform leaf) and chat logging from a plain thread.
    if CW_BIN:
        monkeypatch.setenv('DIARY_CORPUS_WRITE_IMPL', 'rust')
        appmod._tenant_states.clear()
        now = datetime.now().astimezone().isoformat(timespec='seconds')
        rid = str(uuid.uuid4())
        appended = client.post('/api/entries/append', headers=hb, json={'text': 'Synthetic note', 'requestId': rid, 'entryTime': now})
        assert appended.status_code == 200, appended.text
        edit = client.post('/api/entries/edit', headers=hb, json={'xid': rid, 'me': 'Synthetic note, edited', 'assistant': ''})
        assert edit.status_code == 200, edit.text
        st = appmod._tenant_state(plain_request(B))
        errors = []

        def chat_logging():
            try:
                st.store.log_exchange(day=date.today(), sub_header='Synthetic', me_text='Synthetic chat',
                                      claude_text='Synthetic reply', now=datetime.now())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        worker = threading.Thread(target=chat_logging)
        worker.start()
        worker.join(30)
        assert not errors, errors
        # External sources import.
        ext = tmp_path / 'external'
        ext.mkdir()
        (ext / '2026-08-15.md').write_text('Synthetic external note.', encoding='utf-8')
        monkeypatch.setenv('DIARY_EXTERNAL_SOURCES', str(ext))
        imported = client.post('/api/external-sources/import', headers=hb, json={'source_path': str(ext), 'rel_path': '2026-08-15.md'})
        assert imported.status_code == 200, imported.text
        called += ['POST /api/entries/append', 'POST /api/entries/edit', 'chat logging', 'POST /api/external-sources/import']
    # The background reindex of one day (a plain thread: a fresh context, waits for the lock).
    worker = threading.Thread(target=appmod._reindex_today, args=(st, date.today()))
    worker.start()
    worker.join(30)
    # Storage backup: the failure bookkeeping (the remote is unreachable) under the lock.
    with tenant_writer.hold(B), st.managed.db() as db:
        st.managed.set_meta(db, 'due', 0)
    storage = base64.urlsafe_b64encode(json.dumps({'kind': 'webdav', 'baseUrl': 'http://127.0.0.1:9', 'username': 'qa',
                                                   'corpusRoot': 'Diary', 'secret': 'synthetic'}).encode()).decode()
    backup = client.post('/api/storage-backup', headers={**hb, 'X-Cowork-Storage': storage}).json()
    assert backup['backup'] == 'failed', backup
    called.append('POST /api/storage-backup')
    # Embedding re-index bookkeeping (when the vector extension loads here).
    migration = appmod.embedding_migration()
    active = migration.config.active()
    if active is not None:
        migration._drop_everywhere(active)
    # Storage import of the legacy volume tenant A (activation of a verified copy).
    (vol / 'Documents/Diary/Entries').mkdir(parents=True, exist_ok=True)
    (vol / 'Documents/Diary/Entries/a.md').write_bytes(b'# synthetic original')
    ha = {'X-Cowork-User-ID': A}
    preview = client.post('/api/storage-import', headers=ha, json={}).json()
    done = client.post('/api/storage-import', headers=ha, json={'fingerprint': preview['fingerprint']})
    assert done.status_code == 200 and done.json()['imported'], done.text
    assert client.get('/api/storage-status', headers=ha).json()['mode'] == 'managed'
    called.append('POST /api/storage-import')
    wait_threads()
    # Tenant deletion.
    assert client.request('DELETE', '/api/internal/tenant', headers=hb).status_code == 200
    called.append('DELETE /api/internal/tenant')
    assert tenant_writer.denials == [], f'unlocked writes: {tenant_writer.denials}'
    assert tenant_writer.max_concurrent and set(tenant_writer.max_concurrent.values()) == {1}
    print('exercised:', ', '.join(called))


def test_reads_of_a_writer_tenant_never_wait_for_its_writer_lock(guarded):
    client, _ = guarded
    hb = {'X-Cowork-User-ID': B}
    assert client.get('/api/storage-status', headers=hb).json()['mode'] == 'managed'
    assert client.put('/api/file', headers=hb, json={'path': 'notes.md', 'content': 'synthetic', 'version': None}).status_code == 200
    assert client.get('/api/files', headers=hb).status_code == 200  # the state is recovered
    release, taken = threading.Event(), threading.Event()

    def long_python_write():  # e.g. a re-index holding the store lock across embedding calls
        with tenant_writer.hold(B):
            taken.set()
            release.wait(10)
    t = threading.Thread(target=long_python_write)
    t.start()
    try:
        assert taken.wait(5)
        started = time.monotonic()
        assert client.get('/api/files', headers=hb).status_code == 200
        assert client.post('/api/file', headers=hb, json={'path': 'notes.md'}).json()['content'] == 'synthetic'
        assert client.get('/api/storage-status', headers=hb).status_code == 200
        assert client.get('/api/months', headers=hb).status_code == 200
        assert time.monotonic() - started < 2.0, 'a read waited for the writer lock'
    finally:
        release.set()
        t.join(5)
    assert tenant_writer.denials == []


# ------------------------------------------------------------------ concurrency

class TimedForward(rust_forward.ForwardMiddleware):
    """Records when each forward to the Rust side is on the wire (it runs only under the lock)."""
    spans = []

    async def _forward(self, server, scope, deadline, **kwargs):
        started = time.monotonic()
        try:
            return await super()._forward(server, scope, deadline, **kwargs)
        finally:
            TimedForward.spans.append((started, time.monotonic()))


@contextmanager
def python_write_spans(tenant):
    """Records when a Python connection to the tenant's managed-diary.db wrote (connect to commit)."""
    spans, real = [], ManagedCorpusBackend.db

    @contextmanager
    def timed(self):
        cm = real(self)
        db = cm.__enter__()
        started, before = time.monotonic(), db.total_changes
        try:
            yield db
        except BaseException as exc:
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise
        wrote = db.total_changes > before
        if wrote and self.tenant == tenant:
            time.sleep(0.05)  # widen the window a racing write would fall into
        cm.__exit__(None, None, None)
        if wrote and self.tenant == tenant:
            spans.append((started, time.monotonic()))
    ManagedCorpusBackend.db = timed
    try:
        yield spans
    finally:
        ManagedCorpusBackend.db = real


def overlaps(a, b):
    return [(x, y) for i, x in enumerate(a) for j, y in enumerate(b)
            if not (a is b and i >= j) and x[0] < y[1] and y[0] < x[1]]


def run_concurrently(jobs, workers=3):
    errors, sem = [], threading.Semaphore(workers)

    def one(job):
        with sem:
            try:
                job()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
    threads = [threading.Thread(target=one, args=(job,)) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads), 'a writer hung'
    assert not errors, errors


def test_python_only_and_forwarded_writes_of_one_tenant_never_overlap(volume, monkeypatch):
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    monkeypatch.setattr(appmod, '_deleted_tenants', {})
    monkeypatch.setattr(rust_forward, '_server', None)
    saved = install_forwarder(frozenset({B}), TimedForward)
    child = FakeChild('slow')
    try:
        client = TestClient(appmod.app)
        hb = {'X-Cowork-User-ID': B}
        assert client.get('/api/storage-status', headers=hb).json()['mode'] == 'managed'
        for i in range(4):
            assert client.put('/api/file', headers=hb, json={'path': f'n{i}.md', 'content': 'synthetic', 'version': None}).status_code == 200
        st = appmod._tenant_state(plain_request(B))
        server = rust_forward.RustServer(lambda p, t, f: {}, port=child.port, restart_delays=())
        server.ready, server.forward_secret = True, 'synthetic'
        monkeypatch.setattr(rust_forward, '_server', server)
        TimedForward.spans = []

        def forwarded(i):
            return lambda: TestClient(appmod.app).put('/api/file', headers=hb, json={'path': f'r{i}.md', 'content': 'x', 'version': None}).raise_for_status()

        def python_op(i):  # Python-only route
            return lambda: TestClient(appmod.app).post('/api/workspace-ops', headers=hb, json={
                'op': 'copy', 'path': f'n{i}.md', 'destination': f'c{i}.md'}).raise_for_status()

        def background(i):  # a plain thread writing through the store (what chat logging and reindex do)
            return lambda: file_write(st.store, {'path': f'b{i}.md', 'content': 'synthetic', 'version': None})

        with python_write_spans(B) as spans:
            jobs = []
            for i in range(4):
                jobs += [forwarded(i), python_op(i), background(i)]
            run_concurrently(jobs, workers=6)
        assert len(TimedForward.spans) == 4 and len(spans) >= 8
        assert overlaps(spans, TimedForward.spans) == [], 'a Python write ran while a forwarded write was on the wire'
        assert overlaps(spans, spans) == [] and overlaps(TimedForward.spans, TimedForward.spans) == []
        assert tenant_writer.denials == []
    finally:
        child.close()
        appmod.app.user_middleware[:] = saved
        appmod.app.middleware_stack = None


@needs_binary
@needs_server
def test_python_only_and_native_writes_never_overlap_on_the_real_binary(writers, monkeypatch):
    client, start, state = writers
    modes = start()
    for i in range(4):  # written natively
        assert signed(client, DAILY, 'PUT', '/api/file', {'path': f'n{i}.md', 'content': 'synthetic', 'version': None}).status_code == 200
    st = next(s for k, s in appmod._tenant_states.items() if k.startswith(DAILY + ':'))
    spans_fwd, real_forward = [], rust_forward.ForwardMiddleware._forward

    async def timed_forward(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return await real_forward(self, *args, **kwargs)
        finally:
            spans_fwd.append((started, time.monotonic()))
    monkeypatch.setattr(rust_forward.ForwardMiddleware, '_forward', timed_forward)
    Recorder.served = []

    def native(i):
        return lambda: signed(TestClient(appmod.app), DAILY, 'PUT', '/api/file',
                              {'path': f'r{i}.md', 'content': 'x' * 4096, 'version': None}).raise_for_status()

    def python_op(i):
        return lambda: signed(TestClient(appmod.app), DAILY, 'POST', '/api/workspace-ops',
                              {'op': 'copy', 'path': f'n{i}.md', 'destination': f'c{i}.md'}).raise_for_status()

    def background(i):
        return lambda: file_write(st.store, {'path': f'b{i}.md', 'content': 'synthetic', 'version': None})

    with python_write_spans(DAILY) as spans:
        jobs = []
        for i in range(4):
            jobs += [native(i), python_op(i), background(i)]
        run_concurrently(jobs, workers=3)  # the assertion leaf allows 4 concurrent checks
    # Python's workspace ops leave index work behind, after which the Rust side declines (Python then
    # serves those under the same lock): at least the first writes are native.
    assert True in Recorder.served and len(spans_fwd) >= 4, Recorder.served
    assert len(spans) >= 8
    assert overlaps(spans, spans_fwd) == [], 'a Python write ran while a native write was in flight'
    assert tenant_writer.denials == []
    del modes


# ------------------------------------------------------------------ lock order and deadline

def test_lock_order_is_enforced_and_waits_are_bounded():
    tenant_writer.configure(frozenset({A, B}))
    with appmod._tenant_lock:
        with pytest.raises(tenant_writer.LockOrderError):
            with tenant_writer.hold(B):
                pass
    with appmod._backup_backends_lock:
        with pytest.raises(tenant_writer.LockOrderError):
            with tenant_writer.hold(B):
                pass
    with tenant_writer.hold(B):
        with tenant_writer.hold(B):  # re-entrant in one context
            assert tenant_writer.held(B)
        with pytest.raises(tenant_writer.LockOrderError):
            with tenant_writer.hold(A):
                pass
        # The registry lock is taken under the writer lock (the right order).
        with appmod._tenant_lock:
            pass
    # A non-writer tenant is never locked, whatever is held.
    with appmod._tenant_lock, tenant_writer.hold(C):
        assert not tenant_writer.held(C)
    release, taken = threading.Event(), threading.Event()

    def holder():
        with tenant_writer.hold(B):
            taken.set()
            release.wait(10)
    t = threading.Thread(target=holder)
    t.start()
    assert taken.wait(5)
    started = time.monotonic()
    with pytest.raises(tenant_writer.WriterBusy):
        with tenant_writer.hold(B, deadline_s=0.3):
            pass
    assert 0.25 < time.monotonic() - started < 2
    # Other tenants proceed meanwhile.
    with tenant_writer.hold(A):
        pass
    release.set()
    t.join(5)


def test_a_forwarded_write_waits_at_most_the_deadline_then_503(monkeypatch):
    child = FakeChild('native')
    try:
        _, spy, mw = forwarder(child.port, monkeypatch)
        monkeypatch.setattr(tenant_writer, 'DEADLINE_S', 0.3)
        release, taken = threading.Event(), threading.Event()
        user = dict(put()[0])[b'x-cowork-user-id'].decode()

        def holder():
            with tenant_writer.hold(user):
                taken.set()
                release.wait(10)
        t = threading.Thread(target=holder)
        t.start()
        assert taken.wait(5)
        headers, body = put()
        status, reply = asyncio.run(asgi(mw, headers, [body]))
        assert status == 503 and reply == rust_forward.BUSY
        assert child.requests == [] and spy.calls == []
        release.set()
        t.join(5)
        status, reply = asyncio.run(asgi(mw, headers, [body]))
        assert status == 200 and reply == b'{"native":true}'
    finally:
        child.close()


def test_a_tenant_that_is_not_recovered_still_declines_to_rust(monkeypatch):
    """The writer lock changes nothing about recovery: the forward says not recovered, and the Rust
    side declines (tests/test_rust_forward_d5.py runs that decline against the real binary)."""
    child = FakeChild('decline')
    try:
        _, spy, mw = forwarder(child.port, monkeypatch)
        headers, body = put()
        status, reply = asyncio.run(asgi(mw, headers, [body]))
        assert status == 200 and reply == b'{"python":true}'
        sent = child.requests[0][0]
        assert sent[b'x-noevia-diary-recovered'] == b'0' and sent[b'x-noevia-diary-writer'] == b'1'
        assert spy.calls == [('/api/file', body)]
    finally:
        child.close()


# ------------------------------------------------------------------ review ba3a712: P1, P2

class StubChild:
    """Stands in for the Rust child process: does not exit until `exit()`; wait() times out until
    then (a child whose exit is not yet confirmed)."""

    def __init__(self):
        self.exited, self.kills = threading.Event(), 0

    def poll(self):
        return 0 if self.exited.is_set() else None

    def kill(self):
        self.kills += 1

    def wait(self, timeout=None):
        import subprocess
        if not self.exited.wait(timeout):
            raise subprocess.TimeoutExpired('noevia-diary', timeout)
        return 0

    def exit(self):
        self.exited.set()


def blocked_writer(uid, deadline=0.3):
    """From a plain thread (a Python-only writer): None when the lock was taken, else the reason."""
    out = []

    def run():
        try:
            with tenant_writer.hold(uid, deadline_s=deadline):
                out.append(None)
        except tenant_writer.WriterBusy as exc:
            out.append(exc.reason or 'busy')
    t = threading.Thread(target=run)
    t.start()
    t.join(10)
    return out[0]


def test_a_cancelled_forward_is_uncertain_and_keeps_writers_out_until_the_child_exits(monkeypatch):
    """P1: a request cancelled after it may have reached the child must not free the tenant's
    lock while the child can still write."""
    monkeypatch.setattr(rust_forward, 'REAP_WAIT_S', 0.05)
    child, uncertain = FakeChild('hang'), []
    stub = StubChild()
    try:
        server, spy, mw = forwarder(child.port, monkeypatch, deadline=10, uncertain=uncertain,
                                    writers=frozenset({DAILY, MONTHLY}))
        server._proc = stub
        user = DAILY

        async def run():
            headers, body = put()
            task = asyncio.ensure_future(asgi(mw, headers, [body]))
            for _ in range(500):
                if child.requests:
                    break
                await asyncio.sleep(0.01)
            assert child.requests, 'the write reached the child'
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        asyncio.run(run())
        assert uncertain == [user], 'the tenant is marked recovery-pending'
        assert not server.ready and server.reaping == 1
        assert tenant_writer._slots == {}, 'the request itself released its reference'
        # Every writer, of this tenant or any other, waits for the child's confirmed exit.
        assert blocked_writer(user) == 'restarting'
        assert blocked_writer(MONTHLY) == 'restarting'
        assert spy.calls == []
        stub.exit()
        deadline = time.monotonic() + 5
        while server.reaping and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.reaping == 0 and stub.kills >= 1
        assert blocked_writer(user) is None
    finally:
        stub.exit()
        child.close()


def test_python_only_writers_wait_for_a_lost_child_until_its_exit_is_confirmed(monkeypatch):
    """P1: the reaping barrier is in the common admission path, not only the forwarded routes, and
    it is lowered only once the child has exited (not when kill/wait fail)."""
    monkeypatch.setattr(rust_forward, 'REAP_WAIT_S', 0.05)
    child, uncertain = FakeChild('hang'), []
    stub = StubChild()
    try:
        server, spy, mw = forwarder(child.port, monkeypatch, deadline=0.3, uncertain=uncertain,
                                    writers=frozenset({DAILY, MONTHLY}))
        server._proc = stub
        user = DAILY
        headers, body = put()
        assert asyncio.run(asgi(mw, headers, [body])) == (503, rust_forward.UNCERTAIN)
        assert uncertain == [user]
        # The reaper's wait has timed out several times by now: the barrier is still up.
        time.sleep(0.3)
        assert server.reaping == 1
        assert blocked_writer(user) == 'restarting'      # a Python-only write of the tenant
        assert blocked_writer(MONTHLY) == 'restarting'   # and of any other writer tenant
        monkeypatch.setattr(tenant_writer, 'DEADLINE_S', 0.2)
        with pytest.raises(tenant_writer.WriterBusy):    # a store write (CorpusStore._write_lock)
            with tenant_writer.TenantWriteLock(user):
                pass
        with tenant_writer.hold(C):                      # a non-writer tenant is never held up
            pass
        stub.exit()
        deadline = time.monotonic() + 5
        while server.reaping and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.reaping == 0
        assert blocked_writer(user) is None and blocked_writer(MONTHLY) is None
    finally:
        stub.exit()
        child.close()


def test_a_tenant_deletion_waits_for_the_barrier_too(guarded, monkeypatch):
    client, _ = guarded
    hb = {'X-Cowork-User-ID': B}
    assert client.get('/api/storage-status', headers=hb).status_code == 200
    server = rust_forward.RustServer(lambda p, t, f: {}, port=free_port_local(), restart_delays=())
    monkeypatch.setattr(tenant_writer, 'DEADLINE_S', 0.3)
    server.reaping = 1
    try:
        resp = client.request('DELETE', '/api/internal/tenant', headers=hb)
        assert resp.status_code == 503 and resp.json() == json.loads(rust_forward.RESTARTING)
    finally:
        server.reaping = 0
    assert client.request('DELETE', '/api/internal/tenant', headers=hb).status_code == 200


def free_port_local():
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def test_a_cached_statement_is_authorized_again_at_every_execution(tmp_path):
    """P2: the guard holds for prepared/cached statements: a statement first run under the lock is
    refused when run again without it."""
    import sqlite3
    from agent.journal import Journal
    tenant_writer.configure(frozenset({B}))
    path = tmp_path / 'users' / B / 'managed-index.db'
    journal = Journal(path)  # a new file: its schema is created exempt
    with tenant_writer.hold(B):
        journal.mark_dirty('a.md')
    errors = []

    def unlocked():
        try:
            journal.mark_dirty('b.md')  # the same statement text, on the same connection
        except sqlite3.DatabaseError as exc:
            errors.append(str(exc))
    t = threading.Thread(target=unlocked)
    t.start()
    t.join(10)
    assert errors and 'not authorized' in errors[0]
    assert journal.dirty_documents() == ['a.md']
    assert tenant_writer.denials


def test_attach_is_refused_so_no_connection_reaches_a_writer_database_under_another_name(tmp_path):
    import sqlite3
    tenant_writer.configure(frozenset({B}))
    target = tmp_path / 'users' / B / 'managed-diary.db'
    ManagedCorpusBackend(target.parent, B)
    other = tmp_path / 'users' / C / 'managed-diary.db'
    ManagedCorpusBackend(other.parent, C)
    for conn in (tenant_writer.connect(':memory:'), tenant_writer.connect(str(other))):
        with tenant_writer.hold(C):  # even holding another tenant's lock
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute('ATTACH DATABASE ? AS b', (str(target),))
        conn.close()


def test_agent_code_opens_sqlite_only_through_tenant_writer_connect():
    offenders = []
    for path in sorted(AGENT.glob('*.py')):
        if path.name == 'tenant_writer.py':
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ('sqlite3.connect', 'connect'):
                offenders.append(f'{path.name}:{node.lineno}')
    assert not offenders, f'use tenant_writer.connect: {offenders}'


def test_opening_a_writer_tenants_index_writes_its_schema_only_under_the_lock(tmp_path, monkeypatch):
    """P2: schema initialization is covered: an existing database whose schema is in place is
    opened without any write; a missing object is created only under the tenant's lock."""
    import sqlite3
    from agent.journal import Journal
    from agent.retrieval import Retriever
    tenant_writer.configure(frozenset({B}))
    path = tmp_path / 'users' / B / 'managed-index.db'
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:  # an older file: the journal table only
        db.execute('CREATE TABLE journal (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, '
                   'payload TEXT NOT NULL, applied INTEGER NOT NULL DEFAULT 0, applied_at TEXT, '
                   'attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT)')
    monkeypatch.setattr(tenant_writer, 'DEADLINE_S', 0.3)
    release, taken = threading.Event(), threading.Event()

    def holder():
        with tenant_writer.hold(B):
            taken.set()
            release.wait(10)
    t = threading.Thread(target=holder)
    t.start()
    assert taken.wait(5)
    try:
        with pytest.raises(tenant_writer.WriterBusy):
            Journal(path)  # needs to create dirty_documents: waits for the lock
    finally:
        release.set()
        t.join(5)
    Journal(path)  # lock free: created under it
    Retriever(path, llm=None)
    with sqlite3.connect(path) as db:
        names = {r[0] for r in db.execute('SELECT name FROM sqlite_master')}
    assert {'dirty_documents', 'journal', 'chunks', 'vec_meta'} <= names
    # Schema in place: opening again from a plain thread writes nothing (no lock, no denial).
    tenant_writer.denials.clear()
    errors = []

    def reopen():
        try:
            Journal(path)
            Retriever(path, llm=None)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    w = threading.Thread(target=reopen)
    w.start()
    w.join(10)
    assert not errors and tenant_writer.denials == []


def test_recovery_after_a_lost_write_invalidates_documents_created_since_the_build(guarded, monkeypatch):
    """P2: a document a lost Rust write created after the state was built (no dirty row survived)
    is reindexed by the surviving state's recovery."""
    client, _ = guarded
    hb = {'X-Cowork-User-ID': B}
    assert client.get('/api/storage-status', headers=hb).status_code == 200
    st = appmod._tenant_state(plain_request(B))
    assert st.recovered
    new = 'Entries/2030/January/January 7, 2030.md'
    with tenant_writer.hold(B):  # what the lost child left: the corpus saved, no dirty row
        assert st.managed.put(new, b'## Monday, January 7, 2030\n')[0]
    assert new not in st.journal.dirty_documents()
    seen = []

    def record(state):
        seen.extend(state.journal.dirty_documents())
    monkeypatch.setattr(appmod, '_reindex_dirty', record)
    appmod._rust_write_uncertain(B)  # what the forwarder's on_uncertain does
    assert not st.recovered
    assert client.get('/api/files', headers=hb).status_code == 200
    assert st.recovered and new in seen


def test_a_parent_served_write_recovers_without_sweeping_every_document(guarded, monkeypatch):
    """S6b (main mode): after the parent served a native write, the state recovers again, but only
    the documents the journal marks are reindexed: the parent marks its document dirty before it
    writes, so no sweep of every Markdown document is needed (unlike a lost forwarded write)."""
    client, _ = guarded
    hb = {'X-Cowork-User-ID': B}
    assert client.get('/api/storage-status', headers=hb).status_code == 200
    st = appmod._tenant_state(plain_request(B))
    assert st.recovered
    marked, other = 'Entries/2030/January/January 8, 2030.md', 'Entries/2030/January/January 9, 2030.md'
    with tenant_writer.hold(B):  # what the parent's native write leaves: the mark, then the corpus
        st.journal.mark_dirty(marked)
        assert st.managed.put(marked, b'## Tuesday, January 8, 2030\n')[0]
        assert st.managed.put(other, b'## Wednesday, January 9, 2030\n')[0]
    seen = []

    def record(state):
        seen.extend(state.journal.dirty_documents())
    monkeypatch.setattr(appmod, '_reindex_dirty', record)
    appmod._rust_wrote(B)  # what ParentMiddleware's on_written does
    assert not st.recovered and not getattr(st, 'dirty_on_recover', False)
    assert client.get('/api/files', headers=hb).status_code == 200
    assert st.recovered and marked in seen and other not in seen


def test_every_fallback_path_bounds_the_body_and_its_deadline(monkeypatch):
    """P2: with the Rust process unavailable, a writer tenant's write still never runs on Python
    with a body over the bound or unfinished at the deadline."""
    child = FakeChild('native')
    try:
        monkeypatch.setattr(rust_forward, 'MAX_WRITE_BODY', 64)
        server, spy, mw = forwarder(child.port, monkeypatch, deadline=0.3)
        server.ready = False  # Rust unavailable: Python serves
        big = b'{"path":"a.md","content":"' + b'x' * 80 + b'","version":null}'
        chunked = [h for h in put(body=big)[0] if h[0] != b'content-length'] + [(b'transfer-encoding', b'chunked')]
        assert asyncio.run(asgi(mw, chunked, [big[:40], big[40:]])) == (413, rust_forward.TOO_LARGE)
        headers, body = put()
        assert asyncio.run(asgi(mw, headers, [body[:5], body[5:]], gap=0.25)) == (408, rust_forward.BODY_TIMEOUT)
        assert spy.calls == []
        # A complete, bounded body is served by Python with exactly its bytes.
        assert asyncio.run(asgi(mw, headers, [body[:5], body[5:]])) == (200, b'{"python":true}')
        assert spy.calls == [('/api/file', body)] and child.requests == []
    finally:
        child.close()


def test_a_handler_thread_keeps_the_lock_after_its_request_is_cancelled():
    """The middleware's reference and the handler thread's joined reference: the lock is freed by
    the last one, so a sync handler still writing after a cancellation is never overlapped."""
    tenant_writer.configure(frozenset({B}))
    inside, release, done = threading.Event(), threading.Event(), threading.Event()

    async def request():
        h = await tenant_writer.acquire_async(B, time.monotonic() + 1)
        ctx = __import__('contextvars').copy_context()

        def handler():  # what Starlette's threadpool runs, in the request's context
            with tenant_writer.hold(B):
                inside.set()
                release.wait(10)
            done.set()
        threading.Thread(target=lambda: ctx.run(handler)).start()
        inside.wait(5)
        h.release()  # the request is cancelled: the middleware lets go
    asyncio.run(request())
    assert blocked_writer(B, deadline=0.2) == 'busy', 'the handler thread still holds the lock'
    release.set()
    assert done.wait(5)
    assert blocked_writer(B) is None
