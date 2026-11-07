"""Real retrieval/vector/schema publication must retain the primary SQLite reservation.

Only deterministic prepared vectors are used; no embedding or model endpoint runs.
"""
import sqlite3
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest
from agent import tenant_writer, retrieval, vector_slots as slots
from agent.embedding import Target
from agent.embedding_migration import EmbeddingMigration
from agent.managed_storage import ManagedCorpusBackend

UID = '12345678-1234-4234-8234-123456789abc'


def load_vec(conn):
    import sqlite_vec
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
    return True


class CommitPause:
    def __init__(self, conn, entered, resume, rollback=False):
        self.conn, self.entered, self.resume = conn, entered, resume
        self.paused = False
        self.pause_rollback = rollback
    def __getattr__(self, name):
        return getattr(self.conn, name)
    def commit(self):
        if self.pause_rollback:
            raise sqlite3.OperationalError("synthetic commit failure")
        if not self.paused:
            self.paused = True
            self.entered.set()
            assert self.resume.wait(6)
        return self.conn.commit()
    def rollback(self):
        if self.pause_rollback and not self.paused:
            self.paused = True
            self.entered.set()
            assert self.resume.wait(6)
        return self.conn.rollback()
    def __enter__(self):
        return self
    def __exit__(self, kind, value, trace):
        if kind is None:
            self.commit()
        else:
            self.conn.rollback()


@pytest.fixture
def index_factory(tmp_path, monkeypatch):
    tenant_writer.configure_data_root(tmp_path)
    monkeypatch.setattr(tenant_writer, '_writers', frozenset())
    monkeypatch.setattr(retrieval, '_load_vec_extension', load_vec)
    root = tmp_path / 'users' / UID
    backend = ManagedCorpusBackend(root, UID)
    backend.activate({}, {})
    index = root / 'managed-index.db'
    opened = []

    def prepare(mode, entered=None, resume=None, rollback=False):
        return prepare_action(mode, backend, index, monkeypatch, opened, entered, resume, rollback)

    yield backend, index, prepare
    for r in opened:
        r.close()
    tenant_writer.configure_data_root(None)


def prepare_action(mode, backend, index, monkeypatch, opened, entered=None, resume=None, rollback=False):
    if mode == 'schema':
        connect = tenant_writer.connect
        def hooked(database, **kwargs):
            conn = connect(database, **kwargs)
            name = Path(str(database).split('?', 1)[0].removeprefix('file:')).name
            if name == 'managed-index.db' and entered is not None:
                paused = False
                def trace(sql):
                    nonlocal paused
                    if not paused and sql.lstrip().upper().startswith('CREATE'):
                        paused = True
                        entered.set()
                        assert resume.wait(6)
                conn.set_trace_callback(trace)
            return conn
        monkeypatch.setattr(tenant_writer, 'connect', hooked)
        return lambda: retrieval.ensure_schema_file(index)

    r = retrieval.Retriever(index, SimpleNamespace(embed_model='synthetic'))
    opened.append(r)
    target = r.embedding.active()
    plan = retrieval.ReindexPlan('synthetic.md', [('2030-01-01','synthetic','synthetic text','hash')],
                                 {'hash':[1.0,0.0]}, target=target)
    if mode in ('fill', 'drop'):
        assert r.apply_reindex(plan) == 1
    if mode == 'adopt':
        r._conn.execute('CREATE VIRTUAL TABLE vec_items USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[2])')
        r._conn.execute("INSERT INTO vec_meta VALUES ('dim','2')")
        r._conn.commit()
    if mode in ('reindex', 'adopt'):
        if entered is not None:
            r._conn = CommitPause(r._conn, entered, resume, rollback)
        return (lambda: r.apply_reindex(plan)) if mode == 'reindex' else r._active_slot

    config = SimpleNamespace(version=1, active=lambda: target)
    embedder = SimpleNamespace(embed=lambda _target, texts, *a, **k: [[0.0,1.0] for _ in texts])
    migration = EmbeddingMigration(SimpleNamespace(config=config, embedder=embedder), index.parent.parent.parent/'index.db',
                                   load_vec=load_vec, sleep=lambda _: None)
    migration._files = lambda: [index]
    original_open = migration._open
    if entered is not None:
        def paused_open(path):
            conn = original_open(path)
            return CommitPause(conn, entered, resume, rollback) if conn is not None else None
        migration._open = paused_open
    if mode == 'fill':
        return lambda: migration._fill(index, Target(None, 'synthetic-next'), 1, budget=[1])
    return lambda: migration._drop_everywhere(target)


def snapshot(index):
    if not index.exists():
        return []
    conn = sqlite3.connect(index)
    try:
        load_vec(conn)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return [(table, conn.execute('SELECT * FROM "'+table+'"').fetchall()) for table in tables]
    finally:
        conn.close()


@pytest.mark.parametrize('mode', ['reindex','fill','drop','adopt','schema'])
def test_actual_index_commit_excludes_late_primary_intent(index_factory, mode):
    backend, index, prepare = index_factory
    entered, resume = threading.Event(), threading.Event()
    work = prepare(mode, entered, resume)
    errors = []
    def run():
        try:
            work()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5), errors
        blocked = False
        with sqlite3.connect(backend.db_path, timeout=.03) as contender:
            try:
                contender.execute("INSERT INTO meta VALUES ('backup_effect_pending','synthetic-late-intent')")
                contender.commit()
            except sqlite3.OperationalError:
                blocked = True
        assert blocked, 'an intent published after index DML authorization but before the actual index commit'
    finally:
        resume.set()
        thread.join(timeout=7)
    assert not thread.is_alive()
    assert errors == []
    with sqlite3.connect(backend.db_path) as contender:
        contender.execute("INSERT INTO meta VALUES ('backup_effect_pending','next-effect-after-index-settled')")


@pytest.mark.parametrize('mode', ['reindex', 'fill', 'drop'])
def test_failed_publication_retains_primary_until_index_rollback(index_factory, mode):
    backend, index, prepare = index_factory
    entered, resume = threading.Event(), threading.Event()
    work = prepare(mode, entered, resume, rollback=True)
    before = snapshot(index)
    errors = []
    def run():
        try:
            work()
        except sqlite3.OperationalError as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        with sqlite3.connect(backend.db_path, timeout=.03) as contender:
            with pytest.raises(sqlite3.OperationalError):
                contender.execute("INSERT INTO meta VALUES ('backup_effect_pending','too-early-during-rollback')")
    finally:
        resume.set()
        thread.join(timeout=7)
    assert not thread.is_alive()
    assert len(errors) == (0 if mode == 'drop' else 1)
    assert snapshot(index) == before


@pytest.mark.parametrize('mode', ['reindex','fill','drop','adopt','schema'])
def test_primary_intent_wins_without_index_changes(index_factory, mode):
    backend, index, prepare = index_factory
    work = prepare(mode)
    before = snapshot(index)
    with sqlite3.connect(backend.db_path) as db:
        db.execute("INSERT INTO meta VALUES ('backup_effect_pending','synthetic-first-intent')")
    try:
        work()
    except (tenant_writer.WriterBusy, sqlite3.Error):
        pass  # Preserve each existing caller's refusal/failed-pass behavior.
    assert snapshot(index) == before


@pytest.mark.parametrize('seeded', [False, True])
def test_slot_and_stats_reads_do_not_acquire_primary_when_no_migration_needed(index_factory, seeded):
    backend, index, _ = index_factory
    r = retrieval.Retriever(index, SimpleNamespace(embed_model='synthetic'))
    try:
        if seeded:
            plan = retrieval.ReindexPlan('read.md', [('2030-01-01','synthetic','text','read-hash')],
                                         {'read-hash':[1.0,0.0]}, target=r.embedding.active())
            assert r.apply_reindex(plan) == 1
        with sqlite3.connect(backend.db_path) as db:
            db.execute("INSERT INTO meta VALUES ('backup_effect_pending','uncertain')")
        assert r.stats()['chunks'] == (1 if seeded else 0)
        _, found = r._active_slot()
        assert (found is not None) is seeded
        # Exercise the real vector SELECT and its slot=None fallback under the instance lock;
        # neither may attempt legacy adoption or reserve the primary from inside that lock.
        with r._lock:
            assert len(r._search_vector([1.0, 0.0], 3, 0.0)) == (1 if seeded else 0)
    finally:
        r.close()


def test_prepared_reindex_borrows_existing_primary_transaction(index_factory):
    backend, index, _ = index_factory
    r = retrieval.Retriever(index, SimpleNamespace(embed_model='synthetic'))
    try:
        plan = retrieval.ReindexPlan('borrow.md', [('2030-01-01','synthetic','text','borrow-hash')],
                                     {'borrow-hash':[1.0,0.0]}, target=r.embedding.active())
        with backend.db() as db:
            db.execute('BEGIN IMMEDIATE')
            assert r.apply_reindex(plan) == 1
            with sqlite3.connect(backend.db_path, timeout=.03) as contender:
                with pytest.raises(sqlite3.OperationalError):
                    contender.execute("INSERT INTO meta VALUES ('backup_effect_pending','too-early')")
        assert r.stats()['chunks'] == 1
    finally:
        r.close()


@pytest.mark.parametrize('gap', [0, 1], ids=['initial-get','terminal-next-intent'])
@pytest.mark.parametrize('child_loss', [False, True])
@pytest.mark.parametrize('mode', ['reindex','fill'])
def test_real_publication_orders_with_cancelled_dav_caller(lab, monkeypatch, gap, child_loss, mode):
    import asyncio
    import base64
    import json
    import os
    import time
    import httpx
    from agent import rust_forward
    from tests.test_rust_forward_d5 import asgi
    backend, state, storage, _, base = lab
    monkeypatch.setattr(tenant_writer, '_writers', frozenset({UID}))
    monkeypatch.setattr(retrieval, '_load_vec_extension', load_vec)
    # Setup is itself a real publication, before any backup is admitted.
    opened = []
    entered, resume = threading.Event(), threading.Event()
    with tenant_writer.hold(UID):
        action = prepare_action(mode, backend, backend.root/'managed-index.db', monkeypatch, opened, entered, resume)
    errors = []
    def publish():
        try:
            with tenant_writer.hold(UID):
                action()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=publish)
    server = rust_forward.RustServer(lambda *args: {}, port=httpx.URL(base).port, restart_delays=())
    server.ready, server.forward_secret, server._token, server._proc = True, 'forward', 'ctl', state['proc']
    state['hold_get_objects'] = gap
    state['block'] = False

    async def run():
        mw = rust_forward.ForwardMiddleware(None, server=server, routes={}, writers=frozenset({UID}))
        request = asyncio.create_task(asgi(mw, [(b'x-cowork-user-id', UID.encode()), (b'content-length', b'0'),
                    (b'x-cowork-storage', base64.urlsafe_b64encode(json.dumps(storage).encode()).rstrip(b'='))],
                    [b''], method='POST', path='/api/storage-backup'))
        try:
            assert await asyncio.to_thread(state['get_entered'].wait, 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            thread.start()
            assert await asyncio.to_thread(entered.wait, 5), errors
            # Actual last-DML/pre-COMMIT pause: the primary reservation must still be held.
            blocked = False
            with sqlite3.connect(backend.db_path, timeout=.03) as contender:
                try:
                    contender.execute("INSERT INTO meta VALUES ('backup_effect_pending','too-early')")
                    contender.commit()
                except sqlite3.OperationalError:
                    blocked = True
            assert blocked
            state['block'] = True
            state['get_release'].set()
            # The native task cannot dispatch its next PUT while the real index commit is paused.
            assert not await asyncio.to_thread(state['entered'].wait, .1)
            resume.set()
            await asyncio.to_thread(thread.join, 6)
            assert not thread.is_alive() and errors == []
            assert await asyncio.to_thread(state['entered'].wait, 5)
            before = snapshot(backend.root/'managed-index.db')
            if child_loss:
                async def fail(_):
                    raise httpx.ReadTimeout('synthetic unrelated read timeout')
                other = rust_forward.ForwardMiddleware(None, routes={})
                other._client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
                other._client_loop = asyncio.get_running_loop()
                try:
                    await other._forward(server, {'method':'GET','path':'/api/months','headers':[]}, time.monotonic()+2)
                finally:
                    await other._client.aclose()
                await asyncio.to_thread(state['proc'].wait, timeout=5)
            # Reverse ordering: the actual native intent now won. No second index publication.
            r = opened[-1]
            def attempt_late():
                with tenant_writer.hold(UID):
                    if mode == 'reindex':
                        r.apply_reindex(retrieval.ReindexPlan('later.md', [('2030-01-01','later','later','later-hash')],
                                        {'later-hash':[0.0,1.0]}, target=r.embedding.active()))
                    else:
                        cfg = SimpleNamespace(version=1, active=r.embedding.active)
                        e = SimpleNamespace(embed=lambda _t,texts,*a,**k: [[0.0,1.0] for _ in texts])
                        m = EmbeddingMigration(SimpleNamespace(config=cfg,embedder=e), backend.root.parent.parent/'index.db',
                                               load_vec=load_vec, sleep=lambda _:None)
                        m._fill(backend.root/'managed-index.db', Target(None,'synthetic-later'), 1, budget=[1])
            with pytest.raises(tenant_writer.WriterBusy):
                await asyncio.to_thread(attempt_late)
            assert snapshot(backend.root/'managed-index.db') == before
        finally:
            resume.set()
            state['get_release'].set()
            state['release'].set()
            if thread.ident is not None:
                await asyncio.to_thread(thread.join, 7)
                assert not thread.is_alive()
            if not request.done():
                request.cancel()
                try:
                    await request
                except asyncio.CancelledError:
                    pass
    try:
        asyncio.run(run())
    finally:
        for r in opened:
            r.close()


# Keep the pure SQLite regressions runnable without any binary; only these eight cases need
# the already-built synthetic native server. No Cargo/image/model invocation is implied.
from tests.test_rust_storage_backup import lab  # noqa: E402,F401
import os as _os
if not _os.environ.get('DIARY_RUST_SERVER_BIN'):
    test_real_publication_orders_with_cancelled_dav_caller = pytest.mark.skip(reason='native server required')(
        test_real_publication_orders_with_cancelled_dav_caller)
