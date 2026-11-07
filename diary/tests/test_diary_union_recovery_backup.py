"""Union of D6 native journal recovery, D8 native verification and the R4 backup authority.

- The forwarder's refusal of a mutation while a backup effect is unsettled answers only a verified
  request: one delegated to the Rust verifier is verified here first (401 for a bad assertion, 503
  for a good one, its nonce consumed), and nothing reaches the child.
- Native recovery through the real forwarder and binary: while an intent is pending nothing is
  replayed, transferred or marked; once it is settled, one request replays the journal and drains
  the import outbox under its own primary reservation (no self-contention), exactly once.
Synthetic tenants, keys and intents only.
"""
import asyncio
import json
import sqlite3
import time

import agent.app as appmod
from agent import corpus as fmt, rust_forward, tenant_writer
from agent import tenant_assertion as ta
from agent.backup_effect import MESSAGE
from tests.tenant_binary import needs_binary
from tests.test_diary_verifier_d8 import Child, Spy, VERIFIED_OK, consumed, keyed, leaf_calls  # noqa: F401
from tests.test_rust_forward import DAILY, OTHER, needs_server, world  # noqa: F401
from tests.test_rust_forward_d4b import sign
from tests.test_rust_forward_d5 import (Recorder, _writer_registry, needs_cw, request, storage,  # noqa: F401
                                        writers)

UNVERIFIED = b'{"detail":"invalid tenant assertion"}'
BLOCKED = json.dumps({"detail": MESSAGE}, separators=(",", ":")).encode()
INTENT = '{"id":"synthetic-intent","method":"PUT","path":"/synthetic"}'
BODY = b'{"path":"a.md","content":"x","version":null}'


def put(app, headers, body=BODY):
    scope = {'type': 'http', 'method': 'PUT', 'path': '/api/file', 'raw_path': b'/api/file', 'query_string': b'',
             'headers': [(k.lower().encode(), v.encode()) for k, v in headers.items()]
             + [(b'content-length', str(len(body)).encode())], 'state': {}}
    chunks, sent = [body], []

    async def receive():
        return {'type': 'http.request', 'body': chunks.pop(0) if chunks else b'', 'more_body': False}

    async def send(message):
        sent.append(message)
    asyncio.run(app(scope, receive, send))
    return sent[0]['status'], b''.join(m.get('body', b'') for m in sent[1:])


@needs_binary
def test_a_blocked_write_delegated_to_the_rust_verifier_is_verified_before_its_503(keyed, leaf_calls, monkeypatch):
    child = Child(VERIFIED_OK)
    try:
        server = rust_forward.RustServer(lambda p, t, f: {}, port=child.port, restart_delays=())
        server.ready, server.forward_secret = True, 'synthetic'
        spy = Spy()
        forward = rust_forward.ForwardMiddleware(spy, server=server, routes={'/api/months': 'GET'},
                                                 writers=frozenset({DAILY}))
        app = appmod.TenantAssertionMiddleware(forward)
        blocked = {DAILY}
        monkeypatch.setattr(tenant_writer, 'effect_blocked', lambda uid: uid in blocked)
        # A naive union refused here before any check: an unverified caller learnt the backup state.
        bad = sign(OTHER, 'PUT', '/api/file', BODY, None, b'', 'application/json')
        bad['X-Cowork-User-ID'] = DAILY  # signed for another tenant
        assert put(app, bad) == (401, UNVERIFIED)
        assert len(leaf_calls) == 1 and child.requests == [] and spy.calls == []
        good = sign(DAILY, 'PUT', '/api/file', BODY, None, b'', 'application/json')
        assert put(app, good) == (503, BLOCKED)
        assert len(leaf_calls) == 2 and child.requests == [] and spy.calls == []
        assert consumed(good[ta.HEADER])
        assert ta._pending_count() == 0
        # Its nonce is spent: a replay is refused by the middleware, never answered 503 again.
        assert put(app, good)[0] == 401
        # Control: once the effect is settled the write is delegated and verified natively again.
        blocked.clear()
        fresh = sign(DAILY, 'PUT', '/api/file', BODY, None, b'', 'application/json')
        calls = len(leaf_calls)
        assert put(app, fresh) == (200, b'{"native":true}')
        assert len(child.requests) == 1 and child.requests[0][b'x-noevia-diary-verify'] == b'1'
        assert len(leaf_calls) == calls and consumed(fresh[ta.HEADER])
    finally:
        child.close()


def _outbox(tenant):
    with sqlite3.connect(tenant / 'managed-diary.db') as db:
        return [r[0] for r in db.execute('SELECT path FROM workspace_index_outbox ORDER BY path')]


def _dirty(tenant):
    with sqlite3.connect(tenant / 'managed-index.db') as db:
        return {r[0] for r in db.execute('SELECT document FROM dirty_documents')}


def _markers(tenant, xid):
    with sqlite3.connect(tenant / 'managed-diary.db') as db:
        return sum(bytes(r[0]).count(xid.encode()) for r in db.execute('SELECT data FROM files'))


@needs_binary
@needs_server
@needs_cw
def test_native_recovery_waits_for_backup_settlement_then_drains_once_without_self_contention(writers):
    client, start, state = writers
    start()
    tenant = state / 'users' / DAILY
    queued = ['Imports/union-a.md', 'Imports/union-b.md']
    tenant_writer.configure_data_root(state)
    try:
        payload = {'day': '2030-01-06', 'xid': 'abcdef34', 'sub_header': '12:00 | Synthetic',
                   'body': fmt.render_exchange('Synthetic union note', '', 'abcdef34')}
        with sqlite3.connect(tenant / 'managed-index.db') as db:
            db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                       ('union', '2030-01-10', 'exchange', json.dumps(payload)))
        with sqlite3.connect(tenant / 'managed-diary.db') as db:
            for path in queued:
                db.execute('INSERT OR IGNORE INTO workspace_index_outbox VALUES (?)', (path,))
            db.execute("INSERT INTO meta VALUES ('backup_effect_pending', ?)", (INTENT,))
        markers, per = _markers(tenant, 'abcdef34'), payload['body'].count('abcdef34')
        assert markers == 0 and per >= 1
        before, dirty_before = storage(state, DAILY), _dirty(tenant)
        Recorder.served = []
        r = request(client, DAILY, 'POST', '/api/directory', {'path': 'Union'})
        assert (r.status_code, r.content) == (503, BLOCKED)
        assert Recorder.served == []
        assert storage(state, DAILY) == before and _dirty(tenant) == dirty_before
        assert _outbox(tenant) == queued and _markers(tenant, 'abcdef34') == markers
        # Synthetic settlement (test-only removal of the synthetic intent).
        with sqlite3.connect(tenant / 'managed-diary.db') as db:
            db.execute("DELETE FROM meta WHERE key='backup_effect_pending'")
        started = time.monotonic()
        r = request(client, DAILY, 'POST', '/api/directory', {'path': 'Union'})
        # A drain that opened a second primary reservation per mark waited for its own (30 s), then 500.
        assert r.status_code == 200, r.text
        assert time.monotonic() - started < 10
        assert Recorder.served == [True]
        assert _outbox(tenant) == [] and set(queued) <= _dirty(tenant)
        assert _markers(tenant, 'abcdef34') == per
        with sqlite3.connect(tenant / 'managed-index.db') as db:
            assert db.execute("SELECT applied, attempts FROM journal WHERE id='union'").fetchone() == (1, 0)
        # The next native request finds nothing left: no second replay or transfer.
        r = request(client, DAILY, 'POST', '/api/directory', {'path': 'Union again'})
        assert r.status_code == 200, r.text
        assert Recorder.served == [True, True]
        assert _outbox(tenant) == [] and _markers(tenant, 'abcdef34') == per
    finally:
        tenant_writer.configure_data_root(None)


@needs_binary
@needs_server
@needs_cw
def test_an_intent_committed_while_native_waits_for_the_held_primary_is_pythons_busy(writers):
    """Admitted with no intent; the intent commits while the native write waits for the one primary
    reservation. The native refusal is app.py's writer_busy reply byte for byte (status, Retry-After,
    content type, body) and nothing of the request is stored. Before G1 it was a 500."""
    import concurrent.futures
    import threading
    client, start, state = writers
    start()
    tenant = state / 'users' / DAILY
    tenant_writer.configure_data_root(state)
    held, go = threading.Event(), threading.Event()

    def holder():
        db = sqlite3.connect(tenant / 'managed-diary.db', isolation_level=None, timeout=30)
        try:
            db.execute('BEGIN IMMEDIATE')
            held.set()
            go.wait(10)
            db.execute("INSERT INTO meta VALUES ('backup_effect_pending', ?)", (INTENT,))
            db.execute('COMMIT')
        finally:
            db.close()
    try:
        before, dirty_before, outbox_before = storage(state, DAILY), _dirty(tenant), _outbox(tenant)
        expected = asyncio.run(appmod.writer_busy(None, tenant_writer.WriterBusy(DAILY, 'backup settlement uncertain')))
        Recorder.served = []
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            hold = pool.submit(holder)
            assert held.wait(10)
            pending = pool.submit(request, client, DAILY, 'POST', '/api/directory', {'path': 'Busy union'})
            time.sleep(0.5)
            assert not pending.done(), 'the native write must wait for the held primary'
            go.set()
            hold.result(timeout=10)
            r = pending.result(timeout=20)
        assert Recorder.served == [True]
        assert (r.status_code, r.headers.get('retry-after'), r.headers.get('content-type'), r.content) == \
            (expected.status_code, expected.headers['retry-after'], expected.headers['content-type'], expected.body)
        assert (r.status_code, r.headers.get('retry-after'), r.content) == (503, '1', rust_forward.BUSY)
        assert storage(state, DAILY) == before and _dirty(tenant) == dirty_before and _outbox(tenant) == outbox_before
        # The intent stays the exclusion: refused before forwarding from here on, nothing stored.
        r = request(client, DAILY, 'POST', '/api/directory', {'path': 'Busy union'})
        assert (r.status_code, r.content) == (503, BLOCKED) and Recorder.served == [True]
        assert storage(state, DAILY) == before
    finally:
        tenant_writer.configure_data_root(None)
