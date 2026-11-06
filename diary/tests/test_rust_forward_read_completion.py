"""Read completion: real Python handlers and native process, synthetic state only."""
import hashlib
import json
import sqlite3

import pytest

import agent.app as appmod
from agent import rust_forward
from tests.tenant_binary import needs_binary
from tests.test_rust_forward import DAILY, MONTHLY, needs_server, world, b64
from tests.test_rust_forward_d4b import pair, check, db, get, post_file, quiesce, sign, call, wire


def test_file_read_is_opt_in(monkeypatch):
    monkeypatch.delenv('DIARY_RUST_ROUTES', raising=False)
    assert rust_forward.configured_routes() == {}
    assert rust_forward.configured_routes('/api/file') == {'/api/file': 'POST'}


@needs_binary
@needs_server
def test_file_read_differential(pair):
    client, start, state = pair
    modes = start()
    quiesce(state, DAILY, MONTHLY)
    check(client, modes, [
        ('file', post_file(DAILY, {'path': 'memory.md'}), 200, True),
        ('unicode', post_file(DAILY, {'path': 'Notes/note.md'}), 200, True),
        ('missing', post_file(DAILY, {'path': 'missing.md'}), 200, True),
        ('traversal', post_file(DAILY, {'path': '../outside.md'}), 400, True),
        ('absolute', post_file(DAILY, {'path': '/outside.md'}), 400, True),
        ('large', post_file(DAILY, {'path': 'big.md'}), 413, True),
        ('wrong path type', post_file(DAILY, {'path': 5}), 400, True),
        ('invalid JSON', post_file(DAILY, b'{bad'), 422, False),
        ('array', post_file(DAILY, [1]), 422, False),
        ('surrogate', post_file(DAILY, b'{"path":"memory.md","unused":"\\ud800"}'), 200, False),
        ('invalid tenant', post_file('bad', {'path': 'memory.md'}), 400, True),
    ])
    # Fencing applies to writes. The existing read semantics remain available.
    assert modes.server.fence(DAILY)
    check(client, modes, [('fenced read', post_file(DAILY, {'path': 'memory.md'}), 200, True)])
    with db(state, DAILY) as conn:
        conn.execute("UPDATE meta SET value='0' WHERE key='active'")
    check(client, modes, [('non-managed decline', post_file(DAILY, {'path': 'memory.md'}), 503, False)])


@needs_binary
@needs_server
def test_health_detail_differential(pair):
    client, start, state = pair
    modes = start()
    quiesce(state, DAILY, MONTHLY)
    target = get(DAILY, '/api/health', extra={'X-Cowork-Storage': b64({'kind': 'local'})})
    check(client, modes, [('health tenant', target, 200, True)])
    facts = appmod._rust_retrieval_facts(DAILY)
    slot = hashlib.sha256((facts['profile'] + '\0' + facts['model']).encode()).hexdigest()[:16]
    with db(state, DAILY, 'managed-index.db') as conn:
        conn.execute('INSERT OR REPLACE INTO vec_meta VALUES (?,?)', ('slot:' + slot,
                     json.dumps({'table': 'vec_items', 'map': None, 'dim': 7})))
    key = f"{DAILY}:{hashlib.sha256(b'managed').hexdigest()[:16]}"
    retrieval = appmod._tenant_states[key].retrieval
    retrieval.vec_available = True  # synthetic runtime extension-availability fact, no vectors run
    check(client, modes, [('health dimension', target, 200, True)])
    assert call(client, target).json()['retrieval']['dim'] == 7
    with db(state, DAILY, 'managed-index.db') as conn:
        conn.execute("INSERT INTO journal(id,created_at,kind,payload,applied,last_error) VALUES ('pending','synthetic','index_update','{}',0,NULL),('quarantined','synthetic','index_update','{}',1,'quarantined:synthetic')")
        conn.execute("INSERT INTO chunks(day,file,header,body,content_hash) VALUES ('2030-01-01','synthetic.md','heading','text','hash')")
    check(client, modes, [('health counts', target, 200, True)])
    counts = call(client, target).json()
    assert counts['journal_pending'] == counts['journal_quarantined'] == counts['retrieval']['chunks'] == 1
    key = f"{DAILY}:{hashlib.sha256(b'managed').hexdigest()[:16]}"
    retrieval = appmod._tenant_states[key].retrieval
    available = retrieval.vec_available
    retrieval.vec_available = False
    try:
        check(client, modes, [('health without extension', target, 200, True)])
    finally:
        retrieval.vec_available = available
    assert modes.server.fence(DAILY)
    check(client, modes, [('health fenced', target, 200, True)])
    # An absent index must decline without creating a file. An already open Python handler
    # still reads the same inode (Unix unlink), giving a real-handler fallback oracle.
    path = state / 'users' / DAILY / 'managed-index.db'
    path.unlink()
    check(client, modes, [('health absent index', target, 200, False)])
    assert not path.exists()


@pytest.mark.parametrize('limit,decline', [(100, False), (100, True), (3, False)])
def test_read_body_replay_and_no_writer_admission(monkeypatch, limit, decline):
    import asyncio
    from types import SimpleNamespace
    from agent import tenant_writer
    def forbidden(*args, **kwargs):
        raise AssertionError('a read must not acquire writer admission')
    monkeypatch.setattr(tenant_writer, 'acquire_async', forbidden)
    monkeypatch.setattr(rust_forward, 'MAX_WRITE_BODY', limit)
    original = [{'type': 'http.request', 'body': b'ab', 'more_body': True},
                {'type': 'http.request', 'body': b'cd', 'more_body': False}]
    incoming = list(original)
    seen, sent, forwarded = [], [], []
    async def receive():
        return incoming.pop(0)
    async def send(message):
        sent.append(message)
    async def fallback(scope, receive, send):
        while True:
            message = await receive()
            seen.append(message)
            if not message.get('more_body'):
                break
    middleware = rust_forward.ForwardMiddleware(fallback, server=SimpleNamespace(ready=True),
                                               routes={'/api/file': 'POST'})
    async def forward(server, scope, deadline, **kwargs):
        forwarded.append(kwargs['body'])
        return None if decline else (200, [], b'ok')
    middleware._forward = forward
    scope = {'type': 'http', 'path': '/api/file', 'method': 'POST', 'headers': []}
    asyncio.run(middleware(scope, receive, send))
    if limit == 3 or decline:
        assert seen == original and not sent
    else:
        assert not seen and sent[-1]['body'] == b'ok'
    assert forwarded == ([] if limit == 3 else [b'abcd'])
