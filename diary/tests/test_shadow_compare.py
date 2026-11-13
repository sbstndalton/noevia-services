"""Unit tests for the Diary shadow comparison (diary/scripts/shadow_compare.py). No binary, no network, synthetic data."""
import copy
import hashlib
import json
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shadow_compare as sc  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'shadow' / 'ops.v1.json'
SYNTH = '00000000-0000-4000-8000-0000000000a1'


def write_ops(tmp_path, data):
    path = tmp_path / 'ops.json'
    path.write_text(json.dumps(data))
    return path


def base_ops(**changes):
    data = {'version': 1, 'tenant': SYNTH, 'ops': [{'name': 'health', 'method': 'GET', 'path': '/api/health'}]}
    data.update(changes)
    return data


def test_the_committed_fixture_loads_and_is_synthetic():
    ops = sc.load_ops(FIXTURE)
    assert ops['tenant'].startswith(sc.SYNTHETIC_PREFIX)
    assert {op['method'] for op in ops['ops']} == {'GET', 'PUT', 'POST'}


def test_fixture_base_versions_are_the_digest_of_the_previous_content():
    # A file's version is the sha256 of its text (workspace_files.version); a hand-made value would turn an
    # intended overwrite into a stale-version conflict on both paths and test nothing.
    current = {}
    for op in sc.load_ops(FIXTURE)['ops']:
        body = op.get('body') or {}
        if op['path'] != '/api/file' or op['name'].endswith('stale version'):
            continue
        before = current.get(body['path'])
        expected = None if before is None else hashlib.sha256(before.encode()).hexdigest()
        assert body['version'] == expected, op['name']
        current[body['path']] = body['content']


def test_a_real_looking_tenant_is_refused(tmp_path):
    with pytest.raises(sc.OpsError):
        sc.load_ops(write_ops(tmp_path, base_ops(tenant=str(uuid.uuid4()))))


@pytest.mark.parametrize('changes', [
    {'version': 2},
    {'ops': []},
    {'ops': [{'name': 'x', 'method': 'DELETE', 'path': '/api/health'}]},
    {'ops': [{'name': 'x', 'method': 'GET', 'path': '/api/internal/secret'}]},
    {'ops': [{'name': 'x', 'method': 'GET', 'path': '/api/health', 'headers': {}}]},
    {'ops': [{'name': 'x', 'method': 'GET', 'path': '/api/health', 'body': {}}]},
    {'ops': [{'name': 'x', 'method': 'GET', 'path': '/api/health'},
             {'name': 'x', 'method': 'GET', 'path': '/api/months'}]},
])
def test_bad_operation_lists_are_refused(tmp_path, changes):
    with pytest.raises(sc.OpsError):
        sc.load_ops(write_ops(tmp_path, base_ops(**changes)))


def test_volatile_and_internal_headers_are_dropped_and_order_does_not_matter():
    a = sc.canonical_reply(200, {'Date': 'x', 'Server': 'uvicorn', 'Content-Type': 'application/json'}, b'{}')
    b = sc.canonical_reply(200, [(b'content-type', b'application/json'), (b'x-noevia-diary-decline', b'1'),
                                 (b'Date', b'y')], b'{}')
    assert a == b
    assert sc.diff_replies(a, b) == []


def test_a_header_value_difference_is_reported():
    a = sc.canonical_reply(200, {'Content-Type': 'application/json'}, b'{}')
    b = sc.canonical_reply(200, {'Content-Type': 'text/plain'}, b'{}')
    assert sc.diff_replies(a, b) == ['headers']


def test_status_and_body_differences_are_named():
    a = sc.canonical_reply(200, {}, b'{"a":1}')
    b = sc.canonical_reply(409, {}, b'{"a":2}')
    assert sc.diff_replies(a, b) == ['status', 'body']


def test_masked_keys_are_ignored_and_other_keys_are_not():
    a = sc.canonical_reply(200, {}, b'{"day":"2030-01-01","log":"x"}', mask=('day',))
    b = sc.canonical_reply(200, {}, b'{"day":"2031-02-02","log":"x"}', mask=('day',))
    c = sc.canonical_reply(200, {}, b'{"day":"2031-02-02","log":"y"}', mask=('day',))
    assert sc.diff_replies(a, b) == []
    assert sc.diff_replies(a, c) == ['body']


def make_sqlite(path, rows):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE files(path TEXT PRIMARY KEY, data BLOB NOT NULL, version TEXT NOT NULL, updated REAL NOT NULL)')
    for row in rows:
        db.execute('INSERT INTO files VALUES (?,?,?,?)', row)
    db.commit()
    db.close()


def test_sqlite_digest_ignores_row_order_and_time_columns(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.mkdir(); right.mkdir()
    make_sqlite(left / 'managed-diary.db', [('a.md', b'one', 'v1', 1.0), ('b.md', b'two', 'v2', 2.0)])
    make_sqlite(right / 'managed-diary.db', [('b.md', b'two', 'v2', 99.0), ('a.md', b'one', 'v1', 77.0)])
    assert sc.diff_storage(sc.storage_digest(left), sc.storage_digest(right)) == []


def test_sqlite_digest_sees_a_content_difference(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.mkdir(); right.mkdir()
    make_sqlite(left / 'managed-diary.db', [('a.md', b'one', 'v1', 1.0)])
    make_sqlite(right / 'managed-diary.db', [('a.md', b'onf', 'v1', 1.0)])
    assert sc.diff_storage(sc.storage_digest(left), sc.storage_digest(right)) == [
        {'path': 'managed-diary.db', 'reason': 'content'}]


def test_plain_files_are_compared_by_bytes_and_missing_files_are_named(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    (left / 'Notes').mkdir(parents=True); (right / 'Notes').mkdir(parents=True)
    (left / 'Notes' / 'x.md').write_bytes(b'same\n')
    (right / 'Notes' / 'x.md').write_bytes(b'same\n')
    (left / 'only.md').write_bytes(b'l')
    (right / 'extra.md').write_bytes(b'r')
    (right / 'Notes' / 'x.md').write_bytes(b'same\r\n')
    assert sc.diff_storage(sc.storage_digest(left), sc.storage_digest(right)) == [
        {'path': 'Notes/x.md', 'reason': 'content'},
        {'path': 'extra.md', 'reason': 'only-rust'},
        {'path': 'only.md', 'reason': 'only-python'},
    ]


def test_sqlite_sidecars_are_not_compared_on_their_own(tmp_path):
    make_sqlite(tmp_path / 'managed-diary.db', [('a.md', b'one', 'v1', 1.0)])
    (tmp_path / 'managed-diary.db-wal').write_bytes(b'anything')
    assert sorted(sc.storage_digest(tmp_path)) == ['managed-diary.db']


def ok(status=200, body=b'{}', served='rust'):
    return {'status': status, 'headers': {'Content-Type': 'application/json'}, 'body': body, 'served': served}


def test_run_ops_counts_differences_and_fallbacks(tmp_path):
    ops = sc.load_ops(FIXTURE)
    report = sc.run_ops(ops, lambda op: ok(), lambda op: ok())
    assert report['differences'] == 0
    assert report['rust_served'] == len(ops['ops'])
    assert report['fallbacks'] == 0

    def rust(op):
        if op['name'] == 'months':
            return ok(status=500, body=b'{"error":"x"}')
        if op['name'] == 'storage status':
            return ok(served='python')
        return ok()
    report = sc.run_ops(ops, lambda op: ok(), rust)
    names = {op['name']: op for op in report['operations']}
    assert names['months']['differences'] == ['status', 'body']
    assert names['storage status']['served'] == 'python'
    assert report['differences'] == 1
    assert report['fallbacks'] == 1


def test_compare_adds_storage_differences_to_the_total(tmp_path):
    ops = sc.load_ops(FIXTURE)
    left, right = tmp_path / 'py', tmp_path / 'rs'
    left.mkdir(); right.mkdir()
    (left / 'a.md').write_bytes(b'1')
    (right / 'a.md').write_bytes(b'2')
    report = sc.compare(ops, lambda op: ok(), lambda op: ok(), left, right)
    assert report['differences'] == 1
    assert report['storage_differences'] == [{'path': 'a.md', 'reason': 'content'}]
    text = sc.summarise(report)
    assert 'DIFF storage a.md: content' in text
    assert text.splitlines()[-1].startswith('operations=%d differences=1' % len(ops['ops']))


def test_the_fixture_is_not_changed_by_a_run(tmp_path):
    ops = sc.load_ops(FIXTURE)
    before = copy.deepcopy(ops)
    sc.run_ops(ops, lambda op: ok(), lambda op: ok())
    assert ops == before


# ---------------- R9: exchange replies, fresh ids and the vector index ----------------

SSE = {'content-type': 'text/event-stream; charset=utf-8'}


def _frames(*events, keep_alive=False, compact=False):
    out = []
    for i, e in enumerate(events):
        if keep_alive and i == 1:
            out.append(': keep-alive\n\n')
        text = json.dumps(e, separators=(',', ':'), sort_keys=True) if compact else json.dumps(e)
        out.append('data: ' + text + '\n\n')
    return ''.join(out).encode()


def test_the_exchange_route_is_known():
    assert '/v1/chat/completions' in sc.PATHS
    ops = base_ops(ops=[{'name': 'ask', 'method': 'POST', 'path': '/v1/chat/completions', 'body': {'messages': []}}])
    assert sc.load_ops(write_ops_dir(ops))['ops'][0]['path'] == '/v1/chat/completions'


def write_ops_dir(data, _box=[]):
    import tempfile
    d = Path(tempfile.mkdtemp(prefix='qa-shadow-'))
    _box.append(d)
    return write_ops(d, data)


def test_event_streams_compare_as_parsed_events():
    events = [{'type': 'status', 'text': 'Opening diary context…'}, {'type': 'delta', 'text': 'Synthetic é'},
              {'type': 'diary', 'reply': 'Synthetic é', 'decision': 'logged', 'xid': 'x'}, {'type': 'done'}]
    python = sc.canonical_reply(200, SSE, _frames(*events))
    rust = sc.canonical_reply(200, SSE, _frames(*events, keep_alive=True, compact=True))
    assert sc.diff_replies(python, rust) == []
    changed = [dict(e) for e in events]
    changed[1]['text'] = 'Different'
    assert sc.diff_replies(python, sc.canonical_reply(200, SSE, _frames(*changed))) == ['body']
    # A missing event is a difference; a frame that is not JSON is too.
    assert sc.diff_replies(python, sc.canonical_reply(200, SSE, _frames(*events[:-1]))) == ['body']
    broken = _frames(*events) + b'data: {not json\n\n'
    assert sc.diff_replies(python, sc.canonical_reply(200, SSE, broken)) == ['body']


def test_masked_keys_apply_inside_stream_events():
    a = sc.canonical_reply(200, SSE, _frames({'type': 'diary', 'xid': 'one'}), mask=('xid',))
    b = sc.canonical_reply(200, SSE, _frames({'type': 'diary', 'xid': 'two'}), mask=('xid',))
    assert sc.diff_replies(a, b) == []


def test_fresh_uuids_compare_by_where_they_appear():
    def reply(x, j):
        body = json.dumps({'id': 'chatcmpl-' + x, 'diary': {'xid': x, 'journal': j}}).encode()
        return body
    x1, j1, x2, j2 = (str(uuid.uuid4()) for _ in range(4))
    left, right = sc.UuidNumbering(), sc.UuidNumbering()
    a = sc.canonical_reply(200, {'content-type': 'application/json'}, reply(x1, j1), numbering=left)
    b = sc.canonical_reply(200, {'content-type': 'application/json'}, reply(x2, j2), numbering=right)
    assert sc.diff_replies(a, b) == []
    # One id used in both places where the other side used two different ids is a difference.
    c = sc.canonical_reply(200, {'content-type': 'application/json'}, reply(x2, x2), numbering=sc.UuidNumbering())
    assert sc.diff_replies(a, c) == ['body']
    # Without numbering, fresh ids differ.
    assert sc.diff_replies(sc.canonical_reply(200, {}, reply(x1, j1)), sc.canonical_reply(200, {}, reply(x2, j2))) == ['body']


def _tenant_root(tmp_path, side, xid, journal_id, order):
    root = tmp_path / side
    (root / 'corpus').mkdir(parents=True)
    (root / 'corpus' / '2031-03.md').write_text(f'### 09:00\n\n**Me:** synthetic\n\n<!-- xid:{xid} -->\n')
    db = sqlite3.connect(root / 'managed-index.db')
    db.execute('CREATE TABLE journal (id TEXT, payload TEXT, created_at TEXT)')
    db.execute('INSERT INTO journal VALUES (?, ?, ?)', (journal_id, json.dumps({'xid': xid}), side))
    db.execute('CREATE TABLE chunks (id INTEGER PRIMARY KEY, body TEXT, updated_at TEXT)')
    for i in order:
        db.execute('INSERT INTO chunks VALUES (?, ?, ?)', (i, 'chunk %d' % i, side + str(i)))
    db.commit()
    db.close()
    return root


def test_storage_with_fresh_ids_and_clock_columns_compares_equal(tmp_path):
    a = _tenant_root(tmp_path, 'python', str(uuid.uuid4()), str(uuid.uuid4()), [1, 2])
    b = _tenant_root(tmp_path, 'rust', str(uuid.uuid4()), str(uuid.uuid4()), [2, 1])
    assert sc.diff_storage(sc.storage_digest(a, normalise_uuids=True), sc.storage_digest(b, normalise_uuids=True)) == []
    # Without numbering the corpus file and the journal differ (and nothing else: updated_at is a clock column).
    reasons = sc.diff_storage(sc.storage_digest(a), sc.storage_digest(b))
    assert [r['path'] for r in reasons] == ['corpus/2031-03.md', 'managed-index.db']


def _load_vec(conn):
    sqlite_vec = pytest.importorskip('sqlite_vec')
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _vec_index(path, rows):
    db = sqlite3.connect(path)
    _load_vec(db)
    db.execute('CREATE VIRTUAL TABLE vec_items USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[2])')
    for chunk_id, vector in rows:
        import struct
        db.execute('INSERT INTO vec_items (chunk_id, embedding) VALUES (?, ?)', (chunk_id, struct.pack('2f', *vector)))
    db.commit()
    db.close()


def test_vector_tables_compare_by_their_rows(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    (tmp_path / 'c').mkdir()
    _vec_index(tmp_path / 'a' / 'index.db', [(1, (1.0, 0.0)), (2, (0.0, 1.0))])
    _vec_index(tmp_path / 'b' / 'index.db', [(2, (0.0, 1.0)), (1, (1.0, 0.0))])
    _vec_index(tmp_path / 'c' / 'index.db', [(1, (1.0, 0.0)), (2, (0.5, 1.0))])
    da = sc.storage_digest(tmp_path / 'a', load_vec=_load_vec)
    assert 'error' not in da['index.db'], da
    assert da['index.db']['tables'] == ['vec_items']
    assert sc.diff_storage(da, sc.storage_digest(tmp_path / 'b', load_vec=_load_vec)) == []
    assert sc.diff_storage(da, sc.storage_digest(tmp_path / 'c', load_vec=_load_vec)) == [{'path': 'index.db', 'reason': 'content'}]


def test_compare_passes_the_r9_options_through(tmp_path):
    x1, x2 = str(uuid.uuid4()), str(uuid.uuid4())
    ops = {'tenant': SYNTH, 'ops': [{'name': 'ask', 'method': 'POST', 'path': '/v1/chat/completions'}]}
    def send(x, served):
        return lambda op: {'status': 200, 'headers': SSE, 'body': _frames({'type': 'diary', 'xid': x}), 'served': served}
    a = _tenant_root(tmp_path, 'python', x1, str(uuid.uuid4()), [1])
    b = _tenant_root(tmp_path, 'rust', x2, str(uuid.uuid4()), [1])
    report = sc.compare(ops, send(x1, 'python'), send(x2, 'rust'), a, b, normalise_uuids=True)
    assert report['differences'] == 0 and report['rust_served'] == 1
