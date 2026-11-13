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


def test_the_exchange_route_is_known(tmp_path):
    assert '/v1/chat/completions' in sc.PATHS
    ops = base_ops(ops=[{'name': 'ask', 'method': 'POST', 'path': '/v1/chat/completions', 'body': {'messages': []}}])
    assert sc.load_ops(write_ops(tmp_path, ops))['ops'][0]['path'] == '/v1/chat/completions'


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
    import sqlite_vec  # a hard requirement (the CI requirements pin sqlite-vec): missing means fail, not skip
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


# ---------------- R9 review fixes ----------------

JSON = {'content-type': 'application/json'}
OTHER = '00000000-0000-4000-8000-0000000000b2'


def _ids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


def _reply(body, numbering, headers=JSON, mask=()):
    return sc.canonical_reply(200, headers, body, mask=mask, numbering=numbering)


def test_h1_binary_bodies_and_files_with_different_bytes_stay_different(tmp_path):
    a = _reply(b'\xff\x00', sc.UuidNumbering(), {})
    b = _reply(b'\xfe\x00', sc.UuidNumbering(), {})
    assert sc.diff_replies(a, b) == ['body']
    assert sc.diff_replies(a, _reply(b'\xff\x00', sc.UuidNumbering(), {})) == []
    for side, raw in (('l', b'\xff\x00'), ('r', b'\xfe\x00')):
        (tmp_path / side).mkdir()
        (tmp_path / side / 'blob.bin').write_bytes(raw)
    assert sc.diff_storage(sc.storage_digest(tmp_path / 'l', normalise_uuids=True),
                           sc.storage_digest(tmp_path / 'r', normalise_uuids=True)) == [
        {'path': 'blob.bin', 'reason': 'content'}]
    # Binary content with an id in it is still numbered, on the raw bytes.
    x, y = _ids(2)
    n1, n2 = sc.UuidNumbering(), sc.UuidNumbering()
    assert n1.raw(b'\xff' + x.encode() + b'\x00') == n2.raw(b'\xff' + y.encode() + b'\x00') == b'\xff<uuid:1>\x00'


def test_h2_the_tenant_and_known_ids_are_never_numbered():
    x1, x2 = _ids(2)
    body = lambda tenant, x: json.dumps({'tenant': tenant, 'xid': x}).encode()
    a = _reply(body(SYNTH, x1), sc.UuidNumbering(keep={SYNTH}))
    assert sc.diff_replies(a, _reply(body(SYNTH, x2), sc.UuidNumbering(keep={SYNTH}))) == []
    # Another tenant's id where the first side answered the ops tenant is a difference.
    assert sc.diff_replies(a, _reply(body(OTHER, x2), sc.UuidNumbering(keep={SYNTH}))) == ['body']
    # Without keep the leak would pass.
    assert sc.diff_replies(_reply(body(SYNTH, x1), sc.UuidNumbering()),
                           _reply(body(OTHER, x2), sc.UuidNumbering())) == []
    numbering = sc.UuidNumbering(keep={SYNTH})
    assert numbering.text('%s %s %s' % (SYNTH, x1, x2)) == '%s <uuid:1> <uuid:2>' % SYNTH


def test_h2_compare_keeps_the_ops_tenant_and_caller_given_ids(tmp_path):
    known_a, known_b, fresh1, fresh2 = _ids(4)
    ops = {'tenant': SYNTH, 'ops': [{'name': 'ask', 'method': 'POST', 'path': '/v1/chat/completions'}]}
    for side in ('py', 'rs'):
        (tmp_path / side).mkdir()

    def send(doc, served='rust'):
        return lambda op: {'status': 200, 'headers': JSON, 'body': json.dumps(doc).encode(), 'served': served}
    same = sc.compare(ops, send({'t': SYNTH, 'known': known_a, 'new': fresh1}),
                                   send({'t': SYNTH, 'known': known_a, 'new': fresh2}),
                                   tmp_path / 'py', tmp_path / 'rs', normalise_uuids=True, keep_uuids={known_a, known_b})
    assert same['differences'] == 0
    # A different existing id, or another tenant, where the python side answered with the known one.
    wrong_known = sc.compare(ops, send({'t': SYNTH, 'known': known_a, 'new': fresh1}),
                             send({'t': SYNTH, 'known': known_b, 'new': fresh2}),
                             tmp_path / 'py', tmp_path / 'rs', normalise_uuids=True, keep_uuids={known_a, known_b})
    assert wrong_known['differences'] == 1
    wrong_tenant = sc.compare(ops, send({'t': SYNTH}), send({'t': OTHER}),
                              tmp_path / 'py', tmp_path / 'rs', normalise_uuids=True)
    assert wrong_tenant['differences'] == 1


def test_h3_uuid_case_is_part_of_the_key():
    x = _ids(1)[0]
    lower = _reply(json.dumps({'a': x, 'b': x}).encode(), sc.UuidNumbering())
    mixed = _reply(json.dumps({'a': x, 'b': x.upper()}).encode(), sc.UuidNumbering())
    assert sc.diff_replies(lower, mixed) == ['body']
    n = sc.UuidNumbering()
    assert n.text(x + ' ' + x.upper()) == '<uuid:1> <uuid:2>'


def test_h4_one_numbering_per_side_across_replies_and_storage(tmp_path):
    x1, x2, x3, j1, j2, j3 = _ids(6)
    ops = {'tenant': SYNTH, 'ops': [{'name': 'ask', 'method': 'POST', 'path': '/v1/chat/completions'}]}

    def send(x):
        return lambda op: {'status': 200, 'headers': JSON, 'body': json.dumps({'xid': x}).encode(), 'served': 'rust'}
    a = _tenant_root(tmp_path, 'python', x1, j1, [1])
    consistent = _tenant_root(tmp_path, 'rust', x2, j2, [1])
    assert sc.compare(ops, send(x1), send(x2), a, consistent, normalise_uuids=True)['differences'] == 0
    # Rust answered x3 but stored x2 (and the journal under another id): the same id in the reply and the store
    # is what the python side shows, so it is a difference. Separate numberings would call both sides equal.
    (tmp_path / 'other').mkdir()
    stored = _tenant_root(tmp_path / 'other', 'rust', x2, j2, [1])
    report = sc.compare(ops, send(x1), send(x3), a, stored, normalise_uuids=True)
    assert report['differences'] > 0
    assert [d['path'] for d in report['storage_differences']] == ['corpus/2031-03.md', 'managed-index.db']
    assert report['operations'][0]['differences'] == []


def _broken_db(path):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE t (a TEXT)')
    db.execute("INSERT INTO t VALUES (CAST(x'ff' AS TEXT))")
    db.commit()
    db.close()


def test_h5_unreadable_databases_differ_when_load_vec_is_given(tmp_path):
    for side in ('l', 'r'):
        (tmp_path / side).mkdir()
        _broken_db(tmp_path / side / 'index.db')
    # Without load_vec the two error digests are equal, as before.
    plain = [sc.storage_digest(tmp_path / s) for s in ('l', 'r')]
    assert sc.diff_storage(*plain) == []
    assert 'unreadable' not in plain[0]['index.db']
    # With a loader that works, a table that cannot be read is marked and reported.
    digests = [sc.storage_digest(tmp_path / s, load_vec=lambda conn: None) for s in ('l', 'r')]
    assert digests[0]['index.db']['unreadable'] is True
    assert sc.diff_storage(*digests) == [{'path': 'index.db', 'reason': 'unreadable'}]


def test_h5_a_failing_loader_is_a_difference_not_two_equal_errors(tmp_path):
    make_sqlite_dirs = []
    for side in ('l', 'r'):
        (tmp_path / side).mkdir()
        make_sqlite(tmp_path / side / 'managed-diary.db', [('a.md', b'one', 'v1', 1.0)])
        make_sqlite_dirs.append(tmp_path / side)

    def failing(conn):
        raise RuntimeError('no extension')
    digests = [sc.storage_digest(d, load_vec=failing) for d in make_sqlite_dirs]
    assert digests[0] == digests[1]
    assert digests[0]['managed-diary.db']['unreadable'] is True
    assert sc.diff_storage(*digests) == [{'path': 'managed-diary.db', 'reason': 'unreadable'}]
    assert sc.diff_storage(*[sc.storage_digest(d, load_vec=lambda conn: None) for d in make_sqlite_dirs]) == []
    # An unopenable file with a loader is unreadable too.
    (make_sqlite_dirs[0] / 'broken.db').write_bytes(b'not a database')
    (make_sqlite_dirs[1] / 'broken.db').write_bytes(b'not a database')
    got = sc.diff_storage(*[sc.storage_digest(d, load_vec=lambda conn: None) for d in make_sqlite_dirs])
    assert got == [{'path': 'broken.db', 'reason': 'unreadable'}]


def _sse(text):
    return sc.canonical_reply(200, SSE, text.encode())


def test_h6_events_end_at_blank_lines_and_data_lines_join_with_a_newline():
    one = _sse('data: {"a":\ndata: 1}\n\n')  # two data lines of one event: joined with a newline, still JSON
    assert sc.stream_events(b'data: {"a":\ndata: 1}\n\n') == [{'data': {'a': 1}}]
    assert sc.diff_replies(one, _sse('data: {"a":1}\n\n')) == []
    # Two events, not one: a blank line separates them.
    assert sc.diff_replies(one, _sse('data: {"a":\n\ndata: 1}\n\n')) == ['body']
    assert sc.stream_events(b'data: one\ndata: two\n\n') == [{'data_text': 'one\ntwo'}]


def test_h6_event_and_id_are_compared_and_only_comments_and_retry_are_dropped():
    base = _sse('event: delta\nid: 7\ndata: {"a":1}\n\n')
    assert sc.diff_replies(base, _sse(': hi\nretry: 100\nevent: delta\nid: 7\ndata: {"a":1}\n\n: more\n\n')) == []
    assert sc.diff_replies(base, _sse('event: other\nid: 7\ndata: {"a":1}\n\n')) == ['body']
    assert sc.diff_replies(base, _sse('event: delta\nid: 8\ndata: {"a":1}\n\n')) == ['body']
    assert sc.diff_replies(base, _sse('event: delta\ndata: {"a":1}\n\n')) == ['body']
    # A data payload that is not JSON stays a difference; so does a stream that ends inside an event.
    assert sc.stream_events(b'data: {oops\n\n') == [{'data_text': '{oops'}]
    assert sc.diff_replies(_sse('data: {"a":1}\n\n'), _sse('data: {"a":1}\n')) == ['body']
    assert sc.stream_events(b'data: 1\r\n\r\n') == [{'data': 1}]


def test_h7_only_the_vec_extensions_shadow_tables_are_skipped(tmp_path):
    for side in 'abc':
        (tmp_path / side).mkdir()
    _vec_index(tmp_path / 'a' / 'index.db', [(1, (1.0, 0.0))])
    _vec_index(tmp_path / 'b' / 'index.db', [(1, (1.0, 0.0))])
    _vec_index(tmp_path / 'c' / 'index.db', [(1, (1.0, 0.0))])
    for side, note in (('a', 'x'), ('b', 'x'), ('c', 'y')):
        db = sqlite3.connect(tmp_path / side / 'index.db')
        db.execute('CREATE TABLE vec_items_notes (note TEXT)')  # a user table that only looks like a shadow table
        db.execute('INSERT INTO vec_items_notes VALUES (?)', (note,))
        db.commit()
        db.close()
    da, db_, dc = [sc.storage_digest(tmp_path / s, load_vec=_load_vec) for s in 'abc']
    assert da['index.db']['tables'] == ['vec_items', 'vec_items_notes']
    assert sc.diff_storage(da, db_) == []
    assert sc.diff_storage(da, dc) == [{'path': 'index.db', 'reason': 'content'}]
    shadow = ('vec_items_info', 'vec_items_chunks', 'vec_items_rowids', 'vec_items_vector_chunks00',
              'vec_items_metadatatext01', 'vec_items_other')
    assert [sc._is_vec_shadow(n, {'vec_items'}) for n in shadow] == [True, True, True, True, True, False]


def test_f1_key_order_does_not_change_the_numbering():
    x, j, x2, j2 = _ids(4)
    a = _reply(('{"xid":"%s","journal":"%s"}' % (x, j)).encode(), sc.UuidNumbering())
    b = _reply(('{"journal":"%s","xid":"%s"}' % (j2, x2)).encode(), sc.UuidNumbering())
    assert sc.diff_replies(a, b) == []
    # The same holds inside stream events, and the mask is applied before the ids are numbered.
    s1 = sc.canonical_reply(200, SSE, ('data: {"xid":"%s","journal":"%s"}\n\n' % (x, j)).encode(), numbering=sc.UuidNumbering())
    s2 = sc.canonical_reply(200, SSE, ('data: {"journal":"%s","xid":"%s"}\n\n' % (j2, x2)).encode(), numbering=sc.UuidNumbering())
    assert sc.diff_replies(s1, s2) == []
    m1 = _reply(('{"stamp":"%s","xid":"%s"}' % (x, j)).encode(), sc.UuidNumbering(), mask=('stamp',))
    m2 = _reply(('{"xid":"%s"}' % j2).encode().replace(b'}', b',"stamp":"1"}'), sc.UuidNumbering(), mask=('stamp',))
    assert sc.diff_replies(m1, m2) == []
    # A body that is not JSON is numbered raw; without numbering a body is compared as sent.
    assert sc.diff_replies(_reply(('id=%s' % x).encode(), sc.UuidNumbering()),
                           _reply(('id=%s' % x2).encode(), sc.UuidNumbering())) == []
    raw_a = sc.canonical_reply(200, JSON, b'{"a":1,"b":2}')
    raw_b = sc.canonical_reply(200, JSON, b'{"b":2,"a":1}')
    assert sc.diff_replies(raw_a, raw_b) == ['body']


def _fresh_file_root(root, ids, order):
    """Two files named by fresh ids (alpha/beta bodies naming their own id) and an index naming both."""
    (root / 'Notes').mkdir(parents=True)
    names = {'alpha': ids[0], 'beta': ids[1]}
    for label in order:
        (root / 'Notes' / ('%s.md' % names[label])).write_text('%s of %s\n' % (label, names[label]))
    (root / 'index.txt').write_text('alpha=%s\nbeta=%s\n' % (names['alpha'], names['beta']))
    db = sqlite3.connect(root / 'managed-index.db')
    db.execute('CREATE TABLE journal (id TEXT, kind TEXT)')
    for label in order:
        db.execute('INSERT INTO journal VALUES (?, ?)', (names[label], label))
    db.commit()
    db.close()


def test_f2_file_and_row_order_do_not_cross_number_fresh_ids(tmp_path):
    low, high = '11111111-1111-4111-8111-111111111111', 'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee'
    _fresh_file_root(tmp_path / 'l', [low, high], ['alpha', 'beta'])       # alpha sorts first by raw path
    _fresh_file_root(tmp_path / 'r', [high, low], ['beta', 'alpha'])       # alpha sorts last; rows inserted reversed
    left = sc.storage_digest(tmp_path / 'l', normalise_uuids=True)
    right = sc.storage_digest(tmp_path / 'r', normalise_uuids=True)
    assert sc.diff_storage(left, right) == []
    # A real swap (beta's file named by alpha's id in the index) is still seen.
    (tmp_path / 'r' / 'index.txt').write_text('alpha=%s\nbeta=%s\n' % (low, high))
    assert [d['path'] for d in sc.diff_storage(left, sc.storage_digest(tmp_path / 'r', normalise_uuids=True))] == ['index.txt']


def test_f3_unicode_line_separators_inside_json_strings_do_not_split_events():
    raw = ('data: {"t":"a\u2028b\u0085c"}\n\n').encode()
    escaped = b'data: {"t":"a\\u2028b\\u0085c"}\n\n'
    assert sc.stream_events(raw) == [{'data': {'t': 'a\u2028b\u0085c'}}]
    assert sc.diff_replies(sc.canonical_reply(200, SSE, raw), sc.canonical_reply(200, SSE, escaped)) == []


def test_updated_at_is_a_time_column_of_the_chunks_table_only(tmp_path):
    for side, stamp in (('l', 'a'), ('r', 'b')):
        (tmp_path / side).mkdir()
        db = sqlite3.connect(tmp_path / side / 'x.db')
        db.execute('CREATE TABLE chunks (id INTEGER, updated_at TEXT)')
        db.execute('CREATE TABLE notes (id INTEGER, updated_at TEXT)')
        db.execute('INSERT INTO chunks VALUES (1, ?)', (stamp,))
        db.execute('INSERT INTO notes VALUES (1, ?)', ('same',))
        db.commit()
        db.close()
    assert sc.diff_storage(sc.storage_digest(tmp_path / 'l'), sc.storage_digest(tmp_path / 'r')) == []
    db = sqlite3.connect(tmp_path / 'r' / 'x.db')
    db.execute("UPDATE notes SET updated_at = 'changed'")
    db.commit()
    db.close()
    assert sc.diff_storage(sc.storage_digest(tmp_path / 'l'), sc.storage_digest(tmp_path / 'r')) == [
        {'path': 'x.db', 'reason': 'content'}]
