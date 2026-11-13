"""Unit tests for the Diary shadow comparison (diary/scripts/shadow_compare.py). No binary, no network, synthetic data."""
import copy
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
