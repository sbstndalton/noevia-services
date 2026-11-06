"""Real-handler differential coverage for native managed workspace mutations."""
import hashlib
import json
import sqlite3

import pytest

from tests.tenant_binary import needs_binary
from tests.test_rust_forward import DAILY, needs_server, world  # noqa: F401
from tests.test_rust_forward_d5 import (Recorder, request, restore, snapshot,
                                      wire, writers, storage, _writer_registry)  # noqa: F401

def warm(client, modes):
    modes.python()
    assert request(client, DAILY, 'GET', '/api/months', {}).status_code == 200


def differential(client, modes, state, cases):
    for name, uid, method, path, body, status, native in cases:
        warm(client, modes)
        snap = snapshot(state, uid)
        p = request(client, uid, method, path, body)
        effect = storage(state, uid)
        restore(state, uid, snap)
        warm(client, modes)
        modes.rust()
        Recorder.served = []
        r = request(client, uid, method, path, body)
        assert wire(p) == wire(r), (name, p.text, r.text)
        assert r.status_code == status, (name, r.text)
        assert effect == storage(state, uid), name
        assert Recorder.served == [native], (name, Recorder.served)


PATH = '/api/workspace-ops'
TRASH = '/api/workspace-trash'
ID = '00000000-0000-4000-8000-000000000123'


def seed_ops(state):
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        for p, raw in [('Notes/a.md', b'hello'), ('Notes/b.md', b'other'),
                       ('AI Memory/a.md', b'protected'), ('Folder/z.md', b'z'),
                       ('Folder/sub/a.md', b'a'), ('odd.bin', b'opaque')]:
            db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,1.0)',
                       (p, raw, hashlib.sha256(raw).hexdigest()))
        db.execute("INSERT OR IGNORE INTO directories VALUES ('Folder/empty')")


@needs_binary
@needs_server
def test_ops_wire_and_state(writers):
    client, start, state = writers
    seed_ops(state)
    modes = start()
    h = hashlib.sha256(b'hello').hexdigest()
    cases = []
    def case(name, body, status=200):
        cases.append((name, DAILY, 'POST', PATH, body, status, True))
    case('stat file', {'op': 'stat', 'path': '/Notes/a.md/'})
    case('stat folder', {'op': 'stat', 'path': 'Folder'})
    case('unknown', {'op': 'mkdir'}, 400)
    for op in ['stat', 'delete', 'move', 'copy', 'preserve']:
        case(op + ' traversal', {'op': op, 'path': '../a.md', 'destination': 'b.md'}, 403)
        case(op + ' root', {'op': op, 'path': '/', 'destination': 'b.md'}, 403)
        case(op + ' invalid', {'op': op, 'path': 8, 'destination': 'b.md'}, 400)
    case('missing', {'op': 'delete', 'path': 'missing.md'}, 404)
    case('protected', {'op': 'delete', 'path': 'AI Memory/a.md', 'version': h}, 403)
    case('no version', {'op': 'delete', 'path': 'Notes/a.md'}, 428)
    case('stale', {'op': 'delete', 'path': 'Notes/a.md', 'version': 'stale'}, 412)
    case('self', {'op': 'copy', 'path': 'Folder', 'destination': 'Folder/new'}, 403)
    case('destination absent', {'op': 'copy', 'path': 'Notes/a.md', 'destination': 'Absent/new.md'}, 409)
    case('destination protected', {'op': 'copy', 'path': 'Notes/a.md', 'destination': 'AI Memory/new.md'}, 403)
    case('destination exists', {'op': 'copy', 'path': 'Notes/a.md', 'destination': 'Notes/b.md'}, 412)
    case('destination precondition', {'op': 'move', 'path': 'Notes/a.md', 'destination': 'Notes/b.md', 'version': h, 'overwrite': True}, 428)
    case('copy', {'op': 'copy', 'path': 'Notes/a.md', 'destination': 'Notes/copy.md'})
    case('move', {'op': 'move', 'path': 'Notes/a.md', 'destination': 'Notes/moved.md', 'version': h})
    case('copy folder', {'op': 'copy', 'path': 'Folder', 'destination': 'FolderCopy'})
    # The differential leaves each successful operation applied for the next case.
    differential(client, modes, state, cases)


@needs_binary
@needs_server
def test_trash_restore_preserve_delete_and_capsules(writers):
    client, start, state = writers
    seed_ops(state)
    modes = start()
    h = hashlib.sha256(b'hello').hexdigest()
    # Run each sequence from the same DB snapshot. Dynamic UUID/time fields are compared
    # separately; every deterministic reply byte and all capsule content are compared exactly.
    snap = snapshot(state, DAILY)
    results = []
    for mode in (modes.python, modes.rust):
        restore(state, DAILY, snap)
        warm(client, modes)
        mode()
        out = []
        def call(path, body):
            warm(client, modes)
            mode()
            Recorder.served = []
            r = request(client, DAILY, 'POST', path, body)
            if mode == modes.rust:
                assert Recorder.served == [True], (body, r.text)
            out.append((r.status_code, r.json()))
            return r
        r = call(TRASH, {'action': 'trash', 'id': ID, 'path': 'Notes/a.md', 'version': h})
        assert r.status_code == 200
        call(TRASH, {'action': 'trash', 'id': ID, 'path': 'Notes/a.md', 'version': h})
        call(TRASH, {'action': 'restore', 'id': ID})
        call(TRASH, {'action': 'restore', 'id': ID})
        call(TRASH, {'action': 'purge', 'id': ID})
        call(TRASH, {'action': 'trash', 'id': ID, 'path': 'Notes/b.md', 'version': h})
        call(PATH, {'op': 'preserve', 'path': 'Notes/a.md', 'version': h})
        call(PATH, {'op': 'delete', 'path': 'Notes/a.md', 'version': h})
        with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
            capsules = [json.loads(row[0]) for row in db.execute("SELECT data FROM files WHERE path LIKE '.noevia-trash/%'")]
            assert len(capsules) == 3
            for cap in capsules:
                assert cap['version'] == h
                assert cap['data'] == 'aGVsbG8='
                cap.pop('id')
                assert isinstance(cap.pop('trashedAt'), float)
                cap.pop('restoredAt', None)
                if '(replaced ' in cap['path']:
                    cap['path'] = 'Notes/a (replaced TIMESTAMP).md'
        for _, body in out:
            if 'trashedAt' in body:
                assert isinstance(body.pop('trashedAt'), float)
            if 'trash' in body:
                body['trash'] = ['DYNAMIC'] if isinstance(body['trash'], list) else 'DYNAMIC'
        results.append((out, sorted(capsules, key=lambda c: (c['state'], c['path']))))
    assert results[0] == results[1]


@needs_binary
@needs_server
def test_corrupt_capsule_and_unsupported_actions(writers):
    client, start, state = writers
    modes = start()
    with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
        raw = b'[]'
        db.execute('INSERT INTO files VALUES (?,?,?,1.0)',
                   (f'.noevia-trash/{ID}.json', raw, hashlib.sha256(raw).hexdigest()))
    cases = [(name, DAILY, 'POST', TRASH, body, status, True) for name, body, status in [
        ('corrupt', {'action': 'restore', 'id': ID}, 409),
        ('purge unsupported', {'action': 'purge', 'id': ID}, 409),
        ('bad id', {'action': 'restore', 'id': 'bad'}, 409),
    ]]
    differential(client, modes, state, cases)



@needs_binary
@needs_server
@pytest.mark.parametrize('inactive', [False, True])
def test_pending_journal_and_nonmanaged_tenant_decline_to_real_python(writers, inactive):
    client, start, state = writers
    seed_ops(state)
    modes = start()
    warm(client, modes)
    if inactive:
        with sqlite3.connect(state / 'users' / DAILY / 'managed-diary.db') as db:
            db.execute("UPDATE meta SET value='0' WHERE key='active'")
    else:
        with sqlite3.connect(state / 'users' / DAILY / 'managed-index.db') as db:
            db.execute("INSERT INTO journal (id,created_at,kind,payload) VALUES ('pending','2030-01-01','index_update','{}')")
    for path, body in [(PATH, {'op': 'delete', 'path': 'Notes/a.md', 'version': hashlib.sha256(b'hello').hexdigest()}),
                       (TRASH, {'action': 'restore', 'id': ID})]:
        modes.python()
        p = request(client, DAILY, 'POST', path, body)
        modes.rust()
        Recorder.served = []
        r = request(client, DAILY, 'POST', path, body)
        assert wire(p) == wire(r)
        assert r.status_code == (503 if inactive else 409)
        assert Recorder.served == [False]


@needs_binary
@needs_server
@pytest.mark.parametrize('action,state_name', [('restore', 'trashed'), ('trash', 'trashed'), ('restore', 'restored')])
@pytest.mark.parametrize('timestamp_json', [
    '{"z":1,"a":2}',
    '[{"z":1e-7,"a":1e20},-0.0,0.000001]',
    '{"n":18446744073709551617,"minus":-18446744073709551617}',
    '{"z":{"b":"\\u00e9\\n","a":[true,null,"x"]},"z":{"last":2,"first":1}}',
])
def test_legacy_capsule_summary_bytes_and_nested_values(writers, action, state_name, timestamp_json):
    client, start, state = writers
    modes = start()
    warm(client, modes)
    h = hashlib.sha256(b'hello').hexdigest()
    raw = ('{"format":"noevia-trash-v1","id":"' + ID + '","path":"Notes/a.md",'
           '"version":"' + h + '","data":"aGVsbG8=","state":"' + state_name + '",'
           '"trashedAt":' + timestamp_json + ',"unused":{"z":18446744073709551617,"a":1e-7}}').encode()
    db_path = state / 'users' / DAILY / 'managed-diary.db'
    capsule_path = f'.noevia-trash/{ID}.json'
    with sqlite3.connect(db_path) as db:
        db.execute('INSERT INTO files VALUES (?,?,?,1.0)', (capsule_path, raw, hashlib.sha256(raw).hexdigest()))
    snap = snapshot(state, DAILY)
    replies, effects = [], []
    for mode in (modes.python, modes.rust):
        restore(state, DAILY, snap)
        warm(client, modes)
        mode()
        Recorder.served = []
        response = request(client, DAILY, 'POST', TRASH, {'action': action, 'id': ID, 'path': 'Notes/a.md', 'version': h})
        assert response.status_code == 200, response.text
        if mode == modes.rust:
            assert Recorder.served == [True]
        replies.append(wire(response))
        with sqlite3.connect(db_path) as db:
            data, version = db.execute('SELECT data,version FROM files WHERE path=?', (capsule_path,)).fetchone()
            assert version == hashlib.sha256(data).hexdigest()
            if action == 'restore' and state_name == 'trashed':
                record = json.loads(data)
                assert data == json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
                assert isinstance(record.pop('restoredAt'), float)
                # Capsule encoding sorts recursively, while HTTP summary preserves insertion order.
                normalized = json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
                assert b'"unused": {"a": 1e-07, "z": 18446744073709551617}' in normalized
                effects.append(normalized)
                assert db.execute('SELECT data FROM files WHERE path=?', ('Notes/a.md',)).fetchone()[0] == b'hello'
            else:
                assert data == raw, 'retry must leave the legacy capsule bytes unchanged'
                effects.append(data)
    assert replies[0] == replies[1]
    assert effects[0] == effects[1]
