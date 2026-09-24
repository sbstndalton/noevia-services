import hashlib
from types import SimpleNamespace
import pytest
from agent.managed_storage import ManagedCorpusBackend
from agent.corpus_store import CorpusStore
from agent.workspace_ops import OpError, operate, stat, MAX_ENTRIES
from agent.workspace_trash import change, list_trash, PREFIX
from tests.test_dedicated_storage import B, volume  # noqa: F401 (fixture)

sha = lambda b: hashlib.sha256(b).hexdigest()


@pytest.fixture
def store(tmp_path):
    backend = ManagedCorpusBackend(tmp_path / 'managed', B)
    backend.activate({
        'notes/a.md': b'# A\n', 'notes/sub/b.md': b'# B\n', 'solo/only.md': b'# only\n',
        'AI Memory/profile.md': b'# memory\n', 'Entries/2026/September/September 14, 2026.md': b'raw',
        'INDEX.md': b'# index\n', 'capture.md': b'text <!-- xid:1234abcd-0000 -->\n', 'notes/image.bin': b'\x00',
    }, {})
    cfg = SimpleNamespace(get=lambda key, default=None: {'corpus.entry_layout': 'daily'}.get(key, default))
    return CorpusStore(cfg, backend, None)


def files(store):
    with store.backend.db() as db:
        return {r[0]: bytes(r[1]) for r in db.execute('SELECT path,data FROM files') if not r[0].startswith(PREFIX)}


def expect(status, fn):
    with pytest.raises(OpError) as exc:
        fn()
    assert exc.value.status == status, exc.value
    return exc.value


def test_delete_file_goes_to_trash_and_restores(store):
    body = {'op': 'delete', 'path': 'solo/only.md', 'version': sha(b'# only\n')}
    result = operate(store, body)
    assert 'solo/only.md' not in files(store)
    assert stat(store, 'solo')['isDir'], 'the emptied folder stays listed'
    assert list_trash(store)['records'][0]['path'] == 'solo/only.md'
    change(store, {'action': 'restore', 'id': result['trash'][0]})
    assert files(store)['solo/only.md'] == b'# only\n'
    # A replay with the same precondition is a 412 or success, never a second effect.
    operate(store, body)
    expect(404, lambda: operate(store, body))


def test_preconditions(store):
    expect(428, lambda: operate(store, {'op': 'delete', 'path': 'solo/only.md'}))
    expect(412, lambda: operate(store, {'op': 'delete', 'path': 'solo/only.md', 'version': sha(b'stale')}))
    assert 'solo/only.md' in files(store)


@pytest.mark.parametrize('path', ['AI Memory/profile.md', 'AI Memory', 'Entries', 'Entries/2026/September/September 14, 2026.md', 'INDEX.md'])
def test_protected_paths_cannot_be_deleted_or_moved(store, path):
    before = files(store)
    version = stat(store, path)['version']
    expect(403, lambda: operate(store, {'op': 'delete', 'path': path, 'version': version}))
    expect(403, lambda: operate(store, {'op': 'move', 'path': path, 'destination': 'elsewhere/x.md' if path.endswith('.md') else 'elsewhere', 'version': version}))
    assert files(store) == before


@pytest.mark.parametrize('destination', ['AI Memory/new.md', 'Entries/new.md', 'INDEX.md'])
def test_protected_destinations(store, destination):
    before = files(store)
    expect(403, lambda: operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': destination, 'version': sha(b'# only\n'), 'overwrite': True}))
    expect(403, lambda: operate(store, {'op': 'copy', 'path': 'solo/only.md', 'destination': destination}))
    assert files(store) == before


def test_folder_delete_is_all_or_nothing(store):
    version = stat(store, 'notes')['version']
    expect(409, lambda: operate(store, {'op': 'delete', 'path': 'notes', 'version': version}))  # holds a non-Markdown file
    assert 'notes/a.md' in files(store) and 'notes/sub/b.md' in files(store)
    with store.backend.db() as db:
        db.execute("DELETE FROM files WHERE path='notes/image.bin'")
    version = stat(store, 'notes')['version']
    result = operate(store, {'op': 'delete', 'path': 'notes', 'version': version})
    assert len(result['trash']) == 2 and not any(p.startswith('notes') for p in files(store))


def test_folder_version_changes_with_children(store):
    before = stat(store, 'notes')['version']
    _, tag = store.backend.get('notes/a.md')
    store.backend.put('notes/a.md', b'# changed\n', if_match=tag)
    expect(412, lambda: operate(store, {'op': 'delete', 'path': 'notes', 'version': before}))


def test_capture_records_are_not_trashed(store):
    expect(403, lambda: operate(store, {'op': 'delete', 'path': 'capture.md', 'version': stat(store, 'capture.md')['version']}))


def test_move_file_and_folder(store):
    operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': 'notes/renamed.md', 'version': sha(b'# only\n')})
    assert files(store)['notes/renamed.md'] == b'# only\n' and 'solo/only.md' not in files(store)
    version = stat(store, 'notes')['version']
    operate(store, {'op': 'move', 'path': 'notes', 'destination': 'archive', 'version': version})
    assert {'archive/a.md', 'archive/sub/b.md', 'archive/renamed.md', 'archive/image.bin'} <= set(files(store))
    with store.backend.db() as db:
        outbox = {r[0] for r in db.execute('SELECT path FROM workspace_index_outbox')}
    assert {'notes/a.md', 'archive/a.md'} <= outbox


def test_move_rules(store):
    v = sha(b'# only\n')
    expect(409, lambda: operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': 'missing/x.md', 'version': v}))
    expect(412, lambda: operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': 'notes/a.md', 'version': v}))
    expect(428, lambda: operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': 'notes/a.md', 'version': v, 'overwrite': True}))
    expect(403, lambda: operate(store, {'op': 'move', 'path': 'notes', 'destination': 'notes/sub/inner', 'version': stat(store, 'notes')['version']}))
    expect(403, lambda: operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': 'solo/only.txt', 'version': v}))
    expect(403, lambda: operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': '../escape.md', 'version': v}))
    expect(403, lambda: operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': '.noevia-trash/x.md', 'version': v}))
    expect(409, lambda: operate(store, {'op': 'move', 'path': 'solo', 'destination': 'notes', 'version': stat(store, 'solo')['version'], 'overwrite': True}))
    result = operate(store, {'op': 'move', 'path': 'solo/only.md', 'destination': 'notes/a.md', 'version': v, 'overwrite': True, 'destinationVersion': sha(b'# A\n')})
    assert result['replaced'] and files(store)['notes/a.md'] == b'# only\n'
    assert any(r['path'] == 'notes/a.md' for r in list_trash(store)['records']), 'replaced destination is recoverable'


def test_copy(store):
    operate(store, {'op': 'copy', 'path': 'notes/sub', 'destination': 'copied'})
    assert files(store)['copied/b.md'] == b'# B\n' and files(store)['notes/sub/b.md'] == b'# B\n'
    expect(412, lambda: operate(store, {'op': 'copy', 'path': 'notes/sub', 'destination': 'copied'}))


def test_bounds(store, monkeypatch):
    import agent.workspace_ops as ops
    monkeypatch.setattr(ops, 'MAX_ENTRIES', 2)
    expect(507, lambda: operate(store, {'op': 'copy', 'path': 'notes', 'destination': 'big'}))
    assert not any(p.startswith('big') for p in files(store))


def test_requires_managed_storage(tmp_path):
    cfg = SimpleNamespace(get=lambda key, default=None: {'corpus.entry_layout': 'daily'}.get(key, default))
    local = CorpusStore(cfg, SimpleNamespace(), None)
    expect(409, lambda: operate(local, {'op': 'delete', 'path': 'a.md', 'version': 'x'}))
    expect(400, lambda: operate(local, {'op': 'rename'}))


def test_http_auth_tenant_pending_and_status_codes(volume, monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as appmod
    from tests.test_dedicated_storage import request, A
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: pytest.fail('File operations must not call embeddings'))
    client = TestClient(appmod.app)
    state = appmod._tenant_state(request(B), recover=False)
    state.backend.put('notes.md', b'synthetic')
    headers = {'X-Cowork-User-ID': B}
    body = {'op': 'delete', 'path': 'notes.md', 'version': sha(b'synthetic')}
    assert client.post('/api/workspace-ops', json={'op': 'stat', 'path': 'notes.md'}, headers=headers).json()['version'] == sha(b'synthetic')
    monkeypatch.setattr(state.journal, 'pending_count', lambda: 1)
    assert client.post('/api/workspace-ops', json=body, headers=headers).status_code == 409
    monkeypatch.setattr(state.journal, 'pending_count', lambda: 0)
    assert client.post('/api/workspace-ops', json={**body, 'version': sha(b'x')}, headers=headers).status_code == 412
    assert client.post('/api/workspace-ops', json={'op': 'delete', 'path': 'INDEX.md', 'version': 'x'}, headers=headers).status_code in (403, 404)
    other = client.post('/api/workspace-ops', json=body, headers={'X-Cowork-User-ID': A})
    assert other.status_code in (404, 409), 'another tenant never sees this file'
    response = client.post('/api/workspace-ops', json=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    assert 'notes.md' in state.journal.dirty_documents()
    monkeypatch.setattr(appmod, 'check_auth', lambda r: False)
    assert client.post('/api/workspace-ops', json=body, headers=headers).status_code == 401


def test_preserve_keeps_the_previous_version_restorable_beside_the_file(store):
    result = operate(store, {'op': 'preserve', 'path': 'notes/a.md', 'version': sha(b'# A\n')})
    assert files(store)['notes/a.md'] == b'# A\n', 'preserve never changes the file itself'
    record = [r for r in list_trash(store)['records'] if r['id'] == result['trash']][0]
    assert record['path'].startswith('notes/a (replaced ') and record['path'].endswith(').md')
    # Restoring never overwrites: it lands beside the current file.
    change(store, {'action': 'restore', 'id': result['trash']})
    assert files(store)[record['path']] == b'# A\n'
    assert files(store)['notes/a.md'] == b'# A\n'


def test_preserve_refuses_protected_stale_and_missing(store):
    for path in ['INDEX.md', 'AI Memory/profile.md', 'Entries/2026/September/September 14, 2026.md']:
        expect(428, lambda: operate(store, {'op': 'preserve', 'path': path, 'version': stat(store, path)['version']}))
    expect(412, lambda: operate(store, {'op': 'preserve', 'path': 'notes/a.md', 'version': sha(b'stale')}))
    expect(404, lambda: operate(store, {'op': 'preserve', 'path': 'notes/missing.md', 'version': sha(b'x')}))
    expect(404, lambda: operate(store, {'op': 'preserve', 'path': 'notes', 'version': 'x'}))


def test_preserve_of_a_near_max_length_path_stays_listable_and_restorable(store):
    # rel_path allows up to 500 characters; the " (replaced ...)" suffix used to be
    # appended on top of that without limit, so the stored capsule's own path field
    # could exceed 500 and later fail safe_path() when decoded (both on restore and
    # while listing Trash). The stem must now be shortened to keep the full replaced
    # name at or under 500 characters, with the extension and suffix intact.
    long_stem = 'notes/' + 'x' * (490 - len('notes/.md'))
    long_path = long_stem + '.md'
    assert len(long_path) == 490
    with store.backend.db() as db:
        db.execute('INSERT INTO files VALUES (?,?,?,?)', (long_path, b'# long\n', sha(b'# long\n'), 0))
        db.execute('INSERT OR IGNORE INTO directories VALUES (?)', ('notes',))

    result = operate(store, {'op': 'preserve', 'path': long_path, 'version': sha(b'# long\n')})

    listing = list_trash(store)
    matches = [r for r in listing['records'] if r['id'] == result['trash']]
    assert len(matches) == 1, 'the preserved capsule must still be listed, not hidden by a 409'
    record = matches[0]
    assert record.get('invalid') is not True
    assert len(record['path']) <= 500
    assert record['path'].endswith(').md') and ' (replaced ' in record['path']

    change(store, {'action': 'restore', 'id': result['trash']})
    assert files(store)[record['path']] == b'# long\n'
    assert files(store)[long_path] == b'# long\n', 'preserve never changes the original file'


def test_list_trash_skips_a_corrupt_capsule_instead_of_failing_the_page(store):
    # Hand-plant a capsule whose recorded path is invalid (over 500 chars), simulating
    # data written before this fix, or any other form of corruption. It must not make
    # every other record on the page (or the whole listing) return a 409/error.
    result = operate(store, {'op': 'delete', 'path': 'solo/only.md', 'version': sha(b'# only\n')})
    good_id = result['trash'][0]

    import base64
    import json
    import time as time_mod
    from agent.workspace_trash import PREFIX as TRASH_PREFIX
    bad_id = '00000000-0000-0000-0000-000000000000'
    bad_record = {
        'format': 'noevia-trash-v1', 'id': bad_id, 'path': 'x' * 600 + '.md',
        'version': sha(b'junk'), 'data': base64.b64encode(b'junk').decode(),
        'state': 'trashed', 'trashedAt': time_mod.time(),
    }
    encoded = json.dumps(bad_record, sort_keys=True, ensure_ascii=False).encode()
    with store.backend.db() as db:
        db.execute('INSERT INTO files VALUES (?,?,?,?)', (TRASH_PREFIX + bad_id + '.json', encoded, sha(encoded), 0))

    listing = list_trash(store)
    ids = {r['id'] for r in listing['records']}
    assert good_id in ids, 'a valid record must still be listed alongside a corrupt one'
    bad = [r for r in listing['records'] if r['id'] == bad_id][0]
    assert bad.get('invalid') is True
