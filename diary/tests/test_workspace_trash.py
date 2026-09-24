import hashlib
import json
import sqlite3
import time
import uuid
from types import SimpleNamespace
import pytest
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_trash import change, list_trash, PREFIX
from agent.workspace_export import archive
from agent.workspace_restore import verify, restore as zip_restore
from agent.backup_restore import restore as remote_restore
from agent.local_storage import LocalCorpusBackend
from agent.corpus_store import CorpusStore
from tests.test_dedicated_storage import A, B, volume


@pytest.fixture
def fixture(tmp_path):
    backend = ManagedCorpusBackend(tmp_path / 'managed', B)
    backend.activate({'notes/a.md': b'---\r\ntags: [test]\r\n---\r\n# Exact source\r\n'}, {})
    cfg = SimpleNamespace(get=lambda key, default=None: {'corpus.entry_layout': 'daily'}.get(key, default))
    store = CorpusStore(cfg, backend, None)
    body = {'action': 'trash', 'id': str(uuid.uuid4()), 'path': 'notes/a.md', 'version': backend.get('notes/a.md')[1][8:-1]}
    return store, body


def test_roundtrip_retry_after_restore_and_restart(fixture):
    store, body = fixture
    original = store.backend.get(body['path'])[0]
    assert not change(store, body)['alreadyApplied']
    assert store.backend.get(body['path'])[0] is None
    assert list_trash(store)['records'][0]['state'] == 'trashed'
    store.backend = ManagedCorpusBackend(store.backend.root, B)
    assert change(store, body)['alreadyApplied']
    restore = {'action': 'restore', 'id': body['id']}
    assert change(store, restore)['state'] == 'restored'
    assert store.backend.get(body['path'])[0] == original
    _, tag = store.backend.get(body['path'])
    store.backend.put(body['path'], b'new edit', if_match=tag)
    assert change(store, restore)['alreadyApplied']
    assert change(store, body)['alreadyApplied']
    assert store.backend.get(body['path'])[0] == b'new edit'
    with store.backend.db() as db:
        assert store.backend.meta(db, 'generation') == '4'
        assert [r[0] for r in db.execute('SELECT path FROM workspace_index_outbox')] == [body['path']]


@pytest.mark.parametrize('path', ['Entries/2026/September/September 14, 2026.md', 'INDEX.md', '2026-09.md'])
def test_capture_paths_protected(fixture, path):
    store, body = fixture
    store.backend.put(path, b'raw')
    with pytest.raises(ValueError, match='capture files'):
        change(store, {**body, 'path': path, 'version': hashlib.sha256(b'raw').hexdigest()})
    assert store.backend.get(path)[0] == b'raw'


def test_ai_memory_path_protected(fixture):
    # workspace_ops.protected() covers AI Memory/**, and the trash endpoint
    # must refuse the same paths, not just capture/month/index files.
    store, body = fixture
    store.backend.put('AI Memory/notes.md', b'memory')
    with pytest.raises(ValueError, match='AI Memory'):
        change(store, {**body, 'path': 'AI Memory/notes.md', 'version': hashlib.sha256(b'memory').hexdigest()})
    assert store.backend.get('AI Memory/notes.md')[0] == b'memory'


def test_marker_and_changed_version_protected(fixture):
    store, body = fixture
    _, tag = store.backend.get(body['path'])
    raw = b'<!-- xid:1234-abcd -->'
    store.backend.put(body['path'], raw, if_match=tag)
    with pytest.raises(ValueError, match='changed'):
        change(store, body)
    with pytest.raises(ValueError, match='Capture records'):
        change(store, {**body, 'version': hashlib.sha256(raw).hexdigest()})


@pytest.mark.parametrize('collision', ['notes/a.md', 'notes/a.md/child.md', 'notes'])
def test_restore_collision_and_id_mismatch(fixture, collision):
    store, body = fixture
    change(store, body)
    store.backend.put(collision, b'competing file')
    with pytest.raises(ValueError, match='occupied'):
        change(store, {'action': 'restore', 'id': body['id']})
    with pytest.raises(ValueError, match='different file version'):
        change(store, {**body, 'version': 'changed'})
    assert list_trash(store)['records'][0]['state'] == 'trashed'
    assert store.backend.get(collision)[0] == b'competing file'


def test_rollback_on_outbox_failure(fixture):
    store, body = fixture
    with store.backend.db() as db:
        db.execute("CREATE TRIGGER synthetic_failure BEFORE INSERT ON workspace_index_outbox BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        change(store, body)
    assert store.backend.get(body['path'])[0] is not None
    assert list_trash(store)['records'] == []
    with store.backend.db() as db:
        assert store.backend.meta(db, 'generation') == '1'


def test_zip_and_remote_backup_preserve_recoverable_bytes(fixture, tmp_path):
    store, body = fixture
    original = store.backend.get(body['path'])[0]
    change(store, body)
    capsule = PREFIX + body['id'] + '.json'
    zipped = archive(store.backend, '', {})
    files, _, _ = verify(zipped)
    assert body['path'] not in files and capsule in files
    destination = tmp_path / 'zip-restored'
    zip_restore(zipped, destination, hashlib.sha256(zipped).hexdigest())
    assert (destination / capsule).read_bytes() == store.backend.get(capsule)[0]
    remote_root = tmp_path / 'remote'
    remote = LocalCorpusBackend(str(remote_root))
    storage = {'kind': 'webdav', 'baseUrl': 'https://synthetic.invalid', 'username': 'fixture', 'corpusRoot': 'Diary'}
    assert store.backend.backup(remote, storage, now=time.time()+10)['backup'] == 'complete'
    manifest = next((remote_root / f'Diary/noevia-backups/{B}/manifests').glob('*.json'))
    recovered = tmp_path / 'remote-restored'
    remote_restore(remote_root, manifest, recovered)
    assert (recovered / capsule).read_bytes() == files[capsule]
    restored_backend = ManagedCorpusBackend(tmp_path / 'new-managed', B)
    restored_backend.activate({capsule: (recovered / capsule).read_bytes()}, {})
    store.backend = restored_backend
    change(store, {'action': 'restore', 'id': body['id']})
    assert store.backend.get(body['path'])[0] == original


def test_http_auth_tenant_pending_and_no_inference(volume, monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as appmod
    from tests.test_dedicated_storage import request
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: pytest.fail('Trash must not call embeddings'))
    client = TestClient(appmod.app)
    state = appmod._tenant_state(request(B), recover=False)
    state.backend.put('notes.md', b'synthetic')
    body = {'action': 'trash', 'id': str(uuid.uuid4()), 'path': 'notes.md', 'version': hashlib.sha256(b'synthetic').hexdigest()}
    headers = {'X-Cowork-User-ID': B}
    monkeypatch.setattr(state.journal, 'pending_count', lambda: 1)
    assert client.post('/api/workspace-trash', json=body, headers=headers).status_code == 409
    monkeypatch.setattr(state.journal, 'pending_count', lambda: 0)
    response = client.post('/api/workspace-trash', json=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()['indexPending'] and not state.recovered
    assert 'notes.md' in state.journal.dirty_documents()
    assert client.get('/api/workspace-trash', headers=headers).headers['cache-control'] == 'no-store'
    assert client.post('/api/workspace-trash', json=body, headers={'X-Cowork-User-ID': A}).status_code == 409
    monkeypatch.setattr(appmod, 'check_auth', lambda r: False)
    assert client.get('/api/workspace-trash', headers=headers).status_code == 401
    assert client.post('/api/workspace-trash', json=body, headers=headers).status_code == 401


def test_operator_capsule_extraction_never_overwrites(fixture, tmp_path):
    from agent.workspace_trash import recover_capsule
    store, body = fixture
    original = store.backend.get(body['path'])[0]
    change(store, body)
    capsule = tmp_path / (body['id'] + '.json')
    capsule.write_bytes(store.backend.get(PREFIX + capsule.name)[0])
    destination = tmp_path / 'recovered.md'
    assert recover_capsule(capsule, destination)['bytes'] == len(original)
    assert destination.read_bytes() == original
    with pytest.raises(FileExistsError):
        recover_capsule(capsule, destination)
    corrupted = json.loads(capsule.read_bytes());corrupted['data'] = 'eA=='
    capsule.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match='invalid'):
        recover_capsule(capsule, tmp_path / 'bad.md')
    assert not (tmp_path / 'bad.md').exists()


def test_concurrent_trash_has_one_winner(fixture):
    import concurrent.futures
    store, body = fixture
    def attempt(operation):
        try:
            return change(store, {**body, 'id': operation})['state']
        except ValueError:
            return 'conflict'
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(attempt, [str(uuid.uuid4()), str(uuid.uuid4())])) == ['conflict', 'trashed']
    assert len(list_trash(store)['records']) == 1
