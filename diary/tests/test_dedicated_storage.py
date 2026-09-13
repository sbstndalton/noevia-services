import base64
import json
from datetime import date
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
import agent.app as appmod
from agent.config import Config
from agent.dedicated_storage import StorageUnavailable, tenant_volume
from agent.local_storage import LocalCorpusBackend
from agent.workspace_files import file_read, file_write

A = '11111111-1111-4111-8111-111111111111'
B = '22222222-2222-4222-8222-222222222222'
PREFIX = 'Documents/Diary'


def request(user=A, storage=None):
    headers = [(b'x-cowork-user-id', user.encode())]
    if storage:
        headers.append((b'x-cowork-storage', base64.urlsafe_b64encode(json.dumps(storage).encode())))
    return Request({'type': 'http', 'headers': headers})


@pytest.fixture
def volume(tmp_path, monkeypatch):
    root = tmp_path / 'volume'
    root.mkdir()
    (root / '.noevia-diary-volume').write_text(A)
    (root / PREFIX).mkdir(parents=True)
    monkeypatch.setenv('DIARY_LOCAL_VOLUMES', json.dumps({A: {'root': str(root), 'prefix': PREFIX}}))
    cfg = Config({
        'corpus': {'backend': 'local', 'root': '', 'monthly_prefix': '',
                   'entry_layout': 'daily', 'entries_prefix': 'Entries', 'index_enabled': False,
                   'local': {'root': str(tmp_path / 'default')}},
        'retrieval': {'db_path': str(tmp_path / 'state' / 'index.db')},
        'llm': {'base_url': 'http://stub', 'chat_model': 'm', 'embed_model': 'e',
                'aux': {'base_url': 'http://stub', 'model': 'a'}},
        'ui': {'auth_token': ''},
    })
    monkeypatch.setattr(appmod, '_base_cfg', cfg)
    monkeypatch.setattr(appmod, 'check_auth', lambda r: True)
    appmod._tenant_states.clear()
    yield root
    for state in appmod._tenant_states.values():
        appmod._close_state(state)
    appmod._tenant_states.clear()


def test_operator_volume_preserves_paths_and_state(volume):
    state = appmod._tenant_state(request(storage={'kind': 'webdav', 'baseUrl': 'http://not-used'}))
    assert state.backend.root == volume
    assert state.store._join('Entries/note.md') == PREFIX + '/Entries/note.md'
    assert appmod._tenant_state(request(storage={'kind': 's3', 'bucket': 'not-used'})) is state
    state.store.log_exchange(date(2026, 9, 12), 'Synthetic', 'synthetic note', 'synthetic answer')
    assert (volume / PREFIX / 'Entries/2026/September/September 12, 2026.md').is_file()
    assert state.journal.pending_count() == 0
    assert not list(volume.rglob('*.db*'))
    other = appmod._tenant_state(request(B))
    assert other.backend.root != volume
    assert other.backend.get_text(PREFIX + '/Entries/2026/September/September 12, 2026.md')[0] is None
    assert tenant_volume(B) is None


def test_guarded_write_conflict_and_dirty_index(volume):
    state = appmod._tenant_state(request())
    first = file_write(state.store, {'path': 'AI Memory/test.md', 'content': 'one', 'version': None})
    file_write(state.store, {'path': first['path'], 'content': 'two', 'version': first['version']})
    with pytest.raises(Exception) as caught:
        file_write(state.store, {'path': first['path'], 'content': 'stale', 'version': first['version']})
    assert caught.value.status_code == 409
    assert file_read(state.store, first['path'])['content'] == 'two'
    assert state.journal.dirty_documents() == [PREFIX + '/AI Memory/test.md']


@pytest.mark.parametrize('operation', ['get', 'put', 'exists', 'list_dir'])
def test_disappearing_volume_fails_even_for_cached_backend(volume, operation):
    state = appmod._tenant_state(request())
    (volume / '.noevia-diary-volume').unlink()
    with pytest.raises(StorageUnavailable):
        appmod._tenant_state(request())
    with pytest.raises(StorageUnavailable):
        getattr(state.backend, operation)(PREFIX + '/test.md', *([b'bad'] if operation == 'put' else []))
    assert not (volume / PREFIX / 'test.md').exists()


def test_absent_volume_does_not_get_created(tmp_path):
    missing = tmp_path / 'missing'
    with pytest.raises(StorageUnavailable):
        LocalCorpusBackend(str(missing), volume_identity=A)
    assert not missing.exists()


def test_wrong_volume_identity_is_service_unavailable(volume):
    (volume / '.noevia-diary-volume').write_text(B)
    response = TestClient(appmod.app).get('/api/files', headers={'X-Cowork-User-ID': A})
    assert response.status_code == 503
    assert 'unavailable' in response.json()['detail']


@pytest.mark.parametrize('raw', ['[]', '{', json.dumps({A: {'root': 'relative', 'prefix': ''}}),
    json.dumps({A: {'root': '/tmp/a', 'prefix': '../other'}}),
    json.dumps({A: {'root': '/tmp/a', 'prefix': ''}, B: {'root': '/tmp/a/b', 'prefix': ''}})])
def test_bad_operator_config_never_falls_back(monkeypatch, raw):
    monkeypatch.setenv('DIARY_LOCAL_VOLUMES', raw)
    with pytest.raises(StorageUnavailable):
        tenant_volume(A)


def test_atomic_replacement_preserves_reader_access(volume):
    import os
    backend = LocalCorpusBackend(str(volume), volume_identity=A, reader_uid=os.getuid())
    ok, version, _ = backend.put(PREFIX + '/reader.md', b'first')
    assert ok
    assert backend.put(PREFIX + '/reader.md', b'second', if_match=version)[0]
    target = volume / PREFIX / 'reader.md'
    assert target.stat().st_uid == os.getuid()
    assert target.stat().st_mode & 0o777 == 0o600


def test_directory_creation_fails_closed_when_volume_identity_changes(volume):
    from agent.workspace_files import directory_create
    state = appmod._tenant_state(request())
    directory_create(state.store, 'SyntheticFolder')
    assert (volume / PREFIX / 'SyntheticFolder').is_dir()
    (volume / '.noevia-diary-volume').write_text(B)
    with pytest.raises(StorageUnavailable):
        directory_create(state.store, 'MustNotExist')
    assert not (volume / PREFIX / 'MustNotExist').exists()
