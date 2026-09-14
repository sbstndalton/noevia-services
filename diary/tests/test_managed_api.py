import base64
import json
from fastapi.testclient import TestClient
import agent.app as appmod
from agent.local_storage import LocalCorpusBackend
from tests.test_dedicated_storage import volume, A, B, PREFIX, request


def test_new_account_defaults_to_app_and_connection_change_cannot_replace_it(volume, monkeypatch):
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    client = TestClient(appmod.app)
    headers = {'X-Cowork-User-ID': B}
    assert client.get('/api/storage-status', headers=headers).json()['mode'] == 'managed'
    body = {'path': 'MEMORY.md', 'content': 'synthetic owned data', 'version': None}
    assert client.put('/api/file', headers=headers, json=body).status_code == 200
    changed = dict(headers, **{'X-Cowork-Storage': base64.urlsafe_b64encode(json.dumps({'kind': 'webdav', 'baseUrl': 'http://offline.invalid', 'corpusRoot': 'Other'}).encode()).decode()})
    assert client.post('/api/file', headers=changed, json={'path': 'MEMORY.md'}).json()['content'] == body['content']
    assert client.get('/api/storage-status', headers=changed).json()['backup'] == 'pending'
    appmod._tenant_states.clear()  # process-cache loss, not persistent-state loss
    assert client.post('/api/file', headers=changed, json={'path': 'MEMORY.md'}).json()['content'] == body['content']


def test_explicit_verified_cutover_retains_original_and_editor_conflicts(volume, monkeypatch):
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    local = LocalCorpusBackend(str(volume))
    local.put(PREFIX + '/Entries/a.md', b'# synthetic original')
    client = TestClient(appmod.app)
    headers = {'X-Cowork-User-ID': A}
    assert client.get('/api/storage-status', headers=headers).json()['mode'] == 'legacy'
    preview = client.post('/api/storage-import', headers=headers, json={}).json()
    assert preview['fileCount'] == 1
    _, v = local.get(PREFIX + '/Entries/a.md')
    local.put(PREFIX + '/Entries/a.md', b'# changed', if_match=v)
    assert client.post('/api/storage-import', headers=headers, json={'fingerprint': preview['fingerprint']}).status_code == 409
    assert client.get('/api/storage-status', headers=headers).json()['mode'] == 'legacy'
    preview = client.post('/api/storage-import', headers=headers, json={}).json()
    assert client.post('/api/storage-import', headers=headers, json={'fingerprint': preview['fingerprint']}).json()['imported']
    assert client.get('/api/storage-status', headers=headers).json()['mode'] == 'managed'
    current = client.post('/api/file', headers=headers, json={'path': 'Entries/a.md'}).json()
    assert current['content'] == '# changed'
    edit = {**current, 'content': '# app edit'}
    assert client.put('/api/file', headers=headers, json=edit).status_code == 200
    assert client.put('/api/file', headers=headers, json={**edit, 'content': 'stale'}).status_code == 409
    assert local.get(PREFIX + '/Entries/a.md')[0] == b'# changed'
    assert client.post('/api/storage-import', headers=headers, json={'fingerprint': preview['fingerprint']}).status_code == 409


def test_backup_worker_never_initializes_legacy_or_unidentified_tenant(volume):
    client = TestClient(appmod.app)
    assert client.post('/api/storage-backup').status_code == 400
    assert client.post('/api/storage-backup', headers={'X-Cowork-User-ID': B}).json()['mode'] == 'legacy'
    assert not list(volume.rglob('*.db'))


def test_blocked_backup_never_blocks_managed_primary_or_opens_legacy(volume, monkeypatch):
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    client = TestClient(appmod.app)
    headers = {'X-Cowork-User-ID': B}
    assert client.get('/api/storage-status', headers=headers).json()['mode'] == 'managed'
    headers.update({'X-Cowork-Storage-Blocked': '1', 'X-Cowork-Storage': base64.urlsafe_b64encode(b'{"kind":"blocked"}').decode()})
    assert client.put('/api/file', headers=headers, json={'path': 'offline.md', 'content': 'retained', 'version': None}).status_code == 200
    assert client.get('/api/storage-status', headers=headers).json()['backup'] == 'failed'
    headers['X-Cowork-User-ID'] = A
    assert client.get('/api/storage-status', headers=headers).status_code == 403
