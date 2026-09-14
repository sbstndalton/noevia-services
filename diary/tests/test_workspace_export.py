import hashlib
import io
import json
import zipfile
import pytest
from fastapi.testclient import TestClient
import agent.app as appmod
from agent.local_storage import LocalCorpusBackend
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from tests.test_dedicated_storage import volume, A, B, PREFIX


def unpack(body):
    z = zipfile.ZipFile(io.BytesIO(body))
    manifest = json.loads(z.read('manifest.json'))
    for row in manifest['files']:
        data = z.read('workspace/' + row['path'])
        assert len(data) == row['bytes']
        assert hashlib.sha256(data).hexdigest() == row['sha256']
    return z, manifest


def test_managed_binary_empty_folders_and_manifest_collision(tmp_path):
    backend = ManagedCorpusBackend(tmp_path, B)
    data = b'\x00\xff\x89PNG\r\n'
    backend.activate({'note.md': b'---\ntags: unknown\n---\n# Synthetic\r\n', 'assets/pic.png': data,
                      'manifest.json': b'user source'}, {}, ['empty/nested'])
    z, manifest = unpack(archive(backend, '', {}))
    assert z.read('workspace/assets/pic.png') == data
    assert z.read('workspace/manifest.json') == b'user source'
    assert 'workspace/empty/nested/' in z.namelist()
    assert 'workspace/empty/' in z.namelist()
    assert manifest['fileCount'] == 3
    assert 'tenant' not in manifest


def test_managed_empty_export(tmp_path):
    backend = ManagedCorpusBackend(tmp_path, B)
    backend.activate({}, {}, ['empty'])
    z, manifest = unpack(archive(backend, '', {}))
    assert manifest['fileCount'] == 0
    assert 'workspace/empty/' in z.namelist()


def test_legacy_changed_source_refused(tmp_path):
    backend = LocalCorpusBackend(str(tmp_path))
    backend.put('note.md', b'first')
    original = backend.get_bounded
    count = 0
    def changing(*args):
        nonlocal count
        count += 1
        data, etag = original(*args)
        if count == 1:
            backend.put('note.md', b'changed', if_match=etag)
        return data, etag
    backend.get_bounded = changing
    with pytest.raises(ValueError, match='changed during export'):
        archive(backend, '', {})


def test_api_export_is_scoped_and_read_only(volume, monkeypatch):
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: pytest.fail('Export must not reindex'))
    backend = LocalCorpusBackend(str(volume))
    backend.put(PREFIX + '/synthetic.md', b'legacy synthetic')
    client = TestClient(appmod.app)
    response = client.get('/api/workspace-export', headers={'X-Cowork-User-ID': A})
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    z, manifest = unpack(response.content)
    assert z.read('workspace/synthetic.md') == b'legacy synthetic'
    other = client.get('/api/workspace-export', headers={'X-Cowork-User-ID': B})
    assert other.status_code == 200
    assert 'synthetic.md' not in [r['path'] for r in unpack(other.content)[1]['files']]
    assert backend.get(PREFIX + '/synthetic.md')[0] == b'legacy synthetic'
    monkeypatch.setattr(appmod, 'check_auth', lambda r: False)
    assert client.get('/api/workspace-export', headers={'X-Cowork-User-ID': A}).status_code == 401


def test_export_defers_recovery_until_normal_access(volume, monkeypatch):
    from tests.test_dedicated_storage import request
    calls = []
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: calls.append(st))
    from agent.corpus_store import CorpusStore
    monkeypatch.setattr(CorpusStore, 'apply_pending', lambda st: calls.append('replay'))
    client = TestClient(appmod.app)
    assert client.get('/api/workspace-export', headers={'X-Cowork-User-ID': B}).status_code == 200
    assert calls == []
    state = appmod._tenant_state(request(B))
    assert calls == ['replay', state]
    appmod._tenant_state(request(B))
    assert calls == ['replay', state]


def test_export_refuses_pending_writes(volume, monkeypatch):
    from tests.test_dedicated_storage import request
    state = appmod._tenant_state(request(B), recover=False)
    monkeypatch.setattr(state.journal, 'pending_count', lambda: 1)
    response = TestClient(appmod.app).get('/api/workspace-export', headers={'X-Cowork-User-ID': B})
    assert response.status_code == 409
    assert 'pending' in response.json()['detail']


def test_export_caps_checked_before_loading_blobs(tmp_path, monkeypatch):
    import agent.workspace_export as export
    backend = ManagedCorpusBackend(tmp_path, B)
    backend.activate({'synthetic.bin': b'1234'}, {})
    monkeypatch.setattr(export, 'MAX_FILE', 3)
    with pytest.raises(ValueError, match='safety limit'):
        archive(backend, '', {})
