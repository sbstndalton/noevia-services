import io
import json
import zipfile
import pytest
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from agent.workspace_restore import preview, restore
from tests.test_dedicated_storage import B


@pytest.fixture
def bundle(tmp_path):
    backend = ManagedCorpusBackend(tmp_path / 'source', B)
    backend.activate({'raw/note.md': b'---\nunknown: yes\n---\nSynthetic\r\n', 'assets/image.bin': b'\x00\xff'}, {}, ['empty'])
    return archive(backend, '', {})


def mutate(body, change):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(body)) as source, zipfile.ZipFile(output, 'w') as dest:
        rows = {info.filename: source.read(info) for info in source.infolist()}
        change(rows)
        for path, data in rows.items():
            dest.writestr(path, data)
    return output.getvalue()


def test_preview_restore_roundtrip_and_no_overwrite(bundle, tmp_path):
    report = preview(bundle)
    target = tmp_path / 'restored'
    assert not target.exists()
    assert restore(bundle, target, report['fingerprint'])['fileCount'] == 2
    assert (target / 'assets/image.bin').read_bytes() == b'\x00\xff'
    assert (target / 'raw/note.md').read_bytes() == b'---\nunknown: yes\n---\nSynthetic\r\n'
    assert (target / 'empty').is_dir()
    with pytest.raises(ValueError, match='new directory'):
        restore(bundle, target, report['fingerprint'])
    with pytest.raises(ValueError, match='changed'):
        restore(bundle, tmp_path / 'other', '0' * 64)
    assert not (tmp_path / 'other').exists()


@pytest.mark.parametrize('path', ['../escape', '/absolute', 'workspace/../../escape', 'workspace/a\\b', 'workspace/extra'])
def test_traversal_and_unlisted_refused(bundle, path):
    changed = mutate(bundle, lambda rows: rows.update({path: b'bad'}))
    with pytest.raises(ValueError):
        preview(changed)


def test_corrupt_source_refused(bundle):
    with pytest.raises(ValueError, match='checksum'):
        preview(mutate(bundle, lambda rows: rows.update({'workspace/assets/image.bin': b'changed'})))


def test_interrupted_restore_removes_only_new_destination(bundle, tmp_path, monkeypatch):
    import agent.workspace_restore as module
    report = preview(bundle)
    original = tmp_path / 'untouched'
    original.write_bytes(b'keep')
    monkeypatch.setattr(module.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('disk failure')))
    target = tmp_path / 'interrupted'
    with pytest.raises(OSError):
        restore(bundle, target, report['fingerprint'])
    assert not target.exists()
    assert original.read_bytes() == b'keep'


@pytest.mark.parametrize('path', ['workspace/', 'workspace/empty/'])
def test_directory_payload_refused_before_restore(bundle, tmp_path, path):
    changed = mutate(bundle, lambda rows: rows.update({path: b'unchecked payload'}))
    with pytest.raises(ValueError, match='must be empty'):
        preview(changed)
    target = tmp_path / 'refused'
    import hashlib
    with pytest.raises(ValueError):
        restore(changed, target, hashlib.sha256(changed).hexdigest())
    assert not target.exists()


@pytest.mark.parametrize('manifest', [[], None, {'format': 'noevia-workspace-v1', 'files': [None], 'directories': []},
    {'format': 'noevia-workspace-v1', 'files': [], 'directories': [], 'fileCount': False, 'bytes': 0}])
def test_malformed_manifest_is_validation_error(bundle, manifest):
    changed = mutate(bundle, lambda rows: rows.update({'manifest.json': json.dumps(manifest).encode()}))
    with pytest.raises(ValueError):
        preview(changed)


def test_invalid_zip_is_validation_error():
    with pytest.raises(ValueError, match='Invalid workspace archive'):
        preview(b'not a zip')


def test_missing_manifest_is_validation_error(bundle):
    with pytest.raises(ValueError, match='Invalid workspace archive'):
        preview(mutate(bundle, lambda rows: rows.pop('manifest.json')))


def test_directory_mode_without_directory_path_refused(bundle):
    import stat
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(bundle)) as source, zipfile.ZipFile(output, 'w') as dest:
        for info in source.infolist():
            data = source.read(info)
            if info.filename == 'workspace/assets/image.bin':
                info.external_attr = (stat.S_IFDIR | 0o700) << 16
            dest.writestr(info, data)
    with pytest.raises(ValueError, match='type does not match'):
        preview(output.getvalue())
