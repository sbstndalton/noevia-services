import concurrent.futures
import json
import time
import uuid

import pytest
from agent.managed_storage import ManagedCorpusBackend
from agent.local_storage import LocalCorpusBackend
from agent.diary_migration import snapshot, LegacyWriteGuard


@pytest.fixture
def managed(tmp_path):
    backend = ManagedCorpusBackend(tmp_path / 'app', str(uuid.uuid4()), debounce=3)
    backend.activate({}, {'entry_layout': 'daily'})
    return backend


@pytest.fixture
def storage():
    return {'kind': 'webdav', 'baseUrl': 'https://synthetic.invalid/', 'username': 'fixture', 'corpusRoot': 'Diary'}


def test_transactional_save_survives_restart_and_debounces(managed, tmp_path, storage):
    remote = LocalCorpusBackend(str(tmp_path / 'remote'))
    assert managed.put('Entries/day.md', b'one')[0]
    restored = ManagedCorpusBackend(managed.root, managed.tenant)
    assert restored.get('Entries/day.md')[0] == b'one'
    assert restored.backup(remote, storage)['backup'] == 'pending'
    assert remote.list_dir('') == []
    assert restored.backup(remote, storage, now=time.time() + 10)['backup'] == 'complete'
    assert restored.status(storage)['lastBackedUp']
    manifest_dir = f'Diary/noevia-backups/{managed.tenant}/manifests'
    rows = remote.list_dir(manifest_dir)
    manifest = json.loads(remote.get(rows[0]['path'])[0])
    assert manifest['files'][0]['path'] == 'Entries/day.md'
    assert remote.get(manifest['files'][0]['object'])[0] == b'one'


def test_atomic_cas_across_instances(managed):
    managed.put('a.md', b'base')
    other = ManagedCorpusBackend(managed.root, managed.tenant)
    _, version = managed.get('a.md')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda b: b.put('a.md', str(id(b)).encode(), if_match=version), [managed, other]))
    assert sorted(row[0] for row in results) == [False, True]


def test_failed_backup_does_not_block_save_and_retries_after_restart(managed, tmp_path, storage):
    class Offline:
        def get(self, path):
            raise OSError('secret URL must not be persisted')
    managed.put('a.md', b'one')
    result = managed.backup(Offline(), storage, now=time.time() + 10)
    assert result['backup'] == 'failed'
    assert 'secret URL' not in result['error']
    assert managed.put('b.md', b'two')[0]
    restored = ManagedCorpusBackend(managed.root, managed.tenant)
    assert restored.status(storage)['backup'] == 'failed'
    assert restored.backup(LocalCorpusBackend(str(tmp_path / 'remote')), storage, now=time.time() + 100)['backup'] == 'complete'


def test_concurrent_edit_during_backup_stays_pending(managed, tmp_path, storage):
    remote = LocalCorpusBackend(str(tmp_path / 'remote'))
    managed.put('a.md', b'one')
    original = remote.put
    def put(path, data, **kwargs):
        if '/objects/' in path:
            _, v = managed.get('a.md')
            managed.put('a.md', b'two', if_match=v)
        return original(path, data, **kwargs)
    remote.put = put
    assert managed.backup(remote, storage, now=time.time()+10)['backup'] == 'pending'
    assert managed.get('a.md')[0] == b'two'


def test_remote_conflict_preserves_remote_and_local(managed, tmp_path, storage):
    remote = LocalCorpusBackend(str(tmp_path / 'remote'))
    managed.put('a.md', b'one')
    assert managed.backup(remote, storage, now=time.time()+10)['backup'] == 'complete'
    objects = list((tmp_path / 'remote').rglob('a.md'))
    objects[0].write_bytes(b'external edit')
    managed.put('b.md', b'two')
    assert managed.backup(remote, storage, now=time.time()+20)['backup'] == 'failed'
    assert objects[0].read_bytes() == b'external edit'
    assert managed.get('a.md')[0] == b'one'


def test_destination_change_requires_new_backup(managed, tmp_path, storage):
    remote = LocalCorpusBackend(str(tmp_path / 'remote'))
    managed.put('a.md', b'one')
    assert managed.backup(remote, storage, now=time.time()+10)['backup'] == 'complete'
    changed = dict(storage, corpusRoot='New')
    assert managed.status(changed)['backup'] == 'pending'
    assert managed.backup(remote, changed, now=time.time()+10)['backup'] == 'complete'
    assert remote.list_dir('Diary') and remote.list_dir('New')


def test_verified_import_preserves_binary_paths_and_blocks_old_writes(tmp_path):
    local = LocalCorpusBackend(str(tmp_path / 'old'))
    local.put('Diary/Entries/a.md', b'# hello')
    local.put('Diary/Raw Sources/source.bin', b'\x00\xff')
    files, preview = snapshot(local, 'Diary')
    assert preview['fileCount'] == 2
    managed = ManagedCorpusBackend(tmp_path / 'app', str(uuid.uuid4()))
    guarded = LegacyWriteGuard(local, managed)
    with managed.migration_lock():
        verified, second = snapshot(guarded, 'Diary')
        assert second['fingerprint'] == preview['fingerprint']
        managed.activate(verified, {})
    assert managed.get('Raw Sources/source.bin')[0] == b'\x00\xff'
    assert local.get('Diary/Entries/a.md')[0] == b'# hello'
    with pytest.raises(RuntimeError):
        guarded.put('Diary/new.md', b'must fail')
    with pytest.raises(ValueError):
        managed.activate(files, {})


def test_empty_or_changing_source_cannot_be_mistaken_for_verified_import(tmp_path):
    remote = LocalCorpusBackend(str(tmp_path / 'remote'))
    with pytest.raises(ValueError, match='No source files'):
        snapshot(remote, 'Diary')
    remote.put('Diary/a.md', b'one')
    _, before = snapshot(remote, 'Diary')
    _, version = remote.get('Diary/a.md')
    remote.put('Diary/a.md', b'two', if_match=version)
    _, after = snapshot(remote, 'Diary')
    assert before['fingerprint'] != after['fingerprint']


def test_tenant_isolation_and_path_guards(managed, tmp_path):
    other = ManagedCorpusBackend(tmp_path / 'other', str(uuid.uuid4()))
    other.activate({}, {})
    managed.put('a.md', b'private')
    assert other.get('a.md') == (None, None)
    for path in ('../a.md', '/a.md', 'x/../a.md', 'x\\a.md', ''):
        with pytest.raises(ValueError):
            managed.put(path, b'x')


def test_backup_restores_exact_markdown_and_binary_files(managed, tmp_path, storage):
    from agent.backup_restore import restore
    remote_root = tmp_path / 'remote'
    managed.put('Entries/a.md', b'# portable\n')
    managed.put('Raw Sources/a.bin', b'\x00\xff')
    managed.create_directory('Empty')
    managed.backup(LocalCorpusBackend(str(remote_root)), storage, now=time.time()+10)
    manifest = next(remote_root.rglob('*.json'))
    out = tmp_path / 'restored'
    report = restore(remote_root, manifest, out)
    assert report['files'] == 2
    assert (out / 'Entries/a.md').read_bytes() == b'# portable\n'
    assert (out / 'Raw Sources/a.bin').read_bytes() == b'\x00\xff'
    assert (out / 'Empty').is_dir()
    with pytest.raises(ValueError, match='new directory'):
        restore(remote_root, manifest, out)
    next(remote_root.rglob('a.md')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        restore(remote_root, manifest, tmp_path / 'corrupt-restore')
    assert not (tmp_path / 'corrupt-restore').exists()


def test_missing_managed_database_fails_closed(managed):
    from agent.dedicated_storage import StorageUnavailable
    managed.db_path.unlink()
    with pytest.raises(StorageUnavailable):
        ManagedCorpusBackend(managed.root, managed.tenant)
    with pytest.raises(Exception):
        managed.get('a.md')
    assert not managed.db_path.exists()


def _process_guarded_write(root, tenant, remote_root, ready, result):
    backend = ManagedCorpusBackend(root, tenant)
    guarded = LegacyWriteGuard(LocalCorpusBackend(remote_root), backend)
    ready.set()
    try:
        with guarded.write_transaction():
            guarded.put('late.md', b'never written')
        result.put('wrote')
    except RuntimeError:
        result.put('blocked')


def test_cutover_blocks_queued_writer_in_another_process(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    managed = ManagedCorpusBackend(tmp_path / 'app', str(uuid.uuid4()))
    ready, result = ctx.Event(), ctx.Queue()
    with managed.migration_lock():
        process = ctx.Process(target=_process_guarded_write, args=(managed.root, managed.tenant, str(tmp_path / 'old'), ready, result))
        process.start()
        assert ready.wait(15)
        managed.activate({'existing.md': b'original'}, {})
    process.join(15)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail('Writer deadlocked during cutover')
    assert process.exitcode == 0
    assert result.get(timeout=2) == 'blocked'
    assert not (tmp_path / 'old/late.md').exists()


def test_import_preserves_empty_folders_and_fingerprints_them(tmp_path):
    remote = LocalCorpusBackend(str(tmp_path / 'remote'))
    remote.put('Diary/a.md', b'original')
    _, before = snapshot(remote, 'Diary')
    remote.create_directory('Diary/Empty')
    files, after = snapshot(remote, 'Diary')
    assert before['fingerprint'] != after['fingerprint']
    managed = ManagedCorpusBackend(tmp_path / 'app', str(uuid.uuid4()))
    managed.activate(files, {}, after['directories'])
    assert any(row['name'] == 'Empty' and row['is_dir'] for row in managed.list_dir(''))


@pytest.mark.parametrize('kind', ['local', 'webdav', 's3'])
def test_import_reader_bounds_data_before_loading_whole_file(tmp_path, kind):
    import httpx
    if kind == 'local':
        backend = LocalCorpusBackend(str(tmp_path / 'files'))
        backend.put('large.md', b'x' * 100)
    else:
        if kind == 'webdav':
            from agent.webdav import WebDAVCorpusBackend
            backend = WebDAVCorpusBackend('http://synthetic.invalid', '', '')
        else:
            from agent.s3_storage import S3CorpusBackend
            backend = S3CorpusBackend('http://synthetic.invalid', 'bucket')
        backend._client.close()
        backend._client = httpx.Client(base_url='http://synthetic.invalid', transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b'x' * 100)))
    try:
        with pytest.raises(ValueError, match='safety limit'):
            backend.get_bounded('large.md', 10)
        assert backend.get_bounded('large.md', 100)[0] == b'x' * 100
    finally:
        backend.close()


def test_backup_rejects_escaping_destination_before_remote_io(managed, tmp_path, storage):
    remote = LocalCorpusBackend(str(tmp_path / 'remote'))
    managed.put('a.md', b'private')
    assert managed.backup(remote, dict(storage, corpusRoot='../escape'), now=time.time()+10)['backup'] == 'failed'
    assert remote.list_dir('') == []
