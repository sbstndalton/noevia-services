"""Synthetic opt-in managed import transaction adapter and HTTP checks."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pytest

from agent import workspace_import as imports
from agent import workspace_import_native as native
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from tests.test_dedicated_storage import A, B, volume

REAL_BIN = os.environ.get('WORKSPACE_IMPORT_APPLY_TEST_BIN')


@pytest.fixture(autouse=True)
def default_selection(monkeypatch):
    monkeypatch.delenv('WORKSPACE_IMPORT_APPLY_IMPL', raising=False)
    monkeypatch.setattr(imports, '_warned_invalid_impl', False)


@pytest.fixture
def bundle(tmp_path):
    source = ManagedCorpusBackend(tmp_path / 'source', A)
    source.activate({'raw/note.MD': b'# Synthetic\r\n', '日記/image.bin': b'\x00\xff'},
                    {'entry_layout': 'daily'}, ['empty/nested'])
    target = ManagedCorpusBackend(tmp_path / 'target', B)
    target.activate({'keep.md': b'keep'}, {'entry_layout': 'monthly'})
    return target, archive(source, '', source.settings())


@pytest.fixture
def binary(monkeypatch):
    if not REAL_BIN:
        pytest.skip('WORKSPACE_IMPORT_APPLY_TEST_BIN is required for real native integration')
    path = Path(REAL_BIN).resolve()
    assert path.is_file()
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(path))
    monkeypatch.setenv('WORKSPACE_IMPORT_APPLY_IMPL', 'rust')
    return path


def fake(tmp_path, source):
    path = tmp_path / 'synthetic-native-child'
    path.write_text(f'#!{sys.executable}\n' + source)
    path.chmod(0o755)
    return path


def emit(tmp_path, out, code=0, stderr=b''):
    return fake(tmp_path, 'import sys\nsys.stdin.buffer.read()\n'
                f'sys.stdout.buffer.write({out!r})\nsys.stdout.buffer.flush()\n'
                f'sys.stderr.buffer.write({stderr!r})\nsys.exit({code})\n')


def apply(bundle):
    target, body = bundle
    return imports.apply(target, body, 'Copy', imports.fingerprint(body, 'Imports/Copy'))


def snapshot(target):
    with target.db() as db:
        return {table: [tuple(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY 1')]
                for table in ('files', 'directories', 'meta', 'workspace_imports', 'workspace_index_outbox')}


def test_default_and_invalid_selection_never_spawn(bundle, monkeypatch):
    monkeypatch.setattr(native.subprocess, 'Popen', lambda *a, **k: pytest.fail('native child spawned'))
    assert imports.impl_choice() == 'python'
    monkeypatch.setenv('WORKSPACE_IMPORT_APPLY_IMPL', 'invalid')
    assert imports.impl_choice() == 'python'
    assert not apply(bundle)['alreadyApplied']


def test_real_apply_closes_prepare_context_without_backend_close(bundle, binary, monkeypatch):
    target, body = bundle
    active = 0
    original_db = target.db
    @contextmanager
    def tracked_db():
        nonlocal active
        with original_db() as db:
            active += 1
            try:
                yield db
            finally:
                active -= 1
    monkeypatch.setattr(target, 'db', tracked_db)
    monkeypatch.setattr(target, 'close', lambda: pytest.fail('close is no-op, not quiescence'))
    original_spawn = native.subprocess.Popen
    def spawn(*args, **kwargs):
        assert active == 0, 'prepare SQL context still open'
        assert args[0] == [str(binary), 'apply-v1']
        assert kwargs['env'] == {'PATH': '/usr/local/bin:/usr/bin:/bin'}
        return original_spawn(*args, **kwargs)
    monkeypatch.setattr(native.subprocess, 'Popen', spawn)
    assert not apply(bundle)['alreadyApplied']
    assert target.get('Imports/Copy/日記/image.bin')[0] == b'\x00\xff'
    assert target.settings() == {'entry_layout': 'monthly'}
    _, etag = target.get('Imports/Copy/raw/note.MD')
    target.put('Imports/Copy/raw/note.MD', b'edited', if_match=etag)
    before = snapshot(target)
    assert apply(bundle)['alreadyApplied']
    assert snapshot(target) == before


@pytest.mark.parametrize('fault', ['deleted', 'replacement', 'wrong_expected_tenant'])
def test_real_identity_and_no_create_fail_closed(bundle, binary, tmp_path, monkeypatch, fault):
    target, body = bundle
    if fault == 'replacement':
        other = ManagedCorpusBackend(tmp_path / 'other', A)
        other.activate({}, {})
    original_spawn = native.subprocess.Popen
    def spawn(*args, **kwargs):
        if fault == 'deleted':
            target.db_path.unlink()
        elif fault == 'replacement':
            shutil.copyfile(other.db_path, target.db_path)
        return original_spawn(*args, **kwargs)
    if fault == 'wrong_expected_tenant':
        monkeypatch.setattr(target, 'tenant', A)
    monkeypatch.setattr(native.subprocess, 'Popen', spawn)
    with pytest.raises(native.ImportUnavailable):
        apply(bundle)
    if fault == 'deleted':
        assert not target.db_path.exists()


def test_real_transaction_fault_rolls_back(bundle, binary):
    target, _ = bundle
    with target.db() as db:
        db.execute("CREATE TRIGGER synthetic_fault BEFORE INSERT ON directories BEGIN SELECT RAISE(ABORT,'synthetic fault'); END")
    before = snapshot(target)
    with pytest.raises(native.ImportUncertain):
        apply(bundle)
    assert snapshot(target) == before


def test_real_lost_reply_is_uncertain_and_identical_retry_is_idempotent(bundle, binary, tmp_path, monkeypatch):
    target, _ = bundle
    wrapper = fake(tmp_path, 'import subprocess,sys\n'
                   f'subprocess.run([{str(binary)!r},"apply-v1"],input=sys.stdin.buffer.read(),capture_output=True,check=True)\n'
                   'sys.exit(3)\n')
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(wrapper))
    with pytest.raises(native.ImportUncertain):
        apply(bundle)
    with target.db() as db:
        assert target.meta(db, 'generation') == '2'
        assert db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0] == 1
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(binary))
    assert apply(bundle)['alreadyApplied']


@pytest.mark.parametrize('output', [
    b'{}\n', b'accept deadbeef\n', b'{"version":2}\n',
    b'{"version":1,"destination":"Imports/Copy","fileCount":2,"alreadyApplied":false}\ntrailing',
    b'{"version":1,"version":1,"destination":"Imports/Copy","fileCount":2,"alreadyApplied":false}\n',
    b'{"version":1,"destination":"Imports/Elsewhere","fileCount":2,"alreadyApplied":false}\n',
    b'{"version":1,"destination":"Imports/Copy","fileCount":true,"alreadyApplied":false}\n',
])
def test_untrusted_success_never_falls_back_or_mutates(bundle, tmp_path, monkeypatch, output):
    monkeypatch.setenv('WORKSPACE_IMPORT_APPLY_IMPL', 'rust')
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(emit(tmp_path, output)))
    before = snapshot(bundle[0])
    with pytest.raises(native.ImportUncertain):
        apply(bundle)
    assert snapshot(bundle[0]) == before


@pytest.mark.parametrize(('code', 'output', 'error'), [
    (2, b'reject: conflict\n', ValueError),
    (2, b'reject: frame\n', native.ImportUnavailable),
    (3, b'unavailable: tenant\n', native.ImportUnavailable),
    (3, b'unavailable: database\n', native.ImportUncertain),
    (0, b'reject: conflict\n', native.ImportUncertain),
    (2, b'reject: unknown\n', native.ImportUncertain),
])
def test_exit_and_refusal_mapping(bundle, tmp_path, monkeypatch, code, output, error):
    monkeypatch.setenv('WORKSPACE_IMPORT_APPLY_IMPL', 'rust')
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(emit(tmp_path, output, code)))
    with pytest.raises(error) as raised:
        apply(bundle)
    assert type(raised.value) is error


@pytest.mark.parametrize('kind', ['sleep', 'stdout', 'stderr'])
def test_timeout_overflow_reaps_child_and_keeps_source_private(bundle, tmp_path, monkeypatch, caplog, kind):
    scripts = {'sleep': 'import time\ntime.sleep(30)\n',
               'stdout': 'import os\nwhile True: os.write(1,b"x"*65536)\n',
               'stderr': 'import os\nwhile True: os.write(2,b"synthetic-private"*4096)\n'}
    monkeypatch.setenv('WORKSPACE_IMPORT_APPLY_IMPL', 'rust')
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(fake(tmp_path, scripts[kind])))
    monkeypatch.setattr(native, 'TIMEOUT_S', .3)
    children = []
    original_spawn = native.subprocess.Popen
    def spawn(*args, **kwargs):
        child = original_spawn(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(native.subprocess, 'Popen', spawn)
    before = snapshot(bundle[0])
    started = time.monotonic()
    with pytest.raises(native.ImportUncertain):
        apply(bundle)
    assert time.monotonic() - started < 3
    assert all(child.poll() is not None for child in children)
    assert snapshot(bundle[0]) == before
    assert 'synthetic-private' not in caplog.text and 'note.MD' not in caplog.text


def test_missing_binary_is_unavailable_without_fallback(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv('WORKSPACE_IMPORT_APPLY_IMPL', 'rust')
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(tmp_path / 'missing'))
    with pytest.raises(native.ImportUnavailable):
        apply(bundle)


def test_http_auth_lock_scope_statuses_and_no_drain_on_failure(volume, bundle, binary, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as app
    from agent import managed_storage
    from tests.test_dedicated_storage import request
    client = TestClient(app.app)
    body = bundle[1]
    headers = {'X-Cowork-User-ID': B, 'Content-Type': 'application/zip'}
    base = '/api/workspace-import?name=Copy&action='
    preview = client.post(base+'preview', content=body, headers=headers)
    assert preview.status_code == 200
    reviewed = preview.json()['fingerprint']
    url = base+'apply&fingerprint='+reviewed
    state = app._tenant_state(request(B), recover=False)
    original_spawn = native.subprocess.Popen
    def spawn(*args, **kwargs):
        assert state.store._write_lock._is_owned()
        assert managed_storage._migration_depth.paths[str(state.managed.root)] > 0
        child = original_spawn(*args, **kwargs)
        original_wait = child.wait
        def wait(*args, **kwargs):
            assert state.store._write_lock._is_owned()
            assert managed_storage._migration_depth.paths[str(state.managed.root)] > 0
            return original_wait(*args, **kwargs)
        child.wait = wait
        return child
    monkeypatch.setattr(native.subprocess, 'Popen', spawn)
    original_drain = imports.drain_index_outbox
    monkeypatch.setattr(imports, 'drain_index_outbox', lambda *a: pytest.fail('drain on failed import'))
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(tmp_path / 'missing'))
    unavailable = client.post(url, content=body, headers=headers)
    assert unavailable.status_code == 503
    assert 'indexPending' not in unavailable.json()
    assert client.post(base+'preview', content=b'bad', headers=headers).status_code == 400
    assert client.post(url, content=b'bad', headers=headers).status_code == 409
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(emit(tmp_path, b'reject: conflict\n', 2)))
    assert client.post(url, content=body, headers=headers).status_code == 409
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(emit(tmp_path, b'unavailable: database\n', 3)))
    uncertain = client.post(url, content=body, headers=headers)
    assert uncertain.status_code == 503 and 'uncertain' in uncertain.json()['detail']
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(binary))
    monkeypatch.setattr(imports, 'drain_index_outbox', original_drain)
    result = client.post(url, content=body, headers=headers)
    assert result.status_code == 200, result.text
    assert result.json()['indexPending']
    assert not state.recovered
    assert 'Imports/Copy/raw/note.MD' in state.journal.dirty_documents()
    assert client.post(url, content=body, headers=headers).json()['alreadyApplied']
    monkeypatch.setattr(app, 'check_auth', lambda _: False)
    assert client.post(url, content=body, headers=headers).status_code == 401


def test_http_real_uncertain_commit_and_later_recovery_exactly_once(volume, bundle, binary, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as app
    from tests.test_dedicated_storage import request
    client = TestClient(app.app)
    body = bundle[1]
    headers = {'X-Cowork-User-ID': B, 'Content-Type': 'application/zip'}
    base = '/api/workspace-import?name=Copy&action='
    preview = client.post(base+'preview', content=body, headers=headers).json()
    url = base+'apply&fingerprint='+preview['fingerprint']
    state = app._tenant_state(request(B), recover=False)
    state.recovered = True
    original_reindex = app._reindex_dirty
    original_drain = imports.drain_index_outbox
    monkeypatch.setattr(app, '_reindex_dirty', lambda _: pytest.fail('No model/indexing during import'))
    monkeypatch.setattr(imports, 'drain_index_outbox', lambda *a: pytest.fail('No drain after failed native operation'))
    wrapper = fake(tmp_path, 'import subprocess,sys\n'
                   f'subprocess.run([{str(binary)!r},"apply-v1"],input=sys.stdin.buffer.read(),capture_output=True,check=True)\n'
                   'sys.exit(3)\n')
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(wrapper))
    response = client.post(url, content=body, headers=headers)
    assert response.status_code == 503 and 'uncertain' in response.json()['detail']
    assert not state.recovered and 'indexPending' not in response.json()
    with state.backend.db() as db:
        generation = state.backend.meta(db, 'generation')
        assert db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0] == 1
        assert db.execute('SELECT path FROM workspace_index_outbox').fetchall()[0][0] == 'Imports/Copy/raw/note.MD'
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(binary))
    def journal_temporarily_unavailable(*args):
        raise OSError('synthetic journal transfer failure')
    monkeypatch.setattr(imports, 'drain_index_outbox', journal_temporarily_unavailable)
    retry = client.post(url, content=body, headers=headers)
    assert retry.status_code == 200 and retry.json()['alreadyApplied']
    assert not state.recovered
    with state.backend.db() as db:
        assert state.backend.meta(db, 'generation') == generation
        assert db.execute('SELECT count(*) FROM workspace_index_outbox').fetchone()[0] == 1
    marked, indexed = [], []
    original_mark = state.journal.mark_dirty
    def mark(path):
        marked.append(path)
        return original_mark(path)
    monkeypatch.setattr(state.journal, 'mark_dirty', mark)
    monkeypatch.setattr(state.retrieval, 'reindex_file', lambda path, text: indexed.append(path))
    monkeypatch.setattr(state.retrieval, 'pending_embeddings', False, raising=False)
    monkeypatch.setattr(imports, 'drain_index_outbox', original_drain)
    monkeypatch.setattr(app, '_reindex_dirty', original_reindex)
    app._tenant_state(request(B))
    app._tenant_state(request(B))
    assert marked == indexed == ['Imports/Copy/raw/note.MD']
    with state.backend.db() as db:
        assert db.execute('SELECT count(*) FROM workspace_index_outbox').fetchone()[0] == 0
    assert state.recovered


def test_http_real_native_conflict_is_409(volume, bundle, binary, monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as app
    from tests.test_dedicated_storage import request
    client = TestClient(app.app)
    headers = {'X-Cowork-User-ID': B}
    base = '/api/workspace-import?name=Copy&action='
    body = bundle[1]
    preview = client.post(base+'preview', content=body, headers=headers).json()
    state = app._tenant_state(request(B), recover=False)
    state.backend.put('Imports/Copy/changed-after-preview', b'existing')
    monkeypatch.setattr(imports, 'drain_index_outbox', lambda *a: pytest.fail('drain on native conflict'))
    response = client.post(base+'apply&fingerprint='+preview['fingerprint'], content=body, headers=headers)
    assert response.status_code == 409
    assert 'conflicts' in response.json()['detail']
    assert state.backend.get('Imports/Copy/changed-after-preview')[0] == b'existing'
    assert state.backend.get('Imports/Copy/raw/note.MD')[0] is None
