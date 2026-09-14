import sqlite3
import pytest
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from agent.workspace_import import prepare, apply, drain_index_outbox
from tests.test_dedicated_storage import A, B, volume


@pytest.fixture
def setup(tmp_path):
    source = ManagedCorpusBackend(tmp_path / 'source', A)
    source.activate({'raw/note.md': b'original\r\n', 'image.bin': b'\x00\xff'}, {'entry_layout': 'daily'}, ['empty'])
    target = ManagedCorpusBackend(tmp_path / 'target', B)
    target.activate({'keep.md': b'original\r\n'}, {'entry_layout': 'monthly'})
    return target, archive(source, '', source.settings())


def test_preview_apply_and_uncertain_retry(setup):
    target, body = setup
    files, dirs, report = prepare(target, body, 'Copy')
    assert report['duplicates'] == ['raw/note.md']
    assert report['conflicts'] == []
    assert target.get('Imports/Copy/raw/note.md')[0] is None
    result = apply(target, body, 'Copy', report['fingerprint'])
    assert not result['alreadyApplied']
    assert target.get('Imports/Copy/image.bin')[0] == b'\x00\xff'
    assert target.get('Imports/Copy/raw/note.md')[0] == b'original\r\n'
    assert target.settings() == {'entry_layout': 'monthly'}
    assert 'empty' in [r['name'] for r in target.list_dir('Imports/Copy')]
    assert target.get('keep.md')[0] == b'original\r\n'
    with target.db() as db:
        assert target.meta(db, 'generation') == '2'
        assert [r[0] for r in db.execute('SELECT path FROM workspace_index_outbox')] == ['Imports/Copy/raw/note.md']
    _, etag = target.get('Imports/Copy/raw/note.md')
    target.put('Imports/Copy/raw/note.md', b'edited', if_match=etag)
    assert apply(target, body, 'Copy', report['fingerprint'])['alreadyApplied']
    assert target.get('Imports/Copy/raw/note.md')[0] == b'edited'


@pytest.mark.parametrize('collision', ['Imports', 'Imports/Copy', 'Imports/Copy/child.md'])
def test_conflict_after_preview_is_refused(setup, collision):
    target, body = setup
    report = prepare(target, body, 'Copy')[2]
    target.put(collision, b'existing')
    assert collision in prepare(target, body, 'Copy')[2]['conflicts']
    with pytest.raises(ValueError, match='conflicts'):
        apply(target, body, 'Copy', report['fingerprint'])
    assert target.get(collision)[0] == b'existing'
    assert target.get('Imports/Copy/image.bin')[0] is None


def test_changed_destination_and_archive_refused(setup):
    target, body = setup
    reviewed = prepare(target, body, 'Copy')[2]['fingerprint']
    with pytest.raises(ValueError, match='changed'):
        apply(target, body, 'Other', reviewed)
    with pytest.raises(ValueError, match='changed'):
        apply(target, body + b'changed', 'Copy', reviewed)


def test_mid_transaction_failure_rolls_back_everything(setup):
    target, body = setup
    reviewed = prepare(target, body, 'Copy')[2]['fingerprint']
    with target.db() as db:
        db.execute("CREATE TRIGGER synthetic_failure BEFORE INSERT ON directories BEGIN SELECT RAISE(ABORT, 'synthetic disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        apply(target, body, 'Copy', reviewed)
    with target.db() as db:
        assert target.meta(db, 'generation') == '1'
        assert db.execute('SELECT count(*) FROM files').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM workspace_index_outbox').fetchone()[0] == 0


def test_outbox_survives_journal_failure_and_restart(setup):
    target, body = setup
    apply(target, body, 'Copy', prepare(target, body, 'Copy')[2]['fingerprint'])
    class Broken:
        def mark_dirty(self, path):
            raise OSError('journal unavailable')
    with pytest.raises(OSError):
        drain_index_outbox(target, Broken())
    reopened = ManagedCorpusBackend(target.root, B)
    class Journal:
        def __init__(self):
            self.paths = []
        def mark_dirty(self, path):
            self.paths.append(path)
    journal = Journal()
    drain_index_outbox(reopened, journal)
    assert journal.paths == ['Imports/Copy/raw/note.md']
    drain_index_outbox(reopened, journal)
    assert len(journal.paths) == 1


@pytest.mark.parametrize('name', ['', '../escape', '/absolute', 'a/b', 'a\\b', '.hidden', 'x' * 101])
def test_invalid_destination(setup, name):
    with pytest.raises(ValueError):
        prepare(*setup, name)


def test_http_preview_apply_scope_pending_and_no_inference(volume, monkeypatch, setup):
    from fastapi.testclient import TestClient
    import agent.app as appmod
    from tests.test_dedicated_storage import request
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: pytest.fail('Import must not call embeddings'))
    client = TestClient(appmod.app)
    body = setup[1]
    headers = {'X-Cowork-User-ID': B, 'Content-Type': 'application/zip'}
    url = '/api/workspace-import?name=Synthetic&action='
    response = client.post(url + 'preview', content=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    report = response.json()
    state = appmod._tenant_state(request(B), recover=False)
    monkeypatch.setattr(state.journal, 'pending_count', lambda: 1)
    assert client.post(url + 'apply&fingerprint=' + report['fingerprint'], content=body, headers=headers).status_code == 409
    monkeypatch.setattr(state.journal, 'pending_count', lambda: 0)
    result = client.post(url + 'apply&fingerprint=' + report['fingerprint'], content=body, headers=headers)
    assert result.status_code == 200, result.text
    assert result.json()['indexPending']
    assert 'Imports/Synthetic/raw/note.md' in state.journal.dirty_documents()
    assert not state.recovered
    assert client.post(url + 'apply&fingerprint=' + report['fingerprint'], content=body, headers=headers).json()['alreadyApplied']
    legacy = client.post(url + 'preview', content=body, headers={**headers, 'X-Cowork-User-ID': A})
    assert legacy.status_code == 409
    assert client.post(url + 'invalid', content=body, headers=headers).status_code == 400
    monkeypatch.setattr(appmod, 'check_auth', lambda r: False)
    assert client.post(url + 'preview', content=body, headers=headers).status_code == 401
