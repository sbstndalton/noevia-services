from datetime import date
from types import SimpleNamespace
import threading
import pytest
from fastapi import HTTPException
from agent.workspace_files import MemoryBackend, file_list, file_read, file_write, memory_text, safe_path
from agent.app import entry_target

class Journal:
    def __init__(self): self.dirty = []
    def mark_dirty(self, path): self.dirty.append(path)

def store(files):
    return SimpleNamespace(backend=MemoryBackend(files), _join=lambda p: 'tenant/' + p, _write_lock=threading.RLock(), journal=Journal())

def test_file_guarded_save_and_conflict_preserves_other_writer():
    st = store({'tenant/memory/a.md':'first'})
    row = file_read(st, 'memory/a.md')
    file_write(st, {**row, 'content':'second'})
    with pytest.raises(HTTPException) as exc:
        file_write(st, {**row, 'content':'stale'})
    assert exc.value.status_code == 409
    assert st.backend.files['tenant/memory/a.md'] == 'second'
    assert st.journal.dirty == ['tenant/memory/a.md']

def test_new_file_cannot_overwrite_existing_file():
    st = store({'tenant/MEMORY.md':'keep'})
    with pytest.raises(HTTPException):
        file_write(st, {'path':'MEMORY.md', 'content':'overwrite', 'version':None})
    assert file_read(st, 'MEMORY.md')['content'] == 'keep'

@pytest.mark.parametrize('path', ['../a.md','/a.md','a/../../b.md','a\\b.md','.env','a.txt','a//b.md'])
def test_rejects_unsafe_paths(path):
    with pytest.raises(HTTPException): safe_path(path)

def test_memory_is_read_after_edits_and_scoped():
    st = store({'tenant/MEMORY.md':'original', 'tenant/memory/note.md':'second', 'other/MEMORY.md':'private'})
    assert {f['name'] for f in file_list(st)} == {'MEMORY.md','memory'}
    row = file_read(st,'MEMORY.md')
    file_write(st,{**row,'content':'updated'})
    assert 'updated' in memory_text(st)
    assert 'private' not in memory_text(st)

def test_browser_midnight_and_selected_day():
    now, day = entry_target({'entryTime':'2026-09-07T00:05:00+14:00'})
    assert day == date(2026,9,7)
    assert now.hour == 0
    now, day = entry_target({'entryTime':'2026-09-07T00:05:00-04:00', 'entryDay':'2026-07-08'})
    assert day == date(2026,7,8)
    assert now.hour == 0

@pytest.mark.parametrize('body', [{'entryTime':'bad'}, {'entryTime':'2026-09-07T00:00:00'}, {'entryDay':'2026-02-30'}, {'entryTime':'2026-09-07T12:00:00+00:00','entryDay':'2026-09-08'}])
def test_invalid_dates_rejected(body):
    with pytest.raises(HTTPException): entry_target(body)

def test_local_backend_copies_and_enforces_limits():
    original = {'MEMORY.md':'private'}
    backend = MemoryBackend(original)
    backend.put('MEMORY.md', b'changed', if_match=backend.get_text('MEMORY.md')[1])
    assert original['MEMORY.md'] == 'private'
    assert backend.original == original
    with pytest.raises(HTTPException): MemoryBackend({'big.md':'x'*(512*1024+1)})

@pytest.mark.parametrize("stream", [False, True])
def test_local_exchange_uses_memory_only_and_selected_date(monkeypatch, stream):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    strip_marker = appmod.LLMClient.strip_log_marker
    class LLM:
        strip_log_marker = staticmethod(strip_marker)
        def __init__(self, **kwargs): pass
        def chat(self, *args, **kwargs):
            from agent.llm import ChatReply
            return ChatReply('A reply. [LOG: ok]', 'Synthetic provider reasoning')
        def chat_stream(self, messages, emit, **kwargs):
            reply = self.chat()
            emit({'type':'reasoning','text':reply.reasoning})
            emit({'type':'answer','text':'A reply.'})
            return reply
        def embed(self, *args, **kwargs): return []
        def close(self): pass
    monkeypatch.setattr(appmod, 'LLMClient', LLM)
    monkeypatch.setattr(appmod, 'check_auth', lambda r: True)
    monkeypatch.setattr(appmod.LoggingPipeline, '_aux_call', lambda self, key, *a, **kw: 'LOG' if key == 'skip_classifier' else 'A summary.')
    monkeypatch.setattr(appmod, '_tenant_state', lambda r: pytest.fail('local mode must not resolve remote storage'))
    client = TestClient(appmod.app)
    original = {'MEMORY.md':'I prefer short answers.'}
    r = client.post('/api/local-exchange', headers={'X-Cowork-User-ID':'11111111-1111-4111-8111-111111111111'}, json={
        'stream':stream, 'files':original, 'message':'A good day outside.', 'entryTime':'2026-09-07T10:05:00-04:00', 'entryDay':'2026-07-08', 'history':[]})
    assert r.status_code == 200, r.text
    import json
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: ')] if stream else []
    value = next(e for e in events if e['type']=='diary') if stream else r.json()
    if stream:
        assert [e['type'] for e in events].index('answer') < next(i for i,e in enumerate(events) if e.get('text')=='Saving diary entry…')
        assert events[-1]['type']=='done'
    assert value['decision'] == 'logged'
    assert value['reasoning'] == 'Synthetic provider reasoning'
    changed = value['files']
    assert all('Synthetic provider reasoning' not in text for text in changed.values())
    assert changed
    assert 'MEMORY.md' not in changed
    assert any('July 8, 2026' in text for text in changed.values())
    assert original == {'MEMORY.md':'I prefer short answers.'}
    assert not any(key.startswith('11111111-1111-4111-8111-111111111111:') for key in appmod.SESSIONS)

def test_local_directory_create_is_exclusive_and_tenant_relative(tmp_path):
    from agent.local_storage import LocalCorpusBackend
    from agent.workspace_files import directory_create
    backend = LocalCorpusBackend(str(tmp_path))
    (tmp_path / 'tenant').mkdir()
    st = store({})
    st.backend = backend
    assert directory_create(st, 'Research') == {'path': 'Research', 'isDir': True}
    assert (tmp_path / 'tenant/Research').is_dir()
    for path, status in [('Research', 405), ('', 405), ('missing/child', 409), ('../escape', 400)]:
        with pytest.raises(HTTPException) as exc:
            directory_create(st, path)
        assert exc.value.status_code == status
    assert not (tmp_path / 'tenant/missing').exists()
    assert st.journal.dirty == []
    file_write(st, {'path':'Research/a.md', 'content':'Synthetic text', 'version':None})
    assert file_read(st, 'Research/a.md')['content'] == 'Synthetic text'
    assert not (tmp_path / 'Research').exists()


def test_directory_create_refuses_other_backends_and_outside_symlinks(tmp_path):
    from agent.local_storage import LocalCorpusBackend
    from agent.workspace_files import directory_create
    st = store({})
    with pytest.raises(HTTPException) as exc:
        directory_create(st, 'folder')
    assert exc.value.status_code == 405
    root = tmp_path / 'root'
    outside = tmp_path / 'outside'
    outside.mkdir()
    st.backend = LocalCorpusBackend(str(root))
    (root / 'tenant').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        directory_create(st, 'folder')
    assert not (outside / 'folder').exists()


def test_directory_endpoint_checks_auth_and_uses_tenant_store(monkeypatch, tmp_path):
    import agent.app as appmod
    from agent.local_storage import LocalCorpusBackend
    from fastapi.testclient import TestClient
    st = store({}); st.backend = LocalCorpusBackend(str(tmp_path))
    (tmp_path / 'tenant').mkdir()
    monkeypatch.setattr(appmod, '_tenant_state', lambda request: SimpleNamespace(store=st))
    monkeypatch.setattr(appmod, 'check_auth', lambda request: False)
    client = TestClient(appmod.app)
    assert client.post('/api/directory', json={'path':'folder'}).status_code == 401
    assert not (tmp_path / 'tenant/folder').exists()
    monkeypatch.setattr(appmod, 'check_auth', lambda request: True)
    assert client.post('/api/directory', json={'path':'folder'}).status_code == 200
    assert client.post('/api/directory', json={'path':'folder'}).status_code == 405
