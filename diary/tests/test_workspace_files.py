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

def test_local_exchange_uses_memory_only_and_selected_date(monkeypatch):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    strip_marker = appmod.LLMClient.strip_log_marker
    class LLM:
        strip_log_marker = staticmethod(strip_marker)
        def __init__(self, **kwargs): pass
        def chat(self, *args, **kwargs):
            from agent.llm import ChatReply
            return ChatReply('A reply. [LOG: ok]', 'Synthetic provider reasoning')
        def embed(self, *args, **kwargs): return []
        def close(self): pass
    monkeypatch.setattr(appmod, 'LLMClient', LLM)
    monkeypatch.setattr(appmod, 'check_auth', lambda r: True)
    monkeypatch.setattr(appmod.LoggingPipeline, '_aux_call', lambda self, key, *a, **kw: 'LOG' if key == 'skip_classifier' else 'A summary.')
    monkeypatch.setattr(appmod, '_tenant_state', lambda r: pytest.fail('local mode must not resolve remote storage'))
    client = TestClient(appmod.app)
    original = {'MEMORY.md':'I prefer short answers.'}
    r = client.post('/api/local-exchange', headers={'X-Cowork-User-ID':'11111111-1111-4111-8111-111111111111'}, json={
        'files':original, 'message':'A good day outside.', 'entryTime':'2026-09-07T10:05:00-04:00', 'entryDay':'2026-07-08', 'history':[]})
    assert r.status_code == 200, r.text
    assert r.json()['decision'] == 'logged'
    assert r.json()['reasoning'] == 'Synthetic provider reasoning'
    changed = r.json()['files']
    assert all('Synthetic provider reasoning' not in text for text in changed.values())
    assert changed
    assert 'MEMORY.md' not in changed
    assert any('July 8, 2026' in text for text in changed.values())
    assert original == {'MEMORY.md':'I prefer short answers.'}
    assert not any(key.startswith('11111111-1111-4111-8111-111111111111:') for key in appmod.SESSIONS)
