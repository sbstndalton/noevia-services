"""First-entry initialization uses synthetic corpus files and durable replay."""
from datetime import date
import pytest
from agent.config import Config
from agent.corpus_store import CorpusStore
from agent.journal import Journal
from agent.workspace_files import MemoryBackend, memory_text

DAY = date(2026, 9, 10)
SEEDS = ['Entries/README.md', 'AI Memory/README.md', 'Raw Sources/README.md']

def store(tmp_path, backend, layout='daily'):
    return CorpusStore(Config({'corpus': {'root': 'tenant', 'entry_layout': layout, 'index_enabled': False}}),
                       backend, Journal(tmp_path/'journal.db'))

@pytest.mark.parametrize('layout', ['daily', 'monthly'])
def test_first_entry_creates_only_missing_seeds_and_preserves_user_files(tmp_path, layout):
    backend = MemoryBackend({'tenant/AI Memory/README.md': 'My own instructions', 'other/README.md': 'Other tenant'})
    st = store(tmp_path, backend, layout)
    assert backend.files == backend.original  # constructing/reading never scaffolds
    st.log_exchange(DAY, 'Synthetic', 'Synthetic user', 'Synthetic reply')
    assert all('tenant/'+p in backend.files for p in SEEDS)
    assert backend.files['tenant/AI Memory/README.md'] == 'My own instructions'
    assert backend.files['other/README.md'] == 'Other tenant'
    assert 'My own instructions' in memory_text(st)
    del backend.files['tenant/Raw Sources/README.md']
    st.log_exchange(DAY, 'Second', 'Another synthetic entry', 'Reply')
    assert 'tenant/Raw Sources/README.md' not in backend.files
    assert st.journal.pending_count() == 0
    st.journal.close()

def test_partial_scaffold_replays_after_restart_without_duplicate_exchange(tmp_path):
    class Interrupted(MemoryBackend):
        fail = True
        def put(self, path, data, **kwargs):
            if self.fail and path.endswith('AI Memory/README.md'):
                raise OSError('Synthetic interruption')
            return super().put(path, data, **kwargs)
    backend = Interrupted({})
    st = store(tmp_path, backend)
    xid = st.log_exchange(DAY, 'Synthetic', 'ONE RAW MESSAGE', 'Reply')
    assert st.journal.pending_count() == 1
    assert 'tenant/Entries/README.md' in backend.files
    st.journal.close()
    backend.fail = False
    st = store(tmp_path, backend)
    assert st.apply_pending() == 1
    assert all('tenant/'+p in backend.files for p in SEEDS)
    assert backend.files[st.document_path(DAY)].count('ONE RAW MESSAGE') == 1
    assert xid in backend.files[st.document_path(DAY)]
    st.journal.close()

def test_existing_corpus_without_journal_is_not_migrated(tmp_path):
    backend = MemoryBackend({'tenant/2026-08.md': 'Existing imported diary'})
    st = store(tmp_path, backend, 'monthly')
    st.log_exchange(DAY, 'Synthetic', 'New message', 'Reply')
    assert all('tenant/'+p not in backend.files for p in SEEDS)
    st.journal.close()

def test_concurrent_readme_creation_is_preserved_on_etag_retry(tmp_path):
    class Concurrent(MemoryBackend):
        def put(self, path, data, **kwargs):
            if path.endswith('Raw Sources/README.md') and path not in self.files:
                self.files[path] = 'Concurrent user content'
            return super().put(path, data, **kwargs)
    backend = Concurrent({})
    st = store(tmp_path, backend)
    st.log_exchange(DAY, 'Synthetic', 'Message', 'Reply')
    assert backend.files['tenant/Raw Sources/README.md'] == 'Concurrent user content'
    assert st.journal.pending_count() == 0
    st.journal.close()
