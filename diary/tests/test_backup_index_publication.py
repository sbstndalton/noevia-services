"""Index publication borrows/retains primary SQLite coverage through its commit."""
import sqlite3
import pytest
from agent import tenant_writer
from agent.journal import Journal
from agent.managed_storage import ManagedCorpusBackend

UID = '12345678-1234-4234-8234-123456789abc'


@pytest.fixture
def primary(tmp_path, monkeypatch):
    tenant_writer.configure_data_root(tmp_path)
    monkeypatch.setattr(tenant_writer, '_writers', frozenset())
    root = tmp_path / 'users' / UID
    backend = ManagedCorpusBackend(root, UID)
    backend.activate({}, {})
    journal = Journal(root / 'managed-index.db')
    yield backend, journal
    journal._conn.close()
    tenant_writer.configure_data_root(None)


def test_index_writer_checks_late_intent(primary):
    backend, journal = primary
    with backend.db() as db:
        db.execute("INSERT INTO meta VALUES ('backup_effect_pending','synthetic-late')")
    with pytest.raises(tenant_writer.WriterBusy):
        journal.mark_dirty('must-not-commit.md')
    assert journal.dirty_documents() == []


def test_outbox_transaction_is_borrowed_until_index_publication_settles(primary):
    backend, journal = primary
    with backend.db() as db:
        db.execute('BEGIN IMMEDIATE')
        journal.mark_dirty('synthetic.md')
        with sqlite3.connect(backend.db_path, timeout=.01) as contender:
            with pytest.raises(sqlite3.OperationalError):
                contender.execute("INSERT INTO meta VALUES ('backup_effect_pending','next-effect')")
    assert journal.dirty_documents() == ['synthetic.md']
    with backend.db() as db:
        db.execute("INSERT INTO meta VALUES ('backup_effect_pending','next-effect')")
