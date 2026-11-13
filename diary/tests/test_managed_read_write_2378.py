"""#2378: constructing a tenant's storage (every request does it) must not write when the tenant is already set up.

Synthetic tenant only; the database is a temporary file.
"""
import sqlite3
import uuid

import pytest

from agent.managed_storage import ManagedCorpusBackend, StorageUnavailable


@pytest.fixture
def managed(tmp_path):
    backend = ManagedCorpusBackend(tmp_path / 'app', str(uuid.uuid4()), debounce=3)
    backend.activate({}, {'entry_layout': 'daily'})
    return backend


def test_reconstruction_of_an_initialised_tenant_leaves_the_database_untouched(managed):
    before = managed.db_path.read_bytes()
    stat_before = managed.db_path.stat()
    ManagedCorpusBackend(managed.root, managed.tenant)
    ManagedCorpusBackend(managed.root, managed.tenant)
    assert managed.db_path.read_bytes() == before
    assert managed.db_path.stat().st_mtime_ns == stat_before.st_mtime_ns


def test_reconstruction_still_repairs_a_missing_table(managed):
    with sqlite3.connect(managed.db_path) as db:
        db.execute('DROP TABLE backups')
    ManagedCorpusBackend(managed.root, managed.tenant)
    with sqlite3.connect(managed.db_path) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'backups' in names


def test_reconstruction_with_another_identity_still_refuses(managed):
    other = str(uuid.uuid4())
    with pytest.raises(StorageUnavailable):
        ManagedCorpusBackend(managed.root, other)
