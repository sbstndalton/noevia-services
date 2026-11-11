"""A writer killed inside a transaction leaves a hot rollback journal on the tenant's primary.

The backup-effect check must recover it (as any read-write open does) instead of reading the
read-only refusal as an unsettled backup and pausing every save until something else opens the
file read-write. Synthetic tenant and data only.
"""
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from agent import backup_effect

UID = '12345678-1234-4234-8234-123456789abc'

# The child takes the write lock, records an intent and then enough rows to spill pages to the
# database file (a one-page cache), and is killed before COMMIT: the journal stays hot.
_KILLED_WRITER = textwrap.dedent('''
    import os, signal, sqlite3, sys
    db = sqlite3.connect(sys.argv[1], isolation_level=None)
    db.execute('PRAGMA cache_size=1')
    db.execute('BEGIN IMMEDIATE')
    if sys.argv[2] == 'intent':
        db.execute("INSERT INTO meta VALUES ('backup_effect_pending','synthetic-uncommitted')")
    for _ in range(64):
        db.execute('INSERT INTO files VALUES (?, ?)', ('synthetic.md', os.urandom(4096)))
    os.kill(os.getpid(), signal.SIGKILL)
''')


def _tenant(tmp_path, committed_intent):
    root = tmp_path / 'users' / UID
    root.mkdir(parents=True)
    path = root / 'managed-diary.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)')
        db.execute('CREATE TABLE files(path TEXT, body BLOB)')
        if committed_intent:
            db.execute("INSERT INTO meta VALUES ('backup_effect_pending','synthetic-committed')")
    db.close()
    return root, path


def _kill_writer_mid_transaction(path, uncommitted_intent):
    child = subprocess.run([sys.executable, '-c', _KILLED_WRITER, str(path),
                            'intent' if uncommitted_intent else 'rows'], timeout=30)
    assert child.returncode == -signal.SIGKILL
    journal = path.with_name(path.name + '-journal')
    assert journal.exists() and journal.stat().st_size > 0
    # Witness: the journal is hot, and a read-only connection is refused, not answered.
    with pytest.raises(sqlite3.OperationalError) as refused:
        ro = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1)
        try:
            ro.execute('SELECT 1 FROM meta').fetchall()
        finally:
            ro.close()
    assert refused.value.sqlite_errorcode == 776  # SQLITE_READONLY_ROLLBACK
    return journal


@pytest.mark.parametrize('committed_intent,uncommitted_intent,expected', [
    (False, False, False),  # a crashed corpus write never means an unsettled backup
    (False, True, False),   # an intent the dead writer never committed never began its effect
    (True, False, True),    # a committed intent survives recovery and still pauses writes
])
def test_hot_journal_is_recovered_not_read_as_unsettled(tmp_path, committed_intent,
                                                        uncommitted_intent, expected):
    root, path = _tenant(tmp_path, committed_intent)
    journal = _kill_writer_mid_transaction(path, uncommitted_intent)
    assert backup_effect.pending(root) is expected
    assert not journal.exists()
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM files').fetchone()[0] == 0
        assert (db.execute("SELECT value FROM meta WHERE key='backup_effect_pending'").fetchone()
                == (('synthetic-committed',) if committed_intent else None))
    db.close()
    # Settled for good: a second check answers the same from the recovered file.
    assert backup_effect.pending(root) is expected


def test_other_unreadable_primary_still_fails_closed(tmp_path):
    root = tmp_path / 'users' / UID
    root.mkdir(parents=True)
    (root / 'managed-diary.db').write_bytes(b'synthetic bytes that are not a database' * 64)
    assert backup_effect.pending(root) is True
