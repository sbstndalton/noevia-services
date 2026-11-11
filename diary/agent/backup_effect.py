"""Read the native backup's durable write-ahead intent. Never infer settlement here.

The remote DAV protocol has no operation-receipt endpoint. A transport failure or
process loss therefore keeps this intent indefinitely, even if later GETs find
matching bytes. Recovery requires externally establishing that the original
remote request cannot execute again; there is deliberately no automatic reset.
"""
from pathlib import Path
import sqlite3
from fastapi import HTTPException

MESSAGE = 'Diary backup settlement is uncertain. Saving and deletion remain paused.'


# SQLITE_READONLY_ROLLBACK: a read-only connection found a hot rollback journal, left by a
# writer that died inside a transaction (a crashed or killed native server), and may not roll it
# back itself.
_READONLY_ROLLBACK = 776


def _read(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone():
        return False  # construction has not exposed a managed tenant to the native path
    return bool(db.execute("SELECT 1 FROM meta WHERE key='backup_effect_pending'").fetchone())


def pending(root):
    path = (Path(root) / 'managed-diary.db').resolve()
    if not path.exists():
        return False
    from . import tenant_writer
    from contextlib import closing
    try:
        with closing(tenant_writer.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1)) as db:
            return _read(db)
    except sqlite3.Error as exc:
        if getattr(exc, 'sqlite_errorcode', None) != _READONLY_ROLLBACK:
            return True
    except OSError:
        return True
    # Recover the dead writer's transaction the way any read-write open does: SQLite takes the
    # exclusive lock and restores the last committed state, so an intent that was never committed
    # is gone and a committed one stays. Without this a crash mid-write would refuse every save
    # as "backup unsettled" until something else happened to open the file read-write. The file
    # must already exist (mode=rw never creates it); only a read is issued.
    try:
        with closing(tenant_writer.connect(path.as_uri() + '?mode=rw', uri=True, timeout=1)) as db:
            return _read(db)
    except (OSError, sqlite3.Error):
        return True


def require_clear(root):
    if pending(root):
        raise HTTPException(503, MESSAGE)
