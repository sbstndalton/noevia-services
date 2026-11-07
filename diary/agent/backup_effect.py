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


def pending(root):
    path = (Path(root) / 'managed-diary.db').resolve()
    if not path.exists():
        return False
    try:
        from . import tenant_writer
        with tenant_writer.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone():
                return False  # construction has not exposed a managed tenant to the native path
            return bool(db.execute("SELECT 1 FROM meta WHERE key='backup_effect_pending'").fetchone())
    except (OSError, sqlite3.Error):
        return True


def require_clear(root):
    if pending(root):
        raise HTTPException(503, MESSAGE)
