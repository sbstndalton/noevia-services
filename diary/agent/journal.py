"""Write-ahead journal — the durable outbox that makes diary appends atomic and idempotent.

Flow per exchange:
  1. enqueue()    — the intended log block is durably recorded here BEFORE any remote write.
  2. apply()      — corpus appends it (ETag-guarded WebDAV PUT); on success mark_applied().
  3. recover()    — on startup, unapplied entries are replayed; the corpus dedupes by
                    xid marker scan, so replays can never duplicate content.

The journal lives in the same SQLite file as the vector index (one file to back up).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
import threading
from functools import wraps

def synchronized(fn):
    @wraps(fn)
    def call(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return call

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dirty_documents (document TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS journal (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    kind        TEXT NOT NULL,             -- 'exchange' | 'index_update'
    payload     TEXT NOT NULL,             -- JSON
    applied     INTEGER NOT NULL DEFAULT 0,
    applied_at  TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT
);
CREATE INDEX IF NOT EXISTS idx_journal_unapplied ON journal(applied, created_at);
"""


@dataclass
class JournalEntry:
    id: str
    created_at: str
    kind: str
    payload: dict
    applied: bool
    attempts: int
    last_error: Optional[str]


class Journal:
    def __init__(self, db_path: Path):
        self._lock = threading.RLock()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------- write-ahead ----------------

    @synchronized
    def enqueue(self, kind: str, payload: dict) -> str:
        jid = str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn:
            self._conn.execute(
                "INSERT INTO journal (id, created_at, kind, payload) VALUES (?, ?, ?, ?)",
                (jid, now, kind, json.dumps(payload, ensure_ascii=False)),
            )
        log.info("journal enqueue %s kind=%s", jid, kind)
        return jid

    @synchronized
    def mark_applied(self, jid: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE journal SET applied = 1, applied_at = ?, last_error = NULL WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), jid),
            )

    @synchronized
    def mark_failed(self, jid: str, error: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE journal SET attempts = attempts + 1, last_error = ? WHERE id = ?",
                (error[:500], jid),
            )

    # ---------------- recovery ----------------

    @synchronized
    def unapplied(self, limit: int = 100) -> List[JournalEntry]:
        rows = self._conn.execute(
            "SELECT * FROM journal WHERE applied = 0 ORDER BY created_at, rowid LIMIT ?", (limit,)
        ).fetchall()
        return [
            JournalEntry(
                id=r["id"],
                created_at=r["created_at"],
                kind=r["kind"],
                payload=json.loads(r["payload"]),
                applied=bool(r["applied"]),
                attempts=r["attempts"],
                last_error=r["last_error"],
            )
            for r in rows
        ]

    @synchronized
    def mark_dirty(self, document: str) -> None:
        with self._conn:
            self._conn.execute("INSERT OR IGNORE INTO dirty_documents(document) VALUES (?)", (document,))

    @synchronized
    def dirty_documents(self) -> list:
        return [r[0] for r in self._conn.execute("SELECT document FROM dirty_documents")]

    @synchronized
    def clear_dirty(self, document: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM dirty_documents WHERE document=?", (document,))

    @synchronized
    def is_applied(self, jid: str) -> bool:
        row = self._conn.execute("SELECT applied FROM journal WHERE id=?", (jid,)).fetchone()
        return bool(row and row[0])

    @synchronized
    def pending_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM journal WHERE applied = 0").fetchone()[0])

    @synchronized
    def close(self) -> None:
        self._conn.close()
