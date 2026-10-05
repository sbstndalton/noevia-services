"""Retrieval — sqlite-vec backed chunk index over the diary corpus.

Design:
  - Chunk unit: each `### <time/topic>` subsection (prefixed with its day header for date context).
  - Incremental: chunk content hashes are stored; only new/changed chunks are re-embedded.
  - sqlite-vec: single-file DB (shared with the write-ahead journal — one file to back up).
  - Graceful degradation: if the sqlite-vec extension can't load, search returns [] and the
    agent still works with today + standing sections.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from functools import wraps

def synchronized(fn):
    @wraps(fn)
    def call(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return call

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

from . import corpus as fmt
from .llm import LLMClient

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dirty_documents (document TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    xid TEXT,                    -- exchange xid if the chunk came from a logged exchange
    day TEXT NOT NULL,           -- ISO date of the owning day section
    file TEXT NOT NULL,          -- source month file
    header TEXT NOT NULL,        -- '### ' header text (time — topic)
    body TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_day ON chunks(day);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(file, content_hash);

CREATE TABLE IF NOT EXISTS vec_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec

        sqlite_vec.load(conn)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("sqlite-vec unavailable (%s) — retrieval disabled, agent still functional", exc)
        return False


@dataclass
class ReindexPlan:
    """Parsed chunks of one document plus the embeddings computed for them."""
    file_name: str
    rows: List[Tuple[str, str, str, str]]  # (day, header, chunk text, content hash)
    embeddings: dict = field(default_factory=dict)  # content hash -> vector
    pending: bool = False  # some chunk could not be embedded this pass


class Retriever:
    def __init__(self, db_path: Path, llm: LLMClient):
        self._lock = threading.RLock()
        self.pending_embeddings = False
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        # Shares index.db with the journal; wait for its short writes.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.vec_available = _load_vec_extension(self._conn)
        self._dim: Optional[int] = None
        self._ensure_vec_table()

    # ---------------- schema ----------------

    def _ensure_vec_table(self) -> None:
        if not self.vec_available:
            return
        row = self._conn.execute("SELECT value FROM vec_meta WHERE key='dim'").fetchone()
        if row:
            self._dim = int(row["value"])
        else:
            self._dim = None  # set on first indexing

    # ---------------- indexing ----------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _parse_chunks(self, month_text: Optional[str]) -> List[Tuple[str, str, str, str]]:
        rows: List[Tuple[str, str, str, str]] = []  # (day, header, body, hash)
        for d in fmt.parse_diary(month_text or ""):
            if d.date is None:
                continue
            day_iso = d.date.isoformat()
            for sub in d.subsections:
                body = "\n".join(
                    (f"**Me:** {ex.me}\n\n**Assistant:** {ex.claude}" if ex.claude else f"**Me:** {ex.me}")
                    for ex in sub.exchanges
                ).strip()
                if not body:
                    continue
                chunk_text = f"[{day_iso}] {d.header}\n### {sub.header}\n{body}"
                rows.append((day_iso, sub.header, chunk_text, self._hash(chunk_text)))
        return rows

    def _indexed(self, file_name: str) -> dict:
        """{content_hash: (chunk id, has a vector)} for one file. Read-only."""
        rows = self._conn.execute("SELECT id, content_hash FROM chunks WHERE file = ?", (file_name,)).fetchall()
        out = {}
        for row in rows:
            has_vector = bool(
                self.vec_available and self._dim
                and self._conn.execute("SELECT chunk_id FROM vec_items WHERE chunk_id=?", (row["id"],)).fetchone()
            )
            out[row["content_hash"]] = (row["id"], has_vector)
        return out

    def prepare_reindex(self, file_name: str, month_text: Optional[str]) -> "ReindexPlan":
        """Parse one document and embed its new/changed chunks — no write lock.

        Embedding can take minutes against a slow or cold embed server. The
        index database is shared with the write-ahead journal, so nothing here
        opens a write transaction: holding one across embed calls made journal
        writes fail with 'database is locked' (#858). apply_reindex() writes
        the result in one short transaction.
        """
        rows = self._parse_chunks(month_text)
        with self._lock:
            indexed = self._indexed(file_name)
            vec_available, dim = self.vec_available, self._dim
        embeddings = {}
        pending = False
        if vec_available:
            for _, _, chunk_text, h in rows:
                if h in embeddings or (h in indexed and indexed[h][1]):
                    continue
                emb = self._embed_one(chunk_text)
                if emb is not None and dim is None:
                    dim = len(emb)  # the first vector fixes a new table's width
                if emb is None or len(emb) != dim:
                    pending = True
                    log.warning("embedding failed for a chunk in %s — will retry on next reindex", file_name)
                    continue
                embeddings[h] = emb
        return ReindexPlan(file_name, rows, embeddings, pending)

    @synchronized
    def apply_reindex(self, plan: "ReindexPlan") -> int:
        """Write a prepared plan in one short transaction. Returns chunks embedded.

        The index is re-read inside the transaction, so a plan whose state
        moved on (another reindex of the same file landed first) is still
        applied correctly: present rows are kept, missing embeddings stay
        pending for the next reindex.
        """
        file_name = plan.file_name
        keep_hashes = {h for _, _, _, h in plan.rows}
        pending = plan.pending
        embedded = 0
        new_dim = None
        if self._conn.in_transaction:
            self._conn.commit()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            indexed = self._indexed(file_name)
            # Supersede cleanup runs even when nothing new is embedded (rows
            # may be empty — e.g. the file was emptied): any stored row for
            # this file whose content hash is no longer present in the parsed
            # text is a stale version of an edited subsection (same day+header
            # identity, new hash) or removed content. Delete it, embedding
            # included, so retrieval cannot serve the old text next to the
            # corrected one.
            for h, (chunk_id, _) in indexed.items():
                if h not in keep_hashes:
                    if self.vec_available:
                        self._conn.execute("DELETE FROM vec_items WHERE chunk_id = ?", (chunk_id,))
                    self._conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            done = set()
            for day_iso, header, chunk_text, h in plan.rows:
                if h in done:
                    continue
                done.add(h)
                existing = indexed.get(h)
                if existing and (not self.vec_available or existing[1]):
                    continue
                emb = plan.embeddings.get(h) if self.vec_available else None
                if self.vec_available:
                    dim = self._dim or new_dim
                    if emb is None or (dim is not None and len(emb) != dim):
                        # A failed embed inserts no row: the hash-indexed row is
                        # the dedupe record, so inserting it would make this
                        # chunk skip every future reindex. With no row (or a
                        # vector-less one), the next reindex simply retries.
                        pending = True
                        continue
                    if existing:
                        # Recover rows written while vector support was unavailable.
                        self._conn.execute("DELETE FROM chunks WHERE id=?", (existing[0],))
                    if dim is None:
                        new_dim = len(emb)
                        self._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{new_dim}])")
                        self._conn.execute("INSERT OR REPLACE INTO vec_meta (key, value) VALUES ('dim', ?)", (str(new_dim),))
                cur = self._conn.execute(
                    "INSERT INTO chunks (day, file, header, body, content_hash) VALUES (?, ?, ?, ?, ?)",
                    (day_iso, file_name, header, chunk_text, h),
                )
                if emb is not None:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO vec_items (chunk_id, embedding) VALUES (?, ?)",
                        (cur.lastrowid, _serialize_f32(emb)),
                    )
                embedded += 1
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        if new_dim is not None:
            self._dim = new_dim
        self.pending_embeddings = pending
        return embedded

    def reindex_file(self, file_name: str, month_text: Optional[str], day_of_month: Optional[int] = None) -> int:
        """Parse one month file into chunks and upsert embeddings incrementally.

        Callers must pass the full current text of the month file. A chunk is
        identified by (day, header); its row is keyed by content hash. After
        indexing, rows for this file whose hash no longer appears in the
        parsed text (edited or removed subsections) are deleted so a corrected
        entry never stays searchable alongside its stale version.

        Embeds first, then writes in one short transaction (#858). Returns the
        number of chunks embedded this pass; pending_embeddings reports
        whether any chunk still needs a retry.
        """
        return self.apply_reindex(self.prepare_reindex(file_name, month_text))

    def _embed_one(self, text: str) -> Optional[List[float]]:
        try:
            out = self.llm.embed([text])
            return out[0] if out else None
        except Exception as exc:  # noqa: BLE001
            log.warning("embed call failed: %s", exc)
            return None

    # ---------------- search ----------------

    def search(self, query: str, top_k: int = 8, min_score: float = 0.30) -> List[dict]:
        """Cosine-similarity search over past chunks. Returns [{'day','header','text','score'}]."""
        if not self.vec_available or self._dim is None:
            return []
        # Embed outside the lock: apply_reindex runs under the store write
        # lock and must not wait behind a slow query embedding.
        q = self._embed_one(query)
        with self._lock:
            return self._search_vector(q, top_k, min_score)

    def _search_vector(self, q: Optional[List[float]], top_k: int, min_score: float) -> List[dict]:
        if q is None or self._dim is None or len(q) != self._dim:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT c.id, c.day, c.header, c.body,
                       1.0 - vec_distance_cosine(vec_items.embedding, ?) AS score
                FROM vec_items
                JOIN chunks c ON c.id = vec_items.chunk_id
                WHERE c.file NOT IN (SELECT document FROM dirty_documents)
                  AND 1.0 - vec_distance_cosine(vec_items.embedding, ?) >= ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (_serialize_f32(q), _serialize_f32(q), min_score, top_k),
            ).fetchall()
        except sqlite3.Error as exc:
            log.warning("vector search failed: %s", exc)
            return []
        return [
            {"day": r["day"], "header": r["header"], "text": r["body"], "score": round(r["score"], 4)}
            for r in rows
        ]

    @synchronized
    def stats(self) -> dict:
        n = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"chunks": n, "vec_available": self.vec_available, "dim": self._dim}

    @synchronized
    def close(self) -> None:
        self._conn.close()


def _serialize_f32(vec: List[float]) -> bytes:
    import struct

    return struct.pack(f"{len(vec)}f", *vec)
