"""Retrieval — sqlite-vec backed chunk index over the diary corpus.

Design:
  - Chunk unit: each `### <time/topic>` subsection (prefixed with its day header for date context).
  - Incremental: chunk content hashes are stored; only new/changed chunks are re-embedded.
  - sqlite-vec: single-file DB (shared with the write-ahead journal — one file to back up).
  - Graceful degradation: if the sqlite-vec extension can't load, search returns [] and the
    agent still works with today + standing sections.
  - #1636: vectors live in per-model slots (vector_slots.py). Search and indexing use the active
    embedding target (embedding.py) and its slot only; a re-index fills another model's slot in the
    same file (embedding_migration.py) without touching what search uses.
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
from . import vector_slots as slots
from .embedding import EmbeddingView, Target
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
    target: Optional[Target] = None  # the embedding target the vectors were made with


class _LLMOnly:
    """The embedding of a Retriever built without a shared configuration (tests, tools): the given
    client's own model, raw text (the nomic-v1 profile), exactly as before #1636."""

    def __init__(self, llm):
        self._view = EmbeddingView(config=None, embedder=None, own_llm=llm,
                                   own_target=Target(None, str(getattr(llm, "embed_model", "") or "default"), "nomic-v1"))

    def active(self) -> Target:
        return self._view.own_target

    def embed(self, target, texts, kind="document", titles=None):
        return self._view.embed(target, texts, kind, titles)


class Retriever:
    def __init__(self, db_path: Path, llm: LLMClient, embedding=None):
        self._lock = threading.RLock()
        self.pending_embeddings = False
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        # #1636: active target + embed(target, texts, kind) (embedding.EmbeddingView).
        self.embedding = embedding if embedding is not None else _LLMOnly(llm)
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
        self._active_slot()

    def _active_slot(self) -> Tuple[Target, Optional[dict]]:
        """The active target and its slot (None until it has vectors here). Updates ``_dim``."""
        target = self.embedding.active()
        if not self.vec_available:
            return target, None
        with self._lock:
            if self._conn.in_transaction:
                slots.adopt_legacy(self._conn, target)
            else:
                with self._conn:
                    slots.adopt_legacy(self._conn, target)
            slot = slots.find_slot(self._conn, target)
        self._dim = slot["dim"] if slot else None
        return target, slot

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

    def _indexed(self, file_name: str, target: Optional[Target] = None) -> dict:
        """{content_hash: (chunk id, has a current vector for ``target``)} for one file. Read-only."""
        slot = slots.find_slot(self._conn, target) if (self.vec_available and target is not None) else None
        rows = self._conn.execute("SELECT id, content_hash FROM chunks WHERE file = ?", (file_name,)).fetchall()
        out = {}
        for row in rows:
            has_vector = bool(slot) and slots.has_vector(self._conn, slot, row["id"], row["content_hash"])
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
        target, slot = self._active_slot()
        with self._lock:
            indexed = self._indexed(file_name, target)
            vec_available, dim = self.vec_available, (slot["dim"] if slot else None)
        embeddings = {}
        pending = False
        if vec_available:
            for _, _, chunk_text, h in rows:
                if h in embeddings or (h in indexed and indexed[h][1]):
                    continue
                emb = self._embed_one(chunk_text, target, "document")
                if emb is not None and dim is None:
                    dim = len(emb)  # the first vector fixes a new slot's width
                if emb is None or len(emb) != dim:
                    pending = True
                    log.warning("embedding failed for a chunk in %s — will retry on next reindex", file_name)
                    continue
                embeddings[h] = emb
        return ReindexPlan(file_name, rows, embeddings, pending, target)

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
        # The plan's vectors belong to the target they were made with. If the active target changed
        # meanwhile (a switch or rollback), they still go to their own slot, never to another model's.
        target = plan.target or self.embedding.active()
        if self._conn.in_transaction:
            self._conn.commit()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if self.vec_available:
                slots.adopt_legacy(self._conn, self.embedding.active())
            slot = slots.find_slot(self._conn, target) if self.vec_available else None
            indexed = self._indexed(file_name, target)
            # Supersede cleanup runs even when nothing new is embedded (rows
            # may be empty — e.g. the file was emptied): any stored row for
            # this file whose content hash is no longer present in the parsed
            # text is a stale version of an edited subsection (same day+header
            # identity, new hash) or removed content. Delete it, embeddings
            # included (every model's slot), so retrieval cannot serve the old
            # text next to the corrected one.
            stale = [chunk_id for h, (chunk_id, _) in indexed.items() if h not in keep_hashes]
            if self.vec_available:
                slots.drop_chunks(self._conn, stale)
            for chunk_id in stale:
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
                    dim = slot["dim"] if slot else None
                    if emb is None or (dim is not None and len(emb) != dim):
                        # A failed embed inserts no row: the hash-indexed row is
                        # the dedupe record, so inserting it would make this
                        # chunk skip every future reindex. With no row (or a
                        # vector-less one), the next reindex simply retries.
                        pending = True
                        continue
                    if slot is None:
                        slot = slots.ensure_slot(self._conn, target, len(emb))
                if existing:
                    # A row without a vector for this model (written while vector support was
                    # unavailable, or before this model's re-index reached it): keep the row (other
                    # models' vectors point at it) and add the missing vector.
                    chunk_id = existing[0]
                else:
                    cur = self._conn.execute(
                        "INSERT INTO chunks (day, file, header, body, content_hash) VALUES (?, ?, ?, ?, ?)",
                        (day_iso, file_name, header, chunk_text, h),
                    )
                    chunk_id = cur.lastrowid
                if emb is not None:
                    # A new id may be one SQLite freed earlier: clear any vector another model left on it.
                    if not existing:
                        slots.drop_chunks(self._conn, [chunk_id])
                    slots.write_vectors(self._conn, slot, [(chunk_id, h, emb)])
                embedded += 1
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        if self.vec_available and target == self.embedding.active():
            self._dim = slot["dim"] if slot else self._dim
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

    def _embed_one(self, text: str, target: Optional[Target] = None, kind: str = "document") -> Optional[List[float]]:
        try:
            out = self.embedding.embed(target or self.embedding.active(), [text], kind)
            return out[0] if out else None
        except Exception as exc:  # noqa: BLE001
            log.warning("embed call failed: %s", exc)
            return None

    # ---------------- search ----------------

    def search(self, query: str, top_k: int = 8, min_score: float = 0.30, with_ids: bool = False) -> List[dict]:
        """Cosine-similarity search over past chunks. Returns [{'day','header','text','score'}].

        ``with_ids`` (retrieval tracing only) adds each chunk's ``id``; results are otherwise identical."""
        if not self.vec_available:
            return []
        # #1636: only the active model's slot, with a query embedded by that model. No slot yet
        # (its re-index has not reached this file) means no vector search, never another model's.
        target, slot = self._active_slot()
        if slot is None:
            return []
        # Embed outside the lock: apply_reindex runs under the store write
        # lock and must not wait behind a slow query embedding.
        q = self._embed_one(query, target, "query")
        with self._lock:
            return self._search_vector(q, top_k, min_score, slot, with_ids)

    def _search_vector(self, q: Optional[List[float]], top_k: int, min_score: float, slot: Optional[dict] = None,
                       with_ids: bool = False) -> List[dict]:
        if slot is None:
            slot = self._active_slot()[1]
        if q is None or slot is None or len(q) != slot["dim"]:
            return []
        try:
            rows = self._conn.execute(
                f"""
                SELECT c.id, c.day, c.header, c.body,
                       1.0 - vec_distance_cosine(v.embedding, ?) AS score
                FROM {slots.join_sql(slot)}
                WHERE c.file NOT IN (SELECT document FROM dirty_documents)
                  AND 1.0 - vec_distance_cosine(v.embedding, ?) >= ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (_serialize_f32(q), _serialize_f32(q), min_score, top_k),
            ).fetchall()
        except sqlite3.Error as exc:
            log.warning("vector search failed: %s", exc)
            return []
        if with_ids:
            return [
                {"id": r["id"], "day": r["day"], "header": r["header"], "text": r["body"], "score": round(r["score"], 4)}
                for r in rows
            ]
        return [
            {"day": r["day"], "header": r["header"], "text": r["body"], "score": round(r["score"], 4)}
            for r in rows
        ]

    @synchronized
    def stats(self) -> dict:
        n = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        target = self.embedding.active()
        slot = slots.find_slot(self._conn, target) if self.vec_available else None
        return {"chunks": n, "vec_available": self.vec_available, "dim": slot["dim"] if slot else None,
                "model": target.model, "profile": target.profile}

    @synchronized
    def close(self) -> None:
        self._conn.close()


def _serialize_f32(vec: List[float]) -> bytes:
    return slots.serialize_f32(vec)
