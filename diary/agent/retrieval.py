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
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

from . import corpus as fmt
from .llm import LLMClient

log = logging.getLogger(__name__)

_SCHEMA = """
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


class Retriever:
    def __init__(self, db_path: Path, llm: LLMClient, embed_batch_size: int = 8):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.embed_batch_size = embed_batch_size
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
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

    def _create_vec_table(self, dim: int) -> None:
        self._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{dim}])")
        self._conn.execute("INSERT OR REPLACE INTO vec_meta (key, value) VALUES ('dim', ?)", (str(dim),))
        self._conn.commit()
        self._dim = dim

    # ---------------- indexing ----------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def reindex_file(self, file_name: str, month_text: Optional[str], day_of_month: Optional[int] = None) -> int:
        """Parse one month file into chunks and upsert embeddings incrementally.

        Returns the number of chunks (re)embedded.
        """
        if not month_text:
            return 0
        days = fmt.parse_diary(month_text)
        rows: List[Tuple[str, str, str, str]] = []  # (day, header, body, hash)
        for d in days:
            if d.date is None:
                continue
            day_iso = d.date.isoformat()
            for sub in d.subsections:
                body = "\n".join(
                    (f"**Me:** {ex.me}\n\n**Claude:** {ex.claude}" if ex.claude else f"**Me:** {ex.me}")
                    for ex in sub.exchanges
                ).strip()
                if not body:
                    continue
                chunk_text = f"[{day_iso}] {d.header}\n### {sub.header}\n{body}"
                rows.append((day_iso, sub.header, chunk_text, self._hash(chunk_text)))
        if not rows:
            return 0

        embedded = 0
        for day_iso, header, chunk_text, h in rows:
            existing = self._conn.execute(
                "SELECT id FROM chunks WHERE file = ? AND content_hash = ?", (file_name, h)
            ).fetchone()
            if existing:
                continue  # unchanged chunk — skip re-embedding
            cur = self._conn.execute(
                "INSERT INTO chunks (day, file, header, body, content_hash) VALUES (?, ?, ?, ?, ?)",
                (day_iso, file_name, header, chunk_text, h),
            )
            chunk_id = cur.lastrowid
            if self.vec_available:
                emb = self._embed_one(chunk_text)
                if emb is not None:
                    if self._dim is None:
                        self._create_vec_table(len(emb))
                    if len(emb) == self._dim:
                        self._conn.execute(
                            "INSERT OR REPLACE INTO vec_items (chunk_id, embedding) VALUES (?, ?)",
                            (chunk_id, _serialize_f32(emb)),
                        )
                    else:
                        log.warning("embedding dim mismatch (%s != %s) — chunk skipped", len(emb), self._dim)
                else:
                    log.warning("embedding failed for chunk %s — will retry on next reindex", chunk_id)
            embedded += 1
        self._conn.commit()
        return embedded

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
        q = self._embed_one(query)
        if q is None or len(q) != self._dim:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT c.id, c.day, c.header, c.body,
                       1.0 - vec_distance_cosine(vec_items.embedding, ?) AS score
                FROM vec_items
                JOIN chunks c ON c.id = vec_items.chunk_id
                WHERE 1.0 - vec_distance_cosine(vec_items.embedding, ?) >= ?
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

    def stats(self) -> dict:
        n = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"chunks": n, "vec_available": self.vec_available, "dim": self._dim}

    def close(self) -> None:
        self._conn.close()


def _serialize_f32(vec: List[float]) -> bytes:
    import struct

    return struct.pack(f"{len(vec)}f", *vec)
