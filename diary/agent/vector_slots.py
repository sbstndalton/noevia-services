"""Per-model vector slots inside one Diary index file (#1636).

Same layout as noevia-core's vector-slots.cjs, so a re-index builds the new model's vectors while the
old ones keep serving, and a rollback still finds the old ones. Vectors of different slots are
never compared.

  vec_meta 'slot:<id>'  JSON {model, profile, dim, table, map}   id = sha256(profile \\0 model)[:16]
  <table>               vec0(chunk_id INTEGER PRIMARY KEY, embedding float[dim])
  <map>                 (chunk_id INTEGER PRIMARY KEY, content_hash TEXT): the text each vector was
                        made from. A vector counts only while its chunk still has that hash, so a
                        reused chunk id (or an older release editing the file without knowing this
                        slot) never attaches a vector to different text.

The legacy ``vec_items`` table (+ vec_meta 'dim') is what every index built before #1636 holds: raw
text (the nomic-v1 profile). It becomes the slot of the first nomic-v1 identity that opens the file,
and a new nomic-v1 slot in a fresh file uses it too, so an older Diary image rolled back to still
finds its vectors. It has no map; every release keeps it in step (chunks and vectors are deleted
together).
"""
from __future__ import annotations

import json
import re
import sqlite3
import struct
from typing import Dict, Iterable, List, Optional

from .embedding import Target

LEGACY_TABLE = "vec_items"
_TABLE_RE = re.compile(r"^(vec_items|vec_[0-9a-f]{16})$")
_MAP_RE = re.compile(r"^vecmap_[0-9a-f]{16}$")


def serialize_f32(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM vec_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO vec_meta (key, value) VALUES (?, ?)", (key, value))


def _exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def _safe(rec: Optional[dict]) -> Optional[dict]:
    if not rec or not _TABLE_RE.match(str(rec.get("table") or "")):
        return None
    if rec.get("map") is not None and not _MAP_RE.match(str(rec["map"])):
        return None
    dim = rec.get("dim")
    if not (isinstance(dim, int) and 0 < dim <= 65536):
        return None
    return rec


def list_slots(conn: sqlite3.Connection) -> List[dict]:
    out = []
    for key, value in conn.execute("SELECT key, value FROM vec_meta WHERE key LIKE 'slot:%'").fetchall():
        try:
            rec = _safe({"id": key[5:], **json.loads(value)})
        except (ValueError, TypeError):
            rec = None
        if rec:
            out.append(rec)
    return out


def _legacy_claimed(conn: sqlite3.Connection) -> bool:
    return any(s["table"] == LEGACY_TABLE for s in list_slots(conn))


def adopt_legacy(conn: sqlite3.Connection, active: Optional[Target]) -> None:
    """Records the pre-#1636 table as ``active``'s slot when ``active`` uses raw text (nomic-v1)."""
    if not active or active.profile != "nomic-v1":
        return
    try:
        dim = int(_meta(conn, "dim") or 0)
    except ValueError:
        return
    if dim <= 0 or not _exists(conn, LEGACY_TABLE) or _legacy_claimed(conn) or _meta(conn, f"slot:{active.slot_id}"):
        return
    _set_meta(conn, f"slot:{active.slot_id}", json.dumps({"model": active.model, "profile": active.profile, "dim": dim, "table": LEGACY_TABLE, "map": None}))


def find_slot(conn: sqlite3.Connection, target: Optional[Target]) -> Optional[dict]:
    if target is None:
        return None
    raw = _meta(conn, f"slot:{target.slot_id}")
    if not raw:
        return None
    try:
        return _safe({"id": target.slot_id, **json.loads(raw)})
    except (ValueError, TypeError):
        return None


def ensure_slot(conn: sqlite3.Connection, target: Target, dim: int) -> dict:
    """``target``'s slot, created for ``dim``-wide vectors if needed. Call inside a write transaction."""
    found = find_slot(conn, target)
    if found:
        return found
    if not (isinstance(dim, int) and 0 < dim <= 65536):
        raise ValueError(f"bad vector width {dim}")
    sid = target.slot_id
    if target.profile == "nomic-v1" and not _legacy_claimed(conn) and not _exists(conn, LEGACY_TABLE):
        rec = {"model": target.model, "profile": target.profile, "dim": dim, "table": LEGACY_TABLE, "map": None}
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {LEGACY_TABLE} USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{dim}])")
        _set_meta(conn, "dim", str(dim))
    else:
        rec = {"model": target.model, "profile": target.profile, "dim": dim, "table": f"vec_{sid}", "map": f"vecmap_{sid}"}
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {rec['table']} USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{dim}])")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {rec['map']} (chunk_id INTEGER PRIMARY KEY, content_hash TEXT NOT NULL)")
    _set_meta(conn, f"slot:{sid}", json.dumps(rec))
    return {"id": sid, **rec}


def write_vectors(conn: sqlite3.Connection, slot: dict, rows: Iterable[tuple]) -> int:
    """rows: (chunk_id, content_hash, vector). Call inside a write transaction. Returns rows written."""
    n = 0
    for chunk_id, content_hash, vector in rows:
        if not isinstance(vector, (list, tuple)) or len(vector) != slot["dim"]:
            continue
        conn.execute(f"DELETE FROM {slot['table']} WHERE chunk_id = ?", (chunk_id,))
        conn.execute(f"INSERT INTO {slot['table']} (chunk_id, embedding) VALUES (?, ?)", (chunk_id, serialize_f32(list(vector))))
        if slot.get("map"):
            conn.execute(f"INSERT OR REPLACE INTO {slot['map']} (chunk_id, content_hash) VALUES (?, ?)", (chunk_id, content_hash))
        n += 1
    return n


def drop_chunks(conn: sqlite3.Connection, ids: Iterable[int]) -> None:
    """Deletes these chunks' vectors from every slot (and the legacy table). Inside a transaction."""
    ids = list(ids)
    if not ids:
        return
    slots = list_slots(conn)
    if not any(s["table"] == LEGACY_TABLE for s in slots) and _exists(conn, LEGACY_TABLE):
        slots.append({"table": LEGACY_TABLE, "map": None})
    for s in slots:
        if not _exists(conn, s["table"]):
            continue
        has_map = bool(s.get("map")) and _exists(conn, s["map"])
        for chunk_id in ids:
            conn.execute(f"DELETE FROM {s['table']} WHERE chunk_id = ?", (chunk_id,))
            if has_map:
                conn.execute(f"DELETE FROM {s['map']} WHERE chunk_id = ?", (chunk_id,))


def _missing_sql(slot: Optional[dict]) -> str:
    if slot is None:
        return "SELECT c.id FROM chunks c"
    if slot.get("map"):
        return (f"SELECT c.id FROM chunks c LEFT JOIN {slot['map']} m ON m.chunk_id = c.id AND m.content_hash = c.content_hash "
                "WHERE m.chunk_id IS NULL")
    return f"SELECT c.id FROM chunks c WHERE c.id NOT IN (SELECT chunk_id FROM {slot['table']})"


def has_vector(conn: sqlite3.Connection, slot: Optional[dict], chunk_id: int, content_hash: str) -> bool:
    if slot is None:
        return False
    if slot.get("map"):
        return conn.execute(f"SELECT 1 FROM {slot['map']} WHERE chunk_id = ? AND content_hash = ?", (chunk_id, content_hash)).fetchone() is not None
    return conn.execute(f"SELECT 1 FROM {slot['table']} WHERE chunk_id = ?", (chunk_id,)).fetchone() is not None


def missing_chunks(conn: sqlite3.Connection, slot: Optional[dict], limit: int = 4, exclude: Iterable[int] = ()) -> List[tuple]:
    """(id, body, content_hash) of chunks with no current vector in ``slot``, minus ``exclude``."""
    skip = ",".join(str(int(i)) for i in exclude)
    extra = f" AND c.id NOT IN ({skip})" if skip else ""
    return [tuple(r) for r in conn.execute(
        f"SELECT c.id, c.body, c.content_hash FROM chunks c WHERE c.id IN ({_missing_sql(slot)}){extra} ORDER BY c.id LIMIT ?", (limit,)).fetchall()]


def coverage(conn: sqlite3.Connection, slot: Optional[dict]) -> Dict[str, int]:
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    missing = conn.execute(f"SELECT COUNT(*) FROM ({_missing_sql(slot)})").fetchone()[0]
    return {"chunks": chunks, "vectors": chunks - missing}


def join_sql(slot: dict) -> str:
    """FROM clause joining a slot's current vectors (v) to chunks (c)."""
    if slot.get("map"):
        return (f"{slot['table']} v JOIN chunks c ON c.id = v.chunk_id "
                f"JOIN {slot['map']} m ON m.chunk_id = c.id AND m.content_hash = c.content_hash")
    return f"{slot['table']} v JOIN chunks c ON c.id = v.chunk_id"


def drop_slot(conn: sqlite3.Connection, target: Target) -> bool:
    """Removes ``target``'s slot. The legacy table stays (an older image opens it) but is emptied."""
    slot = find_slot(conn, target)
    if not slot:
        return False
    if slot["table"] == LEGACY_TABLE:
        if _exists(conn, LEGACY_TABLE):
            conn.execute(f"DELETE FROM {LEGACY_TABLE}")
    else:
        conn.execute(f"DROP TABLE IF EXISTS {slot['table']}")
        if slot.get("map"):
            conn.execute(f"DROP TABLE IF EXISTS {slot['map']}")
    conn.execute("DELETE FROM vec_meta WHERE key = ?", (f"slot:{slot['id']}",))
    return True
