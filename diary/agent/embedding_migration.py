"""Re-embedding the Diary index for a new embedding model, without downtime (#1636 / #1638).

The one path that changes the Diary's embedding model, profile or endpoint:

  set_next(target)  build ``target``'s vectors beside the active ones; search keeps using the active
                    model the whole time.
  status()          active / next / previous and how far each one's vectors cover the index files.
  switch(force)     make next active (one configuration write; every later query and reindex reads
                    it). Refused until next covers every index file unless forced, and, for a move of
                    the same model to another endpoint, until that endpoint gives the same vectors.
                    The old target stays as ``previous`` with its vectors, for rollback; a ``previous``
                    it displaces (one never finished) loses its vectors. The switch is for the ``next``
                    that was checked: if next changed meanwhile, nothing switches (nextChanged).
  rollback()        previous becomes active again; vectors for chunks written since are filled in. A next
                    that was still being built is abandoned and loses its partial vectors.
  finish()          drop previous and its vectors.
  clear_next()      abandon a re-index (its partial vectors are dropped).

The background pass works file by file over the service index and every tenant's ``index.db`` /
``managed-index.db``. It re-embeds the chunk text already stored in each file and writes the vectors
back into that same file: no corpus read, no storage credential, no tenant data crossing files, and
no person reads any entry. It is paced so chat is not starved of the CPU embeddings server: one
batch at a time and, after each, a pause so the pass is busy at most ``duty`` of the wall time
(EMBEDDING_REINDEX_DUTY, default 0.5).
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import vector_slots as slots
from .embedding import EmbeddingConfigError, SharedEmbedding, Target

log = logging.getLogger(__name__)

# Fixed synthetic strings for the same-model endpoint check: never diary content.
PARITY_SAMPLE = [
    "The quick brown fox jumps over the lazy dog.",
    "Quarterly budget review moved to Tuesday afternoon.",
    "function add(a, b) { return a + b; }",
    "Heute regnet es, morgen soll die Sonne scheinen.",
]
PARITY_MIN_COSINE = 0.995


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or len(a) != len(b):
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else float("nan")


def _clamp(value, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return min(hi, max(lo, v))


def index_files(service_db: Path, skip_user: Callable[[str], bool] = lambda _u: False) -> List[Path]:
    """The service index and every tenant's index files, existing ones only (never created here)."""
    out: List[Path] = []
    if service_db.is_file():
        out.append(service_db)
    users = service_db.parent / "users"
    if users.is_dir():
        for tenant in sorted(p for p in users.iterdir() if p.is_dir()):
            if skip_user(tenant.name):
                continue
            for name in ("index.db", "managed-index.db"):
                f = tenant / name
                if f.is_file():
                    out.append(f)
    return out


class EmbeddingMigration:
    def __init__(self, shared: SharedEmbedding, service_db: Path, *, duty=None, batch: int = 4,
                 load_vec: Optional[Callable[[sqlite3.Connection], bool]] = None,
                 skip_user: Callable[[str], bool] = lambda _u: False,
                 sleep: Optional[Callable[[float], None]] = None, now: Callable[[], float] = time.monotonic,
                 retry_s: float = 60.0):
        self.config = shared.config
        self.embedder = shared.embedder
        self.service_db = Path(service_db)
        self.duty = _clamp(duty, 0.5, 0.05, 1.0)
        self.batch = int(_clamp(batch, 4, 1, 16))
        self._load_vec = load_vec
        self._skip_user = skip_user
        self._stop = threading.Event()
        self._sleep = sleep or (lambda s: self._stop.wait(s))
        self._now = now
        self._retry_s = retry_s
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._again = False
        self._retry_timer: Optional[threading.Timer] = None
        self.last_pass: Optional[dict] = None

    # ---------------- files ----------------

    def _files(self) -> List[Path]:
        return index_files(self.service_db, self._skip_user)

    def _open(self, path: Path) -> Optional[sqlite3.Connection]:
        try:
            # mode=rw: a tenant deleted meanwhile must not be recreated by this pass.
            conn = sqlite3.connect(f"file:{path}?mode=rw", uri=True, timeout=30, check_same_thread=False)
        except sqlite3.Error:
            return None
        try:
            if self._load_vec is not None:
                ok = self._load_vec(conn)
            else:
                from .retrieval import _load_vec_extension
                ok = _load_vec_extension(conn)
            if not ok or conn.execute("SELECT 1 FROM sqlite_master WHERE name='chunks'").fetchone() is None \
                    or conn.execute("SELECT 1 FROM sqlite_master WHERE name='vec_meta'").fetchone() is None:
                conn.close()
                return None
        except sqlite3.Error:
            conn.close()
            return None
        return conn

    def _coverage(self, target: Optional[Target]) -> Optional[Dict[str, int]]:
        if target is None:
            return None
        files = self._files()
        active = self.config.active()
        chunks = vectors = unavailable = behind = 0
        for f in files:
            conn = self._open(f)
            if conn is None:
                unavailable += 1
                continue
            try:
                c = slots.coverage(conn, slots.find_slot(conn, target))
                chunks += c["chunks"]
                vectors += c["vectors"]
                # Ready = this file has at least as many vectors for the target as for the active
                # model. Some chunks never get one (a subsection longer than the server's context),
                # under either model; they must not block a switch that loses nothing.
                if target != active:
                    a = slots.coverage(conn, slots.find_slot(conn, active))
                    if c["vectors"] < a["vectors"]:
                        behind += 1
            finally:
                conn.close()
        return {"files": len(files), "chunks": chunks, "vectors": vectors, "unavailable": unavailable,
                "complete": unavailable == 0 and vectors >= chunks, "ready": unavailable == 0 and behind == 0}

    # ---------------- the background pass ----------------

    def _embed_rows(self, target: Target, rows: List[tuple], refused: set) -> List[tuple]:
        """(id, hash, vector) for the rows the server embeds; a refused chunk is skipped this pass."""
        try:
            vectors = self.embedder.embed(target, [r[1] for r in rows], "document")
            return [(cid, h, v) for (cid, _b, h), v in zip(rows, vectors)]
        except Exception:  # noqa: BLE001 — try one by one, so one bad chunk does not block the file
            if len(rows) == 1:
                refused.add(rows[0][0])
                return []
        out = []
        for row in rows:
            out.extend(self._embed_rows(target, [row], refused))
        return out

    def _fill(self, path: Path, target: Target, version: int) -> Dict[str, object]:
        embedded = 0
        refused: set = set()
        while True:
            if self._stop.is_set() or self.config.version != version:
                return {"embedded": embedded, "aborted": True}
            conn = self._open(path)
            if conn is None:
                return {"embedded": embedded, "unavailable": True}
            try:
                rows = slots.missing_chunks(conn, slots.find_slot(conn, target), self.batch, exclude=refused)
                if not rows:
                    return {"embedded": embedded, **({"refused": len(refused)} if refused else {})}
                started = self._now()
                done = self._embed_rows(target, rows, refused)
                if self._stop.is_set() or self.config.version != version:
                    return {"embedded": embedded, "aborted": True}
                n = 0
                if done:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        # The embedding took a while: check again, now that nothing else can write.
                        if self._stop.is_set() or self.config.version != version:
                            conn.rollback()
                            return {"embedded": embedded, "aborted": True}
                        slots.adopt_legacy(conn, self.config.active())
                        slot = slots.ensure_slot(conn, target, len(done[0][2]))
                        # Only while the chunk still holds the text that was embedded.
                        current = [(cid, h, v) for cid, h, v in done
                                   if conn.execute("SELECT 1 FROM chunks WHERE id = ? AND content_hash = ?", (cid, h)).fetchone()]
                        n = slots.write_vectors(conn, slot, current)
                        conn.commit()
                    except BaseException:
                        conn.rollback()
                        raise
            finally:
                conn.close()
            embedded += n
            if n == 0 and not refused:
                return {"embedded": embedded, "stuck": True}
            busy = self._now() - started
            self._sleep(max(0.05, busy * (1 - self.duty) / self.duty))

    def _pass(self) -> dict:
        version = self.config.version
        embedded = failed = refused = 0
        error = None
        targets = [t for t in (self.config.active(), self.config.next()) if t is not None]
        for f in self._files():
            for target in targets:
                try:
                    r = self._fill(f, target, version)
                except Exception as exc:  # noqa: BLE001 — one file or server down does not stop the rest
                    failed += 1
                    error = f"{type(exc).__name__}: {str(exc)[:160]}"
                    continue
                embedded += int(r["embedded"])
                if r.get("stuck") or r.get("unavailable"):
                    failed += 1
                # Chunks the server refuses (e.g. longer than its context) are reported, not retried
                # in a loop: they stay without a vector, as they were before.
                refused += int(r.get("refused") or 0)
                if r.get("aborted"):
                    return {"embedded": embedded, "failed": failed, "refused": refused, "error": error, "aborted": True}
        return {"embedded": embedded, "failed": failed, "refused": refused, "error": error}

    def _run(self) -> None:
        while True:
            with self._lock:
                self._again = False
            try:
                r = self._pass()
            except Exception as exc:  # noqa: BLE001
                r = {"embedded": 0, "failed": 1, "error": f"{type(exc).__name__}"}
            self.last_pass = {"at": time.time(), **r}
            if r["embedded"] or r["failed"]:
                log.info("embedding re-index pass: %d vectors stored, %d file(s) incomplete%s",
                         r["embedded"], r["failed"], f" ({r['error']})" if r.get("error") else "")
            with self._lock:
                if r["failed"] and self.config.next() is not None and not self._stop.is_set() and self._retry_timer is None:
                    self._retry_timer = threading.Timer(self._retry_s, self._retry)
                    self._retry_timer.daemon = True
                    self._retry_timer.start()
                if not self._again or self._stop.is_set():
                    self._thread = None
                    return

    def _retry(self) -> None:
        with self._lock:
            self._retry_timer = None
        self.kick()

    def kick(self) -> None:
        """Starts a background pass, or queues one after the pass under way. Never raises."""
        with self._lock:
            if self._stop.is_set():
                return
            if self._thread is not None:
                self._again = True
                return
            self._thread = threading.Thread(target=self._run, name="embedding-reindex", daemon=True)
            self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._retry_timer is not None:
                self._retry_timer.cancel()
                self._retry_timer = None

    # ---------------- actions ----------------

    def status(self) -> dict:
        c = self.config.get()
        return {**c, "coverage": {"active": self._coverage(self.config.active()), "next": self._coverage(self.config.next())},
                "running": self._thread is not None, "last_pass": self.last_pass, "duty": self.duty, "batch": self.batch}

    def _drop_everywhere(self, target: Target) -> None:
        for f in self._files():
            conn = self._open(f)
            if conn is None:
                continue
            try:
                conn.execute("BEGIN IMMEDIATE")
                slots.drop_slot(conn, target)
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                log.warning("could not drop %s from an index file (%s)", target.model, type(exc).__name__)
            finally:
                conn.close()

    def _drop_if_orphaned(self, target: Optional[Target]) -> None:
        """Drops ``target``'s vectors unless the configuration still holds that model and profile."""
        if target is None:
            return
        held = {t.identity for t in (self.config.active(), self.config.next(), self.config.previous()) if t is not None}
        if target.identity not in held:
            self._drop_everywhere(target)

    def set_next(self, raw) -> dict:
        before = self.config.next()
        self.config.set_next(raw)
        after = self.config.next()
        if before is not None and before.identity != after.identity and before.identity not in (
                self.config.active().identity, (self.config.previous() or Target(None, "", "")).identity):
            self._drop_everywhere(before)
        self.kick()
        return self.status()

    def clear_next(self) -> dict:
        before = self.config.next()
        self.config.clear_next()
        if before is not None and before.identity not in (self.config.active().identity, (self.config.previous() or Target(None, "", "")).identity):
            self._drop_everywhere(before)
        return self.status()

    def switch(self, force: bool = False) -> dict:
        nxt, active = self.config.next(), self.config.active()  # nxt: what the checks below are about
        if nxt is None:
            raise EmbeddingConfigError("No re-index is under way.", "noNext", 409)
        if nxt.identity == active.identity:
            try:
                a = self.embedder.embed(active, PARITY_SAMPLE, "document")
                b = self.embedder.embed(nxt, PARITY_SAMPLE, "document")
                sims = [_cosine(x, y) for x, y in zip(a, b)]
                ok = len(sims) == len(PARITY_SAMPLE) and all(not math.isnan(s) for s in sims) and min(sims) >= PARITY_MIN_COSINE
            except Exception:  # noqa: BLE001
                ok = False
            if not ok and not force:
                raise EmbeddingConfigError("The new embeddings address does not give the same vectors as the current one.", "parityFailed", 409)
        elif not force:
            cov = self._coverage(nxt)
            if not cov or not cov["ready"]:
                err = EmbeddingConfigError("The new model has not finished indexing every Diary file yet.", "notReady", 409)
                err.coverage = cov  # type: ignore[attr-defined]
                raise err
        displaced = self.config.previous()
        self.config.switch_to_next(expected=nxt)
        # An older rollback target is displaced by the new one: its vectors would never be read again.
        self._drop_if_orphaned(displaced)
        log.info("embedding switched to %s/%s%s; previous %s/%s kept for rollback", nxt.profile, nxt.model,
                 " (forced)" if force else "", active.profile, active.model)
        self.kick()
        return self.status()

    def rollback(self) -> dict:
        abandoned = self.config.next()
        self.config.rollback()
        # The re-index that was under way is abandoned by a rollback: no partly built slot stays behind.
        self._drop_if_orphaned(abandoned)
        log.info("embedding rolled back to %s/%s", self.config.active().profile, self.config.active().model)
        self.kick()
        return self.status()

    def finish(self) -> dict:
        old = self.config.previous()
        if old is None:
            raise EmbeddingConfigError("There is no previous embedding model to remove.", "noPrevious", 409)
        self.config.finish()
        nxt = self.config.next()
        if old.identity != self.config.active().identity and (nxt is None or old.identity != nxt.identity):
            self._drop_everywhere(old)
        log.info("embedding removed %s/%s and its vectors", old.profile, old.model)
        return self.status()
