"""R9 indexing (D9): the re-index of a tenant's dirty documents by the Rust Diary process.

DIARY_RUST_INDEX selects it:
- empty (the default): off. `_reindex_dirty` re-indexes every dirty document here, exactly as before.
- ``on``: for a writer tenant (DIARY_RUST_WRITES), `_reindex_dirty` asks the Rust process
  (noevia-rs bins/noevia-diary reindex.rs, ``POST /internal/reindex``) to do it, while this process
  holds the tenant's writer lock and store lock exactly as its own re-index does, and says so with
  the writer header. The Rust side reads each dirty document, embeds new and changed chunks with
  this process's live embedding configuration, writes them inside the tenant's primary reservation
  and clears the mark unless an embedding is still pending.

A reply that is not a complete one (a decline, a refusal, a connection that was never made) falls
back to the Python loop at once, under the same locks: a re-index is idempotent, a document
re-indexed twice from the same text ends in the same index. A call that failed after it may have
reached the child (a timeout or a lost reply) marks the child lost, as a failed forward of a write
does, and does NOT run the Python loop now: the lost child may still be writing. The documents stay
dirty and the next pass re-indexes them once the child is reaped (tenant_writer's reaping barrier).
The call's deadline grows with the number of dirty documents, each bounded by the model client's
own timeout and retries, so it ends only where Python's own loop would also have been stuck.
With nothing dirty, no call is made.

- ``shadow``: Python re-indexes as before (any managed tenant), and first asks the Rust process for
  its plan of the same dirty documents (``plan_only``: the chunk rows and the vectors of the new and
  changed chunks it would write; it writes nothing). Each document's plan is compared with the one
  Python then applies (its ``prepare_reindex``, which ``reindex_file`` is), and only the outcome is
  logged and counted: positions and counts, never text. The plan call is made under Python's locks,
  before Python's own re-index, for at most SHADOW_MAX_DOCUMENTS documents (more is skipped), each
  phase bounded by SHADOW_DEADLINE_S (the Rust side stops planning once the call is given up); a decline, a refusal, a failure or
  a late reply is counted and changes nothing. The embeddings of changed chunks are computed twice
  while it is on. A chunk one side failed to embed (a transient embeddings error) is a mismatch: a
  lead to check, not proof of a fault.

The Rust process serves this endpoint only with the ``answer`` start-up config, so these modes also
send it (rust_answer.answer_config). scripts/shadow_compare.py compares the vector tables of a
Python-indexed and a Rust-indexed copy of the same synthetic tenants offline.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Callable, Optional

import httpx

from . import tenant_writer

log = logging.getLogger("diary.rust.index")

MODES = ("", "on", "shadow")
# The deadline of one re-index call: this, plus per dirty document the model client's worst case
# (llm.timeout_s for each of 1 + llm.max_retries attempts).
REINDEX_BASE_S = 60.0
_warned = set()
# The index shadow's plan call (under Python's locks, before its own re-index): each phase of the
# call is bounded by this, and a pass with more dirty documents than SHADOW_MAX_DOCUMENTS (a first
# index, an import) is not compared, so the shadow never holds the locks for long.
SHADOW_DEADLINE_S = 30.0
SHADOW_MAX_DOCUMENTS = 8
# Two vectors of the same text from the same server: equal but for float parsing.
VECTOR_TOLERANCE = 1e-9
SHADOW_OUTCOMES = ("match", "mismatch", "declined", "refused", "failed", "skipped")
_stats_lock = threading.Lock()
SHADOW_STATS = {k: 0 for k in SHADOW_OUTCOMES}


def mode(raw: Optional[str] = None) -> str:
    value = (os.environ.get("DIARY_RUST_INDEX", "") if raw is None else raw).strip().lower()
    if value in MODES:
        return value
    if value not in _warned:
        _warned.add(value)
        log.warning("DIARY_RUST_INDEX: %r is not one of %s; re-indexing stays in Python", value[:32], MODES[1:])
    return ""


def wanted() -> bool:
    return mode() != ""


def shadowing() -> bool:
    return mode() == "shadow"


def reindex_dirty(st, tenant_id: Optional[str], server_lookup: Optional[Callable[[], object]] = None,
                  transport: Optional[httpx.BaseTransport] = None) -> bool:
    """Re-index the tenant's dirty documents in the Rust process. True when it did (its per-document
    failures are logged as Python logs its own); False when Python must do it. The caller holds
    st.store._write_lock (the tenant writer lock and the store lock) and has drained the outbox."""
    uid = (tenant_id or "").lower()
    if mode() != "on" or not uid or uid == "legacy" or not tenant_writer.is_writer(uid):
        return False
    if not tenant_writer.held(uid):
        return False  # the writer header would be untrue: Python re-indexes under its own locks
    from . import rust_forward
    server = (server_lookup or rust_forward.current)()
    if server is None:
        return False
    token, ready = server.control()
    if not ready or not token:
        return False
    config = getattr(getattr(st.retrieval, "embedding", None), "config", None)
    if config is None:
        return False
    dirty = st.journal.dirty_documents()
    if not dirty:
        return True  # what Python's loop would do: nothing
    cfg = getattr(st, "cfg", None)
    per_document = (float(cfg.get("llm.timeout_s", 300)) * (1 + max(0, int(cfg.get("llm.max_retries", 3))))
                    if cfg is not None else 1200.0)
    try:
        with httpx.Client(base_url=server.base_url, timeout=REINDEX_BASE_S + len(dirty) * per_document,
                          trust_env=False, transport=transport) as client:
            resp = client.post("/internal/reindex", params={"tenant": uid}, json={"embedding": config.get()},
                               headers={rust_forward.CONTROL_HEADER: token,
                                        rust_forward.WRITER_HEADER.decode(): "1"})
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        log.warning("Rust re-index not reached (%s); re-indexing here", type(exc).__name__)
        return False
    except httpx.HTTPError as exc:
        # It may have reached the child, which may still be writing: lost (as after a failed write
        # forward); the documents stay dirty for the next pass.
        log.warning("Rust re-index unconfirmed (%s); left for the next pass", type(exc).__name__)
        server.lost()
        return True
    if resp.status_code != 200:
        if not (resp.status_code == 421 and resp.headers.get(rust_forward.DECLINE_HEADER.decode()) == "1"):
            log.info("Rust re-index answered HTTP %d; re-indexing here", resp.status_code)
        return False
    try:
        documents = resp.json()["documents"]
        if not isinstance(documents, list):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        log.warning("Rust re-index reply unreadable; re-indexing here")
        return False
    for doc in documents:
        error = doc.get("error") if isinstance(doc, dict) else "malformed"
        if error:
            log.warning("edit reindex pending: %s", error)
    return True


# ---------------- DIARY_RUST_INDEX=shadow ----------------


def _count(outcome: str, n: int = 1) -> None:
    with _stats_lock:
        SHADOW_STATS[outcome] += n


def shadow_stats() -> dict:
    with _stats_lock:
        return dict(SHADOW_STATS)


def _close(a, b) -> bool:
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y) and abs(x - y) <= VECTOR_TOLERANCE * max(1.0, abs(x), abs(y))


def compare_plan(python_plan, rust) -> Optional[str]:
    """None when the plans agree, else where they first differ: positions and counts, no text."""
    if not isinstance(rust, dict):
        return "the Rust plan is not an object"
    if rust.get("error"):
        return "the Rust side could not plan it"
    py_rows = [[day, header, h] for day, header, _, h in python_plan.rows]
    rows = rust.get("rows")
    if not isinstance(rows, list):
        return "the Rust plan has no rows"
    for i, (a, b) in enumerate(zip(py_rows, rows)):
        if a != b:
            part = next((name for name, x, y in zip(("day", "header", "hash"), a, b if isinstance(b, list) else [None] * 3)
                         if x != y), "row")
            return f"row {i}: {part} differs"
    if len(py_rows) != len(rows):
        return f"row counts differ ({len(py_rows)} vs {len(rows)})"
    if bool(python_plan.pending) != (rust.get("pending") is True):
        return f"pending differs ({bool(python_plan.pending)} vs {rust.get('pending') is True})"
    target = python_plan.target.to_json() if python_plan.target is not None else None
    if target != rust.get("target"):
        return "the embedding targets differ"
    vectors = rust.get("embeddings")
    if not isinstance(vectors, dict):
        return "the Rust plan has no embeddings"
    py_vectors = python_plan.embeddings or {}
    if set(py_vectors) != set(vectors):
        return f"embedded chunks differ ({len(py_vectors)} vs {len(vectors)})"
    for n, h in enumerate(sorted(py_vectors)):
        a, b = list(py_vectors[h]), vectors[h]
        if not isinstance(b, list) or len(a) != len(b):
            return f"embedded chunk {n}: widths differ"
        if not all(_close(x, y) for x, y in zip(a, b)):
            return f"embedded chunk {n}: vectors differ"
    return None


class IndexShadow:
    """The Rust plans of one re-index pass, compared document by document as Python applies its own."""

    def __init__(self, plans: dict):
        self._plans = plans

    def compare(self, document: str, python_plan) -> str:
        try:
            difference = compare_plan(python_plan, self._plans.get(document))
        except Exception as exc:  # noqa: BLE001 - the shadow never fails a re-index; no content
            _count("failed")
            log.info("Diary index shadow: failed (%s)", type(exc).__name__)
            return "failed"
        if difference is None:
            _count("match")
            log.info("Diary index shadow: plan matches (%d chunks)", len(python_plan.rows))
            return "match"
        _count("mismatch")
        log.warning("Diary index shadow: plan differs: %s", difference)
        return "mismatch"


def _skip(reason: str) -> None:
    _count("skipped")
    log.info("Diary index shadow: skipped (%s)", reason)
    return None


def shadow(st, tenant_id: Optional[str], documents: list, server_lookup: Optional[Callable[[], object]] = None,
           transport: Optional[httpx.BaseTransport] = None) -> Optional[IndexShadow]:
    """The Rust plans for this pass's dirty documents, or None (off, nothing dirty, not applicable,
    or the call did not give a complete reply: counted). Never raises. The caller holds
    st.store._write_lock and has drained the outbox."""
    uid = (tenant_id or "").lower()
    if not shadowing() or not documents or not uid or uid == "legacy":
        return None
    try:
        from . import rust_forward
        from .managed_storage import ManagedCorpusBackend
        server = (server_lookup or rust_forward.current)()
        token, ready = server.control() if server is not None else ("", False)
        if not ready or not token:
            return None
        backend = getattr(st.store, "backend", None)
        if not isinstance(backend, ManagedCorpusBackend) or getattr(backend, "tenant", None) != uid:
            return _skip("not the tenant's managed storage")
        if not callable(getattr(st.retrieval, "prepare_reindex", None)):
            return _skip("no plan step")
        config = getattr(getattr(st.retrieval, "embedding", None), "config", None)
        if config is None:
            return _skip("no shared embedding configuration")
        if len(documents) > SHADOW_MAX_DOCUMENTS:
            return _skip(f"{len(documents)} dirty documents")
        body = {"embedding": config.get(), "plan_only": True, "documents": list(documents)}
        with httpx.Client(base_url=server.base_url, timeout=SHADOW_DEADLINE_S, trust_env=False,
                          transport=transport) as client:
            resp = client.post("/internal/reindex", params={"tenant": uid}, json=body,
                               headers={rust_forward.CONTROL_HEADER: token})
    except httpx.HTTPError as exc:
        _count("failed")
        log.info("Diary index shadow: failed (%s)", type(exc).__name__)
        return None
    except Exception as exc:  # noqa: BLE001 - the shadow never fails a re-index; no content
        _count("failed")
        log.info("Diary index shadow: could not start (%s)", type(exc).__name__)
        return None
    if resp.status_code == 421 and resp.headers.get(rust_forward.DECLINE_HEADER.decode()) == "1":
        _count("declined")
        log.info("Diary index shadow: declined")
        return None
    if resp.status_code != 200:
        _count("refused")
        log.info("Diary index shadow: refused (HTTP %d)", resp.status_code)
        return None
    try:
        plans = {d["document"]: d for d in resp.json()["documents"] if isinstance(d, dict)}
    except (ValueError, KeyError, TypeError):
        _count("failed")
        log.info("Diary index shadow: unreadable reply")
        return None
    return IndexShadow(plans)
