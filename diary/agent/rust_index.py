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

The Rust process serves this endpoint only with the ``answer`` start-up config, so this mode also
sends it (rust_answer.answer_config). The shadow comparison for indexing is offline:
scripts/shadow_compare.py compares the vector tables of a Python-indexed and a Rust-indexed copy
of the same synthetic tenants.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import httpx

from . import tenant_writer

log = logging.getLogger("diary.rust.index")

MODES = ("", "on")
# The deadline of one re-index call: this, plus per dirty document the model client's worst case
# (llm.timeout_s for each of 1 + llm.max_retries attempts).
REINDEX_BASE_S = 60.0
_warned = set()


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


def reindex_dirty(st, tenant_id: Optional[str], server_lookup: Optional[Callable[[], object]] = None,
                  transport: Optional[httpx.BaseTransport] = None) -> bool:
    """Re-index the tenant's dirty documents in the Rust process. True when it did (its per-document
    failures are logged as Python logs its own); False when Python must do it. The caller holds
    st.store._write_lock (the tenant writer lock and the store lock) and has drained the outbox."""
    uid = (tenant_id or "").lower()
    if not wanted() or not uid or uid == "legacy" or not tenant_writer.is_writer(uid):
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
