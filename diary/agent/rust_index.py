"""R9 indexing (D9): the re-index of a tenant's dirty documents by the Rust Diary process.

DIARY_RUST_INDEX selects it:
- empty (the default): off. `_reindex_dirty` re-indexes every dirty document here, exactly as before.
- ``on``: for a writer tenant (DIARY_RUST_WRITES), `_reindex_dirty` asks the Rust process
  (noevia-rs bins/noevia-diary reindex.rs, ``POST /internal/reindex``) to do it, while this process
  holds the tenant's writer lock and store lock exactly as its own re-index does, and says so with
  the writer header. The Rust side reads each dirty document, embeds new and changed chunks with
  this process's live embedding configuration, writes them inside the tenant's primary reservation
  and clears the mark unless an embedding is still pending.

Anything but a complete reply (a decline, a refusal, a transport failure) falls back to the Python
loop at once, under the same locks. That is safe because a re-index is idempotent: a document
re-indexed twice from the same text ends in the same index. A forward that failed after it may
have reached the child marks the child lost, as a failed forward of a write does.

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
# One whole re-index: every dirty document's embedding calls (each bounded by the client's own
# timeout and retries) and the index writes.
REINDEX_DEADLINE_S = 300.0
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
    try:
        with httpx.Client(base_url=server.base_url, timeout=REINDEX_DEADLINE_S, trust_env=False,
                          transport=transport) as client:
            resp = client.post("/internal/reindex", params={"tenant": uid}, json={"embedding": config.get()},
                               headers={rust_forward.CONTROL_HEADER: token,
                                        rust_forward.WRITER_HEADER.decode(): "1"})
    except httpx.HTTPError as exc:
        # It may have reached the child: the child is lost (as after a failed write forward), and
        # the idempotent re-index runs here.
        log.warning("Rust re-index failed (%s); re-indexing here", type(exc).__name__)
        if isinstance(exc, httpx.TransportError) and not isinstance(exc, httpx.ConnectError):
            server.lost()
        return False
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
