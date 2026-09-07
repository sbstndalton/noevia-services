"""Diary Companion Agent — FastAPI application.

Single-user web UI (port 8010) for chatting with the companion and observing the
automatic diary-logging pipeline (logged / skipped, with manual re-log).

Auth model:
  - `ui.auth_token` (env DIARY_AUTH_TOKEN) empty  -> open (LAN-only mode).
  - token set -> all /api/* and /v1/* endpoints require `Authorization: Bearer <token>`
    (or `X-Diary-Token`). The `/` page shell and /static are open; the browser stores
    the token in localStorage and sends it on every call.

OpenAI-compatible surface (for Solair AI / any OpenAI-format client):
  GET  /v1/models
  POST /v1/chat/completions   -> the companion answers THROUGH the diary pipeline:
       every exchange is skip-classified and auto-logged server-side, regardless of
       which client sent it (browser UI or phone app).

Run:  uvicorn agent.app:app --host 0.0.0.0 --port 8010
"""
from __future__ import annotations

import logging
import copy
import base64
import hashlib
import json
import os
import re
import shutil
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import corpus as fmt
from .config import load_config
from .context import ContextAssembler
from .corpus_store import CorpusError, CorpusStore
from .external_sources import (
    external_source_paths,
    infer_date,
    resolve_import_target,
    scan_source,
)
from .journal import Journal
from .llm import LLMClient
from .pipeline import LoggingPipeline
from .retrieval import Retriever
from .storage import create_backend

log = logging.getLogger("diary")

# ---------------- state ----------------


class AppState:
    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        self.auth_token = (cfg.get("ui.auth_token") or "").strip()
        self.journal = Journal(Path(cfg.get("retrieval.db_path")))
        self.backend = backend if backend is not None else create_backend(cfg)
        self.llm_main = LLMClient(
            base_url=cfg.get("llm.base_url"),
            api_key=cfg.get("llm.api_key") or "",
            chat_model=cfg.get("llm.chat_model"),
            embed_model=cfg.get("llm.embed_model"),
            timeout_s=float(cfg.get("llm.timeout_s", 300)),
            max_retries=int(cfg.get("llm.max_retries", 3)),
        )
        self.llm_aux = LLMClient(
            base_url=cfg.get("llm.aux.base_url"),
            api_key=cfg.get("llm.aux.api_key") or "",
            chat_model=cfg.get("llm.aux.model"),
            embed_model=cfg.get("llm.embed_model"),
            timeout_s=float(cfg.get("llm.timeout_s", 300)),
            max_retries=int(cfg.get("llm.max_retries", 3)),
        )
        self.store = CorpusStore(cfg, self.backend, self.journal)
        self.retrieval = Retriever(
            Path(cfg.get("retrieval.db_path")),
            self.llm_main,
            embed_batch_size=int(cfg.get("retrieval.embed_batch_size", 8)),
        )
        self.assembler = ContextAssembler(self.store, self.retrieval, cfg)
        templates_path = Path(__file__).resolve().parent.parent / "config" / "prompts" / "logging.md"
        templates = yaml.safe_load(templates_path.read_text(encoding="utf-8"))
        self.pipeline = LoggingPipeline(self.store, self.llm_main, self.llm_aux, templates)


    def __del__(self):
        for name in ("backend", "llm_main", "llm_aux", "retrieval", "journal"):
            try:
                resource = getattr(self, name, None)
                if resource is not None:
                    resource.close()
            except Exception:
                pass


_state: Optional[AppState] = None
_tenant_states: "OrderedDict[str, AppState]" = OrderedDict()
_tenant_lock = threading.Lock()
_TENANT_STATE_CAP = 32
_TENANT_STATE_TTL_S = 24 * 60 * 60
_base_cfg = load_config()


def _set_cfg(cfg, dotted: str, value) -> None:
    node = cfg.as_dict()
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState(_base_cfg)
    return _state


def _snapshot_sqlite(source: Path, target: Path) -> None:
    """Copy a live SQLite database safely.

    A plain shutil.copy2 of a database that may be mid-write can produce a
    corrupt snapshot (torn pages, missing WAL content). The sqlite3 backup
    API takes a consistent, restartable copy instead; an integrity check on
    the result guarantees the tenant migration starts from a valid database.
    """
    src = sqlite3.connect(str(source))
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
        check = sqlite3.connect(str(target))
        try:
            result = check.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            check.close()
        if result != "ok":
            target.unlink(missing_ok=True)
            raise RuntimeError(f"snapshot integrity check failed: {result}")
    finally:
        src.close()


def _tenant_state(request: Request) -> AppState:
    """Resolve the tenant AppState for this request.

    SECURITY: fails CLOSED. A request with a missing, malformed, or non-UUID
    X-Cowork-User-ID is rejected with 400 — it must never fall back to the
    process-wide legacy AppState, which in a partially-migrated deployment is
    a real user's live corpus, not an empty default. The legacy AppState is
    reachable only via the explicit DIARY_LEGACY_USER_ID env mapping (one
    user, deliberately configured by the operator).

    TRUST MODEL (hard requirement): tenant isolation is enforced by network
    topology plus the shared DIARY_AUTH_TOKEN — any holder of that token can
    act as ANY tenant by supplying an arbitrary X-Cowork-User-ID, including
    permanently deleting that tenant's corpus via DELETE /api/internal/tenant.
    This is safe only while the service is reachable exclusively from the
    apps/web server on an internal network, which authenticates its own users
    before proxying. NEVER expose this service directly to a browser or the
    public internet. See services/diary/README.md.
    """
    user_id = request.headers.get("X-Cowork-User-ID", "") or os.environ.get("DIARY_LEGACY_USER_ID", "")
    if not re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", user_id):
        raise HTTPException(status_code=400, detail="missing or invalid X-Cowork-User-ID")
    storage_header = request.headers.get("X-Cowork-Storage", "")
    state_key = f"{user_id}:{hashlib.sha256(storage_header.encode()).hexdigest()[:16]}"
    with _tenant_lock:
        cached = _tenant_states.get(state_key)
        if cached is not None:
            # LRU touch: move to the most-recently-used end.
            _tenant_states.move_to_end(state_key)
            cached.last_used = time.monotonic()
            return cached
        cfg = copy.deepcopy(_base_cfg)
        tenant_root = Path(cfg.get("retrieval.db_path")).parent / "users" / user_id
        tenant_root.mkdir(parents=True, exist_ok=True)
        tenant_db = tenant_root / "index.db"
        legacy_owner = request.headers.get("X-Cowork-Legacy-Owner") == "1" or user_id == os.environ.get("DIARY_LEGACY_USER_ID")
        source_db = Path(cfg.get("retrieval.db_path"))
        if legacy_owner and source_db.exists() and not tenant_db.exists():
            _snapshot_sqlite(source_db, tenant_db)
        _set_cfg(cfg, "retrieval.db_path", str(tenant_db))
        storage = None
        if storage_header:
            try:
                storage = json.loads(base64.urlsafe_b64decode(storage_header + "=" * (-len(storage_header) % 4)))
            except Exception:  # noqa: BLE001
                storage = None
        if storage and storage.get("kind") in ("nextcloud", "webdav"):
            _set_cfg(cfg, "corpus.backend", "webdav")
            _set_cfg(cfg, "corpus.webdav.base_url", storage.get("baseUrl", ""))
            _set_cfg(cfg, "corpus.webdav.username", storage.get("username", ""))
            _set_cfg(cfg, "corpus.webdav.password", storage.get("secret", ""))
            _set_cfg(cfg, "corpus.root", storage.get("corpusRoot", ""))
        elif storage and storage.get("kind") == "s3":
            _set_cfg(cfg, "corpus.backend", "s3")
            _set_cfg(cfg, "corpus.s3.endpoint_url", storage.get("baseUrl", ""))
            _set_cfg(cfg, "corpus.s3.bucket", storage.get("bucket", ""))
            _set_cfg(cfg, "corpus.s3.access_key", storage.get("username", ""))
            _set_cfg(cfg, "corpus.s3.secret_key", storage.get("secret", ""))
            # The user's chosen folder inside the bucket maps to a key prefix;
            # corpus paths stay bare keys under that prefix.
            _set_cfg(cfg, "corpus.s3.prefix", storage.get("corpusRoot", ""))
            _set_cfg(cfg, "corpus.root", "")
            _set_cfg(cfg, "corpus.monthly_prefix", "")
        elif not legacy_owner:
            _set_cfg(cfg, "corpus.backend", "local")
            _set_cfg(cfg, "corpus.local.root", str(tenant_root / "corpus"))
            _set_cfg(cfg, "corpus.root", "")
        state = AppState(cfg)
        state.store.apply_pending()
        _reindex_dirty(state)
        state.last_used = time.monotonic()
        _tenant_states[state_key] = state
        _evict_tenant_states_locked()
        return state


def _close_state(state: AppState) -> None:
    """Release a tenant AppState's heavyweight resources (SQLite connections,
    HTTP clients). Errors are ignored: eviction must never take the service
    down, and a half-closed state is simply rebuilt from scratch if a later
    request needs the same tenant again."""
    try:
        state.backend.close()
        state.llm_main.close()
        state.llm_aux.close()
        state.retrieval.close()
        state.journal.close()
    except Exception:  # noqa: BLE001
        log.exception("failed to close evicted tenant state")


def _evict_tenant_states_locked() -> None:
    """TTL + LRU eviction of cached tenant states. Caller must hold
    _tenant_lock. Without this, every distinct storage-config change mints a
    new heavyweight entry (SQLite connections, HTTP clients) that is never
    released except via the explicit tenant-delete route."""
    now = time.monotonic()
    stale = [k for k, s in _tenant_states.items() if now - getattr(s, "last_used", now) > _TENANT_STATE_TTL_S]
    for key in stale:
        _tenant_states.pop(key)
        log.info("evicted tenant state %s (TTL expired)", key.split(":")[0])
    while len(_tenant_states) > _TENANT_STATE_CAP:
        key, state = _tenant_states.popitem(last=False)  # least recently used
        log.info("evicted tenant state %s (LRU cap)", key.split(":")[0])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    st = get_state()
    applied = st.store.apply_pending()
    log.info("startup: applied %d pending journal entries (%d still pending)", applied, st.journal.pending_count())
    yield
    st.backend.close()
    st.llm_main.close()
    st.llm_aux.close()
    st.retrieval.close()
    st.journal.close()


app = FastAPI(title="Diary Companion", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


# ---------------- auth ----------------


def _client_token(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Diary-Token") or None


def check_auth(request: Request) -> bool:
    """True when the request may proceed. Open mode when no token is configured.

    This validates only the shared service token. It does NOT authenticate the
    specific tenant: tenant identity comes from the X-Cowork-User-ID header
    supplied by the trusted apps/web proxy (see _tenant_state for the full
    trust model and why direct exposure is forbidden).
    """
    st = get_state()
    if not st.auth_token:
        return True
    supplied = _client_token(request)
    return bool(supplied) and secrets.compare_digest(supplied, st.auth_token)


@app.delete("/api/internal/tenant")
def delete_tenant(request: Request) -> JSONResponse:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    user_id = request.headers.get("X-Cowork-User-ID", "")
    if not re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", user_id):
        return JSONResponse({"error": "invalid user"}, status_code=400)
    root = Path(_base_cfg.get("retrieval.db_path")).parent / "users" / user_id
    with _tenant_lock:
        for key, state in list(_tenant_states.items()):
            if key.startswith(f"{user_id}:"):
                _close_state(state)
                _tenant_states.pop(key, None)
        shutil.rmtree(root, ignore_errors=True)
    return JSONResponse({"ok": True})# ---------------- in-memory session (per tenant, bounded) ----------------

SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
_SESSION_CAP = 256
_SESSION_TTL_S = 24 * 60 * 60
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _session(session_id: str, tenant_id: str = "legacy") -> dict:
    """Conversation scratch state, keyed by tenant + client-supplied session id.

    Both key parts are validated/normalized and the map is TTL+LRU bounded:
    SESSIONS used to grow without limit because both parts were unvalidated
    client input and entries were never evicted. Validation also keeps keys
    to one line (a header cannot smuggle ':' collisions or weird bytes).
    """
    sid = str(session_id or "default")
    if not _SESSION_ID_RE.fullmatch(sid):
        sid = "default"
    tid = str(tenant_id or "legacy")
    if tid != "legacy" and not re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", tid):
        raise HTTPException(status_code=400, detail="invalid tenant")
    key = f"{tid}:{sid}"
    now = time.monotonic()
    if key in SESSIONS:
        SESSIONS.move_to_end(key)
        SESSIONS[key]["last_used"] = now
        return SESSIONS[key]
    if len(SESSIONS) >= _SESSION_CAP:
        expired = [k for k, v in SESSIONS.items() if now - v.get("last_used", now) > _SESSION_TTL_S]
        for k in expired:
            SESSIONS.pop(k, None)
    while len(SESSIONS) >= _SESSION_CAP:
        SESSIONS.popitem(last=False)  # least recently used
    entry = {"turns": [], "log_status": [], "last_used": now}
    SESSIONS[key] = entry
    return entry


# ---------------- request/response models ----------------


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class RelogRequest(BaseModel):
    index: int
    session_id: str = "default"


class EditEntryRequest(BaseModel):
    xid: str
    me: str
    assistant: str = ""
    month: Optional[str] = None  # YYYY-MM hint; discovery scans all months without it


# ---------------- pages ----------------


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    template_path = Path(__file__).parent / "static" / "index.html"
    return template_path.read_text(encoding="utf-8")


# ---------------- shared conversation core (UI + /v1 both use this) ----------------


def _run_exchange(st: AppState, message: str, session_id: str, tenant_id: str = "legacy", entry_time=None, entry_day=None, background=True) -> dict:
    """One full exchange: context build -> main model -> strip marker -> log -> return payload."""
    _reindex_dirty(st)
    sess = _session(session_id, tenant_id)
    now = entry_time or datetime.now()
    day = entry_day or now.date()
    messages = st.assembler.build(day, message, session_turns=sess["turns"])
    reply = st.llm_main.chat(messages, temperature=0.7)

    visible, marker = LLMClient.strip_log_marker(reply)
    if marker is None:
        marker = "ok"  # model omitted the marker — default to logging (never lose content)

    outcome = st.pipeline.log_exchange(user_message=message, assistant_message=visible, now=now, day=day)

    sess["turns"].extend([
        {"role": "user", "content": message},
        {"role": "companion", "content": visible},
    ])
    sess["log_status"].append({
        "decision": outcome.decision,
        "xid": outcome.xid,
        "reason": outcome.reason,
        "user": message,
        "assistant": visible,
    })

    del sess["turns"][:-32]
    del sess["log_status"][:-32]
    if background:
        threading.Thread(target=_reindex_today, args=(st, day), daemon=True).start()
    return {"reply": visible, "decision": outcome.decision, "xid": outcome.xid, "reason": outcome.reason}


def _reindex_dirty(st: AppState) -> None:
    # Serialize against writes; never acknowledge a newer invalidation accidentally.
    with st.store._write_lock:
        for document in st.journal.dirty_documents():
            try:
                text, _ = st.store.backend.get_text(document)
                st.retrieval.reindex_file(document, text)
                if not getattr(st.retrieval, "pending_embeddings", False):
                    st.journal.clear_dirty(document)
            except Exception as exc:
                log.warning("edit reindex pending: %s", exc)


def _reindex_today(st: AppState, day) -> None:
    try:
        month_text, _ = st.store.read_month(day)
        st.retrieval.reindex_file(st.store.document_path(day), month_text)
    except Exception as exc:  # noqa: BLE001
        log.warning("background reindex failed: %s", exc)


def run_in_threadpool_sync(fn, *args):
    """Run a blocking callable off the event loop (FastAPI sync def endpoints
    already run in a threadpool, but month listing does WebDAV I/O worth
    keeping off it even from other contexts)."""
    return fn(*args)


# ---------------- API (browser UI) ----------------


@app.post("/api/chat")
def api_chat(req: ChatRequest, request: Request) -> JSONResponse:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    message = req.message.strip()
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    st = _tenant_state(request)  # fail-closed: outside the try so an identity error is not masked as a model error
    try:
        result = _run_exchange(st, message, req.session_id, request.headers.get("X-Cowork-User-ID", "legacy"))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("chat failed")
        return JSONResponse({"error": f"model error: {exc}"}, status_code=502)
    return JSONResponse(result)


@app.post("/api/relog")
def api_relog(req: RelogRequest, request: Request) -> JSONResponse:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    st = _tenant_state(request)
    sess = _session(req.session_id, request.headers.get("X-Cowork-User-ID", "legacy"))
    try:
        item = sess["log_status"][req.index]
    except IndexError:
        return JSONResponse({"error": "no such exchange"}, status_code=404)
    outcome = st.pipeline.relog_last(
        user_message=item["user"],
        assistant_message=item["assistant"],
        now=datetime.now(),
    )
    item["decision"] = outcome.decision
    item["xid"] = outcome.xid
    return JSONResponse({"decision": outcome.decision, "xid": outcome.xid, "reason": outcome.reason})


@app.get("/api/day")
def api_day(request: Request, month: Optional[str] = None) -> JSONResponse:
    """Today's log (no params — unchanged behavior for existing callers), or a
    whole month's display text when ?month=YYYY-MM is given."""
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    st = _tenant_state(request)
    if month:
        m = re.fullmatch(r"(\d{4})-(\d{2})", month or "")
        if not m:
            return JSONResponse({"error": "month must be YYYY-MM"}, status_code=400)
        year, mon = int(m.group(1)), int(m.group(2))
        if not 1 <= mon <= 12:
            return JSONResponse({"error": "month out of range"}, status_code=400)
        return JSONResponse({
            "month": month,
            "log": st.store.read_month_text(year, mon, include_xids=True),
            "standing": "",
        })
    day = datetime.now().date()
    return JSONResponse({
        "day": day.isoformat(),
        "today_log": st.store.get_day_text(day, max_chars=12000, include_markers=True),
        "standing": st.store.get_standing_sections_text(max_chars=4000),
    })


@app.get("/api/months")
def api_months(request: Request) -> JSONResponse:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    st = _tenant_state(request)
    months = run_in_threadpool_sync(st.store.list_months)
    return JSONResponse({"months": months})


@app.post("/api/entries/edit")
def api_entries_edit(req: EditEntryRequest, request: Request) -> JSONResponse:
    """Edit one logged exchange, matched by its hidden xid marker.

    This is the ONLY supported way to correct past diary text from the app, and
    it exists to keep the diary's integrity guarantee: editing is an explicit,
    human-initiated action on the user's own words — never a silent assistant
    rewrite. Routed through the same write-ahead journal and ETag-guarded write
    as every other corpus mutation (no separate, unguarded write path), and the
    retrieval index is refreshed so the corrected text supersedes the old chunk.
    """
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    xid = (req.xid or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F-]{8,64}", xid):
        return JSONResponse({"error": "invalid xid"}, status_code=400)
    if not req.me.strip():
        return JSONResponse({"error": "me is required"}, status_code=400)
    st = _tenant_state(request)
    try:
        path, day_iso = run_in_threadpool_sync(st.store.edit_exchange, xid, req.me, req.assistant, req.month)
    except CorpusError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503 if "queued" in str(exc) else 404)
    except Exception as exc:  # noqa: BLE001 — persistent write conflict etc.
        log.exception("entry edit failed")
        return JSONResponse({"error": f"edit failed: {exc}"}, status_code=502)
    _reindex_dirty(st)
    return JSONResponse({"ok": True, "document": path, "day": day_iso})


@app.get("/api/external-sources")
def api_external_sources(request: Request) -> JSONResponse:
    """Detect diary-like files in operator-configured external folders.

    Read-only: reports filename, size, and best-guess date per file, plus a
    total count, so the UI can say "we found N entries in other sources".
    Never mutates anything in the source folders.
    """
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    st = _tenant_state(request)  # tenant resolution first: consistent auth surface
    paths = external_source_paths()
    if not paths:
        return JSONResponse({"configured": False, "sources": [], "total": 0})
    results = []
    total = 0
    for p in paths:
        scan = run_in_threadpool_sync(scan_source, p)
        total += scan["total"]
        results.append(scan)
    return JSONResponse({"configured": True, "sources": results, "total": total})


class ImportRequest(BaseModel):
    source_path: str
    rel_path: str


@app.post("/api/external-sources/import")
def api_external_sources_import(body: ImportRequest, request: Request) -> JSONResponse:
    """Import ONE external source file into the corpus, explicitly.

    The file's full text becomes the entry body (verbatim), dated by the same
    inference the scan reported. Runs through the journal-durable store
    applier exactly like a chat-logged exchange — but skips the AI classifier
    and summarizer: the user explicitly asked to import this file, so its
    text is kept as-is. The original file is only ever read.
    """
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    st = _tenant_state(request)

    # The named source must be one of the configured ones — no path smuggling.
    configured = external_source_paths()
    if body.source_path not in configured:
        raise HTTPException(status_code=400, detail="source path is not configured")
    target = resolve_import_target(body.source_path, body.rel_path)
    if target is None:
        raise HTTPException(status_code=404, detail="file not found in source")

    try:
        with target.open("rb") as source:
            raw = source.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Source file exceeds 2 MiB")
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"file unreadable: {exc}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="file is empty")

    file_date, _method = infer_date(target)
    day = file_date or datetime.now().date()
    try:
        xid = st.store.log_exchange(
            day=day,
            sub_header=f"Imported: {target.name}",
            me_text=text,
            claude_text="",
            now=datetime.now().replace(hour=12, minute=0, second=0, microsecond=0),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"import failed: {exc}")

    threading.Thread(
        target=_reindex_today,
        args=(st, day),
        daemon=True,
    ).start()
    return JSONResponse({
        "imported": True,
        "xid": xid,
        "day": day.isoformat(),
        "date_source": _method if file_date else "today",
    })


@app.get("/api/health")
def api_health(request: Request) -> JSONResponse:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    # Health answers for the *process*: the Docker/compose healthchecks probe
    # without an X-Cowork-User-ID header, and a liveness probe must not depend
    # on tenant resolution. Tenant detail is added only when a valid header is
    # supplied (the web server's proxy calls always carry one).
    user_id = request.headers.get("X-Cowork-User-ID", "") or os.environ.get("DIARY_LEGACY_USER_ID", "")
    legacy_user = os.environ.get("DIARY_LEGACY_USER_ID", "")
    # Same two resolution paths _tenant_state supports: an explicit proxy
    # header (user + storage), or the operator-set legacy direct-client map.
    has_tenant = bool(re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", user_id)) and (
        bool(request.headers.get("X-Cowork-Storage", "")) or (bool(legacy_user) and user_id == legacy_user)
    )
    payload: dict = {
        "ok": True,
        "model": _base_cfg.get("llm.chat_model"),
        "auth_required": bool(get_state().auth_token),
    }
    if has_tenant:
        try:
            st = _tenant_state(request)
            payload.update({
                "journal_pending": st.journal.pending_count(),
                "retrieval": st.retrieval.stats(),
            })
        except HTTPException:
            pass  # tenant detail unavailable; probe-level health is still ok
    return JSONResponse(payload)


# ---------------- OpenAI-compatible surface (Solair AI & friends) ----------------


@app.get("/v1/models")
def v1_models(request: Request) -> JSONResponse:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    st = _tenant_state(request)
    model_id = "diary-companion"
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "diary-companion"},
            {"id": st.cfg.get("llm.chat_model") or model_id, "object": "model", "created": 0, "owned_by": "cowork"},
        ],
    })


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request) -> JSONResponse:
    """OpenAI-format endpoint backed by the diary companion.

    The LAST user message becomes the diary exchange; earlier messages in the client's
    thread are passed as session context so short back-and-forth stays coherent. The
    exchange is skip-classified and auto-logged exactly like a UI chat — the client
    cannot bypass the diary pipeline.
    """
    if not check_auth(request):
        return JSONResponse({"error": {"message": "unauthorized", "type": "auth_error"}}, status_code=401)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": {"message": "invalid JSON body", "type": "invalid_request_error"}}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": "invalid request body", "type": "invalid_request_error"}}, status_code=400)
    messages_in = body.get("messages") or []
    user_msgs = [m.get("content", "") for m in messages_in if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs or not str(user_msgs[-1]).strip():
        return JSONResponse({"error": {"message": "no user message", "type": "invalid_request_error"}}, status_code=400)
    last = str(user_msgs[-1]).strip()
    prior = [{"role": m["role"], "content": m["content"]} for m in messages_in[:-1]
             if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)][-16:]

    session_id = str(body.get("session_id") or "openai-client")
    tenant_id = request.headers.get("X-Cowork-User-ID", "legacy")
    st = _tenant_state(request)
    sess = _session(session_id, tenant_id)
    if prior and not sess["turns"]:
        sess["turns"].extend(prior)  # seed short client-thread context

    try:
        # _run_exchange does blocking inference I/O — keep the event loop free.
        entry_time, entry_day = entry_target(body)
        result = await run_in_threadpool(_run_exchange, st, last, session_id, tenant_id, entry_time, entry_day)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("v1 exchange failed")
        return JSONResponse(
            {"error": {"message": f"model error: {exc}", "type": "api_error"}}, status_code=502
        )

    now = int(time.time())
    return JSONResponse({
        "id": f"chatcmpl-{result['xid'] or 'local'}",
        "object": "chat.completion",
        "created": now,
        "model": "diary-companion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["reply"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "diary": {"decision": result["decision"], "xid": result["xid"]},
    })


# ---------------- entrypoint ----------------


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host=os.environ.get("DIARY_HOST", "0.0.0.0"), port=int(os.environ.get("DIARY_PORT", "8010")))


# Diary workspace routes share the same tenant resolution as the conversation.
from .workspace_files import file_list, file_read, file_write, MemoryBackend, reference_text


def entry_target(body):
    """Use the browser's explicit offset, never the container timezone."""
    try:
        raw = body.get("entryTime")
        now = datetime.fromisoformat(raw) if raw else datetime.now().astimezone()
        if raw and now.tzinfo is None:
            raise ValueError("timezone offset required")
        selected = body.get("entryDay")
        day = date.fromisoformat(selected) if selected else now.date()
        if not 1900 <= day.year <= 2100 or day > now.date():
            raise ValueError("choose today or a past date")
        return now, day
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid diary date/time; include a timezone offset")


@app.get("/api/files")
def workspace_files(request: Request, path: str = ""):
    if not check_auth(request):
        raise HTTPException(401, "unauthorized")
    return {"files": file_list(_tenant_state(request).store, path)}


@app.post("/api/file")
def workspace_file(body: dict, request: Request):
    if not check_auth(request):
        raise HTTPException(401, "unauthorized")
    return file_read(_tenant_state(request).store, body.get("path", ""))


@app.put("/api/file")
def workspace_file_save(body: dict, request: Request):
    if not check_auth(request):
        raise HTTPException(401, "unauthorized")
    st = _tenant_state(request)
    result = file_write(st.store, body)
    _reindex_dirty(st)
    return result


@app.post("/api/local-exchange")
def local_exchange(body: dict, request: Request):
    if not check_auth(request):
        raise HTTPException(401, "unauthorized")
    # Resolve/authenticate the owner but never read/write their online corpus.
    tenant_id = request.headers.get("X-Cowork-User-ID", "")
    if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", tenant_id):
        raise HTTPException(400, "invalid tenant")
    backend = MemoryBackend(body.get("files"))
    message = body.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 32000:
        raise HTTPException(400, "message required (maximum 32000 characters)")
    now, day = entry_target(body)
    cfg = copy.deepcopy(_base_cfg)
    _set_cfg(cfg, "retrieval.db_path", ":memory:")
    _set_cfg(cfg, "corpus.root", "")
    _set_cfg(cfg, "corpus.webdav.remote_root", "")
    _set_cfg(cfg, "corpus.monthly_prefix", "")
    _set_cfg(cfg, "corpus.index_enabled", False)
    st = AppState(cfg, backend=backend)
    sid = secrets.token_hex(16)
    try:
        sess = _session(sid, tenant_id)
        sess["turns"] = [{"role": m["role"], "content": m["content"][:32000]}
                         for m in body.get("history", [])[-16:]
                         if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
        # Local context is bounded and selected without durable indexing.
        st.assembler.local_reference = reference_text(backend.files, message)
        result = _run_exchange(st, message, sid, tenant_id, now, day, False)
        if result["decision"] == "error":
            raise HTTPException(503, result["reason"])
        result["files"] = {p: t for p, t in backend.files.items() if backend.original.get(p) != t}
        return result
    finally:
        _close_state(st)
        SESSIONS.pop(f"{tenant_id}:{sid}", None)
