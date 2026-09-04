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
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config
from .context import ContextAssembler
from .corpus_store import CorpusStore
from .journal import Journal
from .llm import LLMClient
from .pipeline import LoggingPipeline
from .retrieval import Retriever
from .storage import create_backend

log = logging.getLogger("diary")

# ---------------- state ----------------


class AppState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.auth_token = (cfg.get("ui.auth_token") or "").strip()
        self.journal = Journal(Path(cfg.get("retrieval.db_path")))
        self.backend = create_backend(cfg)
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


_state: Optional[AppState] = None
_tenant_states: Dict[str, AppState] = {}
_tenant_lock = threading.Lock()
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


def _tenant_state(request: Request) -> AppState:
    user_id = request.headers.get("X-Cowork-User-ID", "") or os.environ.get("DIARY_LEGACY_USER_ID", "")
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", user_id):
        return get_state()
    storage_header = request.headers.get("X-Cowork-Storage", "")
    state_key = f"{user_id}:{hashlib.sha256(storage_header.encode()).hexdigest()[:16]}"
    with _tenant_lock:
        if state_key in _tenant_states:
            return _tenant_states[state_key]
        cfg = copy.deepcopy(_base_cfg)
        tenant_root = Path(cfg.get("retrieval.db_path")).parent / "users" / user_id
        tenant_root.mkdir(parents=True, exist_ok=True)
        tenant_db = tenant_root / "index.db"
        legacy_owner = request.headers.get("X-Cowork-Legacy-Owner") == "1" or bool(os.environ.get("DIARY_LEGACY_USER_ID"))
        source_db = Path(cfg.get("retrieval.db_path"))
        if legacy_owner and source_db.exists() and not tenant_db.exists():
            shutil.copy2(source_db, tenant_db)
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
        elif not legacy_owner:
            _set_cfg(cfg, "corpus.backend", "local")
            _set_cfg(cfg, "corpus.local.root", str(tenant_root / "corpus"))
            _set_cfg(cfg, "corpus.root", "")
        state = AppState(cfg)
        state.store.apply_pending()
        _tenant_states[state_key] = state
        return state


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
    """True when the request may proceed. Open mode when no token is configured."""
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
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", user_id):
        return JSONResponse({"error": "invalid user"}, status_code=400)
    root = Path(_base_cfg.get("retrieval.db_path")).parent / "users" / user_id
    with _tenant_lock:
        for key, state in list(_tenant_states.items()):
            if key.startswith(f"{user_id}:"):
                try:
                    state.backend.close(); state.llm_main.close(); state.llm_aux.close(); state.retrieval.close(); state.journal.close()
                except Exception:  # noqa: BLE001
                    pass
                _tenant_states.pop(key, None)
        shutil.rmtree(root, ignore_errors=True)
    return JSONResponse({"ok": True})


# ---------------- in-memory session (single user) ----------------

SESSIONS: Dict[str, dict] = {}


def _session(session_id: str, tenant_id: str = "legacy") -> dict:
    key = f"{tenant_id}:{session_id}"
    if key not in SESSIONS:
        SESSIONS[key] = {"turns": [], "log_status": []}
    return SESSIONS[key]


# ---------------- request/response models ----------------


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class RelogRequest(BaseModel):
    index: int
    session_id: str = "default"


# ---------------- pages ----------------


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    template_path = Path(__file__).parent / "static" / "index.html"
    return template_path.read_text(encoding="utf-8")


# ---------------- shared conversation core (UI + /v1 both use this) ----------------


def _run_exchange(st: AppState, message: str, session_id: str, tenant_id: str = "legacy") -> dict:
    """One full exchange: context build -> main model -> strip marker -> log -> return payload."""
    sess = _session(session_id, tenant_id)
    day = datetime.now().date()
    messages = st.assembler.build(day, message, session_turns=sess["turns"])
    reply = st.llm_main.chat(messages, temperature=0.7)

    visible, marker = LLMClient.strip_log_marker(reply)
    if marker is None:
        marker = "ok"  # model omitted the marker — default to logging (never lose content)

    outcome = st.pipeline.log_exchange(user_message=message, assistant_message=visible, now=datetime.now())

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

    threading.Thread(target=_reindex_today, args=(st, day), daemon=True).start()
    return {"reply": visible, "decision": outcome.decision, "xid": outcome.xid, "reason": outcome.reason}


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
    try:
        result = _run_exchange(_tenant_state(request), message, req.session_id, request.headers.get("X-Cowork-User-ID", "legacy"))
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
            "log": st.store.read_month_text(year, mon),
            "standing": "",
        })
    day = datetime.now().date()
    return JSONResponse({
        "day": day.isoformat(),
        "today_log": st.store.get_day_text(day, max_chars=12000),
        "standing": st.store.get_standing_sections_text(max_chars=4000),
    })


@app.get("/api/months")
def api_months(request: Request) -> JSONResponse:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    st = _tenant_state(request)
    months = run_in_threadpool_sync(st.store.list_months)
    return JSONResponse({"months": months})


@app.get("/api/health")
def api_health(request: Request) -> JSONResponse:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    st = _tenant_state(request)
    return JSONResponse({
        "ok": True,
        "journal_pending": st.journal.pending_count(),
        "retrieval": st.retrieval.stats(),
        "model": st.cfg.get("llm.chat_model"),
        "auth_required": bool(st.auth_token),
    })


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
    prior = [{"role": "user", "content": str(m)} for m in user_msgs[:-1] if str(m).strip()][-6:]

    session_id = "openai-client"
    tenant_id = request.headers.get("X-Cowork-User-ID", "legacy")
    st = _tenant_state(request)
    sess = _session(session_id, tenant_id)
    if prior and not sess["turns"]:
        sess["turns"].extend(prior)  # seed short client-thread context

    try:
        # _run_exchange does blocking inference I/O — keep the event loop free.
        result = await run_in_threadpool(_run_exchange, st, last, session_id, tenant_id)
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
