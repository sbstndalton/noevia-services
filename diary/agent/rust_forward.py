"""D4: the Rust Diary server process and the forwarding of selected routes to it.

DIARY_RUST_ROUTES names the routes the Rust process (`noevia-diary`, noevia-rs bins/noevia-diary)
serves, comma-separated, e.g. ``/api/months,/api/day``. Empty (the default) means Python serves
everything: no process is started and no request is touched, so behaviour is byte-identical to a
release without this module.

When it is set, the lifespan starts the binary on 127.0.0.1:8011 with this service's resolved
settings as one JSON line on stdin (never argv or its environment). Once it answers
/internal/ready, its replay floor is raised to the current second and the named routes move to it.
A forwarded request goes out with its raw wire path, raw query, every header and the exact body,
because the tenant assertion signs those bytes. The Rust side either serves it exactly as this
app would, or DECLINES it (421 + ``x-noevia-diary-decline: 1``) before checking any credential,
for anything it does not reproduce (non-managed storage, pending journal or reindex work, an
enabled INDEX.md, tenant detail in /api/health ...); a declined request is served here as usual.

Each process keeps its own nonce cache and an assertion is checked by exactly one of them. When a
route changes sides, the side that takes it over refuses assertions signed at or before the switch
second: Rust's floor is raised before routes move to it, and this module raises Python's
(tenant_assertion.raise_floor) when the Rust process is lost. If the process fails to start, dies
or stops answering, Python serves every route, logs the fallback, and restarts it at most
MAX_RESTARTS times with a growing delay, so a broken binary never becomes a crash loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import subprocess
import threading
import time
from typing import Callable, Optional

import httpx

from . import tenant_assertion, workspace_export_native
from .diary_migration import SETTING_KEYS

log = logging.getLogger("diary.rust")

PORT = 8011
DEFAULT_BIN = "/usr/local/bin/noevia-diary"
# The routes the Rust process implements (D4). Anything else named in DIARY_RUST_ROUTES is ignored.
# D4b adds the read side of the file API, storage status and workspace export; `PUT /api/file`
# and every other write stay here (single writer until D5).
SUPPORTED = {"/api/health": "GET", "/api/months": "GET", "/api/day": "GET",
             "/api/files": "GET", "/api/file": "POST", "/api/storage-status": "GET",
             "/api/workspace-export": "GET"}
DECLINE_HEADER = b"x-noevia-diary-decline"
CONTROL_HEADER = "x-noevia-diary-control"
READY_TIMEOUT_S = 10.0
FORWARD_TIMEOUT_S = 20.0
MAX_RESTARTS = 3
RESTART_DELAYS_S = (5.0, 30.0, 120.0)
MAX_FORWARD_BODY = 64 * 1024 * 1024
_HOP = {b"connection", b"keep-alive", b"transfer-encoding", b"te", b"upgrade", b"content-length",
        b"proxy-connection", b"trailer"}
_DROP_REPLY = {b"date", b"server", b"connection", b"keep-alive", b"transfer-encoding"}


def configured_routes(raw: Optional[str] = None) -> dict:
    """{path: method} from DIARY_RUST_ROUTES; unknown names get one warning and are ignored."""
    raw = os.environ.get("DIARY_RUST_ROUTES", "") if raw is None else raw
    routes = {}
    for name in (part.strip() for part in raw.split(",")):
        if not name:
            continue
        if name in SUPPORTED:
            routes[name] = SUPPORTED[name]
        else:
            log.warning("DIARY_RUST_ROUTES: %r is not served by the Rust Diary process; ignoring it", name[:64])
    return routes


def _binary() -> Optional[str]:
    configured = (os.environ.get("DIARY_RUST_SERVER_BIN") or "").strip()
    binary = configured or DEFAULT_BIN
    if configured and not os.path.isabs(configured):
        log.warning("DIARY_RUST_SERVER_BIN is not an absolute path; using %s", DEFAULT_BIN)
        binary = DEFAULT_BIN
    return binary if os.path.isfile(binary) and os.access(binary, os.X_OK) else None


def server_config(base_cfg, state, port: int, control_token: str) -> dict:
    """What the Rust process needs, resolved from this service's own config (config.yaml + env)."""
    def client(c, api_key_setting):
        return {"base_url": c.base_url, "api_key": base_cfg.get(api_key_setting) or "",
                "chat_model": c.chat_model or "", "fallback_model": c.fallback_model,
                "setting_key": c.setting_key, "fallback_setting_key": c.fallback_setting_key}
    db_path = str(base_cfg.get("retrieval.db_path") or "")
    return {
        "port": port,
        "control_token": control_token,
        "auth_token": state.auth_token,
        "tenant_key": tenant_assertion.tenant_key(),
        "data_root": os.path.dirname(db_path) if db_path and db_path != ":memory:" else ".",
        "legacy_user_id": os.environ.get("DIARY_LEGACY_USER_ID", ""),
        "corpus": {
            **{key: base_cfg.get("corpus." + key) for key in
               ("monthly_prefix", "entry_layout", "entries_prefix", "index_file", "month_file_template")},
            "index_enabled": bool(base_cfg.get("corpus.index_enabled", True)),
        },
        "llm": {
            "main": client(state.llm_main, "llm.api_key"),
            "aux": client(state.llm_aux, "llm.aux.api_key"),
            "aux_prefers_main": getattr(state.llm_aux, "_prefer_from", None) is state.llm_main,
            "configured": base_cfg.get("llm.chat_model"),
        },
        # GET /api/workspace-export is reproducible only with the native exporter (the Python
        # exporter stamps the current local time into the ZIP); the Rust side declines otherwise.
        "workspace_export": {
            "impl": os.environ.get("WORKSPACE_EXPORT_IMPL", "python"),
            "binary": workspace_export_native.BINARY,
            "settings": {key: base_cfg.get("corpus." + key) for key in SETTING_KEYS},
        },
    }


class RustServer:
    """The child process, its readiness and its restarts. Thread-safe; `ready` is the only flag
    the request path reads."""

    def __init__(self, config_factory: Callable[[int, str], dict], port: int = PORT,
                 binary: Optional[str] = None, restart_delays=RESTART_DELAYS_S):
        self._config_factory = config_factory
        self.port = port
        self._binary = binary
        self._delays = tuple(restart_delays)
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._token = ""
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ready = False
        self.restarts = 0
        self.base_url = f"http://127.0.0.1:{port}"

    # ---- lifecycle ----
    def start(self) -> None:
        self._thread = threading.Thread(target=self._supervise, name="rust-diary", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._mark_lost(None, quiet=True)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _spawn(self) -> Optional[subprocess.Popen]:
        binary = self._binary or _binary()
        if binary is None:
            log.warning("Rust Diary server binary not found; Python serves every route")
            return None
        token = secrets.token_hex(16)
        try:
            config = json.dumps(self._config_factory(self.port, token), separators=(",", ":"))
            proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=None, close_fds=True,
                                    env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                                         "TZ": os.environ.get("TZ", ""), "HOME": "/nonexistent"})
            proc.stdin.write(config.encode() + b"\n")
            proc.stdin.close()
        except Exception as exc:  # noqa: BLE001 - never take the service down
            log.warning("Rust Diary server could not start (%s); Python serves every route", type(exc).__name__)
            return None
        with self._lock:
            self._proc, self._token = proc, token
        return proc

    def _wait_ready(self, proc: subprocess.Popen) -> bool:
        deadline = time.monotonic() + READY_TIMEOUT_S
        headers = {CONTROL_HEADER: self._token}
        with httpx.Client(base_url=self.base_url, timeout=2.0, trust_env=False) as client:
            while time.monotonic() < deadline and not self._stopping.is_set():
                if proc.poll() is not None:
                    return False
                try:
                    if client.get("/internal/ready", headers=headers).status_code == 200:
                        # Requests this process accepted so far must not be replayable there.
                        floor = client.post("/internal/floor", headers=headers, content=str(int(time.time())))
                        return floor.status_code == 200
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
        return False

    def _supervise(self) -> None:
        attempt = 0
        while not self._stopping.is_set():
            proc = self._spawn()
            if proc is not None and self._wait_ready(proc):
                with self._lock:
                    if self._proc is proc:
                        self.ready = True
                log.info("Rust Diary server ready on 127.0.0.1:%d", self.port)
                while proc.poll() is None and not self._stopping.is_set():
                    self._stopping.wait(0.5)
                if self._stopping.is_set():
                    break
                self._mark_lost(proc)
            else:
                self._mark_lost(proc)
            if self._stopping.is_set():
                break
            if attempt >= min(MAX_RESTARTS, len(self._delays)):
                log.error("Rust Diary server unavailable after %d restarts; Python serves every route", attempt)
                return
            delay = self._delays[attempt]
            attempt += 1
            self.restarts = attempt
            self._stopping.wait(delay)

    def _mark_lost(self, proc: Optional[subprocess.Popen], quiet: bool = False) -> None:
        """Python takes every route back. Assertions signed at or before now may have been accepted
        by the Rust side, so this side refuses them from here on."""
        with self._lock:
            was_ready, self.ready = self.ready, False
            current = self._proc if proc is None else proc
            if proc is None or self._proc is proc:
                self._proc = None
        if was_ready:
            tenant_assertion.raise_floor(int(time.time()))
        if current is not None and current.poll() is None:
            try:
                current.kill()
                current.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        if was_ready and not quiet:
            code = current.returncode if current is not None else None
            log.warning("Rust Diary server lost (exit %s); Python serves every route", code)

    def lost(self) -> None:
        """Called from the request path when a forward fails: kill the process (the supervisor
        restarts it) and take every route back."""
        self._mark_lost(None)


class ForwardMiddleware:
    """Outermost ASGI middleware: forwards the configured routes to the Rust process while it is
    ready. It runs before TenantAssertionMiddleware, so a forwarded request is never verified here
    (and a declined one is verified only here)."""

    def __init__(self, app, server: Optional[RustServer] = None, routes: Optional[dict] = None,
                 is_deleted: Optional[Callable[[str], bool]] = None):
        self.app = app
        self.server = server
        self.routes = routes or {}
        self.is_deleted = is_deleted or (lambda _uid: False)
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop = None

    def _wants(self, scope, server) -> bool:
        if scope.get("type") != "http" or server is None or not server.ready or not self.routes:
            return False
        if self.routes.get(scope.get("path", "")) != scope.get("method"):
            return False
        for name, value in scope.get("headers") or []:
            if name == b"x-cowork-user-id":
                # A tenant deleted in this process (24 h tombstone) is answered here (410).
                if self.is_deleted(value.decode("latin-1").lower()):
                    return False
                break
        return True

    async def __call__(self, scope, receive, send):
        server = self.server or current()
        if not self._wants(scope, server):
            await self.app(scope, receive, send)
            return
        chunks, size, more, overflow = [], 0, True, False
        while more:
            message = await receive()
            if message.get("type") != "http.request":
                break
            chunk = message.get("body", b"")
            size += len(chunk)
            chunks.append(chunk)
            more = message.get("more_body", False)
            if size > MAX_FORWARD_BODY:
                overflow = True
                break
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": overflow and more}
            return await receive()

        reply = None if overflow else await self._forward(server, scope, b"".join(chunks))
        if reply is None:
            await self.app(scope, replay, send)
            return
        status, headers, body = reply
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def _forward(self, server, scope, body: bytes):
        """(status, headers, body) from the Rust side, or None to serve the request here."""
        raw_path = scope.get("raw_path") or scope.get("path", "").encode()
        raw_path = raw_path.split(b"?", 1)[0]
        query = scope.get("query_string", b"")
        target = raw_path + (b"?" + query if query else b"")
        try:
            url = httpx.URL(server.base_url, raw_path=target)
        except Exception:  # noqa: BLE001
            return None
        if url.raw_path != target:
            # httpx would re-encode this request target, and the assertion signs the exact bytes.
            return None
        headers = [(k, v) for k, v in scope.get("headers") or [] if k.lower() not in _HOP]
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            # One pooled client per event loop (uvicorn runs one; a client is bound to its loop).
            self._client, self._client_loop = httpx.AsyncClient(timeout=FORWARD_TIMEOUT_S, trust_env=False), loop
        try:
            resp = await self._client.send(self._client.build_request(
                scope["method"], url, headers=headers, content=body))
            content = await resp.aread()
        except Exception as exc:  # noqa: BLE001 - lost process: Python serves from here on
            log.warning("Rust Diary server did not answer (%s); falling back to Python", type(exc).__name__)
            server.lost()
            return None
        raw = resp.headers.raw
        if any(k.lower() == DECLINE_HEADER for k, _ in raw):
            return None
        kept = [(k.lower(), v) for k, v in raw if k.lower() not in _DROP_REPLY]
        # Starlette's order: the response's own headers, then content-length, then content-type.
        order = {b"content-length": 1, b"content-type": 2}
        kept.sort(key=lambda kv: order.get(kv[0], 0))
        return resp.status_code, kept, content


_server: Optional[RustServer] = None


def start(config_factory: Callable[[int, str], dict], routes: dict) -> Optional[RustServer]:
    """Lifespan hook: start the Rust process when any route is configured."""
    global _server
    if not routes:
        return None
    _server = RustServer(config_factory)
    _server.start()
    return _server


def stop() -> None:
    global _server
    if _server is not None:
        _server.stop()
        _server = None


def current() -> Optional[RustServer]:
    return _server
