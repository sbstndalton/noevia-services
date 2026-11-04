"""D4: the Rust Diary server process and the forwarding of selected routes to it.

DIARY_RUST_ROUTES names the routes the Rust process (`noevia-diary`, noevia-rs bins/noevia-diary)
serves, comma-separated, e.g. ``/api/months,/api/day``. Empty (the default) means Python serves
everything: no process is started and no middleware is installed, so behaviour is byte-identical to
a release without this module.

ONE verifier, ONE nonce cache. While both processes exist, the tenant assertion is checked only by
this process, by TenantAssertionMiddleware exactly as before (same refusals, bytes and order).
ForwardMiddleware sits INSIDE it, so it only ever sees requests that middleware let through. It
forwards the configured routes with their raw wire path, raw query, headers and body plus three
loopback-only headers: the per-process forward secret (sent to the child on stdin, never argv or
environment; a client-supplied copy of any of these headers is dropped), whether this request's
tenant was asserted, and whether this process already holds a recovered managed state for the
tenant. The Rust side refuses anything without the secret with this app's own 401 bytes, never
verifies an assertion itself, and either serves the request exactly as this app would or DECLINES it
(421 + ``x-noevia-diary-decline: 1``) for anything it does not reproduce (non-managed storage, a
tenant this process has not recovered yet, pending journal / dirty / outbox work, an enabled
INDEX.md, tenant detail in /api/health ...). A declined request, like one the child never answered,
is served here without verifying it again, so each nonce is consumed exactly once whichever process
answers, across declines, restarts and a dead child.

Only requests that declare no body are forwarded (the native routes are GETs that ignore one); a
request with a body is never read here and goes to the app untouched, so its timing is exactly
Python's. A forward has ONE absolute deadline from the moment it is chosen, covering preparation,
the call and reading the reply; when it passes, or the child fails, the request falls back to Python
at once and killing/reaping the child happens on a background thread, never on the request path.

If the process fails to start, dies or stops answering, Python serves every route, logs the fallback, and restarts it at most MAX_RESTARTS times with a
growing delay, so a broken binary never becomes a crash loop.

D5, WRITES. The write routes (WRITE_ROUTES) are never chosen per route: DIARY_RUST_WRITES names the
TENANTS (UUIDs, or ``*``) whose writes go to the Rust process, all of them together. For such a
tenant every write request takes ONE per-tenant lock here, held until the request is answered by
whichever process answers it (a native reply, or Python after a decline or a dead child), so at most
one write of the tenant runs at any moment across both processes. A write body is forwarded only
when it is complete, JSON (or untyped), at most MAX_WRITE_BODY (Python's own bound) and read within
the forward's deadline; otherwise nothing was sent and Python serves it with the bytes it was given.
A decline means the Rust side wrote nothing. A forward that fails after its request may have reached
the child is NOT served again here (that could apply a write twice): it answers 503 UNCERTAIN, the
tenant's state is marked unrecovered, and the next write waits until the old child is reaped.

The switch is inert while PYTHON_ONLY_WRITES is non-empty: those writes have no Rust executor yet, so
a tenant's writer cannot move without being split across two processes (see the D5 design note).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
             "/api/files": "GET", "/api/storage-status": "GET",
             "/api/workspace-export": "GET"}
# D5: a tenant's writes move together (DIARY_RUST_WRITES), never route by route.
WRITE_ROUTES = {"/api/file": "PUT", "/api/directory": "POST", "/api/entries/append": "POST",
                "/api/entries/edit": "POST"}
# The writes that only this process performs today. While any is listed, DIARY_RUST_WRITES selects
# no tenant: moving a tenant's writer would split it across two processes.
PYTHON_ONLY_WRITES = (
    "/v1/chat/completions logging (log_exchange)",
    "retrieval index maintenance (_reindex_dirty, _reindex_today, embedding re-index)",
    "journal replay and index-outbox drain when a tenant state recovers",
    "/api/workspace-ops and /api/workspace-trash (workspace transaction receipt schema undecided)",
    "/api/workspace-import",
    "/api/storage-import",
    "/api/storage-backup",
    "/api/external-sources/import",
)
WRITER_HEADER = b"x-noevia-diary-writer"
# app.MAX_SIGNED_BODY: the largest JSON body this service buffers.
MAX_WRITE_BODY = 64 * 1024 * 1024
UNCERTAIN = b'{"detail":"The Diary did not confirm this save. Reload before trying again."}'
TOO_LARGE = b'{"detail":"request body too large"}'
RESTARTING = b'{"detail":"Diary saving is restarting. Try again in a moment."}'
REAP_WAIT_S = 10.0
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
DECLINE_HEADER = b"x-noevia-diary-decline"
CONTROL_HEADER = "x-noevia-diary-control"
FORWARD_HEADER = b"x-noevia-diary-forward"
ASSERTED_HEADER = b"x-noevia-diary-asserted"
RECOVERED_HEADER = b"x-noevia-diary-recovered"
INTERNAL_PREFIX = b"x-noevia-diary-"
READY_TIMEOUT_S = 10.0
# Per read/connect inactivity, and the absolute bound on one forward (a trickled reply cannot
# stretch it).
FORWARD_TIMEOUT_S = 20.0
FORWARD_DEADLINE_S = 30.0
MAX_RESTARTS = 3
RESTART_DELAYS_S = (5.0, 30.0, 120.0)
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


ALL_TENANTS = "*"
_gate_logged = False


def configured_writers(raw: Optional[str] = None):
    """The tenants whose writes go to the Rust process: frozenset of UUIDs, ALL_TENANTS, or empty.
    Empty (the default) while any write in PYTHON_ONLY_WRITES remains."""
    global _gate_logged
    raw = os.environ.get("DIARY_RUST_WRITES", "") if raw is None else raw
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        return frozenset()
    if PYTHON_ONLY_WRITES:
        if not _gate_logged:
            _gate_logged = True
            log.error("DIARY_RUST_WRITES is set but ignored: these writes have no Rust executor yet, so a "
                      "tenant's writer would be split across two processes: %s", "; ".join(PYTHON_ONLY_WRITES))
        return frozenset()
    if ALL_TENANTS in names:
        return ALL_TENANTS
    tenants = set()
    for name in names:
        if _UUID_RE.fullmatch(name):
            tenants.add(name)
        else:
            log.warning("DIARY_RUST_WRITES: %r is not a tenant UUID; ignoring it", name[:64])
    return frozenset(tenants)


def _binary() -> Optional[str]:
    configured = (os.environ.get("DIARY_RUST_SERVER_BIN") or "").strip()
    binary = configured or DEFAULT_BIN
    if configured and not os.path.isabs(configured):
        log.warning("DIARY_RUST_SERVER_BIN is not an absolute path; using %s", DEFAULT_BIN)
        binary = DEFAULT_BIN
    return binary if os.path.isfile(binary) and os.access(binary, os.X_OK) else None


def server_config(base_cfg, state, port: int, control_token: str, forward_secret: str) -> dict:
    """What the Rust process needs, resolved from this service's own config (config.yaml + env)."""
    def client(c, api_key_setting):
        return {"base_url": c.base_url, "api_key": base_cfg.get(api_key_setting) or "",
                "chat_model": c.chat_model or "", "fallback_model": c.fallback_model,
                "setting_key": c.setting_key, "fallback_setting_key": c.fallback_setting_key}
    db_path = str(base_cfg.get("retrieval.db_path") or "")
    return {
        "port": port,
        "control_token": control_token,
        "forward_secret": forward_secret,
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
        # D5: entries are written natively only with the transforms Python itself uses.
        "corpus_write_impl": (os.environ.get("DIARY_CORPUS_WRITE_IMPL") or "python").strip().lower(),
        "workspace_export": {
            "impl": os.environ.get("WORKSPACE_EXPORT_IMPL", "python"),
            "binary": workspace_export_native.BINARY,
            "settings": {key: base_cfg.get("corpus." + key) for key in SETTING_KEYS},
        },
    }


class RustServer:
    """The child process, its readiness and its restarts. Thread-safe; `ready` and `forward_secret`
    are what the request path reads."""

    def __init__(self, config_factory: Callable[[int, str, str], dict], port: int = PORT,
                 binary: Optional[str] = None, restart_delays=RESTART_DELAYS_S):
        self._config_factory = config_factory
        self.port = port
        self._binary = binary
        self._delays = tuple(restart_delays)
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._token = ""
        self.forward_secret = ""
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ready = False
        self.restarts = 0
        # Lost children still being killed/reaped: a tenant's next write waits for zero.
        self.reaping = 0
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
        token, forward = secrets.token_hex(16), secrets.token_hex(32)
        try:
            config = json.dumps(self._config_factory(self.port, token, forward), separators=(",", ":"))
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
            self._proc, self._token, self.forward_secret = proc, token, forward
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
                        return True
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
            if self._stopping.is_set():
                break
            if attempt >= min(MAX_RESTARTS, len(self._delays)):
                log.error("Rust Diary server unavailable after %d restarts; Python serves every route", attempt)
                return
            delay = self._delays[attempt]
            attempt += 1
            self.restarts = attempt
            self._stopping.wait(delay)

    def _mark_lost(self, proc: Optional[subprocess.Popen], quiet: bool = False, background: bool = False) -> None:
        """Python takes every route back (nothing to hand over: the nonce cache is already here).
        `background`: kill and reap the child on its own thread (the request path never waits)."""
        with self._lock:
            was_ready, self.ready = self.ready, False
            current = self._proc if proc is None else proc
            if proc is None or self._proc is proc:
                self._proc = None
        if was_ready and not quiet:
            log.warning("Rust Diary server lost; Python serves every route")
        if current is None:
            return
        with self._lock:
            self.reaping += 1
        if background:
            threading.Thread(target=self._reap, args=(current,), name="rust-diary-reap", daemon=True).start()
        else:
            self._reap(current)

    def _reap(self, proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        finally:
            with self._lock:
                self.reaping -= 1

    def fence(self, user_id: str, timeout: float = FORWARD_DEADLINE_S) -> bool:
        """DELETE /api/internal/tenant: make the Rust side refuse the tenant's later writes and wait
        for its write in flight. False when the process is not running or did not confirm."""
        with self._lock:
            token, ready = self._token, self.ready
        if not ready or not token:
            return False
        try:
            with httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=False) as client:
                resp = client.post("/internal/fence", params={"tenant": user_id}, headers={CONTROL_HEADER: token})
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def lost(self) -> None:
        """Called from the request path when a forward fails: take every route back now; the child
        is killed and reaped in the background (the supervisor then restarts it)."""
        self._mark_lost(None, background=True)


class ForwardMiddleware:
    """ASGI middleware INSIDE TenantAssertionMiddleware: forwards the configured routes of already
    verified requests to the Rust process while it is ready. A request it does not forward, or one
    the Rust side declines or never answers, continues to the app unchanged (scope state included,
    so it is never verified twice)."""

    def __init__(self, app, server: Optional[RustServer] = None, routes: Optional[dict] = None,
                 is_deleted: Optional[Callable[[str], bool]] = None,
                 is_recovered: Optional[Callable[[str], bool]] = None,
                 writers=frozenset(), on_uncertain: Optional[Callable[[str], None]] = None):
        self.app = app
        self.server = server
        self.routes = routes or {}
        self.writers = writers
        self.is_deleted = is_deleted or (lambda _uid: False)
        self.is_recovered = is_recovered or (lambda _uid: False)
        self.on_uncertain = on_uncertain or (lambda _uid: None)
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop = None
        # D5: one write at a time per writer tenant, across both processes. {uid: [lock, users]}
        self._write_locks: dict = {}

    @staticmethod
    def _user(scope) -> str:
        for name, value in scope.get("headers") or []:
            if name == b"x-cowork-user-id":
                return value.decode("latin-1")
        return ""

    def _wants(self, scope, server) -> bool:
        if scope.get("type") != "http" or server is None or not server.ready or not self.routes:
            return False
        if self.routes.get(scope.get("path", "")) != scope.get("method"):
            return False
        user = self._user(scope)
        # A tenant deleted in this process (24 h tombstone) is answered here (410).
        return not (user and self.is_deleted(user.lower()))

    @staticmethod
    def _declares_body(scope) -> bool:
        for name, value in scope.get("headers") or []:
            name = name.lower()
            if name == b"transfer-encoding":
                return True
            if name == b"content-length" and value.strip() not in (b"", b"0"):
                return True
        return False

    def _writer_tenant(self, scope) -> str:
        """The tenant (lower-case UUID) when this request is a write of a tenant whose writer is the
        Rust process, else ""."""
        if scope.get("type") != "http" or not self.writers:
            return ""
        if WRITE_ROUTES.get(scope.get("path", "")) != scope.get("method"):
            return ""
        user = self._user(scope).lower()
        if not _UUID_RE.fullmatch(user) or self.is_deleted(user):
            return ""
        return user if self.writers == ALL_TENANTS or user in self.writers else ""

    async def __call__(self, scope, receive, send):
        server = self.server or current()
        writer = self._writer_tenant(scope)
        if writer:
            await self._write(writer, server, scope, receive, send)
            return
        # A read with a body is never forwarded or read here: the app handles it as today.
        if not self._wants(scope, server) or self._declares_body(scope):
            await self.app(scope, receive, send)
            return
        deadline = time.monotonic() + FORWARD_DEADLINE_S
        reply = await self._forward(server, scope, deadline)
        if reply is None:
            await self.app(scope, receive, send)
            return
        status, headers, body = reply
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    # ---- D5 writes ----

    async def _write(self, user, server, scope, receive, send):
        entry = self._write_locks.setdefault(user, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                await self._write_locked(user, server, scope, receive, send)
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._write_locks.pop(user, None)

    @staticmethod
    async def _reply(send, status, body, headers=None):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-length", str(len(body)).encode()),
                                (b"content-type", b"application/json"), *(headers or [])]})
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _json_media(scope) -> bool:
        for name, value in scope.get("headers") or []:
            if name.lower() == b"content-type":
                media = value.split(b";", 1)[0].strip().lower()
                return media == b"application/json"
        return True

    @staticmethod
    def _declared_length(scope) -> Optional[int]:
        for name, value in scope.get("headers") or []:
            if name.lower() == b"content-length":
                try:
                    return int(value.strip())
                except ValueError:
                    return None
        return None

    async def _write_locked(self, user, server, scope, receive, send):
        # A child lost mid-write may still be running: never let a second process write meanwhile.
        if server is not None and server.reaping:
            waited = time.monotonic() + REAP_WAIT_S
            while server.reaping and time.monotonic() < waited:
                await asyncio.sleep(0.05)
            if server.reaping:
                await self._reply(send, 503, RESTARTING)
                return
        declared = self._declared_length(scope)
        if declared is not None and declared > MAX_WRITE_BODY:
            # Never read; Python's own bound for a buffered body (writes of this tenant never run
            # on Python with an unbounded body).
            await self._reply(send, 413, TOO_LARGE)
            return
        if server is None or not server.ready or not self._json_media(scope):
            # The Rust process is not running (nothing of it can still write: see reaping above),
            # or the body is not JSON (the app answers 422 without writing).
            await self.app(scope, receive, send)
            return
        deadline = time.monotonic() + FORWARD_DEADLINE_S
        chunks, size, complete, disconnected = [], 0, False, None
        try:
            while not complete:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                message = await asyncio.wait_for(receive(), remaining)
                if message.get("type") != "http.request":
                    disconnected = message
                    break
                chunk = message.get("body", b"")
                size += len(chunk)
                chunks.append(chunk)
                if size > MAX_WRITE_BODY:
                    await self._reply(send, 413, TOO_LARGE)
                    return
                complete = not message.get("more_body", False)
        except asyncio.TimeoutError:
            pass
        body = b"".join(chunks)

        async def replay():
            # Exactly the bytes received so far, then whatever the client still sends.
            nonlocal body, disconnected
            if body is not None:
                prefix, body = body, None
                return {"type": "http.request", "body": prefix, "more_body": not complete}
            if disconnected is not None:
                message, disconnected = disconnected, None
                return message
            return await receive()

        if not complete:
            # Nothing was sent to the Rust side: Python serves what it was given.
            await self.app(scope, replay, send)
            return
        outcome = await self._forward(server, scope, deadline, body=body, writer=True)
        if outcome == "uncertain":
            self.on_uncertain(user)
            await self._reply(send, 503, UNCERTAIN)
            return
        if outcome is None:
            await self.app(scope, replay, send)
            return
        status, headers, content = outcome
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": content})

    async def _forward(self, server, scope, deadline: float, body: bytes = b"", writer: bool = False):
        """(status, headers, body) from the Rust side, or None to serve the request here. Everything
        runs within `deadline` (time.monotonic). A write (`writer`) that may have reached the child
        and got no reply returns "uncertain" instead of None: it must not run again here."""
        raw_path = scope.get("raw_path") or scope.get("path", "").encode()
        raw_path = raw_path.split(b"?", 1)[0]
        query = scope.get("query_string", b"")
        target = raw_path + (b"?" + query if query else b"")
        try:
            url = httpx.URL(server.base_url, raw_path=target)
        except Exception:  # noqa: BLE001
            return None
        if url.raw_path != target:
            # httpx would re-encode this request target: keep it here, unaltered.
            return None
        secret = server.forward_secret
        if not secret:
            return None
        asserted = bool((scope.get("state") or {}).get("tenant_asserted"))
        user = self._user(scope)
        recovered = bool(user) and self.is_recovered(user.lower())
        headers = [(k, v) for k, v in scope.get("headers") or []
                   if k.lower() not in _HOP and not k.lower().startswith(INTERNAL_PREFIX)]
        headers += [(FORWARD_HEADER, secret.encode()), (ASSERTED_HEADER, b"1" if asserted else b"0"),
                    (RECOVERED_HEADER, b"1" if recovered else b"0")]
        if writer:
            headers.append((WRITER_HEADER, b"1"))
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            # One pooled client per event loop (uvicorn runs one; a client is bound to its loop).
            self._client, self._client_loop = httpx.AsyncClient(timeout=FORWARD_TIMEOUT_S, trust_env=False), loop

        async def exchange():
            resp = await self._client.send(self._client.build_request(scope["method"], url, headers=headers,
                                                                       content=body))
            return resp, await resp.aread()

        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("forward deadline passed before the call")
            resp, content = await asyncio.wait_for(exchange(), remaining)
        except Exception as exc:  # noqa: BLE001 - lost or stuck process: Python serves from here on
            log.warning("Rust Diary server did not answer (%s); falling back to Python", type(exc).__name__)
            server.lost()
            if writer and not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                # The request may have reached the child, which may have committed it.
                return "uncertain"
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


def start(config_factory: Callable[[int, str, str], dict], routes: dict, writers=frozenset()) -> Optional[RustServer]:
    """Lifespan hook: start the Rust process when any route or writer tenant is configured."""
    global _server
    if not routes and not writers:
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


def fence_tenant(user_id: str) -> None:
    """DELETE /api/internal/tenant, after Python's tombstone: the Rust side stops writing the tenant
    (and finishes the write in flight) before its root is removed."""
    server = current()
    if server is not None and not server.fence(user_id):
        log.warning("Rust Diary server did not confirm the tenant fence; removing the tenant anyway")
