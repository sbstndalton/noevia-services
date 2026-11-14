"""D4: the Rust Diary server process and the forwarding of selected routes to it.

DIARY_RUST_ROUTES names the routes the Rust process (`noevia-diary`, noevia-rs bins/noevia-diary)
serves, comma-separated, e.g. ``/api/months,/api/day``. Empty (the default) means Python serves
everything: no process is started and no middleware is installed, so behaviour is byte-identical to
a release without this module.

ONE verifier per request (D8 verifier move). A request this forwarder will send to a ready Rust
process (`delegates`) is NOT verified here: TenantAssertionMiddleware lets it through unread and
the Rust side (noevia-rs bins/noevia-diary assertion.rs) verifies it first, with the middleware's
exact refusals (401 ``invalid tenant assertion``, 413 above MAX_SIGNED_BODY). Every other request is
verified here exactly as before. ForwardMiddleware sits INSIDE the middleware. It forwards with the
raw wire path, raw query, headers and body plus loopback-only headers: the per-process forward
secret (sent to the child on stdin, never argv or environment; a client-supplied copy of any
``x-noevia-diary-*`` header is dropped), ``x-noevia-diary-verify: 1`` for a delegated request (else
``x-noevia-diary-asserted`` says whether this process verified it), and whether this process
already holds a recovered managed state for the tenant. The Rust side refuses anything without the
secret with this app's own 401 bytes, and either serves the request exactly as this app would or
DECLINES it (421 + ``x-noevia-diary-decline: 1``) for anything it does not reproduce (non-managed
storage, a tenant this process has not recovered yet, pending journal / dirty / outbox work, an
enabled INDEX.md, tenant detail in /api/health ...).

ONE nonce owner. This process's replay cache is the single admission authority for every nonce,
whichever process checks the signature (tenant_assertion "one nonce owner"). A request is delegated
only after its nonce is RESERVED there, atomically with every other admission; until that
reservation ends, every other request with the same nonce is refused, on either path and with any
body framing. The reservation is settled from what is known about the forward:
- the reply carries ``x-noevia-diary-verified: 1`` (the Rust side verified and accepted it): COMMIT,
  and a declined one is served here as asserted, without a second signature check;
- definitely not accepted (never dispatched: no forward, connection refused; or the child answered
  without the marker): RELEASE, and a request still to be served here is verified here first;
- unknown after dispatch (transport error, timeout, the caller cancelling, a lost reply): HELD until
  the nonce's validity horizon, never released early. The exchange keeps running on its own (a
  detached owner) and commits or releases if an authoritative reply still arrives. A read whose
  forward ended this way answers 503 (UNCONFIRMED) while the child process had not exited; only when
  it HAD exited (the accepted lost-child-before-reply case) is the SAME request served here once,
  verified here with its own reservation as owner; no other request with that nonce is admitted
  either way. A write answers 503 UNCERTAIN as before. The child is marked lost in every case.
Exactly one signature check per accepted request (the lost-child fallback excepted). The Rust
side's own cache stays as a second guard, and a restarted child refuses anything signed before it
started.

Only requests that declare no body are forwarded (the native routes are GETs that ignore one); a
request with a body is never read here and goes to the app untouched, so its timing is exactly
Python's. A forward has ONE absolute deadline from the moment it is chosen, covering preparation,
the call and reading the reply; when it passes, or the child fails, the request falls back to Python
at once and killing/reaping the child happens on a background thread, never on the request path.

If the process fails to start, dies or stops answering, Python serves every route, logs the fallback, and restarts it at most MAX_RESTARTS times with a
growing delay, so a broken binary never becomes a crash loop.

D5, WRITES. The write routes (WRITE_ROUTES) are never chosen per route: DIARY_RUST_WRITES names the
TENANTS (UUIDs, or ``*``) whose writes go to the Rust process, all of them together. For such a
tenant every write request takes the tenant's writer lock (tenant_writer) here, held until the request
is answered by whichever process answers it (a native reply, or Python after a decline or a dead
child). Every write Python makes on its own for such a tenant (chat logging, re-index, recovery,
workspace ops, imports, backup bookkeeping, deletion) takes the same lock, so at most one write of
the tenant runs at any moment across both processes. A write body is forwarded only
when it is complete, JSON (or untyped), at most MAX_WRITE_BODY (Python's own bound) and read within
the forward's deadline; otherwise nothing was sent and Python serves it with the bytes it was given.
A decline means the Rust side wrote nothing. A forward that fails after its request may have reached
the child is NOT served again here (that could apply a write twice): it answers 503 UNCERTAIN, the
tenant's state is marked unrecovered, and the next write waits until the old child is reaped.
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
import weakref
from typing import Callable, Optional

import httpx

from . import rust_answer, tenant_assertion, tenant_writer, workspace_export_native
from .diary_migration import SETTING_KEYS

log = logging.getLogger("diary.rust")

PORT = 8011
DEFAULT_BIN = "/usr/local/bin/noevia-diary"
# The routes the Rust process implements (D4). Anything else named in DIARY_RUST_ROUTES is ignored.
# D4b adds the read side of the file API, storage status and workspace export; `PUT /api/file`
# and every other write stay here (single writer until D5).
SUPPORTED = {"/api/health": "GET", "/api/months": "GET", "/api/day": "GET",
             "/api/files": "GET", "/api/storage-status": "GET",
             "/api/workspace-export": "GET",
             # R9: the external-sources scan (served only with the resolved list, below).
             "/api/external-sources": "GET"}
# R9 main mode only (rust_parent): routes the Rust parent serves natively that this forwarder never
# would (the file read takes a body). Honoured only when this app is the Rust process's child.
MAIN_ONLY = {"/api/file": "POST"}
# D5: a tenant's writes move together (DIARY_RUST_WRITES), never route by route.
WRITE_ROUTES = {"/api/file": "PUT", "/api/directory": "POST", "/api/entries/append": "POST",
                "/api/entries/edit": "POST", "/api/storage-backup": "POST"}
WRITER_HEADER = b"x-noevia-diary-writer"
# app.MAX_SIGNED_BODY: the largest JSON body this service buffers.
MAX_WRITE_BODY = 64 * 1024 * 1024
UNCERTAIN = b'{"detail":"The Diary did not confirm this save. Reload before trying again."}'
TOO_LARGE = b'{"detail":"request body too large"}'
RESTARTING = b'{"detail":"Diary saving is restarting. Try again in a moment."}'
BUSY = b'{"detail":"Diary is saving another change. Try again in a moment."}'
BODY_TIMEOUT = b'{"detail":"The request body did not arrive in time."}'
# D8: a delegated read whose forward ended without an answer while the child had not exited: it may
# have been accepted there, so it is neither served nor verified again here.
UNCONFIRMED = b'{"detail":"The Diary did not answer in time. Try again in a moment."}'
# How long a failed forward waits to see whether the child process has exited (the lost-child case).
EXIT_PROBE_S = 0.2
REAP_WAIT_S = 10.0
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
DECLINE_HEADER = b"x-noevia-diary-decline"
VERIFY_HEADER = b"x-noevia-diary-verify"
VERIFIED_HEADER = b"x-noevia-diary-verified"
UNVERIFIED = b'{"detail":"invalid tenant assertion"}'
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


def configured_routes(raw: Optional[str] = None, main: bool = False) -> dict:
    """{path: method} from DIARY_RUST_ROUTES; unknown names get one warning and are ignored.
    `main` (R9: the Rust process is the parent) also accepts MAIN_ONLY."""
    raw = os.environ.get("DIARY_RUST_ROUTES", "") if raw is None else raw
    known = {**SUPPORTED, **MAIN_ONLY} if main else SUPPORTED
    routes = {}
    for name in (part.strip() for part in raw.split(",")):
        if not name:
            continue
        if name in known:
            routes[name] = known[name]
        else:
            log.warning("DIARY_RUST_ROUTES: %r is not served by the Rust Diary process; ignoring it", name[:64])
    return routes


ALL_TENANTS = tenant_writer.ALL


def configured_writers(raw: Optional[str] = None):
    """The tenants whose writes go to the Rust process: frozenset of UUIDs, ALL_TENANTS, or empty
    (the default)."""
    raw = os.environ.get("DIARY_RUST_WRITES", "") if raw is None else raw
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
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
            "impl": workspace_export_native.impl_value(),
            "binary": workspace_export_native.BINARY,
            "settings": {key: base_cfg.get("corpus." + key) for key in SETTING_KEYS},
        },
        # R9: DIARY_EXTERNAL_SOURCES as GET /api/external-sources reads it (`~` expanded here: the
        # child runs with HOME=/nonexistent). Absent when it cannot be resolved: Python answers.
        **({"external_sources": sources} if (sources := _external_sources()) is not None else {}),
        # R9: present only while DIARY_RUST_ANSWER is on; without it the answer endpoints are 404.
        **({"answer": answer} if (answer := rust_answer.answer_config(
            base_cfg, getattr(state.llm_main, "reasoning_tokens", 0))) is not None else {}),
    }


def _external_sources() -> Optional[list]:
    from .external_sources import external_source_paths
    try:
        paths = list(external_source_paths())
        for p in paths:
            p.encode("utf-8")  # a lone surrogate would make the whole start-up line unreadable there
        return paths
    except Exception as exc:  # noqa: BLE001 - the route then stays with Python (its own error)
        log.info("DIARY_EXTERNAL_SOURCES not resolvable for the Rust process (%s)", type(exc).__name__)
        return None


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
        self._spawning = False
        # Reserved before deletion. New incarnations cannot receive a forwarded request for
        # these tenants, including while the caller has not yet published its tombstone.
        self._deleting = {}
        self._token = ""
        self.forward_secret = ""
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ready = False
        self.restarts = 0
        # Lost children not yet confirmed exited. While non-zero no writer tenant's lock can be newly
        # acquired, by either process's writers (tenant_writer admission barrier).
        self.reaping = 0
        self.base_url = f"http://127.0.0.1:{port}"
        self.answer_only = False
        _servers.add(self)

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
        with self._lock:
            self._spawning = True
        proc = None
        try:
            binary = self._binary or _binary()
            if binary is None:
                log.warning("Rust Diary server binary not found; Python serves every route")
                return None
            token, forward = secrets.token_hex(16), secrets.token_hex(32)
            config = json.dumps(self._config_factory(self.port, token, forward), separators=(",", ":"))
            proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=None, close_fds=True,
                                    env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                                         "TZ": os.environ.get("TZ", ""), "HOME": "/nonexistent"})
            # Publish ownership before feeding config: a broken pipe must never leave an
            # untracked child which an absence check could mistake for settled.
            with self._lock:
                self._proc, self._token, self.forward_secret = proc, token, forward
            proc.stdin.write(config.encode() + b"\n")
            proc.stdin.close()
            return proc
        except Exception as exc:
            log.warning("Rust Diary server could not start (%s); Python serves every route", type(exc).__name__)
            if proc is not None:
                self._mark_lost(proc, quiet=True)
            return None
        finally:
            with self._lock:
                self._spawning = False

    def prepare_delete(self, user_id: str) -> bool:
        """Reserve deletion against all future forwards, then prove old-child coverage.

        The caller checks the durable intent before and after this and retains its tenant
        hold through removal. An absent process is safe only outside spawn/reap transitions.
        A failed live fence keeps the reservation and may be retried; it never authorizes
        tombstoning or removal. No transport error/exit retires a remote effect intent.
        """
        with self._lock:
            if self._spawning or self.reaping:
                return False
            proc = self._proc
            if proc is None:
                self._deleting[user_id] = time.monotonic()
                return True
            if not self.ready or not self._token:
                return False
            self._deleting[user_id] = time.monotonic()
        return self.fence(user_id)

    def forward_secret_for(self, user_id: str) -> str:
        with self._lock:
            reserved = self._deleting.get(user_id.lower())
            # Same 24-hour tombstone lifetime as the Python registry and native fence.
            if reserved is not None and time.monotonic() - reserved < 24 * 60 * 60:
                return ""
            self._deleting.pop(user_id.lower(), None)
            return self.forward_secret

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
            if current is not None:
                # In the same critical section that forgets the child: a caller that finds no
                # child here always finds the barrier up until the child is confirmed exited.
                self.reaping += 1
        if was_ready and not quiet:
            log.warning("Rust Diary server lost; Python serves every route")
        if current is None:
            return
        if background:
            threading.Thread(target=self._reap, args=(current,), name="rust-diary-reap", daemon=True).start()
        else:
            self._reap(current)

    def _reap(self, proc: subprocess.Popen) -> None:
        """Kill the child and lower the barrier only once its exit is confirmed. Until then (kill
        or wait failing, a child that does not exit) the barrier stays up and writes fail closed."""
        attempt = 0
        while True:
            try:
                if proc.poll() is not None:
                    break
                proc.kill()
                proc.wait(timeout=REAP_WAIT_S)
                break
            except Exception as exc:  # noqa: BLE001 - keep trying; never lower the barrier unproven
                attempt += 1
                if attempt in (1, 10) or attempt % 100 == 0:
                    log.error("Rust Diary server did not confirm its exit (%s); Diary writes stay paused",
                              type(exc).__name__)
                time.sleep(0.2)
        with self._lock:
            self.reaping -= 1
        tenant_writer.notify()

    def control(self):
        """(control token, ready): what an internal call (rust_answer) needs."""
        with self._lock:
            return self._token, self.ready

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


_servers: "weakref.WeakSet[RustServer]" = weakref.WeakSet()


def _reaping_barrier() -> Optional[str]:
    return "restarting" if any(server.reaping for server in list(_servers)) else None


tenant_writer.add_barrier(_reaping_barrier)


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
        # D5: the writer tenants' lock is tenant_writer's, shared with every Python write path.
        if writers:
            tenant_writer.configure(writers)

    def delegates(self, scope) -> bool:
        """True when this request will be forwarded to a ready Rust process now, so its assertion is
        the Rust side's to verify (TenantAssertionMiddleware skips it). Call once per request: it
        reserves the nonce (tenant_assertion.reserve, kept in scope state) and is False when it
        cannot, so a nonce this process holds, accepted or reserved, is verified (and refused) here."""
        server = self.server or current()
        if scope.get("type") != "http" or server is None or not server.ready or not server.forward_secret:
            return False
        if self._writer_tenant(scope):
            declared = self._declared_length(scope)
            if not self._json_media(scope) or (declared is not None and declared > MAX_WRITE_BODY):
                return False
        elif not self._wants(scope, server) or self._declares_body(scope):
            return False
        # The single admission decision: reserve the nonce now (refused or full: verified here).
        claim = tenant_assertion.reserve(self._assertion(scope))
        if claim is None:
            return False
        scope.setdefault("state", {})["rust_claim"] = claim
        return True

    @staticmethod
    def _claim(scope):
        return (scope.get("state") or {}).get("rust_claim")

    @staticmethod
    def _delegated(scope) -> bool:
        return bool((scope.get("state") or {}).get("rust_verifies"))

    async def _serve_here(self, scope, receive, send, body: bytes = b""):
        """Serve a request in this process. A delegated request the Rust side did not verify is
        verified here first, exactly as TenantAssertionMiddleware would have (fail closed)."""
        state = scope.setdefault("state", {})
        if self._delegated(scope) and not state.get("tenant_asserted"):
            reason = await self._verify_here(scope, body, self._claim(scope))
            if reason:
                if reason != tenant_assertion.BUSY_REFUSAL:
                    log.warning("tenant assertion rejected: %s", reason)
                await self._reply(send, 401, UNVERIFIED)
                return
            state["tenant_asserted"] = True
        await self.app(scope, receive, send)

    @staticmethod
    async def _verify_here(scope, body: bytes, owner=None) -> Optional[str]:
        from starlette.concurrency import run_in_threadpool
        from starlette.datastructures import Headers
        headers = Headers(scope=scope)
        key = tenant_assertion.tenant_key()
        if not key or not headers.get("x-cowork-user-id", ""):
            return None
        body_hash = (tenant_assertion.sha256_hex(body) if tenant_assertion.body_is_hashed(headers.get("content-type", ""))
                     else tenant_assertion.STREAM)
        return await run_in_threadpool(tenant_assertion.verify, key, headers, scope.get("method", "GET"),
                                       tenant_assertion.wire_path(scope), query=scope.get("query_string", b""),
                                       body_hash=body_hash, owner=owner)

    async def _refuse_effect_blocked(self, scope, receive, send):
        """A mutation while the tenant's backup effect is unsettled: 503 (backup_effect.MESSAGE). A
        request delegated to the Rust verifier is verified here first, exactly as `_serve_here`
        would, so an unverified caller gets the 401 and never learns the tenant's backup state."""
        from .backup_effect import MESSAGE
        state = scope.setdefault("state", {})
        if self._delegated(scope) and not state.get("tenant_asserted"):
            body = b""
            if self._declares_body(scope):
                # Delegation admitted only a bounded JSON write body (see `delegates`).
                body, failure = await self._collect(receive, time.monotonic() + FORWARD_DEADLINE_S)
                if failure == "too_large":
                    await self._reply(send, 413, TOO_LARGE)
                    return
                if failure == "timeout":
                    await self._reply(send, 408, BODY_TIMEOUT)
                    return
                if failure == "disconnected":
                    return
            reason = await self._verify_here(scope, body, self._claim(scope))
            if reason:
                if reason != tenant_assertion.BUSY_REFUSAL:
                    log.warning("tenant assertion rejected: %s", reason)
                await self._reply(send, 401, UNVERIFIED)
                return
            state["tenant_asserted"] = True
        await self._reply(send, 503, json.dumps({"detail": MESSAGE}, separators=(",", ":")).encode())

    async def _after_decline(self, scope, receive, send, body: bytes = b""):
        """A forward that came back without a served reply (declined, or no answer)."""
        state = scope.setdefault("state", {})
        if state.pop("rust_verified", False):
            # The Rust side verified and accepted this assertion (its reservation is committed).
            state["tenant_asserted"] = True
        await self._serve_here(scope, receive, send, body)

    @staticmethod
    def _assertion(scope) -> str:
        for name, value in scope.get("headers") or []:
            if name == b"x-cowork-tenant-assertion":
                return value.decode("latin-1")
        return ""

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
        try:
            await self._call(scope, receive, send)
        finally:
            # A reservation never dispatched to the child (a refusal before sending, a request that
            # was not forwarded after all, a request verified here) is released: it consumed
            # nothing. A dispatched one is settled by its forward (commit / release / hold).
            state = scope.get("state") or {}
            claim = state.get("rust_claim")
            if claim is not None and not state.get("rust_dispatched"):
                tenant_assertion.release(claim)

    async def _call(self, scope, receive, send):
        user = (self._user(scope) or os.environ.get("DIARY_LEGACY_USER_ID", "")).lower()
        if scope.get("type") == "http" and scope.get("method") not in ("GET", "HEAD", "OPTIONS") and tenant_writer.effect_blocked(user):
            await self._refuse_effect_blocked(scope, receive, send)
            return
        server = self.server or current()
        writer = self._writer_tenant(scope)
        if writer:
            await self._write(writer, server, scope, receive, send)
            return
        # A read with a body is never forwarded or read here: the app handles it as today.
        if not self._wants(scope, server) or self._declares_body(scope):
            await self._serve_here(scope, receive, send)
            return
        deadline = time.monotonic() + FORWARD_DEADLINE_S
        reply = await self._forward(server, scope, deadline)
        if reply == "unconfirmed":
            await self._reply(send, 503, UNCONFIRMED, [(b"retry-after", b"1")])
            return
        if reply is None:
            await self._after_decline(scope, receive, send)
            return
        status, headers, body = reply
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    # ---- D5 writes ----

    async def _write(self, user, server, scope, receive, send):
        # One deadline for the wait: a write never queues longer than a forward may take. The wait
        # also covers the admission barrier: a lost child not yet confirmed exited (RESTARTING).
        try:
            lock = await tenant_writer.acquire_async(user, until=time.monotonic() + tenant_writer.DEADLINE_S)
        except tenant_writer.WriterBusy as exc:
            await self._reply(send, 503, RESTARTING if exc.reason == "restarting" else BUSY, [(b"retry-after", b"1")])
            return
        try:
            await self._write_locked(user, server, scope, receive, send)
        finally:
            # Synchronous: runs to completion even when the request task is cancelled. Anything
            # that could leave the child writing has raised the admission barrier before this.
            lock.release()

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

    async def _collect(self, receive, deadline):
        """The whole request body, bounded: (body, None), or (None, reason) with reason one of
        "too_large", "timeout" or "disconnected". Every path of a writer tenant's write uses it, so
        no write of the tenant runs on either process with an unbounded or unfinished body."""
        chunks, size = [], 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "timeout"
            try:
                message = await asyncio.wait_for(receive(), remaining)
            except asyncio.TimeoutError:
                return None, "timeout"
            if message.get("type") != "http.request":
                return None, "disconnected"
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_WRITE_BODY:
                return None, "too_large"
            chunks.append(chunk)
            if not message.get("more_body", False):
                return b"".join(chunks), None

    async def _write_locked(self, user, server, scope, receive, send):
        declared = self._declared_length(scope)
        if declared is not None and declared > MAX_WRITE_BODY:
            # Never read; Python's own bound for a buffered body (writes of this tenant never run
            # on Python with an unbounded body).
            await self._reply(send, 413, TOO_LARGE)
            return
        # One deadline from the lock: the body, then the forward.
        deadline = time.monotonic() + FORWARD_DEADLINE_S
        body, failure = await self._collect(receive, deadline)
        if failure == "too_large":
            await self._reply(send, 413, TOO_LARGE)
            return
        if failure == "timeout":
            # Never handed on unfinished: a stalled sender would hold the tenant's lock.
            await self._reply(send, 408, BODY_TIMEOUT)
            return
        if failure == "disconnected":
            return  # nobody to answer; nothing was written

        sent = False

        async def replay():
            # The whole body once; after it the client can only disconnect.
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        if server is None or not server.ready or not self._json_media(scope):
            # The Rust process is not running (nothing of it can still write: the admission
            # barrier held this lock closed until any lost child was confirmed exited), or the
            # body is not JSON (the app answers 422 without writing).
            await self._serve_here(scope, replay, send, body)
            return
        try:
            outcome = await self._forward(server, scope, deadline, body=body, writer=True)
        except asyncio.CancelledError:
            # The request may have reached the child, which may still be writing: the same as a
            # lost reply. The child is killed (the barrier is up before this lock is released) and
            # the tenant recovers before its next write.
            if scope.get("path") != "/api/storage-backup":
                self._uncertain(server, user)
            # Backup retains its own flock and native writer until transport settles.
            # Killing the process here could unlock while a remote PUT still commits.
            raise
        if outcome == "uncertain":
            self.on_uncertain(user)
            await self._reply(send, 503, UNCERTAIN)
            return
        if outcome is None:
            await self._after_decline(scope, replay, send, body)
            return
        status, headers, content = outcome
        # Native replay and writes leave durable dirty marks for Python retrieval.
        # Invalidate the cached recovery state before any later read can skip indexing.
        self.on_uncertain(user)
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": content})

    def _uncertain(self, server, user) -> None:
        try:
            server.lost()
        finally:
            self.on_uncertain(user)

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
        effective_user = self._user(scope) or os.environ.get("DIARY_LEGACY_USER_ID", "")
        secret = server.forward_secret_for(effective_user)
        if not secret:
            return None
        state = scope.setdefault("state", {})
        state.pop("rust_verified", None)
        claim = self._claim(scope)
        delegated = self._delegated(scope) and not state.get("tenant_asserted")
        if delegated and not tenant_assertion.holds(claim):
            # Not this request's nonce to send for verification (never reserved, or settled).
            return None
        asserted = bool(state.get("tenant_asserted"))
        user = self._user(scope)
        recovered = bool(user) and self.is_recovered(user.lower())
        headers = [(k, v) for k, v in scope.get("headers") or []
                   if k.lower() not in _HOP and not k.lower().startswith(INTERNAL_PREFIX)]
        headers += [(FORWARD_HEADER, secret.encode()), (ASSERTED_HEADER, b"1" if asserted else b"0"),
                    (RECOVERED_HEADER, b"1" if recovered else b"0")]
        if delegated:
            headers.append((VERIFY_HEADER, b"1"))
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

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Never dispatched: the claim (if any) is released by __call__. A write keeps D5's
            # conservative answer (UNCERTAIN), exactly as before.
            log.warning("Rust Diary server did not answer (TimeoutError); falling back to Python")
            if scope.get("path") != "/api/storage-backup":
                server.lost()  # a backup keeps its settling child (as below)
            return "uncertain" if writer else None
        proc = getattr(server, "_proc", None)
        if delegated:
            state["rust_dispatched"] = True
        task = asyncio.ensure_future(exchange())
        try:
            resp, content = await asyncio.wait_for(asyncio.shield(task), remaining)
        except asyncio.CancelledError:
            # The caller went away: the exchange keeps running on its own and settles the claim from
            # its outcome; until then (and if it never answers) the nonce stays reserved.
            if delegated:
                _detach(task, claim)
            else:
                task.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - lost or stuck process: Python serves from here on
            log.warning("Rust Diary server did not answer (%s); falling back to Python", type(exc).__name__)
            not_sent = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
            exited = False
            try:
                if delegated:
                    if not_sent:
                        # Definitely never reached the child: nothing was accepted.
                        tenant_assertion.release(claim)
                    else:
                        # Unknown: keep the reservation; a late authoritative answer still settles it.
                        _detach(task, claim)
                        if not writer:
                            exited = await _exited(proc)
                else:
                    task.cancel()
            finally:
                # Cancellation inside the read's exit probe still retires this child. Writes
                # skip that probe and raise the D5 barrier without awaiting an unused verdict.
                # A backup keeps its child: its flock and native writer stay held until the
                # remote transport settles (killing it could unlock while a PUT still commits).
                if scope.get("path") != "/api/storage-backup":
                    server.lost()
            if writer and not not_sent:
                # The request may have reached the child, which may have committed it.
                return "uncertain"
            if delegated and not not_sent and not exited:
                # The child may have accepted (and served) it: never served a second time here.
                return "unconfirmed"
            return None
        raw = resp.headers.raw
        if delegated:
            if _settle(claim, raw):
                state["rust_verified"] = True
        if any(k.lower() == DECLINE_HEADER for k, _ in raw):
            return None
        kept = [(k.lower(), v) for k, v in raw
                if k.lower() not in _DROP_REPLY and not k.lower().startswith(INTERNAL_PREFIX)]
        # Starlette's order: the response's own headers, then content-length, then content-type.
        order = {b"content-length": 1, b"content-type": 2}
        kept.sort(key=lambda kv: order.get(kv[0], 0))
        return resp.status_code, kept, content


async def _exited(proc) -> bool:
    """Has the child process exited (checked for at most EXIT_PROBE_S)? Unknown counts as no."""
    if proc is None:
        return False
    deadline = time.monotonic() + EXIT_PROBE_S
    while True:
        try:
            if proc.poll() is not None:
                return True
        except Exception:  # noqa: BLE001
            return False
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.01)


def _settle(claim, raw_headers) -> bool:
    """An authoritative reply from the child for a delegated request: commit the reservation when it
    carries the verified marker (True), else release it (the child did not accept the nonce)."""
    if any(k.lower() == VERIFIED_HEADER and v == b"1" for k, v in raw_headers):
        tenant_assertion.commit(claim)
        return True
    tenant_assertion.release(claim)
    return False


# Exchanges whose caller stopped waiting (cancelled, timed out, transport error) but that still own a
# reservation: kept referenced until they finish.
_detached: set = set()


def _detach(task, claim) -> None:
    """Hand a dispatched exchange its claim: a reply that still arrives settles it (commit/release);
    a failure or cancellation leaves it reserved until its validity horizon. Never releases early."""
    def done(t):
        _detached.discard(t)
        if t.cancelled() or t.exception() is not None:
            return
        resp, _ = t.result()
        _settle(claim, resp.headers.raw)
    _detached.add(task)
    task.add_done_callback(done)


_server: Optional[RustServer] = None


def start(config_factory: Callable[[int, str, str], dict], routes: dict, writers=frozenset(),
          answering: bool = False) -> Optional[RustServer]:
    """Lifespan hook: start the Rust process when any route or writer tenant is configured, or (R9)
    the answer shadow is on (rust_answer)."""
    global _server
    if not routes and not writers and not answering:
        return None
    _server = RustServer(config_factory)
    # Started only for the read-only answer shadow: it never writes, so deletion does not wait on it.
    _server.answer_only = not routes and not writers
    _server.start()
    return _server


def stop() -> None:
    global _server
    if _server is not None:
        _server.stop()
        _server = None


def current() -> Optional[RustServer]:
    return _server


def adopt(server) -> None:
    """R9 main mode (rust_parent): the Rust parent is the process the internal calls reach."""
    global _server
    _server = server


def fence_tenant(user_id: str) -> None:
    """DELETE /api/internal/tenant, after Python's tombstone: the Rust side stops writing the tenant
    (and finishes the write in flight) before its root is removed."""
    server = current()
    if server is not None and not getattr(server, "answer_only", False) and not server.prepare_delete(user_id):
        from fastapi import HTTPException
        raise HTTPException(503, "Diary saving did not confirm the tenant fence. Deletion remains paused.")
