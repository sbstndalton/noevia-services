"""R9 main mode: this app as the CHILD of the Rust Diary process (DIARY_MAIN=rust in entrypoint.sh).

Unset (the default) changes nothing: this module is inert unless DIARY_PARENT=rust, which only the
Rust main process (noevia-rs bins/noevia-diary front.rs, `noevia-diary --main`) sets on its child.

In child mode the Rust process listens on the public port and this app on 127.0.0.1:8012. The Rust
side is the only tenant-assertion verifier and nonce owner; it serves the read routes this process
names in DIARY_RUST_ROUTES natively and forwards everything else here. So, in child mode:
- the parent sends one JSON line on stdin: ``control_token``, ``parent_secret``, ``control_port``;
- this app POSTs its start-up config (`rust_forward.server_config`, plus ``native_routes``) to the
  parent's control port at start-up, and refuses to start when the parent does not take it;
- ParentMiddleware replaces TenantAssertionMiddleware and ForwardMiddleware: every request must carry
  the parent secret (else 401 with the middleware's own body), the parent's verdict
  (``x-noevia-diary-asserted: 1``) becomes ``tenant_asserted``, and every reply to a request naming
  a UUID tenant is stamped ``x-noevia-diary-recovered: 1|0`` (does this process hold a recovered
  managed state for it?), which is what lets the parent serve that tenant's reads natively;
- no Rust child is started; `rust_forward.current()` is the parent (ParentServer), so the answer
  endpoint (DIARY_RUST_ANSWER), the index shadow (DIARY_RUST_INDEX=shadow) and the deletion fence
  reach the parent's control port. DIARY_RUST_WRITES is ignored: every write is served here under
  this process's own tenant writer lock (the lock moves to the parent in a later step), so
  DIARY_RUST_INDEX=on has nothing to hand over either.
The parent never restarts this process: when it exits the parent exits and the container restarts.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import sys
import time
from typing import Callable, Optional

import httpx

log = logging.getLogger("diary.rust")

PARENT_HEADER = b"x-noevia-diary-parent"
ASSERTED_HEADER = b"x-noevia-diary-asserted"
RECOVERED_HEADER = b"x-noevia-diary-recovered"
CONTROL_HEADER = "x-noevia-diary-control"
INTERNAL_PREFIX = b"x-noevia-diary-"
UNVERIFIED = b'{"detail":"invalid tenant assertion"}'
CONFIG_DEADLINE_S = 30.0
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def active() -> bool:
    """Child mode: the Rust main process started this app."""
    return (os.environ.get("DIARY_PARENT") or "").strip() == "rust"


def read_handshake(stream=None) -> dict:
    """The parent's one stdin line. Raises (the app then refuses to start) when it is missing."""
    line = (stream or sys.stdin).readline()
    data = json.loads(line)
    token, secret, port = data.get("control_token"), data.get("parent_secret"), data.get("control_port")
    if not (isinstance(token, str) and token and isinstance(secret, str) and secret
            and isinstance(port, int) and 0 < port < 65536):
        raise ValueError("the Rust parent's handshake is incomplete")
    return {"control_token": token, "parent_secret": secret, "control_port": port}


class ParentServer:
    """`rust_forward.RustServer`'s surface for the internal callers (rust_answer, rust_index, the
    deletion fence), pointing at the parent's control port. Always ready while this process runs."""

    def __init__(self, control_token: str, parent_secret: str, control_port: int):
        self._token = control_token
        self.forward_secret = parent_secret
        self.port = control_port
        self.base_url = f"http://127.0.0.1:{control_port}"
        self.ready = True
        self.restarts = 0
        self.reaping = 0
        self.answer_only = False

    def control(self):
        return self._token, True

    def forward_secret_for(self, _user_id: str) -> str:
        return self.forward_secret

    def fence(self, user_id: str, timeout: float = 30.0) -> bool:
        try:
            with httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=False) as client:
                resp = client.post("/internal/fence", params={"tenant": user_id},
                                   headers={CONTROL_HEADER: self._token})
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def prepare_delete(self, user_id: str) -> bool:
        return self.fence(user_id)

    def lost(self) -> None:
        # The parent is the container's main process: it is never restarted from here.
        log.warning("Rust Diary parent did not answer an internal call")

    def stop(self) -> None:
        pass

    def send_config(self, config: dict, deadline_s: float = CONFIG_DEADLINE_S,
                    transport: Optional[httpx.BaseTransport] = None) -> None:
        """POST the start-up config until the parent takes it (200) or refuses it (raises)."""
        body = json.dumps(config, separators=(",", ":"))
        until = time.monotonic() + deadline_s
        with httpx.Client(base_url=self.base_url, timeout=5.0, trust_env=False, transport=transport) as client:
            while True:
                try:
                    resp = client.post("/internal/child/config", content=body,
                                       headers={CONTROL_HEADER: self._token, "content-type": "application/json"})
                    if resp.status_code == 200:
                        return
                    raise RuntimeError(f"the Rust parent refused the start-up config ({resp.status_code})")
                except httpx.TransportError:
                    if time.monotonic() >= until:
                        raise RuntimeError("the Rust parent did not take the start-up config in time") from None
                    time.sleep(0.1)


class ParentMiddleware:
    """Pure ASGI, outermost in child mode (see the module doc)."""

    def __init__(self, app, secret: Callable[[], str], is_recovered: Callable[[str], bool]):
        self.app = app
        self.secret = secret
        self.is_recovered = is_recovered

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        supplied, asserted, kept, user = b"", False, [], ""
        for name, value in scope.get("headers") or []:
            lower = name.lower()
            if lower == PARENT_HEADER:
                supplied = value
            elif lower == ASSERTED_HEADER:
                asserted = value == b"1"
            elif lower.startswith(INTERNAL_PREFIX):
                continue
            else:
                if lower == b"x-cowork-user-id":
                    user = value.decode("latin-1").lower()
                kept.append((name, value))
        expected = self.secret().encode()
        if not expected or not hmac.compare_digest(supplied, expected):
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-length", str(len(UNVERIFIED)).encode()),
                                    (b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": UNVERIFIED})
            return
        scope = dict(scope)
        scope["headers"] = kept
        state = scope.setdefault("state", {})
        if asserted:
            state["tenant_asserted"] = True
        tenant = user if _UUID_RE.fullmatch(user) else ""

        async def stamped(message):
            if tenant and message.get("type") == "http.response.start":
                message = dict(message)
                message["headers"] = [*(message.get("headers") or []),
                                      (RECOVERED_HEADER, b"1" if self.is_recovered(tenant) else b"0")]
            await send(message)

        await self.app(scope, receive, stamped)
