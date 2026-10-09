"""Back-off for a remote storage server that refuses or throttles us (#1166).

Nextcloud answers a rejected login with 401/403 and, after repeated failures, throttles the whole
source IP with 429 (brute-force protection), which also hurts every other client on that server.
So once storage has refused a credential we stop calling it for a cool-down, and after a 429 we
wait out Retry-After. The gate sits in the HTTP transport of the storage client, so every path
(list, month listing, read, write, search, background replay and reindex) obeys it with no
per-route code.

A gate is shared per credential fingerprint (server + account + secret hash): changing the
credentials gives a different fingerprint, so the new login is tried at once; an explicit retry
from the person resets a login cool-down immediately.
"""
from __future__ import annotations

import hashlib
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

LOGIN_COOLDOWN_S = 300.0
DEFAULT_RETRY_AFTER_S = 60.0
MAX_RETRY_AFTER_S = 3600.0
_MAX_GATES = 256
STORAGE_TAG = "noevia_storage"


class StorageBackoff(OSError):
    """Raised instead of calling a storage server that is in a cool-down. An OSError, so the write
    journal treats it as a transient outage and never quarantines an exchange because of it."""

    def __init__(self, kind: str, retry_after: float):
        self.kind = kind  # "login" | "throttle"
        self.retry_after = max(1, int(retry_after + 0.999))
        super().__init__(f"storage {kind} back-off, {self.retry_after}s remaining")


class StorageThrottled(Exception):
    """Storage asked us to slow down (429, or still inside its Retry-After). Answered with 503."""

    def __init__(self, retry_after: float):
        self.retry_after = max(1, int(retry_after))
        super().__init__(throttled_message(self.retry_after))


class StorageUpstreamError(Exception):
    """Storage answered 5xx or could not be reached. Answered with 502."""

    def __init__(self):
        super().__init__(UPSTREAM_MESSAGE)


def throttled_message(seconds: int) -> str:
    return f"Storage is busy and asked us to slow down. Try again in about {seconds} seconds."


UPSTREAM_MESSAGE = "The storage server returned an error or could not be reached. Try again shortly."


def parse_retry_after(value: Optional[str], default: float = DEFAULT_RETRY_AFTER_S) -> float:
    """Retry-After as seconds (delta or HTTP-date), clamped to 1..MAX; default when absent or garbage."""
    if value:
        text = value.strip()
        try:
            seconds = float(text)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(text).timestamp() - time.time()
            except (TypeError, ValueError, IndexError, OverflowError):
                return default
        if seconds == seconds:  # not NaN
            return min(max(seconds, 1.0), MAX_RETRY_AFTER_S)
    return default


class StorageGate:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._until = 0.0
        self._kind = ""

    def check(self) -> None:
        with self._lock:
            remaining = self._until - self._clock()
            if remaining > 0:
                raise StorageBackoff(self._kind, remaining)

    def record(self, status: int, retry_after: Optional[str] = None) -> None:
        with self._lock:
            now = self._clock()
            if status in (401, 403):
                self._kind, self._until = "login", now + LOGIN_COOLDOWN_S
            elif status == 429:
                until = now + parse_retry_after(retry_after)
                if self._kind == "login" and self._until > now:
                    return  # a login cool-down already covers it
                self._kind, self._until = "throttle", max(until, self._until if self._kind == "throttle" else 0.0)
            elif status < 400 and self._kind == "login":
                self._kind, self._until = "", 0.0  # an in-flight success after a rejection: the login works

    def reset_login(self) -> bool:
        """Explicit retry: clear a login cool-down (a throttle still has to be waited out)."""
        with self._lock:
            if self._kind == "login" and self._until > self._clock():
                self._kind, self._until = "", 0.0
                return True
            return False

    def state(self):
        with self._lock:
            remaining = self._until - self._clock()
            return (self._kind, remaining) if remaining > 0 else ("", 0.0)


_gates: "dict[str, StorageGate]" = {}
_gates_lock = threading.Lock()


def gate_for(*credential_parts: str) -> StorageGate:
    """The shared gate for one credential. Secrets are only hashed, never stored."""
    digest = hashlib.sha256("\x00".join(credential_parts).encode()).hexdigest()
    with _gates_lock:
        gate = _gates.get(digest)
        if gate is None:
            if len(_gates) >= _MAX_GATES:
                idle = [k for k, g in _gates.items() if g.state()[0] == ""]
                for key in idle[: _MAX_GATES // 2]:
                    del _gates[key]
            gate = _gates[digest] = StorageGate()
        return gate


class GatedTransport(httpx.BaseTransport):
    """Wraps a real transport: refuse while the gate is closed, learn from each answer, and tag the
    responses/errors that came from storage so the app's handlers map only those."""

    def __init__(self, inner: httpx.BaseTransport, gate: StorageGate):
        self._inner, self.gate = inner, gate

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.gate.check()
        try:
            response = self._inner.handle_request(request)
        except httpx.TransportError as exc:
            setattr(exc, STORAGE_TAG, True)
            raise
        self.gate.record(response.status_code, response.headers.get("Retry-After"))
        response.extensions[STORAGE_TAG] = True
        return response

    def close(self) -> None:
        self._inner.close()


def is_storage_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return bool(exc.response is not None and exc.response.extensions.get(STORAGE_TAG))
    return bool(getattr(exc, STORAGE_TAG, False))


def classify(exc: BaseException):
    """("login", None) | ("throttle", seconds) | ("upstream", None) for a storage failure, else None.

    The caller decides whether the exception is storage's (is_storage_error) or sits on a route
    that only talks to storage."""
    if isinstance(exc, StorageBackoff):
        return exc.kind, exc.retry_after
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        code = exc.response.status_code
        if code in (401, 403):
            return "login", None
        if code == 429:
            return "throttle", int(parse_retry_after(exc.response.headers.get("Retry-After")) + 0.999)
        return "upstream", None
    if isinstance(exc, httpx.HTTPError):
        return "upstream", None
    return None


def as_response_error(exc: BaseException, login_rejected):
    """The typed error the app answers for a storage failure, or None when `exc` is not one.

    `login_rejected` is the StorageLoginRejected class (kept in workspace_files, which owns its
    wording); passing it avoids a circular import."""
    found = classify(exc)
    if found is None:
        return None
    kind, seconds = found
    if kind == "login":
        return login_rejected()
    if kind == "throttle":
        return StorageThrottled(seconds)
    return StorageUpstreamError()
