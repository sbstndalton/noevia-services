"""Per-request tenant assertion (docs/spec-service-boundaries.md §6.2 M2).

The shared DIARY_AUTH_TOKEN proves "this is the web server"; it does not prove
which tenant a request acts for. With DIARY_TENANT_KEY set, the web server signs
every tenant-scoped request (apps/web/server/diary-tenant-assertion.cjs) and
this module verifies it:

  X-Cowork-Tenant-Assertion: v2.<unix seconds>.<nonce hex>.<base64url HMAC-SHA256>

over the newline-joined canonical string
  "cowork-diary-tenant-v2", user id (lower case), ts, nonce, METHOD, path,
  "query=" + sha256(raw query string), "body=" + (sha256(raw body) | "stream"),
  sha256(X-Cowork-Storage or ""), X-Cowork-Legacy-Owner or "",
  X-Cowork-Storage-Blocked or ""

Path is the percent-ENCODED request-target path exactly as sent on the wire
(ASGI raw_path here, WHATWG URL.pathname in web), so both sides hash the same
bytes; the decoded scope["path"] is never signed. The query is the raw query
string without "?" (ASGI query_string, URL.search in web).

Body: a request whose Content-Type is application/json (or absent) must sign
sha256 of its exact body bytes (empty body: sha256 of ""). Any other media type
(ZIP upload, multipart) signs the literal "stream" instead, so its body is not
covered: in-flight tampering of such bodies is out of scope (see
docs/spec-managed-diary.md). A JSON call can never be signed as "stream".

Checks: constant-time signature compare, |now - ts| <= SKEW_S, and each nonce
accepted once while its timestamp is inside the window (in-process cache; the
sidecar runs one uvicorn worker). When the cache is full of unexpired nonces a
new request is refused (NonceCacheFull, answered 503) rather than evicting a
live nonce, which would reopen replay. The key is read per call so an operator
change takes effect on restart and tests can patch the environment.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Mapping, Optional

HEADER = "X-Cowork-Tenant-Assertion"
LABEL = "cowork-diary-tenant-v2"
SKEW_S = 60
STREAM = "stream"
_nonce_cap = 100_000
_ASSERTION_RE = re.compile(r"v2\.(\d{1,12})\.([0-9a-f]{32})\.([A-Za-z0-9_-]{43})")

_seen: "OrderedDict[str, float]" = OrderedDict()
_seen_lock = threading.Lock()


class NonceCacheFull(Exception):
    """Every cached nonce is still inside its window; refuse rather than evict."""


def tenant_key() -> str:
    return (os.environ.get("DIARY_TENANT_KEY") or "").strip()


def sha256_hex(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()


def body_is_hashed(content_type: str) -> bool:
    """JSON (or no Content-Type) bodies are always hashed; other media types sign STREAM."""
    media = (content_type or "").split(";", 1)[0].strip().lower()
    return media in ("", "application/json")


def canonical(user_id: str, ts: int, nonce: str, method: str, path: str, query_hash: str, body_hash: str,
              storage: str, legacy_owner: str, blocked: str) -> str:
    return "\n".join([LABEL, user_id.lower(), str(ts), nonce, method.upper(), path, "query=" + query_hash,
                      "body=" + body_hash, sha256_hex(storage), legacy_owner, blocked])


def sign(key: str, user_id: str, method: str, path: str, headers: Mapping[str, str], ts: int, nonce: str,
         query: bytes = b"", body_hash: Optional[str] = None) -> str:
    """Reference signer (tests, tooling). Web has its own in diary-tenant-assertion.cjs.

    body_hash defaults to sha256 of an empty body."""
    message = canonical(user_id, ts, nonce, method, path, sha256_hex(query), sha256_hex(b"") if body_hash is None else body_hash,
                        headers.get("X-Cowork-Storage", ""), headers.get("X-Cowork-Legacy-Owner", ""),
                        headers.get("X-Cowork-Storage-Blocked", ""))
    sig = base64.urlsafe_b64encode(hmac.new(key.encode(), message.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    return f"v2.{ts}.{nonce}.{sig}"


def _remember_nonce(nonce: str, now: float) -> bool:
    """False when the nonce was already used inside the window.

    Raises NonceCacheFull when no expired entry can make room."""
    with _seen_lock:
        # Insertion order is expiry order (constant window), so expired entries are at the front.
        while _seen:
            oldest, expiry = next(iter(_seen.items()))
            if expiry > now:
                break
            _seen.popitem(last=False)
        if nonce in _seen:
            return False
        if len(_seen) >= _nonce_cap:
            raise NonceCacheFull()
        # Expire after the latest moment the same timestamp could still verify.
        _seen[nonce] = now + 2 * SKEW_S
        return True


def verify(key: str, headers: Mapping[str, str], method: str, path: str, now: Optional[float] = None,
           query: bytes = b"", body_hash: Optional[str] = None) -> Optional[str]:
    """None when the request carries a valid assertion for its X-Cowork-User-ID, else a reason.

    `path` is the encoded wire path, `query` the raw query string, `body_hash`
    sha256 hex of the body or STREAM (see module doc). Reasons are for logs
    only; callers answer every failure the same way. Raises NonceCacheFull."""
    now = time.time() if now is None else now
    user_id = headers.get("X-Cowork-User-ID", "")
    raw = headers.get(HEADER, "")
    if not user_id:
        return "missing tenant"
    match = _ASSERTION_RE.fullmatch(raw or "")
    if not match:
        return "missing or malformed assertion"
    ts, nonce, _ = int(match.group(1)), match.group(2), match.group(3)
    expected = sign(key, user_id, method, path, headers, ts, nonce, query, body_hash)
    if not hmac.compare_digest(expected.encode(), raw.encode()):
        return "bad signature"
    if abs(now - ts) > SKEW_S:
        return "outside clock window"
    if not _remember_nonce(nonce, now):
        return "replayed"
    return None


def wire_path(scope) -> str:
    """The percent-encoded path as sent (raw_path, query stripped), falling back to the decoded path."""
    raw = scope.get("raw_path")
    if raw:
        return raw.decode("latin-1").split("?", 1)[0]
    return scope.get("path", "")


def storage_secret_ref(key: str, user_id: str, secret: str) -> str:
    """Same derivation as web's storageSecretRef (diary-tenant-assertion.cjs)."""
    msg = f"{LABEL}:storage-secret\n{user_id.lower()}\n{secret}"
    return hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()[:32]


def _reset_for_tests() -> None:
    with _seen_lock:
        _seen.clear()
