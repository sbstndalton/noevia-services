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

TENANT_ASSERTION_IMPL=rust additionally runs the same check in the bounded Rust leaf from
sbstndalton/noevia-rs (`tenant-assertion check`, built into the image at the Dockerfile's pinned
NOEVIA_RS_REF). It is AND-composed and FAILS CLOSED: a request is accepted only when Python
accepts AND Rust accepts; a Rust refusal, a missing binary, a nonzero exit, a timeout or any
unexpected output rejects it. The default is python (no subprocess); any other value is python
with one warning. The key, the headers and the secret travel to the child on stdin only, with a
minimal environment; reasons and logs carry codes only. The shared differential fixtures
(tests/fixtures/tenant-assertion.v1.json, generated in noevia-rs from this module) are the
contract between the two.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
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


IMPLS = ("python", "rust")
RUST_TIMEOUT_S = 2.0
_RUST_STDOUT_CAP = 256
_log = logging.getLogger(__name__)
_logged: set = set()
_logged_lock = threading.Lock()


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


def _log_once(key: str, message: str) -> None:
    with _logged_lock:
        if key in _logged:
            return
        _logged.add(key)
    _log.warning(message)


def impl_choice() -> str:
    """The configured implementation: "python" (default) or "rust". Anything else is python."""
    value = (os.environ.get("TENANT_ASSERTION_IMPL") or "python").strip().lower() or "python"
    if value not in IMPLS:
        # The value is an operator setting, never a secret; still, log only its length-bounded repr.
        _log_once("invalid_setting", f"TENANT_ASSERTION_IMPL={value[:32]!r} is not one of {', '.join(IMPLS)}; using python")
        return "python"
    return value


def _rust_binary() -> Optional[str]:
    configured = (os.environ.get("TENANT_ASSERTION_BIN") or "tenant-assertion").strip() or "tenant-assertion"
    if os.sep in configured:
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None
    return shutil.which(configured)


def _rust_decision(payload: dict) -> Optional[str]:
    """None when `tenant-assertion check` accepts `payload`, else a reason code. Fails closed:
    every fault is a refusal. The payload (key included) goes to the child on stdin only."""
    binary = _rust_binary()
    if binary is None:
        _log_once("rust:missing_binary", "tenant-assertion binary not found; refusing tenant requests (TENANT_ASSERTION_IMPL=rust)")
        return "rust unavailable"
    try:
        data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    except (TypeError, ValueError):
        return "rust unavailable"
    try:
        proc = subprocess.run([binary, "check"], input=data, capture_output=True, timeout=RUST_TIMEOUT_S,
                              close_fds=True, env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")})
    except subprocess.TimeoutExpired:
        _log_once("rust:timeout", "tenant-assertion timed out; refusing the request")
        return "rust unavailable"
    except OSError as e:
        _log_once("rust:spawn", f"tenant-assertion could not start ({type(e).__name__}); refusing the request")
        return "rust unavailable"
    out = proc.stdout[:_RUST_STDOUT_CAP + 1]
    if proc.returncode == 0 and out == b"accept\n":
        return None
    if proc.returncode == 1 and out.startswith(b"reject: ") and len(out) <= _RUST_STDOUT_CAP:
        return "rust refused"
    # Exit 2/3, a signal or unexpected output: a fault, never an acceptance. stderr is not logged
    # (it is a fixed string anyway; nothing of the request is in it).
    _log_once(f"rust:exit:{proc.returncode}", f"tenant-assertion failed (exit {proc.returncode}); refusing the request")
    return "rust unavailable"


def _check(key: str, headers: Mapping[str, str], method: str, path: str, now: float, query: bytes,
           body_hash: Optional[str]) -> Optional[str]:
    """Python's stateless decision: None when the assertion is valid, else a reason."""
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
    return None


def verify(key: str, headers: Mapping[str, str], method: str, path: str, now: Optional[float] = None,
           query: bytes = b"", body_hash: Optional[str] = None) -> Optional[str]:
    """None when the request carries a valid assertion for its X-Cowork-User-ID, else a reason.

    `path` is the encoded wire path, `query` the raw query string, `body_hash`
    sha256 hex of the body or STREAM (see module doc). Reasons are for logs
    only; callers answer every failure the same way. Raises NonceCacheFull.
    Under TENANT_ASSERTION_IMPL=rust the Rust check must accept as well (module doc)."""
    now = time.time() if now is None else now
    reason = _check(key, headers, method, path, now, query, body_hash)
    if reason:
        return reason
    if impl_choice() == "rust":
        reason = _rust_decision({
            "op": "verify", "key": key, "user_id": headers.get("X-Cowork-User-ID", ""),
            "assertion": headers.get(HEADER, ""), "method": method, "path": path,
            "query_hex": bytes(query).hex(), "body_hash": sha256_hex(b"") if body_hash is None else body_hash,
            "storage": headers.get("X-Cowork-Storage", ""), "legacy_owner": headers.get("X-Cowork-Legacy-Owner", ""),
            "blocked": headers.get("X-Cowork-Storage-Blocked", ""), "now": repr(float(now))})
        if reason:
            return reason
    nonce = _ASSERTION_RE.fullmatch(headers.get(HEADER, "") or "").group(2)
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


def secret_ref_matches(key: str, user_id: str, secret: str, ref: str) -> bool:
    """app.py's secretRef check: `ref` equals storage_secret_ref (constant time). Under
    TENANT_ASSERTION_IMPL=rust the Rust check must agree as well; any fault is a mismatch."""
    if not hmac.compare_digest(ref, storage_secret_ref(key, user_id, secret)):
        return False
    if impl_choice() == "rust":
        return _rust_decision({"op": "secret_ref", "key": key, "user_id": user_id, "secret": secret, "ref": ref}) is None
    return True


def _reset_for_tests() -> None:
    with _seen_lock:
        _seen.clear()
