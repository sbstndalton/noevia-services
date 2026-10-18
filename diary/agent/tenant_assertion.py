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

Checks: the signature (constant-time), |now - ts| <= SKEW_S and the bounds on every field are the Rust leaf's
(`tenant-assertion check`, below). This module keeps what the leaf cannot know: the shape of the header (its
timestamp and nonce feed the next two checks), ts not before process start - RESTART_SKEW_S (the cache is
memory-only; this closes the post-restart replay window), and each nonce accepted once while its timestamp is
inside the window (in-process cache; the sidecar runs one uvicorn worker). The cache is bucketed by the SIGNED timestamp
and bounded (_nonce_cap). A bucket leaves the cache when its second is outside
the clock window, or, when the cache is full, the second farthest from now first (#1680: the
old cache refused every new request with 503 above ~833 signed req/s). Either
way a past second raises a floor and any assertion with ts <= floor is refused
(a future second is remembered exactly instead, so it cannot raise the floor),
so a replay of an evicted nonce, or one after the clock steps back, is still
caught; a fresh request (ts at the newest second) is never refused, so the
bound is _nonce_cap plus one second of throughput. Under
overload the floor only shortens how late a signed request may arrive. The key
is read per call so an operator change takes effect on restart and tests can
patch the environment.

The signature, clock window and bounds are checked by the bounded Rust leaf from sbstndalton/noevia-rs
(`tenant-assertion check`, built into the image at the Dockerfile's pinned NOEVIA_RS_REF) and by nothing else:
since #1071 there is no Python copy of those rules and no TENANT_ASSERTION_IMPL switch (it is retired: `rust`
and empty are silent, any other value gets one warning at start). It FAILS CLOSED: a request is accepted only when
the leaf accepts; a refusal, a missing binary, a nonzero exit, a timeout or any unexpected output rejects it, and
none of those is ever cached or remembered as accepted. The key, the headers and the secret travel to the child on
stdin only, with a minimal environment; reasons and logs carry codes only. The shared differential fixtures
(tests/fixtures/tenant-assertion.v1.json, generated in noevia-rs from the former Python module) and the frozen
Python answers (tests/fixtures/tenant-assertion-py.v1.json) are the contract between the two.
"""
from __future__ import annotations

import base64
import bisect
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Dict, List, Mapping, Optional, Set

HEADER = "X-Cowork-Tenant-Assertion"
LABEL = "cowork-diary-tenant-v2"
SKEW_S = 60
# Refuse assertions signed before this process started (minus RESTART_SKEW_S): the nonce cache is
# memory-only, so without this a restart would let a captured assertion from the last SKEW_S seconds
# be replayed once (#1811). Web signs a fresh assertion per request, so a legitimate ts is >= the
# second this process started in; 1 s covers ts truncation and rounding, the containers share a clock.
RESTART_SKEW_S = 1
STREAM = "stream"
_nonce_cap = 100_000
_ASSERTION_RE = re.compile(r"v2\.(\d{1,12})\.([0-9a-f]{32})\.([A-Za-z0-9_-]{43})")

# Signed second -> nonces seen with it; _seconds is the sorted key list. _floor: assertions
# with ts <= _floor are refused (their nonces may have been dropped from the cache).
_seen: Dict[int, Set[str]] = {}
_seconds: List[int] = []
_count = 0
_floor = -1
# Future seconds whose bucket was evicted under overload: a ts equal to one is refused (they cannot
# be represented by the floor). At most SKEW_S entries; pruned when outside the window.
_lost: Set[int] = set()
_started_at: Optional[float] = time.time()
_seen_lock = threading.Lock()


RUST_TIMEOUT_S = 0.5
# At most this many leaf children run at once, process-wide. A request that finds every slot taken is refused at
# once (the refusal a bad signature gets), never queued: the leaf now decides every signature, so without a bound a
# flood of bad ones would spawn without limit (each costs a few ms and lives at most RUST_TIMEOUT_S).
MAX_CONCURRENT_LEAVES = 4
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_LEAVES)
_busy_refusals = 0
_busy_lock = threading.Lock()
DEFAULT_BIN = "/usr/local/bin/tenant-assertion"
_RUST_STDOUT_CAP = 256
_log = logging.getLogger(__name__)
_logged: set = set()
_logged_lock = threading.Lock()


class NonceCacheFull(Exception):
    """No longer raised (#1680); kept so older callers that name it still import."""


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


def _drop_second(second: int, now: float) -> None:
    """Drop one bucket. A past/current second raises the floor to it (ts <= floor is refused); a
    FUTURE second is remembered in _lost instead, so a flood of future-dated assertions cannot raise
    the floor above `now` and lock out legitimate traffic (#1811)."""
    global _count, _floor
    _seconds.remove(second)
    _count -= len(_seen.pop(second))
    if second > now:
        _lost.add(second)
    else:
        _floor = max(_floor, second)


def _remember_nonce(nonce: str, ts: int, now: float) -> Optional[str]:
    """None the first time (nonce, ts) is seen inside the window, else a reason. Never refuses
    a fresh request for capacity: when full it drops the bucket farthest from `now` (never the one
    just written), so a future-dated flood evicts itself rather than live traffic."""
    global _count
    with _seen_lock:
        # Seconds that can no longer pass |now - ts| <= SKEW_S are dead; dropping them raises the
        # floor too, so a backwards clock step cannot revive their nonces.
        while _seconds and _seconds[0] < now - SKEW_S:
            _drop_second(_seconds[0], now)
        for second in [x for x in _lost if x < now - SKEW_S]:
            _lost.discard(second)
        if ts <= _floor or ts in _lost:
            return "replayed or older than the replay cache"
        bucket = _seen.get(ts)
        if bucket is None:
            bucket = _seen[ts] = set()
            bisect.insort(_seconds, ts)
        elif nonce in bucket:
            return "replayed"
        bucket.add(nonce)
        _count += 1
        # The farthest second is always the oldest or the newest. The bucket just written is kept:
        # the signed ts has one-second resolution, so dropping it would refuse the rest of that
        # second's fresh requests. The cache is therefore bounded by _nonce_cap plus one second.
        while _count > _nonce_cap and len(_seconds) > 1:
            ends = [x for x in (_seconds[0], _seconds[-1]) if x != ts]
            _drop_second(max(ends, key=lambda x: abs(x - now)), now)
        return None


def _nonce_count() -> int:
    with _seen_lock:
        return _count


def _log_once(key: str, message: str) -> None:
    with _logged_lock:
        if key in _logged:
            return
        _logged.add(key)
    _log.warning(message)


def startup_checks() -> None:
    """Called once when the service starts: a leftover TENANT_ASSERTION_IMPL other than `rust` is ignored with one
    warning (the Rust leaf is always used), and a tenant key with no usable binary is an error in the log (every
    tenant request is refused until the binary is there)."""
    value = (os.environ.get("TENANT_ASSERTION_IMPL") or "").strip().lower()
    if value not in ("", "rust"):
        _log_once("retired_setting", f"TENANT_ASSERTION_IMPL={value[:32]!r} is retired; Rust is always used")
    if tenant_key() and _rust_binary() is None:
        _log_once("rust:missing_binary_at_start", "DIARY_TENANT_KEY is set but the tenant-assertion binary is not usable; refusing every tenant request")


def _rust_binary() -> Optional[str]:
    # A fixed absolute path by default (never a PATH lookup). TENANT_ASSERTION_BIN (tests, operators)
    # is honoured only as an absolute path to an existing, executable regular file; anything else,
    # including a bare name, is ignored with one warning and the default is used (#1832, the same
    # rule as WORKSPACE_RESTORE_VERIFY_BIN in workspace_restore.py).
    configured = (os.environ.get("TENANT_ASSERTION_BIN") or "").strip()
    binary = DEFAULT_BIN
    if configured:
        if os.path.isabs(configured) and os.path.isfile(configured) and os.access(configured, os.X_OK):
            binary = configured
        else:
            _log_once("rust:bad_override", "TENANT_ASSERTION_BIN is not an absolute path to an executable file; ignoring it")
    return binary if os.path.isfile(binary) and os.access(binary, os.X_OK) else None


def _busy_refusal() -> str:
    """Every slot is taken: refuse at once, logged at the 1st, 10th, 100th, 1000th ... refusal, with no input."""
    global _busy_refusals
    with _busy_lock:
        _busy_refusals += 1
        n = _busy_refusals
    if n == 1 or (n >= 10 and re.fullmatch(r"10*", str(n))):
        _log.warning("tenant-assertion: all %d leaf slots are busy; refusing the request%s", MAX_CONCURRENT_LEAVES,
                     f" ({n} times so far)" if n > 1 else "")
    return "rust refused"


def _rust_decision(payload: dict) -> Optional[str]:
    """None when `tenant-assertion check` accepts `payload`, else a reason code. Fails closed:
    every fault is a refusal, and so is finding every slot taken (MAX_CONCURRENT_LEAVES). The payload
    (key included) goes to the child on stdin only."""
    if not _slots.acquire(blocking=False):
        return _busy_refusal()
    try:
        return _run_leaf(payload)
    finally:
        _slots.release()


def _run_leaf(payload: dict) -> Optional[str]:
    binary = _rust_binary()
    if binary is None:
        _log_once("rust:missing_binary", "tenant-assertion binary not found; refusing tenant requests")
        return "rust unavailable"
    try:
        data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    except (TypeError, ValueError):
        return "rust unavailable"
    try:
        proc = subprocess.Popen([binary, "check"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, close_fds=True,
                                env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")})
    except OSError as e:
        _log_once("rust:spawn", f"tenant-assertion could not start ({type(e).__name__}); refusing the request")
        return "rust unavailable"
    out = bytearray()
    overflow = threading.Event()

    def feed():
        try:
            proc.stdin.write(data)
        except (OSError, ValueError):
            pass  # The child exited early; its exit status decides.
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def drain():
        # Read at most cap+1 bytes, then stop; the child is killed on overflow.
        try:
            while len(out) <= _RUST_STDOUT_CAP:
                chunk = proc.stdout.read1(_RUST_STDOUT_CAP + 1 - len(out))
                if not chunk:
                    return
                out.extend(chunk)
            overflow.set()
        except (OSError, ValueError):
            return

    threads = [threading.Thread(target=feed, daemon=True), threading.Thread(target=drain, daemon=True)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + RUST_TIMEOUT_S
    fault = None
    while True:
        try:
            code = proc.wait(timeout=0.01)
            break
        except subprocess.TimeoutExpired:
            if overflow.is_set():
                fault = "output_too_large"
            elif time.monotonic() >= deadline:
                fault = "timeout"
            if fault:
                proc.kill()
                proc.wait()
                break
    for t in threads:
        t.join(timeout=0.2)
    try:
        proc.stdout.close()
    except OSError:
        pass
    if fault or overflow.is_set():
        fault = fault or "output_too_large"
        _log_once(f"rust:{fault}", f"tenant-assertion failed ({fault}); refusing the request")
        return "rust unavailable"
    out = bytes(out)
    if code == 0 and out == b"accept\n":
        return None
    if code == 1 and out.startswith(b"reject: ") and len(out) <= _RUST_STDOUT_CAP:
        return "rust refused"
    # Exit 2/3, a signal or unexpected output: a fault, never an acceptance. stderr is not logged
    # (it is a fixed string anyway; nothing of the request is in it).
    _log_once(f"rust:exit:{code}", f"tenant-assertion failed (exit {code}); refusing the request")
    return "rust unavailable"


def _check(key: str, headers: Mapping[str, str], method: str, path: str, now: float, query: bytes,
           body_hash: Optional[str]) -> Optional[str]:
    """None when the assertion passes everything that needs no cache, else a reason. The shape and the
    restart guard are this module's (they are cheap and need no secret, so they come before the child is
    spawned); the signature, the clock window and the bounds are the Rust leaf's alone."""
    user_id = headers.get("X-Cowork-User-ID", "")
    raw = headers.get(HEADER, "")
    if not user_id:
        return "missing tenant"
    match = _ASSERTION_RE.fullmatch(raw or "")
    if not match:
        return "missing or malformed assertion"
    ts = int(match.group(1))
    if _started_at is not None and ts < int(_started_at) - RESTART_SKEW_S:
        return "signed before this process started"
    return _rust_decision({
        "op": "verify", "key": key, "user_id": user_id, "assertion": raw, "method": method, "path": path,
        "query_hex": bytes(query).hex(), "body_hash": sha256_hex(b"") if body_hash is None else body_hash,
        "storage": headers.get("X-Cowork-Storage", ""), "legacy_owner": headers.get("X-Cowork-Legacy-Owner", ""),
        "blocked": headers.get("X-Cowork-Storage-Blocked", ""), "now": repr(float(now))})


def verify(key: str, headers: Mapping[str, str], method: str, path: str, now: Optional[float] = None,
           query: bytes = b"", body_hash: Optional[str] = None) -> Optional[str]:
    """None when the request carries a valid assertion for its X-Cowork-User-ID, else a reason.

    `path` is the encoded wire path, `query` the raw query string, `body_hash`
    sha256 hex of the body or STREAM (see module doc). Reasons are for logs
    only; callers answer every failure the same way. The Rust leaf must accept
    (module doc), then the nonce cache runs: a nonce is remembered only for an
    assertion everything else accepted. Spawns a child: call from a thread."""
    now = time.time() if now is None else now
    reason = _check(key, headers, method, path, now, query, body_hash)
    if reason:
        return reason
    match = _ASSERTION_RE.fullmatch(headers.get(HEADER, "") or "")
    return _remember_nonce(match.group(2), int(match.group(1)), now)


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
    """app.py's secretRef check: `ref` equals the derivation (storage_secret_ref), decided by the Rust leaf
    alone; any fault is a mismatch (a non-ASCII `ref` is a mismatch too, not an error)."""
    return _rust_decision({"op": "secret_ref", "key": key, "user_id": user_id, "secret": secret, "ref": ref}) is None


def _reset_busy_for_tests() -> None:
    global _busy_refusals, _slots
    with _busy_lock:
        _busy_refusals = 0
    _slots = threading.BoundedSemaphore(MAX_CONCURRENT_LEAVES)


def _reset_for_tests(started_at: Optional[float] = None) -> None:
    """Clear the cache. The restart guard is off unless `started_at` is given (tests sign with
    synthetic clocks)."""
    global _count, _floor, _started_at
    with _seen_lock:
        _lost.clear()
        _started_at = started_at
        _seen.clear()
        _seconds.clear()
        _count = 0
        _floor = -1
