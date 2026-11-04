"""D5: one writer per tenant across the Python and the Rust Diary processes.

For a tenant named in DIARY_RUST_WRITES ("a writer tenant") every write, by either process, runs
while this module's per-tenant lock is held, so at most one write of the tenant is in flight at any
moment. Forwarded writes take it in rust_forward.ForwardMiddleware and hold it until the reply;
Python's own writes take it structurally, not per call site:

- CorpusStore._write_lock is a TenantWriteLock: the tenant lock first, then the store's RLock.
  Every store write (log_exchange, edit_exchange, apply_pending, file writes, re-index passes,
  outbox drain, workspace ops/trash/import, storage import) already runs under it.
- ManagedCorpusBackend.migration_lock() takes the tenant lock before its RLock and flock.
- The few writes outside both (recovery, backup bookkeeping, the embedding re-index, tenant
  deletion, legacy slot adoption) call hold() directly.
- guard(conn) installs an SQLite authorizer on every connection Python opens to a writer tenant's
  managed-diary.db / managed-index.db that REFUSES a write statement made without the lock, so a
  write path added later without the lock fails loudly instead of racing the Rust process.

Ownership is a token, not a thread: the middleware acquires it on the event loop and the request's
threadpool handler, which inherits the request's context, sees it as held (re-entrant). Threads
started with threading.Thread get a fresh context and wait for the lock like any other writer.

Lock order (deadlock-free): tenant writer lock → app._tenant_lock → state.recover_lock →
store RLock → migration RLock/flock → SQLite. A context holds at most one tenant's writer lock, and
the writer lock is never taken while the tenant registry lock (an OrderedLock) is held: both raise
LockOrderError instead of waiting. Acquisition waits at most DEADLINE_S (the forward deadline);
then WriterBusy (503).

For any other tenant (the default: DIARY_RUST_WRITES empty) everything here is a no-op.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger("diary.writer")

DEADLINE_S = 30.0
ALL = "*"
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_GUARDED_FILES = ("managed-diary.db", "managed-index.db")

_writers = frozenset()


class WriterBusy(Exception):
    """The tenant's writer lock was not free within the deadline."""


class LockOrderError(RuntimeError):
    """A lock-order violation: a programming error, never waited out."""


def configure(writers) -> None:
    """The tenants whose writes are shared with the Rust process (rust_forward.configured_writers)."""
    global _writers
    _writers = writers or frozenset()


def is_writer(uid: Optional[str]) -> bool:
    if not uid or not _writers:
        return False
    uid = uid.lower()
    return bool(_UUID_RE.fullmatch(uid)) and (_writers == ALL or uid in _writers)


def tenant_of(path) -> Optional[str]:
    """The tenant of users/<uid>/managed-*.db, else None."""
    p = Path(str(path).split("?", 1)[0].removeprefix("file:"))
    if p.name in _GUARDED_FILES and p.parent.parent.name == "users" and _UUID_RE.fullmatch(p.parent.name):
        return p.parent.name
    return None


# ------------------------------------------------------------------ the lock

class _Slot:
    __slots__ = ("owner", "users")

    def __init__(self):
        self.owner = None
        self.users = 0


_cond = threading.Condition()
_slots: dict = {}
# (uid, token) this context holds; at most one entry.
_held: contextvars.ContextVar = contextvars.ContextVar("diary_tenant_writer", default=())
_exempt: contextvars.ContextVar = contextvars.ContextVar("diary_tenant_writer_exempt", default=())
_ordered_locks: list = []
# Test observability: the most holders one tenant ever had at once (must stay 1).
max_concurrent: dict = {}


class OrderedLock:
    """A plain lock that knows its owner thread, so the writer lock can refuse to be taken under it."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._owner = None
        _ordered_locks.append(self)

    def acquire(self, blocking=True, timeout=-1):
        ok = self._lock.acquire(blocking, timeout)
        if ok:
            self._owner = threading.get_ident()
        return ok

    def release(self):
        self._owner = None
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def held_by_me(self) -> bool:
        return self._owner == threading.get_ident()


def held(uid: Optional[str]) -> bool:
    entry = _held.get()
    if not entry or not uid or entry[0] != uid.lower():
        return False
    with _cond:
        slot = _slots.get(entry[0])
        return slot is not None and slot.owner is entry[1]


def _check_order(uid: str) -> None:
    entry = _held.get()
    if entry and entry[0] != uid:
        raise LockOrderError("a context may hold only one tenant's writer lock")
    for lock in _ordered_locks:
        if lock.held_by_me():
            raise LockOrderError(f"the tenant writer lock is taken before {lock.name}, never under it")


def _try(uid: str, token) -> bool:
    slot = _slots.setdefault(uid, _Slot())
    if slot.owner is None:
        slot.owner = token
        slot.users += 1
        max_concurrent[uid] = max(max_concurrent.get(uid, 0), slot.users)
        return True
    return False


def _release(uid: str, token) -> None:
    with _cond:
        slot = _slots.get(uid)
        if slot is None or slot.owner is not token:
            return
        slot.owner = None
        slot.users -= 1
        if slot.users <= 0:
            _slots.pop(uid, None)
        _cond.notify_all()


@contextmanager
def hold(uid: Optional[str], deadline_s: float = DEADLINE_S):
    """Hold the tenant's writer lock (no-op for a non-writer tenant; re-entrant in one context)."""
    uid = (uid or "").lower()
    if not is_writer(uid) or held(uid):
        yield
        return
    _check_order(uid)
    token = object()
    until = time.monotonic() + deadline_s
    with _cond:
        while not _try(uid, token):
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise WriterBusy(uid)
            _cond.wait(remaining)
    reset = _held.set((uid, token))
    try:
        yield
    finally:
        _held.reset(reset)
        _release(uid, token)


@asynccontextmanager
async def hold_async(uid: str, until: float):
    """hold() for the event loop: never blocks it; `until` is a time.monotonic() deadline."""
    uid = (uid or "").lower()
    if not is_writer(uid) or held(uid):
        yield
        return
    _check_order(uid)
    token, pause = object(), 0.002
    while True:
        with _cond:
            if _try(uid, token):
                break
        if time.monotonic() >= until:
            raise WriterBusy(uid)
        await asyncio.sleep(min(pause, max(0.0, until - time.monotonic())))
        pause = min(pause * 2, 0.05)
    reset = _held.set((uid, token))
    try:
        yield
    finally:
        _held.reset(reset)
        _release(uid, token)


@contextmanager
def exempt(uid: str, reason: str):
    """Allow the guard to pass writes that provably cannot meet a Rust write of the tenant, where
    the lock order forbids taking the writer lock (activating a tenant under the registry lock:
    the Rust side never writes a tenant that is not active yet)."""
    reset = _exempt.set(_exempt.get() + ((uid.lower(), reason),))
    try:
        yield
    finally:
        _exempt.reset(reset)


class TenantWriteLock:
    """CorpusStore._write_lock: the tenant writer lock, then the store's RLock. `local()` is the
    RLock alone, for bookkeeping that writes nothing (closing a state)."""

    def __init__(self, tenant=None):
        self.tenant = tenant
        self._rlock = threading.RLock()
        self._stack = threading.local()

    def __enter__(self):
        cm = hold(self.tenant)
        cm.__enter__()
        try:
            self._rlock.acquire()
        except BaseException:
            cm.__exit__(None, None, None)
            raise
        stack = getattr(self._stack, "items", None)
        if stack is None:
            stack = self._stack.items = []
        stack.append(cm)
        return self

    def __exit__(self, *exc):
        self._rlock.release()
        self._stack.items.pop().__exit__(*exc)
        return False

    def local(self):
        return self._rlock

    def _is_owned(self) -> bool:
        """threading.RLock's ownership check (the store's own RLock), for callers and tests."""
        return self._rlock._is_owned()


# ------------------------------------------------------------------ the guard

_WRITE_ACTIONS = frozenset(getattr(sqlite3, name) for name in (
    "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_TABLE", "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TEMP_TABLE", "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW", "SQLITE_CREATE_VTABLE",
    "SQLITE_DROP_TABLE", "SQLITE_DROP_INDEX", "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW", "SQLITE_DROP_VTABLE",
    "SQLITE_ALTER_TABLE", "SQLITE_REINDEX") if hasattr(sqlite3, name))
denials: list = []


def guard(conn: sqlite3.Connection, path) -> sqlite3.Connection:
    """Refuse writes to a writer tenant's managed database made without its writer lock."""
    uid = tenant_of(path)
    if uid is None:
        return conn

    def authorize(action, arg1, arg2, dbname, trigger):
        if action in _WRITE_ACTIONS and is_writer(uid) and not held(uid) \
                and not any(e[0] == uid for e in _exempt.get()):
            if dbname == "temp":
                return sqlite3.SQLITE_OK
            denials.append((uid, action, arg1))
            log.error("Diary write to a writer tenant's database without its writer lock was refused (%s)", arg1)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorize)
    return conn
