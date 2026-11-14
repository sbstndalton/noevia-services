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
  deletion, the backend construction write) call hold() directly.
- connect() is the only way agent code opens SQLite (tests/test_tenant_writer_d5.py enforces it).
  On a writer tenant's managed-diary.db / managed-index.db it installs an authorizer that REFUSES a
  write made without the lock, with the statement cache off, so every execution is authorized
  again (a statement prepared while the lock was held is never reused without it). ATTACH is
  refused on every connection, so no connection reaches a tenant's database under another name.

Admission barrier. A lock holder whose forwarded write may still be running in the Rust child
(no reply, a timeout, a cancelled request) does not simply release: the child is killed, the
tenant is marked recovery-pending, and a barrier (add_barrier: the child is not yet confirmed
exited) is raised BEFORE the lock is released. Every new acquisition of ANY writer tenant's lock,
Python-only writers included, waits while a barrier is up and fails closed (WriterBusy) at its
deadline. So the lock is free only once the Rust child has provably stopped writing.

Ownership is a token, not a thread: the middleware acquires it on the event loop, and the request's
threadpool handler, which inherits the request's context, JOINS it (a counted reference): the slot
is freed only when the last reference is released, so a handler thread still inside a write keeps
the lock even if its request was cancelled. Threads started with threading.Thread get a fresh
context and wait for the lock like any other writer.

Lock order (deadlock-free): tenant writer lock → app._tenant_lock → state.recover_lock →
store RLock → migration RLock/flock → SQLite. A context holds at most one tenant's writer lock, and
the writer lock is never taken while an OrderedLock is held: both raise LockOrderError instead of
waiting. Acquisition waits at most DEADLINE_S (the forward deadline); then WriterBusy (503).

For any other tenant (the default: DIARY_RUST_WRITES empty) the lock is a no-op.

R9 main mode, S3 (rust_parent; DIARY_MAIN=rust): the Rust process is this process's PARENT and owns
the tenant's writer slot. configure_remote() makes the first acquisition of a writer tenant's lock
also take a LEASE on that slot (outside every lock here, within the same deadline), and the last
release return it; joins stay local. A lease the parent refuses or cannot grant is WriterBusy and
leaves nothing held. While this process holds the lease, the calls it makes to the parent under it
(the deletion fence, the re-index) name it (lease_of) and join it there.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("diary.writer")

DEADLINE_S = 30.0
ALL = "*"
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_GUARDED_FILES = ("managed-diary.db", "managed-index.db")

_writers = frozenset()
_data_root = None


def configure_data_root(root):
    global _data_root
    _data_root = Path(root) if root is not None else None


def effect_blocked(uid):
    from .backup_effect import pending
    return bool(_data_root is not None and uid and _UUID_RE.fullmatch(uid) and pending(_data_root / 'users' / uid))



class WriterBusy(Exception):
    """The tenant's writer lock was not free within the deadline (`reason` "restarting" when an
    admission barrier, a Rust child not yet confirmed exited, kept it closed)."""

    def __init__(self, uid: str, reason: Optional[str] = None):
        super().__init__(uid)
        self.reason = reason


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


# ------------------------------------------------------------------ admission barriers

_barriers: list = []


def add_barrier(fn: Callable[[], Optional[str]]) -> None:
    """`fn()` returns a reason while no writer tenant's lock may be newly acquired, else None."""
    _barriers.append(fn)


def _blocked() -> Optional[str]:
    for fn in list(_barriers):
        try:
            reason = fn()
        except Exception:  # noqa: BLE001 - a broken barrier keeps writes closed
            reason = "barrier failed"
        if reason:
            return reason
    return None


def notify() -> None:
    """A barrier changed: wake waiting writers."""
    with _cond:
        _cond.notify_all()


# ------------------------------------------------------------------ the lock

class _Slot:
    __slots__ = ("owner", "users", "lease")

    def __init__(self):
        self.owner = None
        self.users = 0
        self.lease = None


# R9 main mode: the parent's lease API (rust_parent.ParentLeases), None otherwise.
_remote = None


def configure_remote(remote) -> None:
    """`remote.acquire(uid, wait_s) -> str` (raises WriterBusy) and `remote.release(lease)`."""
    global _remote
    _remote = remote


def lease_of(uid: Optional[str]) -> Optional[str]:
    """The parent lease this context holds for the tenant (main mode), else None."""
    entry = _held.get()
    if not entry or not uid or entry[0] != uid.lower():
        return None
    with _cond:
        slot = _slots.get(entry[0])
        return slot.lease if slot is not None and slot.owner is entry[1] else None


_cond = threading.Condition()
_slots: dict = {}
# (uid, token) this context holds; at most one entry.
_held: contextvars.ContextVar = contextvars.ContextVar("diary_tenant_writer", default=())
_exempt: contextvars.ContextVar = contextvars.ContextVar("diary_tenant_writer_exempt", default=())
_ordered_locks: list = []
# Test observability: the most distinct owners one tenant ever had at once (must stay 1).
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


def _try(uid: str, token) -> Optional[str]:
    """Caller holds _cond. None when acquired, else why not ("busy" or a barrier's reason)."""
    if effect_blocked(uid):
        return "backup settlement uncertain"
    reason = _blocked()
    if reason:
        return reason
    slot = _slots.setdefault(uid, _Slot())
    if slot.owner is None:
        slot.owner, slot.users = token, 1
        owners = max_concurrent.get(uid, 0)
        max_concurrent[uid] = max(owners, 1)
        return None
    return "busy"


def _join(uid: str, token) -> bool:
    """Add a reference to the slot this context's token owns (a handler thread inside the
    middleware's hold)."""
    with _cond:
        slot = _slots.get(uid)
        # users == 0: the last reference is returning the parent's lease; never join that.
        if slot is None or slot.owner is not token or slot.users <= 0:
            return False
        slot.users += 1
        return True


def _release(uid: str, token) -> None:
    with _cond:
        slot = _slots.get(uid)
        if slot is None or slot.owner is not token or slot.users <= 0:
            return
        slot.users -= 1
        if slot.users > 0:
            return
        lease, slot.lease = slot.lease, None
        if lease is None:
            slot.owner = None
            _slots.pop(uid, None)
            _cond.notify_all()
            return
    # Main mode: the slot stays ours (users 0: nobody joins it) until the parent has the lease back.
    try:
        _remote.release(lease)
    except Exception as exc:  # noqa: BLE001 - the parent lists a lease it still holds (/internal/leases)
        log.error("Diary writer lease was not returned to the Rust parent (%s)", type(exc).__name__)
    finally:
        _abandon(uid, token)


def _abandon(uid: str, token) -> None:
    """Free a slot this token owns, whatever its count (a lease that was not granted, or returned)."""
    with _cond:
        slot = _slots.get(uid)
        if slot is not None and slot.owner is token:
            slot.owner, slot.users, slot.lease = None, 0, None
            _slots.pop(uid, None)
            _cond.notify_all()


def _lease(uid: str, token, until: float) -> None:
    """Main mode: take the parent's lease for a slot this token has just taken (outside _cond)."""
    if _remote is None:
        return
    try:
        lease = _remote.acquire(uid, max(0.0, until - time.monotonic()))
    except WriterBusy:
        raise
    except Exception as exc:  # noqa: BLE001 - no lease: nothing may be written
        log.error("Diary writer lease from the Rust parent failed (%s)", type(exc).__name__)
        raise WriterBusy(uid) from None
    with _cond:
        slot = _slots.get(uid)
        if slot is None or slot.owner is not token:
            stale = True
        else:
            slot.lease, stale = lease, False
    if stale:  # cannot happen (only this token frees its slot): never keep a lease nobody holds
        _remote.release(lease)
        raise WriterBusy(uid)


class Hold:
    """One reference to a tenant's writer lock. release() is synchronous and idempotent, so it
    runs to completion whatever cancels the holder."""

    def __init__(self, uid: str, token, reset):
        self.uid, self.token, self._reset, self._done = uid, token, reset, False

    def release(self) -> None:
        if self._done:
            return
        self._done = True
        if self._reset is not None:
            try:
                _held.reset(self._reset)
            except ValueError:  # released from another context: the reference still goes
                pass
        _release(self.uid, self.token)


_NOOP = Hold("", None, None)
_NOOP._done = True


def _enter(uid: str) -> Optional[Hold]:
    """A no-op or joined hold, or None when the caller must acquire."""
    if effect_blocked(uid):
        raise WriterBusy(uid, "backup settlement uncertain")
    if not is_writer(uid):
        return _NOOP
    entry = _held.get()
    if entry and entry[0] == uid and _join(uid, entry[1]):
        return Hold(uid, entry[1], None)
    _check_order(uid)
    return None


def acquire(uid: Optional[str], deadline_s: Optional[float] = None) -> Hold:
    uid = (uid or "").lower()
    deadline_s = DEADLINE_S if deadline_s is None else deadline_s
    hold_ = _enter(uid)
    if hold_ is not None:
        return hold_
    token = object()
    until = time.monotonic() + deadline_s
    with _cond:
        while True:
            reason = _try(uid, token)
            if reason is None:
                break
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise WriterBusy(uid, None if reason == "busy" else "restarting")
            # Barriers are polled: their state changes outside _cond (notify() shortens the wait).
            _cond.wait(min(remaining, 0.05))
    try:
        _lease(uid, token, until)
    except BaseException:
        _abandon(uid, token)
        raise
    return Hold(uid, token, _held.set((uid, token)))


async def acquire_async(uid: str, until: float) -> Hold:
    """acquire() for the event loop: never blocks it; `until` is a time.monotonic() deadline."""
    uid = (uid or "").lower()
    hold_ = _enter(uid)
    if hold_ is not None:
        return hold_
    token, pause = object(), 0.002
    while True:
        with _cond:
            reason = _try(uid, token)
        if reason is None:
            break
        if time.monotonic() >= until:
            raise WriterBusy(uid, None if reason == "busy" else "restarting")
        await asyncio.sleep(min(pause, max(0.0, until - time.monotonic())))
        pause = min(pause * 2, 0.05)
    if _remote is not None:
        # The lease call blocks: run it off the loop. A cancelled caller still settles it: a
        # lease granted after the cancellation is returned (the slot is freed with it).
        future = asyncio.get_running_loop().run_in_executor(None, _lease, uid, token, until)
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(lambda _f: _settle_cancelled(uid, token))
            raise
        except BaseException:
            _abandon(uid, token)
            raise
    return Hold(uid, token, _held.set((uid, token)))


def _settle_cancelled(uid: str, token) -> None:
    with _cond:
        slot = _slots.get(uid)
        lease = slot.lease if slot is not None and slot.owner is token else None
    if lease is not None:
        try:
            _remote.release(lease)
        except Exception as exc:  # noqa: BLE001
            log.error("Diary writer lease was not returned to the Rust parent (%s)", type(exc).__name__)
    _abandon(uid, token)


@contextmanager
def hold(uid: Optional[str], deadline_s: Optional[float] = None):
    """Hold the tenant's writer lock (no-op for a non-writer tenant; re-entrant in one context)."""
    h = acquire(uid, deadline_s)
    try:
        yield
    finally:
        h.release()


@contextmanager
def exempt(uid: str, reason: str):
    """Allow the guard to pass writes that provably cannot meet a Rust write of the tenant, where
    the lock order forbids taking the writer lock (activating a tenant under the registry lock:
    the Rust side never writes a tenant that is not active yet; creating a database file this
    process has just created: the Rust side never creates one)."""
    reset = _exempt.set(_exempt.get() + ((uid.lower(), reason),))
    try:
        yield
    finally:
        _exempt.reset(reset)


def is_exempt(uid: Optional[str]) -> bool:
    return bool(uid) and any(e[0] == uid.lower() for e in _exempt.get())


class TenantWriteLock:
    """CorpusStore._write_lock: the tenant writer lock, then the store's RLock. `local()` is the
    RLock alone, for bookkeeping that writes nothing (closing a state)."""

    def __init__(self, tenant=None):
        self.tenant = tenant
        self._rlock = threading.RLock()
        self._stack = threading.local()

    def __enter__(self):
        h = acquire(self.tenant)
        try:
            self._rlock.acquire()
        except BaseException:
            h.release()
            raise
        stack = getattr(self._stack, "items", None)
        if stack is None:
            stack = self._stack.items = []
        stack.append(h)
        return self

    def __exit__(self, *exc):
        self._rlock.release()
        self._stack.items.pop().release()
        return False

    def local(self):
        return self._rlock

    def _is_owned(self) -> bool:
        """threading.RLock's ownership check (the store's own RLock), for callers and tests."""
        return self._rlock._is_owned()


# A Python journal/index publication shares the same primary authority reservation as native
# helpers. Borrow a primary transaction already held by this context (outbox draining), rather
# than deadlocking by opening a second connection to the same SQLite write reservation.
_primary_scopes = contextvars.ContextVar("diary_primary_scopes", default=())


@contextmanager
def primary_scope(path, conn):
    token = _primary_scopes.set(_primary_scopes.get() + ((str(Path(path).resolve()), conn),))
    try:
        yield
    finally:
        _primary_scopes.reset(token)


@contextmanager
def index_publication(path):
    path = Path(path)
    if path.name != 'managed-index.db' or tenant_of(path) is None:
        yield  # Legacy/non-managed index cannot meet an app-managed backup.
        return
    primary = path.with_name('managed-diary.db').resolve()
    if not primary.exists():
        yield  # No managed primary exists to back up.
        return
    borrowed = next((conn for name, conn in reversed(_primary_scopes.get())
                     if name == str(primary) and conn.in_transaction), None)
    if borrowed is not None:
        # Upgrade a read transaction before checking; a concurrent snapshot change fails closed.
        borrowed.execute('UPDATE meta SET value=value WHERE 0')
        if borrowed.execute("SELECT 1 FROM meta WHERE key='backup_effect_pending'").fetchone():
            raise WriterBusy(tenant_of(path), 'backup settlement uncertain')
        yield
        return
    guard = connect(primary.as_uri() + '?mode=rw', uri=True, timeout=30)
    try:
        guard.execute('BEGIN IMMEDIATE')
        if guard.execute("SELECT 1 FROM meta WHERE key='backup_effect_pending'").fetchone():
            raise WriterBusy(tenant_of(path), 'backup settlement uncertain')
        with primary_scope(primary, guard):
            yield
        guard.commit()
    finally:
        guard.close()


# ------------------------------------------------------------------ connections and the guard

_WRITE_ACTIONS = frozenset(getattr(sqlite3, name) for name in (
    "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_TABLE", "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TEMP_TABLE", "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW", "SQLITE_CREATE_VTABLE",
    "SQLITE_DROP_TABLE", "SQLITE_DROP_INDEX", "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW", "SQLITE_DROP_VTABLE",
    "SQLITE_ALTER_TABLE", "SQLITE_REINDEX") if hasattr(sqlite3, name))
_ATTACH = frozenset(getattr(sqlite3, n) for n in ("SQLITE_ATTACH", "SQLITE_DETACH") if hasattr(sqlite3, n))
denials: list = []


def _authorizer(uid: Optional[str]):
    def authorize(action, arg1, arg2, dbname, trigger):
        if action in _ATTACH:
            # Nothing here attaches databases; refusing it keeps every write on the connection
            # opened for that file, where its tenant's guard applies.
            denials.append((uid, action, arg1))
            return sqlite3.SQLITE_DENY
        if uid is not None and action in _WRITE_ACTIONS and effect_blocked(uid):
            return sqlite3.SQLITE_DENY
        if uid is not None and action in _WRITE_ACTIONS and dbname != "temp" and is_writer(uid) \
                and not held(uid) and not any(e[0] == uid for e in _exempt.get()):
            denials.append((uid, action, arg1))
            log.error("Diary write to a writer tenant's database without its writer lock was refused (%s)", arg1)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return authorize


def connect(database, **kwargs) -> sqlite3.Connection:
    """sqlite3.connect for agent code. A tenant's managed database gets the write guard and no
    statement cache (every execution is authorized again); every connection refuses ATTACH."""
    uid = tenant_of(database)
    if uid is not None:
        kwargs["cached_statements"] = 0
    conn = sqlite3.connect(database, **kwargs)
    conn.set_authorizer(_authorizer(uid))
    return conn


_SCHEMA_OBJECT = re.compile(r"CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+IF\s+NOT\s+EXISTS\s+(\w+)", re.I)


def ensure_schema(conn: sqlite3.Connection, path, script: str, existed: bool) -> None:
    """Run a CREATE ... IF NOT EXISTS schema script only when an object is missing, so opening a
    database whose schema is in place writes nothing. A writer tenant's missing objects are created
    under its writer lock; a database this process has just created is exempt (the Rust process
    never creates one, so it cannot be writing it)."""
    names = _SCHEMA_OBJECT.findall(script)
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if names and all(n in have for n in names):
        return
    uid = tenant_of(path)
    if uid is not None and is_exempt(uid):
        with index_publication(path):
            conn.executescript(script)
            conn.commit()
        return
    if not existed and uid is not None:
        with exempt(uid, "creating a database file this process just created"), index_publication(path):
            conn.executescript(script)
            conn.commit()
        return
    with hold(uid), index_publication(path):
        conn.executescript(script)
        conn.commit()
