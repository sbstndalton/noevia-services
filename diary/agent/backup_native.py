"""Rust backup worker; the Python host keeps the existing flock and gated remote transport.

No credentials enter the child. Only immutable create-and-verify requests are brokered. The
child owns snapshot selection, manifest construction and legacy backups-row transitions.
"""
import base64
import fcntl
import json
import os
import selectors
import subprocess
import time
import httpx
from .leaf_native import LeafUnavailable, selected

SELECTOR = 'DIARY_BACKUP_IMPL'
DEFAULT_BIN = '/usr/local/bin/diary-backup-run'
TIMEOUT_S = 300
MAX_FRAME = 1536 * 1024 * 1024  # one legacy restore-sized object, base64 encoded


def enabled():
    return selected(SELECTOR)


def run(backend, remote, storage, now=None):
    now = time.time() if now is None else now
    destination = backend.destination(storage)
    if not destination or not backend.active():
        return backend.status(storage)
    with (backend.root / 'managed-backup.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return backend.status(storage)
        try:
            return _worker(backend, remote, storage, destination, now)
        except LeafUnavailable:
            # The child is already reaped. Keep an acknowledged committed generation;
            # a lost response never authorizes a Python rerun or a remote-contact claim.
            with backend.db() as db:
                previous = db.execute('SELECT * FROM backups WHERE destination=?', (destination,)).fetchone()
                generation = int(backend.meta(db, 'generation', '0'))
                if not previous or previous['generation'] != generation:
                    failures = (previous['failures'] if previous else 0) + 1
                    db.execute('INSERT OR REPLACE INTO backups VALUES (?,?,?,?,?,?)',
                               (destination, previous['generation'] if previous else -1,
                                previous['last_success'] if previous else None,
                                now + min(300, 5 * 2 ** min(failures - 1, 6)), failures,
                                'Backup could not be verified. Check the connection or remote conflict; app saves are retained.'))
            return backend.status(storage)


def _worker(backend, remote, storage, destination, now):
    request = {'db':str(backend.db_path), 'tenant':backend.tenant, 'destination':destination,
               'now':now, 'corpusRoot':storage.get('corpusRoot', '')}
    proc = None
    deadline = time.monotonic() + TIMEOUT_S
    result = None
    verified_count = 0
    try:
        proc = subprocess.Popen([DEFAULT_BIN, 'run-v1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, close_fds=True,
                                env={'PATH':'/usr/local/bin:/usr/bin:/bin'})
        pending = memoryview(json.dumps(request).encode() + b'\n')
        output, errors = bytearray(), bytearray()
        with selectors.DefaultSelector() as sel:
            for pipe, event in ((proc.stdin, selectors.EVENT_WRITE), (proc.stdout, selectors.EVENT_READ),
                                (proc.stderr, selectors.EVENT_READ)):
                os.set_blocking(pipe.fileno(), False)
                sel.register(pipe, event)
            while sel.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LeafUnavailable('Diary backup deadline exceeded')
                for key, _ in sel.select(min(.05, remaining)):
                    pipe = key.fileobj
                    if pipe is proc.stdin:
                        if not pending:
                            sel.unregister(pipe)
                            continue
                        try:
                            pending = pending[os.write(pipe.fileno(), pending[:65536]):]
                        except BlockingIOError:
                            continue
                    else:
                        target, cap = (output, MAX_FRAME) if pipe is proc.stdout else (errors, 256)
                        try:
                            block = os.read(pipe.fileno(), min(65536, cap+1-len(target)))
                        except BlockingIOError:
                            continue
                        if not block:
                            sel.unregister(pipe)
                            if pipe is proc.stdout and not proc.stdin.closed and proc.stdin.fileno() in sel.get_map():
                                sel.unregister(proc.stdin)
                            continue
                        target.extend(block)
                        if len(target) > cap:
                            raise LeafUnavailable('Diary backup frame exceeds capacity')
                        if pipe is proc.stdout:
                            while b'\n' in output:
                                line, _, rest = output.partition(b'\n')
                                output = bytearray(rest)
                                message = json.loads(line)
                                if result is not None:
                                    raise LeafUnavailable('Diary backup unexpected frame')
                                if message['type'] == 'done':
                                    result = message['outcome']
                                    if proc.stdin.fileno() in sel.get_map():
                                        sel.unregister(proc.stdin)
                                    proc.stdin.close()
                                elif message['type'] == 'verify':
                                    path = message['path']
                                    prefix = '/'.join(p.strip('/') for p in (storage.get('corpusRoot',''), 'noevia-backups', backend.tenant) if p) + '/'
                                    if not isinstance(path,str) or not path.startswith(prefix):
                                        raise LeafUnavailable('Diary backup object namespace')
                                    from .managed_storage import safe_key
                                    safe_key(path)
                                    data = base64.b64decode(message['data'], validate=True)
                                    try:
                                        backend._create_verified(remote, path, data)
                                        reply = 'ok'
                                        verified_count += 1
                                    except Exception as exc:
                                        reply = ('loginRefused' if isinstance(exc, httpx.HTTPStatusError)
                                                 and exc.response is not None and exc.response.status_code in (401,403)
                                                 else 'failed')
                                    pending = memoryview(json.dumps({'result':reply}).encode()+b'\n')
                                    if proc.stdin.fileno() not in sel.get_map():
                                        sel.register(proc.stdin, selectors.EVENT_WRITE)
                                else:
                                    raise LeafUnavailable('Diary backup invalid frame')
        if proc.wait(timeout=max(.001,deadline-time.monotonic())) or errors or output or result is None:
            raise LeafUnavailable('Diary backup worker failed')
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        raise LeafUnavailable('Diary backup worker unavailable') from None
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
    status = backend.status(storage)
    if result['result'] == 'failed' and result.get('loginRefused'):
        return {**status, 'loginRefused':True}
    if result['result'] == 'complete':
        return {**status, 'storageContacted':True}
    if result['result'] == 'abandoned':
        from .managed_storage import log
        log.warning('managed backup abandoned: a save landed mid-run (%s files verified); retrying after the debounce', verified_count)
    return status
