"""Bounded JSON transport for opt-in pure Diary leaves; failures never select Python."""
import json
import os
import selectors
import subprocess
import time

class LeafUnavailable(OSError):
    """Retryable native apply outage; durable journal intent must never quarantine."""

def selected(name):
    return (os.environ.get(name) or 'python').strip().lower() == 'rust'

def call(binary, payload, timeout=10):
    raw = json.dumps(payload, ensure_ascii=False).encode()
    if len(raw) > 4 * 1024 * 1024:
        raise LeafUnavailable('Diary leaf input exceeds capacity')
    proc = None
    try:
        proc = subprocess.Popen([binary, 'json-v1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, close_fds=True,
                                env={'PATH': '/usr/local/bin:/usr/bin:/bin'})
        deadline = time.monotonic() + timeout
        output, errors = bytearray(), bytearray()
        pending = memoryview(raw)
        with selectors.DefaultSelector() as sel:
            for pipe, event in ((proc.stdin, selectors.EVENT_WRITE), (proc.stdout, selectors.EVENT_READ),
                                (proc.stderr, selectors.EVENT_READ)):
                os.set_blocking(pipe.fileno(), False)
                sel.register(pipe, event)
            while sel.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LeafUnavailable('Diary leaf deadline exceeded')
                for key, _ in sel.select(min(remaining, .05)):
                    pipe = key.fileobj
                    if pipe is proc.stdin:
                        if not pending:
                            sel.unregister(pipe)
                            pipe.close()
                            continue
                        try:
                            pending = pending[os.write(pipe.fileno(), pending[:65536]):]
                        except BlockingIOError:
                            continue
                    else:
                        target, cap = (output, 8*1024*1024) if pipe is proc.stdout else (errors, 256)
                        try:
                            chunk = os.read(pipe.fileno(), min(65536, cap+1-len(target)))
                        except BlockingIOError:
                            continue
                        if not chunk:
                            sel.unregister(pipe)
                            pipe.close()
                        else:
                            target.extend(chunk)
                            if len(target) > cap:
                                raise LeafUnavailable('Diary leaf output exceeds capacity')
        if proc.wait(timeout=max(.001, deadline-time.monotonic())) or errors:
            raise LeafUnavailable('Diary leaf failed')
        return json.loads(output)
    except (OSError, ValueError, subprocess.SubprocessError):
        raise LeafUnavailable('Diary leaf unavailable') from None
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
