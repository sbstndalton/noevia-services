"""Bounded IPC for the internal, opt-in managed workspace exporter."""
import hashlib
import json
import logging
import os
import selectors
import struct
import subprocess
import time

BINARY = '/usr/local/bin/diary-workspace-export'
MAGIC = b'NOEVIA-WORKSPACE-EXPORT-1\n'
OUTPUT_MAGIC = b'NOEVIA-WORKSPACE-EXPORT-OK-1\n'
OUTPUT_END = b'NOEVIA-WORKSPACE-EXPORT-END-1\n'
MAX_HEADER = 16 * 1024 * 1024
MAX_OUTPUT = 272 * 1024 * 1024
TIMEOUT = 60.0
ERROR = 'Workspace export could not be completed.'
log = logging.getLogger(__name__)


def selected():
    value = os.environ.get('WORKSPACE_EXPORT_IMPL', 'python')
    if value not in ('python', 'rust'):
        raise ValueError(ERROR)
    return value == 'rust'


def archive(settings, directories, files, bodies):
    """Consume body chunks during the caller's SQLite read transaction.

    No credentials or caller-selected executable/environment reach the child.
    Every pipe is nonblocking: a child that never reads cannot strand a writer.
    """
    try:
        header = json.dumps({'settings': settings, 'directories': directories, 'files': files},
                            ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(ERROR) from exc
    if len(header) > MAX_HEADER:
        raise ValueError(ERROR)

    def chunks():
        yield MAGIC + struct.pack('>I', len(header)) + header
        yield from bodies
    source = iter(chunks())
    pending = memoryview(b'')
    output = bytearray()
    stderr = bytearray()
    process = None
    try:
        process = subprocess.Popen([BINARY, 'export'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env={'LANG': 'C.UTF-8'}, cwd='/', close_fds=True)
        deadline = time.monotonic() + TIMEOUT
        with selectors.DefaultSelector() as selector:
            for stream, event in ((process.stdin, selectors.EVENT_WRITE),
                                  (process.stdout, selectors.EVENT_READ),
                                  (process.stderr, selectors.EVENT_READ)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, event)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError(ERROR)
                for key, _ in selector.select(min(remaining, 0.1)):
                    stream = key.fileobj
                    if stream is process.stdin:
                        if not pending:
                            try:
                                pending = memoryview(next(source))
                            except StopIteration:
                                selector.unregister(stream)
                                stream.close()
                                continue
                        if pending:
                            try:
                                count = os.write(stream.fileno(), pending[:65536])
                                pending = pending[count:]
                            except BlockingIOError:
                                pass
                    else:
                        try:
                            block = os.read(stream.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if not block:
                            selector.unregister(stream)
                            stream.close()
                            continue
                        target = output if stream is process.stdout else stderr
                        target.extend(block)
                        limit = MAX_OUTPUT + len(OUTPUT_MAGIC) + 8 + 32 + len(OUTPUT_END)
                        if len(target) > (limit if stream is process.stdout else 4096):
                            raise ValueError(ERROR)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or process.wait(timeout=remaining) != 0 or stderr:
                raise ValueError(ERROR)
        header_size = len(OUTPUT_MAGIC) + 8 + 32
        if len(output) < header_size + len(OUTPUT_END) or not output.startswith(OUTPUT_MAGIC):
            raise ValueError(ERROR)
        length = struct.unpack('>Q', output[len(OUTPUT_MAGIC):len(OUTPUT_MAGIC) + 8])[0]
        if (length > MAX_OUTPUT or len(output) != header_size + length + len(OUTPUT_END)
                or output[header_size + length:] != OUTPUT_END):
            raise ValueError(ERROR)
        content = memoryview(output)[header_size:header_size + length]
        if hashlib.sha256(content).digest() != output[len(OUTPUT_MAGIC) + 8:header_size]:
            raise ValueError(ERROR)
        return bytes(content)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        log.warning('Native workspace export failed')
        raise ValueError(ERROR) from exc
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
