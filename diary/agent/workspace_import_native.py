"""Opt-in native import transaction IPC; no ZIP parser or mutation fallback.

Linux service pipes are multiplexed without helper threads. The caller retains its
store/migration/journal exclusion until every owned child is reaped.
"""
import json
import logging
import os
import re
import selectors
import struct
import subprocess
import time

from .workspace_export import MAX_ARCHIVE
from .workspace_restore import MESSAGES

DEFAULT_BIN = '/usr/local/bin/diary-workspace-import-apply'
MAGIC = b'noevia-workspace-import-apply-v1\0'
MAX_METADATA = 1024 * 1024
MAX_STDOUT = 4096
MAX_STDERR = 4096
TIMEOUT_S = 120
REJECT = re.compile(rb'reject: ([a-z_]{1,32})\n\Z')
UNAVAILABLE = re.compile(rb'unavailable: ([a-z_]{1,32})\n\Z')
_log = logging.getLogger(__name__)


class ImportUnavailable(ValueError):
    """The native transaction service cannot accept this operation."""


class ImportUncertain(ImportUnavailable):
    """The transaction may have committed; retry the identical reviewed request."""


def _fault(reason, uncertain=True):
    # No paths, child output, source, tenant IDs or fingerprint are logged.
    _log.warning('workspace import native fault (%s)', reason)
    if uncertain:
        return ImportUncertain('Workspace import outcome is uncertain. Retry the same archive and destination with the same reviewed fingerprint.')
    return ImportUnavailable('Workspace import is unavailable. Try again later.')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate response key')
        result[key] = value
    return result


def _response(code, out, report):
    if code == 2:
        match = REJECT.fullmatch(out)
        if match:
            reason = match.group(1).decode('ascii')
            if reason in MESSAGES:
                raise ValueError(MESSAGES[reason])
            if reason == 'destination':
                raise ValueError('Choose a new folder name of 1–100 characters')
            if reason == 'fingerprint':
                raise ValueError('Archive or destination changed. Review a fresh preview.')
            if reason == 'conflict':
                raise ValueError('Destination already exists or conflicts. Choose a new folder.')
            if reason == 'frame':
                raise _fault('frame', uncertain=False)
    if code == 3:
        match = UNAVAILABLE.fullmatch(out)
        if match and match.group(1) in {b'schema', b'tenant', b'inactive', b'clock', b'generation'}:
            raise _fault(match.group(1).decode('ascii'), uncertain=False)
    if code != 0:
        raise _fault('child_exit')
    try:
        if not out.endswith(b'\n') or out.count(b'\n') != 1:
            raise ValueError('line')
        result = json.loads(out[:-1].decode('utf-8'), object_pairs_hook=_pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
        if type(result) is not dict or set(result) != {'version', 'destination', 'fileCount', 'alreadyApplied'}:
            raise ValueError('fields')
        if type(result['version']) is not int or result['version'] != 1:
            raise ValueError('version')
        if result['destination'] != report['destination'] or type(result['fileCount']) is not int or result['fileCount'] != report['fileCount']:
            raise ValueError('scope')
        if type(result['alreadyApplied']) is not bool:
            raise ValueError('receipt')
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise _fault('protocol') from None
    del result['version']
    return result


def apply_native(backend, body, report):
    """Apply only after prepare's SQL context ended; never call backend.close()."""
    if len(body) > MAX_ARCHIVE:
        raise ValueError(MESSAGES['archive_too_large'])
    try:
        metadata = json.dumps({'version': 1, 'dbPath': str(backend.db_path),
                               'expectedTenant': backend.tenant,
                               'destination': report['destination'],
                               'reviewedFingerprint': report['fingerprint'],
                               'debounce': backend.debounce},
                              ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(metadata) > MAX_METADATA:
            raise ValueError('metadata')
    except (TypeError, ValueError, UnicodeError, OverflowError):
        raise _fault('metadata', uncertain=False) from None
    binary = DEFAULT_BIN
    if not (os.path.isabs(binary) and os.path.isfile(binary) and os.access(binary, os.X_OK)):
        raise _fault('missing_binary', uncertain=False)
    selector = None
    try:
        selector = selectors.DefaultSelector()
        child = subprocess.Popen([binary, 'apply-v1'], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                 close_fds=True, env={'PATH': '/usr/local/bin:/usr/bin:/bin'})
    except OSError:
        if selector is not None:
            selector.close()
        raise _fault('spawn', uncertain=False) from None
    segments = [memoryview(part) for part in (
        MAGIC, struct.pack('<I', len(metadata)), metadata,
        struct.pack('<Q', len(body)), body)]
    output = bytearray()
    stderr_bytes = 0
    segment, offset = 0, 0
    fed = False
    input_error = False
    fault = None
    deadline = time.monotonic() + TIMEOUT_S

    def close(stream):
        try:
            selector.unregister(stream)
        except (KeyError, ValueError):
            pass
        stream.close()

    try:
        for stream, kind, event in ((child.stdin, 'input', selectors.EVENT_WRITE),
                                    (child.stdout, 'output', selectors.EVENT_READ),
                                    (child.stderr, 'error', selectors.EVENT_READ)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, event, kind)
        while selector.get_map() or child.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fault = 'timeout'
                break
            for key, _ in selector.select(timeout=min(.05, remaining)):
                stream, kind = key.fileobj, key.data
                try:
                    if kind == 'input':
                        while segment < len(segments) and offset == len(segments[segment]):
                            segment += 1
                            offset = 0
                        if segment == len(segments):
                            close(stream)
                            fed = True
                            continue
                        written = os.write(stream.fileno(), segments[segment][offset:offset + 65536])
                        if written == 0:
                            raise OSError('short write')
                        offset += written
                    else:
                        cap = MAX_STDOUT if kind == 'output' else MAX_STDERR
                        used = len(output) if kind == 'output' else stderr_bytes
                        chunk = os.read(stream.fileno(), min(65536, cap + 1 - used))
                        if not chunk:
                            close(stream)
                            continue
                        if kind == 'output':
                            output.extend(chunk)
                        else:
                            stderr_bytes += len(chunk)
                        if used + len(chunk) > cap:
                            fault = kind + '_overflow'
                            break
                except BlockingIOError:
                    continue
                except OSError:
                    if kind == 'input':
                        input_error = True
                        close(stream)
                    else:
                        fault = 'stream'
                        break
            if fault:
                break
    except (OSError, ValueError):
        fault = 'io'
    finally:
        if child.poll() is None:
            try:
                child.kill()  # Only this owned process; no group/other-runtime action.
            except ProcessLookupError:
                pass
        code = child.wait()
        for stream in (child.stdin, child.stdout, child.stderr):
            if not stream.closed:
                stream.close()
        selector.close()
    if time.monotonic() >= deadline:
        fault = fault or 'timeout'
    if fault or not fed or input_error or stderr_bytes:
        raise _fault(fault or 'input_or_stderr')
    return _response(code, bytes(output), report)
