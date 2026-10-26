"""Bounded private IPC for the Rust workspace decode-v1 command.

A frame becomes trusted only after its marker, exact EOF and successful child exit.
The archive is never parsed as ZIP in this module.
"""
import hashlib
import json
import logging
import os
import re
import struct
import subprocess
import selectors
import time

from .managed_storage import safe_key
from .workspace_export import MAX_ARCHIVE, MAX_FILE, MAX_TOTAL


class DecodeUnavailable(ValueError):
    """The verifier or its private protocol is unavailable; callers may retry."""


UNAVAILABLE = 'Workspace archive verification is unavailable. Try again later.'
DEFAULT_BIN = '/usr/local/bin/workspace-restore-verify'
TIMEOUT_S = 120
MAX_METADATA = 32 * 1024 * 1024
MAX_ENTRIES = 5000
MAGIC = b'noevia-workspace-decode-v1\0'
END = b'\0noevia-workspace-decode-v1-end\0'
MAX_FRAME = len(MAGIC) + 4 + MAX_METADATA + 8 * MAX_ENTRIES + MAX_TOTAL + len(END)
HEX = re.compile(r'[0-9a-f]{64}\Z')
REJECT = re.compile(rb'reject: ([a-z_]{1,32})\n\Z')
_log = logging.getLogger(__name__)


def _unavailable(reason):
    # Only fixed reason codes; never print private stdout, paths, source or metadata.
    _log.warning('workspace decode unavailable (%s)', reason)
    return DecodeUnavailable(UNAVAILABLE)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate metadata key')
        result[key] = value
    return result


def _parse_frame(out, body):
    try:
        view = memoryview(out)
        if len(view) < len(MAGIC) + 4 or view[:len(MAGIC)] != MAGIC:
            raise ValueError('magic')
        offset = len(MAGIC)
        meta_len = struct.unpack_from('<I', view, offset)[0]
        offset += 4
        if meta_len > MAX_METADATA or meta_len > len(view) - offset:
            raise ValueError('metadata length')
        metadata = json.loads(bytes(view[offset:offset + meta_len]).decode('utf-8'),
                              object_pairs_hook=_no_duplicate_keys,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
        offset += meta_len
        if type(metadata) is not dict or type(metadata.get('version')) is not int or metadata['version'] != 1:
            raise ValueError('version')
        fingerprint = metadata.get('fingerprint')
        listing = metadata.get('listingSha256')
        if not isinstance(fingerprint, str) or not HEX.fullmatch(fingerprint) or fingerprint != hashlib.sha256(body).hexdigest():
            raise ValueError('fingerprint')
        if not isinstance(listing, str) or not HEX.fullmatch(listing):
            raise ValueError('listing')
        rows = metadata.get('files')
        directories = metadata.get('directories')
        if type(rows) is not list or len(rows) > MAX_ENTRIES or type(directories) is not list or len(directories) > MAX_ENTRIES:
            raise ValueError('counts')
        if type(metadata.get('bytes')) is not int or not 0 <= metadata['bytes'] <= MAX_TOTAL or 'settings' not in metadata:
            raise ValueError('totals or settings')
        dirs = set()
        previous = None
        for path in directories:
            safe_key(path)
            if previous is not None and path <= previous:
                raise ValueError('directory order')
            dirs.add(path)
            previous = path
        files = {}
        previous = None
        total = 0
        for row in rows:
            if type(row) is not dict or set(row) != {'path', 'bytes', 'sha256'}:
                raise ValueError('file record')
            path = safe_key(row['path'])
            length, checksum = row['bytes'], row['sha256']
            if previous is not None and path <= previous or path in dirs:
                raise ValueError('file order or overlap')
            if type(length) is not int or not 0 <= length <= MAX_FILE or not isinstance(checksum, str) or not HEX.fullmatch(checksum):
                raise ValueError('file fields')
            if total + length > MAX_TOTAL or len(view) - offset < 8:
                raise ValueError('source limit')
            actual_length = struct.unpack_from('<Q', view, offset)[0]
            offset += 8
            if actual_length != length or length > len(view) - offset:
                raise ValueError('payload length')
            data = bytes(view[offset:offset + length])
            offset += length
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError('checksum')
            files[path] = data
            total += length
            previous = path
        if total != metadata['bytes']:
            raise ValueError('source total')
        for path in (*files, *dirs):
            parts = path.split('/')
            if any('/'.join(parts[:i]) in files for i in range(1, len(parts))):
                raise ValueError('parent overlap')
        if len(view) - offset != len(END) or view[offset:offset + len(END)] != END:
            raise ValueError('end marker or trailing data')
        from .workspace_restore import listing_sha256
        if listing_sha256(files, dirs) != listing:
            raise ValueError('listing digest')
        manifest = {'format': 'noevia-workspace-v1', 'settings': metadata['settings'],
                    'directories': sorted(dirs), 'fileCount': len(files), 'bytes': total,
                    'files': [{'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                              for path, data in files.items()]}
        return files, dirs, manifest
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, UnicodeError, RecursionError, struct.error):
        raise _unavailable('protocol') from None


def decode(body, messages):
    """Return the existing files, directories, manifest tuple, or fail closed."""
    if len(body) > MAX_ARCHIVE:
        raise ValueError(messages['archive_too_large'])
    if type(body) is not bytes:
        body = bytes(body)
    # The installed image path is fixed. Tests may monkeypatch DEFAULT_BIN without
    # introducing a production environment/PATH-selected executable.
    binary = DEFAULT_BIN
    if not (os.path.isabs(binary) and os.path.isfile(binary) and os.access(binary, os.X_OK)):
        raise _unavailable('missing_binary')
    output = bytearray()
    offset = 0
    fed = False
    feed_error = False
    fault = None
    deadline = time.monotonic() + TIMEOUT_S
    selector = None
    proc = None
    code = None

    def close(stream):
        try:
            selector.unregister(stream)
        except (KeyError, ValueError):
            pass
        stream.close()

    try:
        selector = selectors.DefaultSelector()
        proc = subprocess.Popen([binary, 'decode-v1'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                                close_fds=True, env={'PATH': '/usr/local/bin:/usr/bin:/bin'})
        for stream, kind, event in ((proc.stdin, 'input', selectors.EVENT_WRITE),
                                    (proc.stdout, 'output', selectors.EVENT_READ)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, event, kind)
        while selector.get_map() or proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fault = 'timeout'
                break
            for key, _ in selector.select(timeout=min(.05, remaining)):
                stream, kind = key.fileobj, key.data
                try:
                    if kind == 'input':
                        if offset == len(body):
                            close(stream)
                            fed = True
                            continue
                        written = os.write(stream.fileno(), memoryview(body)[offset:offset + 65536])
                        if written == 0:
                            raise OSError('short write')
                        offset += written
                    else:
                        cap = MAX_FRAME
                        used = len(output)
                        chunk = os.read(stream.fileno(), min(65536, cap + 1 - used))
                        if not chunk:
                            close(stream)
                            continue
                        output.extend(chunk)
                        if used + len(chunk) > cap:
                            fault = kind + '_overflow'
                            break
                except BlockingIOError:
                    continue
                except OSError:
                    if kind == 'input':
                        feed_error = True
                        close(stream)
                    else:
                        fault = 'stream'
                        break
            if fault:
                break
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        fault = 'io'
    finally:
        # Cover selector construction, spawn and setup, as well as caller
        # cancellation during IO. Success requires reap; no daemon pipe worker
        # can remain blocked on an inherited descriptor.
        try:
            if proc is not None:
                try:
                    if proc.poll() is None:
                        try:
                            proc.kill()  # Only this owned direct child.
                        except ProcessLookupError:
                            pass
                finally:
                    code = proc.wait()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            fault = fault or 'cleanup'
        finally:
            try:
                if proc is not None:
                    for stream in (proc.stdin, proc.stdout):
                        try:
                            stream.close()
                        except (OSError, ValueError, RuntimeError):
                            fault = fault or 'cleanup'
            finally:
                if selector is not None:
                    try:
                        selector.close()
                    except (OSError, ValueError, RuntimeError):
                        fault = fault or 'cleanup'
    if time.monotonic() >= deadline:
        fault = fault or 'timeout'
    if fault:
        raise _unavailable(fault)
    out = output
    if code == 1:
        match = REJECT.fullmatch(out)
        if match:
            reason = match.group(1).decode('ascii')
            if reason in messages:
                raise ValueError(messages[reason])
    if code != 0 or not fed or feed_error:
        raise _unavailable('child_exit_or_input')
    return _parse_frame(out, body)
