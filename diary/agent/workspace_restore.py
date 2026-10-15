"""Preview a portable ZIP; restore only to a new directory, never a live corpus.

python -m agent.workspace_restore ARCHIVE.zip
python -m agent.workspace_restore ARCHIVE.zip --destination NEW --fingerprint REVIEWED_SHA256

`verify` is the gate for this restore, its preview and the in-app ZIP import
(workspace_import.prepare). WORKSPACE_RESTORE_VERIFY_IMPL selects it (noevia#982):

- python (default; also empty, unset or any unknown value, with one warning): `_verify` below.
- rust: the bounded Rust leaf from sbstndalton/noevia-rs (`workspace-restore-verify verify`,
  built into the image at the Dockerfile's pinned NOEVIA_RS_REF) checks the archive FIRST, so its
  stricter container checks run before `zipfile` parses anything; then `_verify` runs as well and
  the two verified listings (paths, sizes, checksums, directories, totals) must hash identically.
  It FAILS CLOSED: a Rust refusal is answered with Python's message for that check; a missing
  binary, a nonzero exit other than a well-formed refusal, a timeout, unexpected output or a
  listing mismatch refuses the archive as "verification unavailable". The archive goes to the
  child on stdin only, with a minimal environment; logs carry codes only, never names.

The shared differential fixtures (tests/fixtures/workspace-restore.v1.json, generated in noevia-rs
by tools/gen-workspace-restore.py from this module) are the contract between the two.
"""
import argparse
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import threading
import time
import zipfile
import zlib
from .managed_storage import safe_key
from .workspace_export import MAX_ARCHIVE, MAX_FILE, MAX_TOTAL


INVALID = 'Invalid workspace archive or manifest'
UNAVAILABLE = 'Workspace archive verification is unavailable. Try again later.'
# Every refusal `_verify` raises, by the code the Rust leaf prints for the same check.
MESSAGES = {
    'archive_too_large': 'Archive exceeds 272 MiB',
    'invalid': INVALID,
    'entries': 'Too many archive entries or duplicate paths',
    'path': 'Invalid corpus path',
    'entry_type': 'Links, special files and encrypted entries are unsupported',
    'entry_limits': 'Archive entry exceeds limits or uses unsupported compression',
    'directory_not_empty': 'Directory entries must be empty',
    'type_mismatch': 'Archive entry type does not match its path',
    'expanded': 'Expanded archive exceeds limits',
    'manifest_size': 'Manifest exceeds 8 MiB',
    'manifest_object': 'Invalid manifest object',
    'format': 'Unsupported workspace format',
    'counts': 'Invalid manifest counts',
    'duplicate_directory': 'Duplicate directory',
    'file_record': 'Invalid manifest file record',
    'overlap': 'Duplicate or overlapping path',
    'checksum': 'Source checksum or size mismatch',
    'parent_overlap': 'File overlaps a parent directory',
    'entries_mismatch': 'Archive contains missing or unlisted entries',
    'totals_invalid': 'Invalid manifest totals',
    'totals_mismatch': 'Manifest totals mismatch',
}
IMPLS = ('python', 'rust')
DEFAULT_BIN = '/usr/local/bin/workspace-restore-verify'
# 272 MiB of SHA-256 plus inflate is a few seconds; this only bounds a hung child.
RUST_TIMEOUT_S = 120
_RUST_STDOUT_CAP = 256
_ACCEPT_RE = re.compile(rb'accept ([0-9a-f]{64})\n')
_REJECT_RE = re.compile(rb'reject: ([a-z_]{1,32})\n')
_log = logging.getLogger(__name__)
_logged = set()
_logged_lock = threading.Lock()


def _log_once(key, message):
    with _logged_lock:
        if key in _logged:
            return
        _logged.add(key)
    _log.warning(message)


def impl_choice():
    """The configured implementation: "python" (default) or "rust". Anything else is python."""
    value = (os.environ.get('WORKSPACE_RESTORE_VERIFY_IMPL') or 'python').strip().lower() or 'python'
    if value not in IMPLS:
        _log_once('invalid_setting', f'WORKSPACE_RESTORE_VERIFY_IMPL={value[:32]!r} is not one of {", ".join(IMPLS)}; using python')
        return 'python'
    return value


def _rust_binary():
    # A fixed absolute path by default (never a PATH lookup); WORKSPACE_RESTORE_VERIFY_BIN is for tests.
    configured = (os.environ.get('WORKSPACE_RESTORE_VERIFY_BIN') or DEFAULT_BIN).strip() or DEFAULT_BIN
    if os.sep in configured:
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None
    return shutil.which(configured)


def _rust_verdict(body):
    body = bytes(body)
    """The Rust leaf's listing digest for an accepted archive. Raises ValueError with Python's
    message for a refusal, or UNAVAILABLE for any fault (fails closed)."""
    binary = _rust_binary()
    if binary is None:
        _log_once('rust:missing_binary', 'workspace-restore-verify binary not found; refusing archives (WORKSPACE_RESTORE_VERIFY_IMPL=rust)')
        raise ValueError(UNAVAILABLE)
    try:
        proc = subprocess.Popen([binary, 'verify'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, close_fds=True,
                                env={'PATH': os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')})
    except OSError as exc:
        _log_once('rust:spawn', f'workspace-restore-verify could not start ({type(exc).__name__}); refusing the archive')
        raise ValueError(UNAVAILABLE) from None
    out = bytearray()
    overflow = threading.Event()

    def feed():
        try:
            proc.stdin.write(body)
        except (OSError, ValueError):
            pass  # The child exited early; its exit status decides.
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def drain():
        # At most cap+1 bytes are kept; the child is killed on overflow.
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
            code = proc.wait(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            if overflow.is_set():
                fault = 'output_too_large'
            elif time.monotonic() >= deadline:
                fault = 'timeout'
            if fault:
                proc.kill()
                code = proc.wait()
                break
    for t in threads:
        t.join(timeout=1)
    try:
        proc.stdout.close()
    except OSError:
        pass
    if fault or overflow.is_set():
        fault = fault or 'output_too_large'
        _log_once(f'rust:{fault}', f'workspace-restore-verify failed ({fault}); refusing the archive')
        raise ValueError(UNAVAILABLE)
    out = bytes(out)
    if len(out) <= _RUST_STDOUT_CAP:
        accepted = _ACCEPT_RE.fullmatch(out) if code == 0 else None
        if accepted:
            return accepted.group(1).decode('ascii')
        refused = _REJECT_RE.fullmatch(out) if code == 1 else None
        if refused and refused.group(1).decode('ascii') in MESSAGES:
            raise ValueError(MESSAGES[refused.group(1).decode('ascii')])
    _log_once(f'rust:exit:{code}', f'workspace-restore-verify failed (exit {code}); refusing the archive')
    raise ValueError(UNAVAILABLE)


def listing_sha256(files, dirs):
    """SHA-256 of the canonical verified listing; noevia-rs workspace-restore-verify's
    `Verified::listing_sha256` hashes the same bytes. Paths are safe keys, so never contain NUL."""
    digest = hashlib.sha256(b'noevia-workspace-verify-v1\0')
    for path in sorted(files):
        digest.update(f'F\0{path}\0{len(files[path])}\0{hashlib.sha256(files[path]).hexdigest()}\0'.encode())
    for directory in sorted(dirs):
        digest.update(f'D\0{directory}\0'.encode())
    digest.update(f'T\0{len(files)}\0{sum(map(len, files.values()))}\0'.encode())
    return digest.hexdigest()


def verify(body):
    choice = impl_choice()
    expected = _rust_verdict(body) if choice == 'rust' else None
    try:
        result = _verify(body)
    # zlib.error (corrupt DEFLATE), EOFError (DEFLATE ending early), NotImplementedError (newer ZIP
    # version or feature) and RecursionError (deeply nested manifest) used to escape as server
    # errors (noevia#982); they are invalid archives like every other parse failure.
    except (zipfile.BadZipFile, KeyError, TypeError, AttributeError, UnicodeError, json.JSONDecodeError,
            zlib.error, EOFError, NotImplementedError, RecursionError) as exc:
        raise ValueError(INVALID) from exc
    if expected is not None:
        try:
            same = listing_sha256(result[0], result[1]) == expected
        except UnicodeError:
            same = False
        if not same:
            _log_once('rust:listing_mismatch', 'workspace-restore-verify and Python verified different listings; refusing the archive')
            raise ValueError(UNAVAILABLE)
    return result


def _verify(body):
    if len(body) > MAX_ARCHIVE:
        raise ValueError('Archive exceeds 272 MiB')
    with zipfile.ZipFile(io.BytesIO(body)) as bundle:
        entries = bundle.infolist()
        if len(entries) > 10002 or len({e.filename for e in entries}) != len(entries):
            raise ValueError('Too many archive entries or duplicate paths')
        total = 0
        for entry in entries:
            safe_key(entry.filename.rstrip('/'))
            mode = entry.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1:
                raise ValueError('Links, special files and encrypted entries are unsupported')
            if entry.file_size > MAX_FILE or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError('Archive entry exceeds limits or uses unsupported compression')
            if entry.is_dir() and entry.file_size != 0:
                raise ValueError('Directory entries must be empty')
            if (stat.S_IFMT(mode) == stat.S_IFDIR and not entry.is_dir()) or (stat.S_IFMT(mode) == stat.S_IFREG and entry.is_dir()):
                raise ValueError('Archive entry type does not match its path')
            total += entry.file_size
        if total > MAX_TOTAL + 8 * 1024 * 1024:
            raise ValueError('Expanded archive exceeds limits')
        if bundle.getinfo('manifest.json').file_size > 8 * 1024 * 1024:
            raise ValueError('Manifest exceeds 8 MiB')
        manifest = json.loads(bundle.read('manifest.json'))
        if not isinstance(manifest, dict):
            raise ValueError('Invalid manifest object')
        if manifest.get('format') != 'noevia-workspace-v1':
            raise ValueError('Unsupported workspace format')
        rows, directories = manifest.get('files'), manifest.get('directories')
        if not isinstance(rows, list) or len(rows) > 5000 or not isinstance(directories, list) or len(directories) > 5000:
            raise ValueError('Invalid manifest counts')
        dirs = set()
        for directory in directories:
            safe_key(directory)
            if directory in dirs:
                raise ValueError('Duplicate directory')
            dirs.add(directory)
        files = {}
        for row in rows:
            if not isinstance(row, dict) or type(row.get('bytes')) is not int or row['bytes'] < 0 or not isinstance(row.get('sha256'), str):
                raise ValueError('Invalid manifest file record')
            path = safe_key(row['path'])
            if path in files or path in dirs:
                raise ValueError('Duplicate or overlapping path')
            data = bundle.read('workspace/' + path)
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Source checksum or size mismatch')
            files[path] = data
        for path in [*files, *dirs]:
            if any(str(parent) in files for parent in Path(path).parents if str(parent) != '.'):
                raise ValueError('File overlaps a parent directory')
        expected = {'manifest.json', 'workspace/'} | {'workspace/' + p for p in files} | {'workspace/' + p + '/' for p in dirs}
        if {e.filename for e in entries} != expected:
            raise ValueError('Archive contains missing or unlisted entries')
        total = sum(map(len, files.values()))
        if type(manifest.get('fileCount')) is not int or type(manifest.get('bytes')) is not int:
            raise ValueError('Invalid manifest totals')
        if total > MAX_TOTAL or manifest.get('fileCount') != len(files) or manifest.get('bytes') != total:
            raise ValueError('Manifest totals mismatch')
    return files, dirs, manifest


def preview(body):
    files, dirs, manifest = verify(body)
    return {'fingerprint': hashlib.sha256(body).hexdigest(), 'fileCount': len(files),
            'bytes': manifest['bytes'], 'files': sorted(files), 'directories': sorted(dirs),
            'mode': 'new-directory-only', 'settings': manifest.get('settings', {})}


def restore(body, destination, fingerprint):
    if hashlib.sha256(body).hexdigest() != fingerprint:
        raise ValueError('Archive changed. Review a fresh preview.')
    files, dirs, manifest = verify(body)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Restore requires a new directory; existing data will not be overwritten')
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        for directory in sorted(dirs):
            (destination / directory).mkdir(mode=0o700, parents=True, exist_ok=True)
        for path, data in files.items():
            target = destination / path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open('xb') as output:
                os.chmod(target, 0o600)
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {'fileCount': len(files), 'bytes': manifest['bytes'], 'settings': manifest.get('settings', {})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--destination')
    parser.add_argument('--fingerprint')
    args = parser.parse_args()
    if bool(args.destination) != bool(args.fingerprint):
        parser.error('--destination requires the reviewed --fingerprint')
    with open(args.archive, 'rb') as source:
        body = source.read(272 * 1024 * 1024 + 1)
    print(json.dumps(restore(body, args.destination, args.fingerprint) if args.destination else preview(body), indent=2))
