"""Bounded portable source export. No credentials, indexes or browser drafts."""
import hashlib
import io
import json
import zipfile
from .diary_migration import snapshot
from .managed_storage import ManagedCorpusBackend, safe_key

MAX_ARCHIVE = 272 * 1024 * 1024
MAX_FILE = 64 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024


def archive(backend, prefix, settings):
    if isinstance(backend, ManagedCorpusBackend):
        # One SQLite read transaction, including directories. Avoid independent
        # per-file reads that could combine different committed generations.
        with backend.db() as db:
            db.execute('BEGIN')
            sizes = db.execute('SELECT count(*),coalesce(sum(length(data)),0),coalesce(max(length(data)),0) FROM files').fetchone()
            if sizes[0] > 5000 or sizes[1] > MAX_TOTAL or sizes[2] > MAX_FILE:
                raise ValueError('Export exceeds the 5,000 file / 64 MiB per file / 256 MiB safety limit')
            directories = [r[0] for r in db.execute('SELECT path FROM directories ORDER BY path LIMIT 5002')]
            if len(directories) > 5000:
                raise ValueError('Export contains too many directories')
            files = {r[0]: bytes(r[1]) for r in db.execute('SELECT path,data FROM files ORDER BY path')}
    else:
        files, report = snapshot(backend, prefix, allow_empty=True)
        _, second = snapshot(backend, prefix, allow_empty=True)
        if report['fingerprint'] != second['fingerprint']:
            raise ValueError('Stored files changed during export. Retry when other writers have finished.')
        directories = report['directories']
    if len(directories) > 5000:
        raise ValueError('Export contains too many directories')
    # Include inferred parents and explicit empty folders.
    for path in [*files, *directories]:
        safe_key(path)
    directories = sorted(set(directories) | {path.rsplit('/', 1)[0] for path in files if '/' in path})
    for path in list(directories):
        parts = path.split('/')
        directories.extend('/'.join(parts[:i]) for i in range(1, len(parts)))
    directories = sorted(set(directories))
    if len(directories) > 5000:
        raise ValueError('Export contains too many directories')
    if any(path in files for path in directories):
        raise ValueError('Stored file and directory paths overlap')
    manifest = {'format': 'noevia-workspace-v1', 'settings': settings,
                'directories': directories, 'fileCount': len(files),
                'bytes': sum(map(len, files.values())),
                'files': [{'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for path, data in sorted(files.items())]}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode()
    if len(manifest_bytes) > 8 * 1024 * 1024:
        raise ValueError('Export manifest exceeds 8 MiB')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr('manifest.json', manifest_bytes)
        bundle.writestr('workspace/', b'')
        for directory in directories:
            bundle.writestr('workspace/' + directory + '/', b'')
        for path, data in sorted(files.items()):
            bundle.writestr('workspace/' + path, data)
    if output.tell() > MAX_ARCHIVE:
        raise ValueError('Export archive exceeds 272 MiB')
    return output.getvalue()
