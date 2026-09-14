"""Explicit, bounded copy-and-verify import; originals are never mutated."""
import hashlib
import json
from contextlib import contextmanager
from .managed_storage import safe_key

SETTING_KEYS = ('monthly_prefix', 'entry_layout', 'entries_prefix', 'index_file', 'month_file_template', 'index_enabled')


def corpus_settings(cfg):
    return {key: cfg.get('corpus.' + key) for key in SETTING_KEYS if cfg.get('corpus.' + key) is not None}


def snapshot(backend, prefix):
    prefix = prefix.strip('/')
    files = {}
    directories = [prefix]
    seen = set()
    folders = []
    total = 0
    while directories:
        directory = directories.pop()
        if directory in seen or len(seen) > 5000:
            raise ValueError('Import contains too many directories or a directory cycle')
        seen.add(directory)
        names = set()
        for row in backend.list_dir(directory):
            name = row.get('name', '')
            if not name or '/' in name or name in ('.', '..') or '\\' in name or name in names:
                raise ValueError('Invalid import listing')
            names.add(name)
            # Backup history is outside the original corpus; never recursively import it.
            if directory == prefix and name == 'noevia-backups':
                continue
            path = directory + '/' + name if directory else name
            if row.get('path', '').rstrip('/') != path:
                raise ValueError('Import listing escaped its directory')
            safe_key(path)
            if row.get('is_dir'):
                directories.append(path)
                folders.append(path[len(prefix) + 1:] if prefix else path)
                continue
            bounded_get = getattr(backend, 'get_bounded', None)
            data, _ = bounded_get(path, min(64 * 1024 * 1024, 256 * 1024 * 1024 - total)) if bounded_get else backend.get(path)
            if data is None:
                raise ValueError('A source file changed during import')
            total += len(data)
            if len(data) > 64 * 1024 * 1024 or total > 256 * 1024 * 1024 or len(files) >= 5000:
                raise ValueError('Import exceeds the 5,000 file / 256 MiB safety limit')
            relative = path[len(prefix) + 1:] if prefix else path
            files[relative] = data
    if not files:
        raise ValueError('No source files found. Verify the original connection and folder before importing.')
    listing = [{'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for path, data in sorted(files.items())]
    fingerprint = hashlib.sha256(json.dumps({'files': listing, 'directories': sorted(folders)}, sort_keys=True).encode()).hexdigest()
    return files, {'fingerprint': fingerprint, 'files': listing, 'directories': sorted(folders), 'fileCount': len(files), 'bytes': total}


class LegacyWriteGuard:
    """Old in-flight requests cannot write to the retired corpus after activation."""
    def __init__(self, backend, managed):
        self.backend, self.managed = backend, managed

    def __getattr__(self, key):
        return getattr(self.backend, key)

    @contextmanager
    def write_transaction(self):
        with self.managed.migration_lock():
            if self.managed.active():
                raise RuntimeError('Diary moved into the app. Reload before writing; no new write was queued.')
            yield

    def put(self, *args, **kwargs):
        with self.managed.migration_lock():
            if self.managed.active():
                raise RuntimeError('Diary moved into the app. Reload before writing; this request was not saved to the old corpus.')
            return self.backend.put(*args, **kwargs)

    def create_directory(self, *args, **kwargs):
        with self.managed.migration_lock():
            if self.managed.active():
                raise RuntimeError('Diary moved into the app. Reload before writing.')
            return self.backend.create_directory(*args, **kwargs)
