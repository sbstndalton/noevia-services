"""Transactional app-owned Markdown and an append-only WebDAV backup outbox.

Corpus bytes and the debounce deadline commit in one SQLite transaction. Network
I/O never runs in a save transaction. Immutable remote objects and manifests make
retry after an uncertain PUT safe without overwriting any remote document.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import threading
import uuid
from pathlib import Path
from contextlib import contextmanager
import fcntl
import os
from .dedicated_storage import StorageUnavailable


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_key(path):
    if not isinstance(path, str) or path.startswith('/') or '\\' in path or '\x00' in path or any(p in ('', '.', '..') for p in path.split('/')):
        raise ValueError('Invalid corpus path')
    return path


_migration_locks = {}
_migration_guard = threading.Lock()
_migration_depth = threading.local()


class ManagedCorpusBackend:
    """SQLite is primary storage; plain Markdown is exposed through the file API."""
    def __init__(self, root, tenant, debounce=3.0):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tenant = str(uuid.UUID(tenant))
        self.db_path = self.root / 'managed-diary.db'
        self.debounce = debounce
        self.marker = self.root / 'managed-diary.active'
        if self.marker.exists() and (not self.db_path.is_file() or self.marker.read_text() != self.tenant):
            raise StorageUnavailable('App Diary storage is missing or has the wrong identity. Restore its backup before continuing.')
        if not self.db_path.exists():
            # Only construction may create a database; operational reads/writes
            # below use mode=rw and never replace a disappeared database.
            sqlite3.connect(self.db_path).close()
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, data BLOB NOT NULL, version TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS directories(path TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS workspace_imports(fingerprint TEXT PRIMARY KEY, destination TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS workspace_index_outbox(path TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS backups(destination TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT -1,
                    last_success REAL, retry_at REAL NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0, error TEXT);
            ''')
            identity = self.meta(db, 'tenant')
            if identity is not None and identity != self.tenant:
                raise StorageUnavailable('App Diary tenant identity mismatch.')
            self.set_meta(db, 'tenant', self.tenant)
            if self.marker.exists() and self.meta(db, 'active') != '1':
                raise StorageUnavailable('App Diary activation is incomplete. Restore or review the staged database.')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.db_path.as_uri() + '?mode=rw', uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def migration_lock(self):
        key = str(self.root)
        with _migration_guard:
            lock = _migration_locks.setdefault(key, threading.RLock())
        with lock:
            depths = getattr(_migration_depth, 'paths', None)
            if depths is None:
                depths = _migration_depth.paths = {}
            if depths.get(key, 0):
                depths[key] += 1
                try:
                    yield
                finally:
                    depths[key] -= 1
                return
            with (self.root / 'managed-diary.lock').open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                depths[key] = 1
                try:
                    yield
                finally:
                    depths.pop(key, None)
                    fcntl.flock(handle, fcntl.LOCK_UN)

    @staticmethod
    def meta(db, key, default=None):
        row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return row[0] if row else default

    @staticmethod
    def set_meta(db, key, value):
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, str(value)))

    def active(self):
        with self.db() as db:
            return self.meta(db, 'active') == '1'

    def activate(self, files, settings, directories=()):
        """Only caller's verified import may activate; transaction never merges."""
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if self.meta(db, 'active') == '1' or db.execute('SELECT 1 FROM files LIMIT 1').fetchone():
                raise ValueError('App diary already exists; automatic merge is refused')
            now = time.time()
            for path, data in files.items():
                safe_key(path)
                db.execute('INSERT INTO files VALUES (?,?,?,?)', (path, data, digest(data), now))
            for directory in directories:
                safe_key(directory)
                db.execute('INSERT INTO directories VALUES (?)', (directory,))
            self.set_meta(db, 'snapshot_time', now)
            self.set_meta(db, 'settings', json.dumps(settings, sort_keys=True))
            self.set_meta(db, 'generation', 1)
            self.set_meta(db, 'due', now + self.debounce)
            self.set_meta(db, 'active', 1)
            # Fail closed even if storage disappears after the cutover. A crash
            # before commit leaves an explicit incomplete-activation condition.
            with self.marker.open('w') as marker:
                marker.write(self.tenant)
                marker.flush()
                os.fsync(marker.fileno())
            fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def settings(self):
        with self.db() as db:
            return json.loads(self.meta(db, 'settings', '{}'))

    def get(self, path):
        safe_key(path)
        with self.db() as db:
            row = db.execute('SELECT data,version FROM files WHERE path=?', (path,)).fetchone()
        return (bytes(row[0]), '"sha256:' + row[1] + '"') if row else (None, None)

    def get_text(self, path):
        data, version = self.get(path)
        return (data.decode('utf-8', errors='replace') if data is not None else None, version)

    def put(self, path, data, if_match=None, if_none_match='*', max_retries=5):
        safe_key(path)
        data = bytes(data)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT version FROM files WHERE path=?', (path,)).fetchone()
            current = '"sha256:' + row[0] + '"' if row else None
            if (if_match is not None and current != if_match) or (if_match is None and if_none_match == '*' and row):
                return False, None, 412
            if db.execute('SELECT 1 FROM directories WHERE path=?', (path,)).fetchone() or db.execute('SELECT 1 FROM files WHERE substr(path,1,?)=? LIMIT 1', (len(path)+1, path+'/')).fetchone():
                raise IsADirectoryError(path)
            for ancestor in Path(path).parents:
                if str(ancestor) != '.' and db.execute('SELECT 1 FROM files WHERE path=?', (str(ancestor),)).fetchone():
                    raise NotADirectoryError(str(ancestor))
            version = digest(data)
            now = time.time()
            db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?)', (path, data, version, now))
            self.set_meta(db, 'generation', int(self.meta(db, 'generation', '0')) + 1)
            self.set_meta(db, 'due', now + self.debounce)
            self.set_meta(db, 'snapshot_time', now)
        return True, '"sha256:' + version + '"', 204 if row else 201

    def exists(self, path):
        return self.get(path)[0] is not None

    def list_dir(self, path):
        if path:
            safe_key(path)
        prefix = path + '/' if path else ''
        with self.db() as db:
            rows = db.execute('SELECT path,version,updated FROM files UNION ALL SELECT path,NULL,NULL FROM directories').fetchall()
        entries = {}
        for row in rows:
            if not row['path'].startswith(prefix):
                continue
            rest = row['path'][len(prefix):]
            if not rest:
                continue
            name = rest.split('/')[0]
            is_dir = '/' in rest or row['version'] is None
            entries[name] = {'name': name, 'path': prefix + name, 'is_dir': is_dir,
                             'etag': None if is_dir else '"sha256:' + row['version'] + '"', 'lastmod': row['updated']}
        return sorted(entries.values(), key=lambda row: row['name'].lower())

    def create_directory(self, path):
        safe_key(path)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            parent = path.rsplit('/', 1)[0] if '/' in path else ''
            all_paths = [row[0] for row in db.execute('SELECT path FROM files UNION SELECT path FROM directories')]
            if path in all_paths or any(p.startswith(path + '/') for p in all_paths):
                raise FileExistsError(path)
            if parent and not any(p.startswith(parent + '/') or p == parent for p in all_paths):
                raise FileNotFoundError(parent)
            if parent and db.execute('SELECT 1 FROM files WHERE path=?', (parent,)).fetchone():
                raise NotADirectoryError(parent)
            db.execute('INSERT INTO directories VALUES (?)', (path,))
            self.set_meta(db, 'generation', int(self.meta(db, 'generation', '0')) + 1)
            now = time.time()
            self.set_meta(db, 'due', now + self.debounce)
            self.set_meta(db, 'snapshot_time', now)

    @staticmethod
    def destination(storage):
        if not storage or storage.get('kind') not in ('webdav', 'nextcloud'):
            return None
        return digest(json.dumps([storage.get(k, '') for k in ('baseUrl', 'username', 'corpusRoot')], separators=(',', ':')).encode())

    def status(self, storage=None):
        if storage and storage.get('kind') == 'blocked':
            return {'mode': 'managed' if self.active() else 'legacy', 'backup': 'failed',
                    'lastBackedUp': None, 'retryAt': None,
                    'error': 'Backup endpoint is not approved. Ask an administrator to review the connection; app saves are retained.'}
        destination = self.destination(storage)
        with self.db() as db:
            generation = int(self.meta(db, 'generation', '0'))
            row = db.execute('SELECT * FROM backups WHERE destination=?', (destination,)).fetchone() if destination else None
            pending = not row or row['generation'] != generation
            return {'mode': 'managed' if self.meta(db, 'active') == '1' else 'legacy',
                    'backup': 'not_configured' if not destination else 'failed' if row and row['error'] else 'pending' if pending else 'complete',
                    'lastBackedUp': row['last_success'] if row else None,
                    'retryAt': row['retry_at'] if row and row['error'] else None,
                    'error': row['error'] if row else None}

    def backup(self, remote, storage, now=None):
        """Idempotent full manifest; immutable content objects are deduplicated."""
        now = time.time() if now is None else now
        destination = self.destination(storage)
        if not destination or not self.active():
            return self.status(storage)
        # Serialize workers across processes, while ordinary SQLite saves proceed.
        with (self.root / 'managed-backup.lock').open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return self.status(storage)
            with self.db() as db:
                db.execute('BEGIN')
                generation = int(self.meta(db, 'generation', '0'))
                row = db.execute('SELECT * FROM backups WHERE destination=?', (destination,)).fetchone()
                if float(self.meta(db, 'due', '0')) > now or (row and (row['generation'] == generation or row['retry_at'] > now)):
                    return self.status(storage)
                # Metadata only: content blobs are fetched one row at a time
                # below (see the loop), so a tenant with a large corpus never
                # holds every file's bytes in memory at once (#219). Loading
                # a bare (path, version) pair per file is cheap even for
                # thousands of documents.
                file_meta = db.execute('SELECT path, version FROM files ORDER BY path').fetchall()
                directories = [r[0] for r in db.execute('SELECT path FROM directories ORDER BY path')]
                settings = self.meta(db, 'settings', '{}')
                snapshot_time = float(self.meta(db, 'snapshot_time', '0'))
            base = '/'.join(p.strip('/') for p in (storage.get('corpusRoot', ''), 'noevia-backups', self.tenant) if p)
            try:
                safe_key(base)
                manifest = {'format': 'noevia-diary-backup-v1', 'tenant': self.tenant, 'generation': generation,
                            'savedAt': snapshot_time, 'settings': json.loads(settings), 'directories': directories, 'files': []}
                for path, version in ((r['path'], r['version']) for r in file_meta):
                    object_path = base + '/objects/' + version + '/' + path.rsplit('/', 1)[-1]
                    # Fetch this file's bytes only when it is actually about to
                    # be verified/uploaded, not all up front (#219). Every
                    # object is still verified against the remote every
                    # generation: objects are supposed to be immutable once
                    # written, but a corrupted/edited remote object must still
                    # be detected rather than silently trusted (see
                    # test_remote_conflict_preserves_remote_and_local).
                    with self.db() as fdb:
                        row = fdb.execute('SELECT data FROM files WHERE path=?', (path,)).fetchone()
                    data = bytes(row[0]) if row else b''
                    self._create_verified(remote, object_path, data)
                    manifest['files'].append({'path': path, 'sha256': version, 'object': object_path})
                body = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
                self._create_verified(remote, base + '/manifests/' + digest(body) + '.json', body)
                with self.db() as db:
                    db.execute('INSERT OR REPLACE INTO backups VALUES (?,?,?,0,0,NULL)', (destination, generation, time.time()))
            except Exception:
                # Never persist server exceptions/URLs: these can contain credentials.
                with self.db() as db:
                    previous = db.execute('SELECT * FROM backups WHERE destination=?', (destination,)).fetchone()
                    failures = (previous['failures'] if previous else 0) + 1
                    db.execute('INSERT OR REPLACE INTO backups VALUES (?,?,?,?,?,?)',
                               (destination, previous['generation'] if previous else -1,
                                previous['last_success'] if previous else None,
                                now + min(300, 5 * 2 ** min(failures - 1, 6)), failures,
                                'Backup could not be verified. Check the connection or remote conflict; app saves are retained.'))
            return self.status(storage)

    @staticmethod
    def _create_verified(remote, path, data):
        bounded_get = getattr(remote, 'get_bounded', None)
        def read():
            return bounded_get(path, len(data)) if bounded_get else remote.get(path)
        existing, _ = read()
        if existing is None:
            ok, _, _ = remote.put(path, data, if_none_match='*', max_retries=1)
            # Re-read even after success: the manifest must describe verified bytes.
            existing, _ = read()
        if existing != data:
            raise ValueError('Remote backup object conflict')

    def close(self):
        pass
