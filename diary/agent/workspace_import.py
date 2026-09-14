"""Verified new-folder imports. Source, receipt and index outbox commit together."""
import hashlib
import time
from .managed_storage import ManagedCorpusBackend, digest, safe_key
from .workspace_restore import verify


def destination_path(name):
    if not isinstance(name, str):
        raise ValueError('Choose a new folder name of 1–100 characters')
    name = name.strip()
    if not name or len(name) > 100 or '/' in name or name.startswith('.') or any(ord(c) < 32 for c in name):
        raise ValueError('Choose a new folder name of 1–100 characters')
    return safe_key('Imports/' + name.strip())


def fingerprint(body, destination):
    return hashlib.sha256(destination.encode() + b'\x00' + body).hexdigest()


def conflicts(db, destination):
    paths = [r[0] for r in db.execute('SELECT path FROM files UNION SELECT path FROM directories')]
    found = [p for p in paths if p == destination or p.startswith(destination + '/')]
    ancestors = destination.split('/')[:-1]
    for i in range(1, len(ancestors) + 1):
        path = '/'.join(ancestors[:i])
        if db.execute('SELECT 1 FROM files WHERE path=?', (path,)).fetchone():
            found.append(path)
    return sorted(set(found))


def prepare(backend, body, name):
    if not isinstance(backend, ManagedCorpusBackend) or not backend.active():
        raise ValueError('ZIP import requires app-managed Diary storage')
    files, directories, manifest = verify(body)
    destination = destination_path(name)
    with backend.db() as db:
        db.execute('BEGIN')
        applied = bool(db.execute('SELECT 1 FROM workspace_imports WHERE fingerprint=?', (fingerprint(body, destination),)).fetchone())
        blocked = conflicts(db, destination)
        hashes = {r[0] for r in db.execute('SELECT version FROM files')}
    return files, directories, {'fingerprint': fingerprint(body, destination),
        'destination': destination, 'fileCount': len(files), 'bytes': manifest['bytes'],
        'files': sorted(files), 'directories': sorted(directories), 'conflicts': blocked,
        'duplicates': sorted(path for path, data in files.items() if digest(data) in hashes),
        'mode': 'new-folder-only', 'alreadyApplied': applied}


def apply(backend, body, name, reviewed):
    files, directories, report = prepare(backend, body, name)
    if reviewed != report['fingerprint']:
        raise ValueError('Archive or destination changed. Review a fresh preview.')
    destination = report['destination']
    with backend.db() as db:
        db.execute('BEGIN IMMEDIATE')
        # A durable receipt makes an uncertain HTTP retry harmless, even if
        # imported files have subsequently been edited. Never reapply source.
        if db.execute('SELECT 1 FROM workspace_imports WHERE fingerprint=?', (reviewed,)).fetchone():
            return {'destination': destination, 'fileCount': len(files), 'alreadyApplied': True}
        if conflicts(db, destination):
            raise ValueError('Destination already exists or conflicts. Choose a new folder.')
        now = time.time()
        for path, data in files.items():
            target = destination + '/' + path
            db.execute('INSERT INTO files VALUES (?,?,?,?)', (target, data, digest(data), now))
            if path.lower().endswith('.md'):
                db.execute('INSERT OR IGNORE INTO workspace_index_outbox VALUES (?)', (target,))
        all_dirs = {'Imports', destination} | {destination + '/' + p for p in directories}
        for path in files:
            parts = (destination + '/' + path).split('/')
            all_dirs.update('/'.join(parts[:i]) for i in range(1, len(parts)))
        for path in sorted(all_dirs):
            db.execute('INSERT OR IGNORE INTO directories VALUES (?)', (path,))
        backend.set_meta(db, 'generation', int(backend.meta(db, 'generation', '0')) + 1)
        backend.set_meta(db, 'due', now + backend.debounce)
        backend.set_meta(db, 'snapshot_time', now)
        db.execute('INSERT INTO workspace_imports VALUES (?,?)', (reviewed, destination))
    return {'destination': destination, 'fileCount': len(files), 'alreadyApplied': False}


def drain_index_outbox(backend, journal):
    """Call under store write lock before indexing. Never clear before durable mark."""
    if not isinstance(backend, ManagedCorpusBackend):
        return
    with backend.db() as db:
        db.execute('BEGIN IMMEDIATE')
        for row in db.execute('SELECT path FROM workspace_index_outbox').fetchall():
            journal.mark_dirty(row[0])
            db.execute('DELETE FROM workspace_index_outbox WHERE path=?', (row[0],))
