"""Reversible managed-file removal, with portable recovery capsules.

Capsules are ordinary hidden JSON files in the managed source transaction, so
existing immutable backups and ZIP exports preserve them without extra tables.
They are never Markdown, never indexed, and retained after restore as receipts.
"""
import base64
import json
import re
import time
import uuid
from .managed_storage import ManagedCorpusBackend, digest
from .workspace_files import safe_path, MAX_FILE
from .corpus import MARKER_RE

PREFIX = '.noevia-trash/'


def identity(value):
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise ValueError('A valid recovery operation ID is required')


def allowed(store, path):
    safe_path(path)
    full = store._join(path)
    entries = store._join(store.entries_prefix).rstrip('/')
    pattern = re.escape(store._join(store.monthly_prefix, store.month_file_template))
    for field in ('year', 'month', 'month02', 'month_name'):
        pattern = pattern.replace(re.escape('{' + field + '}'), '[^/]+')
    if full == store.index_path() or full == entries or full.startswith(entries + '/') or re.fullmatch(pattern, full):
        raise ValueError('Diary capture files and the Diary index cannot be moved to Trash')
    return full


def require_managed(store):
    if not isinstance(store.backend, ManagedCorpusBackend) or not store.backend.active():
        raise ValueError('Trash requires app-managed Diary storage')


def decode(data, operation):
    try:
        record = json.loads(data)
        if record['format'] != 'noevia-trash-v1' or record['id'] != operation or record['state'] not in ('trashed', 'restored'):
            raise ValueError()
        safe_path(record['path'])
        raw = base64.b64decode(record['data'], validate=True)
        if len(raw) > MAX_FILE or digest(raw) != record['version']:
            raise ValueError()
        return record, raw
    except Exception as exc:
        raise ValueError('Recovery record is invalid; preserved bytes have not been changed') from exc


def summary(record):
    return {key: record[key] for key in ('id', 'path', 'version', 'state', 'trashedAt')}


def list_trash(store, after=''):
    require_managed(store)
    if after:
        after = identity(after)
    with store.backend.db() as db:
        rows = db.execute('SELECT path,data FROM files WHERE substr(path,1,?)=? AND path>? ORDER BY path LIMIT 101',
                          (len(PREFIX), PREFIX, PREFIX + after + ('.json' if after else ''))).fetchall()
    records = []
    for row in rows[:100]:
        operation = identity(row['path'][len(PREFIX):-5])
        record, _ = decode(row['data'], operation)
        records.append(summary(record))
    return {'records': records, 'next': records[-1]['id'] if len(rows) > 100 else None}


def change(store, body):
    require_managed(store)
    operation = identity(body.get('id'))
    action = body.get('action')
    if action not in ('trash', 'restore'):
        raise ValueError('Choose trash or restore')
    backend = store.backend
    capsule = PREFIX + operation + '.json'
    with backend.db() as db:
        db.execute('BEGIN IMMEDIATE')
        existing = db.execute('SELECT data FROM files WHERE path=?', (capsule,)).fetchone()
        now = time.time()
        if action == 'trash':
            path = body.get('path')
            full = allowed(store, path)
            if existing:
                record, _ = decode(existing[0], operation)
                if record['path'] != path or record['version'] != body.get('version'):
                    raise ValueError('Operation ID was already used for a different file version')
                return {**summary(record), 'alreadyApplied': True}
            row = db.execute('SELECT data,version FROM files WHERE path=?', (full,)).fetchone()
            if not row or row['version'] != body.get('version'):
                raise ValueError('File changed or is missing. Reopen it before moving it to Trash.')
            raw = bytes(row['data'])
            if len(raw) > MAX_FILE or MARKER_RE.search(raw.decode('utf-8', errors='replace')):
                raise ValueError('Capture records or files over 512 KiB cannot be moved to Trash')
            record = {'format': 'noevia-trash-v1', 'id': operation, 'path': path,
                      'version': row['version'], 'data': base64.b64encode(raw).decode(),
                      'state': 'trashed', 'trashedAt': now}
            # Refuse pre-existing namespace conflicts, including imported files.
            if db.execute('SELECT 1 FROM files WHERE path=?', (PREFIX.rstrip('/'),)).fetchone() or db.execute('SELECT 1 FROM directories WHERE path=?', (capsule,)).fetchone() or db.execute('SELECT 1 FROM files WHERE substr(path,1,?)=? LIMIT 1', (len(capsule)+1, capsule+'/')).fetchone():
                raise ValueError('Recovery namespace conflicts with existing data')
            db.execute('DELETE FROM files WHERE path=?', (full,))
        else:
            if not existing:
                raise ValueError('Recovery record was not found')
            record, raw = decode(existing[0], operation)
            full = allowed(store, record['path'])
            if record['state'] == 'restored':
                return {**summary(record), 'alreadyApplied': True}
            paths = [r[0] for r in db.execute('SELECT path FROM files UNION SELECT path FROM directories')]
            if any(p == full or p.startswith(full + '/') or (full.startswith(p + '/') and db.execute('SELECT 1 FROM files WHERE path=?', (p,)).fetchone()) for p in paths):
                raise ValueError('The original path is occupied. Restore never overwrites existing data.')
            db.execute('INSERT INTO files VALUES (?,?,?,?)', (full, raw, record['version'], now))
            record['state'] = 'restored'
            record['restoredAt'] = now
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
        db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?)', (capsule, encoded, digest(encoded), now))
        db.execute('INSERT OR IGNORE INTO workspace_index_outbox VALUES (?)', (full,))
        backend.set_meta(db, 'generation', int(backend.meta(db, 'generation', '0')) + 1)
        backend.set_meta(db, 'due', now + backend.debounce)
        backend.set_meta(db, 'snapshot_time', now)
    return {**summary(record), 'alreadyApplied': False}


def recover_capsule(source, destination):
    """Explicit operator extraction from an exported capsule into a NEW file."""
    import os
    from pathlib import Path
    source, destination = Path(source), Path(destination)
    with source.open('rb') as handle:
        encoded = handle.read(MAX_FILE * 2 + 1)
    if len(encoded) > MAX_FILE * 2:
        raise ValueError('Recovery record exceeds limits')
    record, raw = decode(encoded, identity(source.stem))
    # Exclusive creation also refuses dangling symlinks. Caller chooses the new
    # destination; capsule metadata can never select a write path.
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink()
        raise
    return {'path': record['path'], 'bytes': len(raw), 'version': record['version']}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Recover one exported Trash capsule to a new file; never overwrite or activate a corpus.')
    parser.add_argument('capsule')
    parser.add_argument('new_file')
    args = parser.parse_args()
    print(json.dumps(recover_capsule(args.capsule, args.new_file), indent=2))
