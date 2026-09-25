"""Bounded DELETE / MOVE / COPY for managed workspace files (docs/dav.md § Storage contract, D6).

One transaction per operation, all-or-nothing:
- delete moves every file into Trash with a recovery capsule (never a hard delete);
- move renames files and explicit directories in place;
- copy duplicates bytes with new timestamps.
Every destructive call carries the source version (a file's sha256 or a folder version derived
from its listing). Protected paths — Diary capture files, month files, the index and
``AI Memory/**`` — are never deleted, moved, overwritten or used as a destination.
"""
import base64
import hashlib
import json
import re
import time
import uuid

from .corpus import MARKER_RE
from .corpus_store import month_name_pattern
from .managed_storage import ManagedCorpusBackend, digest
from .workspace_files import MAX_FILE
from .workspace_trash import PREFIX as TRASH_PREFIX

MAX_ENTRIES = 500
MAX_BYTES = 50 * 1024 * 1024
MEMORY_FOLDER = 'AI Memory'


class OpError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def rel_path(path, allow_root=False):
    if not isinstance(path, str) or len(path) > 500 or '\\' in path or '\x00' in path:
        raise OpError(400, 'Invalid path')
    path = path.strip('/')
    if not path:
        if allow_root:
            return ''
        raise OpError(403, 'The workspace root cannot be changed')
    if any(not part or part in ('.', '..') or part.startswith('.') for part in path.split('/')):
        raise OpError(403, 'Choose a path inside the diary folder')
    return path


def protected(store, full):
    """True when ``full`` (a backend key) is a path clients may never delete, move or replace."""
    entries = store._join(store.entries_prefix).rstrip('/')
    memory = store._join(MEMORY_FOLDER).rstrip('/')
    # Field-specific substitutions (digits for year/month, {month_name} built from the actual
    # locale-dependent %B values month_filename() produces for months 1-12), matching
    # corpus_store.py's list_months() regex exactly (same helper, same [0-9]/ASCII digit
    # matching). A blanket "[^/]+" here previously made any "word-word.md" filename at the
    # corpus root match the default "{year}-{month02}.md" template, wrongly protecting
    # ordinary files from DELETE/MOVE/overwrite; a blanket "[A-Za-z]+" for {month_name} in turn
    # lost protection for locale month names such as "März" or "août".
    pattern = re.escape(store._join(store.monthly_prefix, store.month_file_template))
    pattern = pattern.replace(re.escape('{year}'), r'[0-9]{4}')
    pattern = pattern.replace(re.escape('{month02}'), r'[0-9]{2}')
    pattern = pattern.replace(re.escape('{month}'), r'[0-9]{1,2}')
    pattern = pattern.replace(re.escape('{month_name}'), f'(?:{month_name_pattern()})')
    return (full == store.index_path() or full in (entries, memory) or full.startswith(entries + '/')
            or full.startswith(memory + '/') or re.fullmatch(pattern, full, re.ASCII) is not None)


def require_managed(store):
    if not isinstance(store.backend, ManagedCorpusBackend) or not store.backend.active():
        raise OpError(409, 'This operation requires app-managed Diary storage')


def _tree(db, full):
    """Files (with bytes) and explicit directories at or under ``full``.

    Loads every file body of the subtree; callers that only need path/version
    (identity and folder_version, e.g. ``stat``) should use ``_tree_meta`` instead.
    """
    files = {r['path']: r for r in db.execute('SELECT path,data,version,updated FROM files WHERE path=? OR substr(path,1,?)=?',
                                               (full, len(full) + 1, full + '/'))}
    dirs = [r[0] for r in db.execute('SELECT path FROM directories WHERE path=? OR substr(path,1,?)=?', (full, len(full) + 1, full + '/'))]
    return files, dirs


def _tree_meta(db, full):
    """Files and explicit directories at or under ``full``, without loading file bodies.

    Used by ``stat``/``folder_version`` (e.g. folderTag on every PROPFIND Depth 1
    child), which only ever need path and version, never the bytes.
    """
    files = {r['path']: r for r in db.execute('SELECT path,version,updated FROM files WHERE path=? OR substr(path,1,?)=?',
                                               (full, len(full) + 1, full + '/'))}
    dirs = [r[0] for r in db.execute('SELECT path FROM directories WHERE path=? OR substr(path,1,?)=?', (full, len(full) + 1, full + '/'))]
    return files, dirs


def _kind(files, dirs, full):
    if full in files:
        return 'file'
    if files or dirs:
        return 'dir'
    return None


def folder_version(files, dirs, full):
    lines = sorted([f'f\t{p[len(full):]}\t{r["version"]}' for p, r in files.items()] + [f'd\t{p[len(full):]}' for p in dirs])
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def stat(store, path):
    require_managed(store)
    full = store._join(rel_path(path))
    with store.backend.db() as db:
        files, dirs = _tree_meta(db, full)
    kind = _kind(files, dirs, full)
    if kind is None:
        raise OpError(404, 'File or folder not found')
    return {'path': path.strip('/'), 'isDir': kind == 'dir',
            'version': files[full]['version'] if kind == 'file' else folder_version(files, dirs, full),
            'entries': 1 if kind == 'file' else len(files) + len(dirs)}


def _check_version(kind, files, dirs, full, expected, label='source'):
    if not isinstance(expected, str) or not expected:
        raise OpError(428, f'Read the {label} ETag and send it with If-Match')
    current = files[full]['version'] if kind == 'file' else folder_version(files, dirs, full)
    if current != expected:
        raise OpError(412, f'The {label} changed; read it again')


def _bounded(files, dirs):
    if len(files) + len(dirs) > MAX_ENTRIES or sum(len(bytes(r['data'])) for r in files.values()) > MAX_BYTES:
        raise OpError(507, f'Folder operations are limited to {MAX_ENTRIES} entries and 50 MiB')


def _refuse_protected(store, files, dirs, full):
    if protected(store, full) or any(protected(store, p) for p in list(files) + dirs):
        raise OpError(403, 'Diary capture files, month files, the index and AI Memory are protected')


def _parent_exists(db, full):
    parent = full.rsplit('/', 1)[0] if '/' in full else ''
    if not parent:
        return True
    if db.execute('SELECT 1 FROM files WHERE path=?', (parent,)).fetchone():
        return False
    return bool(db.execute('SELECT 1 FROM directories WHERE path=? UNION SELECT 1 FROM files WHERE substr(path,1,?)=? '
                           'UNION SELECT 1 FROM directories WHERE substr(path,1,?)=? LIMIT 1',
                           (parent, len(parent) + 1, parent + '/', len(parent) + 1, parent + '/')).fetchone())


def _capsule(db, store, full, row, now, restore_as=None):
    raw = bytes(row['data'])
    if len(raw) > MAX_FILE or MARKER_RE.search(raw.decode('utf-8', errors='replace')):
        raise OpError(403, 'Capture records or files over 512 KiB cannot be moved to Trash')
    operation = str(uuid.uuid4())
    root = store._join('').rstrip('/')
    relative = restore_as or (full[len(root) + 1:] if root else full)
    record = {'format': 'noevia-trash-v1', 'id': operation, 'path': relative, 'version': row['version'],
              'data': base64.b64encode(raw).decode(), 'state': 'trashed', 'trashedAt': now}
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
    db.execute('INSERT INTO files VALUES (?,?,?,?)', (TRASH_PREFIX + operation + '.json', encoded, digest(encoded), now))
    return operation


def _keep_parent(db, full):
    """Removing the last child must not make its folder vanish from listings."""
    parent = full.rsplit('/', 1)[0] if '/' in full else ''
    if parent and not db.execute('SELECT 1 FROM files WHERE substr(path,1,?)=? UNION SELECT 1 FROM directories WHERE path=? OR substr(path,1,?)=? LIMIT 1',
                                 (len(parent) + 1, parent + '/', parent, len(parent) + 1, parent + '/')).fetchone():
        db.execute('INSERT OR IGNORE INTO directories VALUES (?)', (parent,))


def _commit(backend, db, paths, now):
    for path in paths:
        db.execute('INSERT OR IGNORE INTO workspace_index_outbox VALUES (?)', (path,))
    backend.set_meta(db, 'generation', int(backend.meta(db, 'generation', '0')) + 1)
    backend.set_meta(db, 'due', now + backend.debounce)
    backend.set_meta(db, 'snapshot_time', now)


def delete(store, body):
    """Move a file, or a folder and everything in it, to Trash."""
    require_managed(store)
    full = store._join(rel_path(body.get('path')))
    backend = store.backend
    with backend.db() as db:
        db.execute('BEGIN IMMEDIATE')
        files, dirs = _tree(db, full)
        kind = _kind(files, dirs, full)
        if kind is None:
            raise OpError(404, 'File or folder not found')
        _refuse_protected(store, files, dirs, full)
        _check_version(kind, files, dirs, full, body.get('version'))
        _bounded(files, dirs)
        if any(p.startswith(TRASH_PREFIX) for p in files):
            raise OpError(403, 'Trash records cannot be deleted')
        if any(not p.lower().endswith('.md') for p in files):
            raise OpError(409, 'The folder holds files this endpoint cannot show; remove it in the app')
        now = time.time()
        trashed = [_capsule(db, store, p, r, now) for p, r in sorted(files.items())]
        db.execute('DELETE FROM files WHERE path=? OR substr(path,1,?)=?', (full, len(full) + 1, full + '/'))
        db.execute('DELETE FROM directories WHERE path=? OR substr(path,1,?)=?', (full, len(full) + 1, full + '/'))
        _keep_parent(db, full)
        _commit(backend, db, list(files), now)
    return {'deleted': body['path'].strip('/'), 'trash': trashed}


def _replaced_name(relative, now):
    """Where a preserved version restores to: beside the file, never over it.

    The result must stay within the 500-character path limit enforced by
    ``safe_path``/``rel_path`` elsewhere, even though ``relative`` itself was only
    checked against that same limit *before* the " (replaced ...)" suffix was
    added. Shorten the stem rather than the suffix, so restores stay grouped by
    timestamp and still land beside the original file.
    """
    stem = relative[:-3] if relative.lower().endswith('.md') else relative
    suffix = f" (replaced {time.strftime('%Y-%m-%d %H.%M.%S', time.gmtime(now))}).md"
    budget = 500 - len(suffix)
    if len(stem) > budget:
        # Keep the folder prefix intact (it is what a person recognizes and what
        # other checks such as `allowed()` reason about) and shorten only the
        # trailing filename segment. Never drop the folder entirely: a name left
        # with no '/' could accidentally collide with a top-level reserved
        # filename pattern (e.g. a month file).
        folder, _, name = stem.rpartition('/')
        prefix = folder + '/' if folder else ''
        room = budget - len(prefix)
        if room < 1:
            # Even the folder alone doesn't fit; fall back to trimming the whole
            # stem from the front so at least a valid, in-budget name results.
            stem = stem[-budget:].lstrip('.').lstrip('/') or 'file'
        else:
            name = name.lstrip('.') or 'file'
            stem = prefix + name[:room]
    return f"{stem}{suffix}"


def preserve(store, body):
    """Keep the current bytes of a file in Trash before a client replaces it without a precondition.

    Clients such as rclone, Finder and Obsidian sync never send If-Match, so their writes cannot be
    checked against what they last read. The previous version is kept instead: it appears in Trash
    and restores beside the current file. Protected files still require If-Match.
    """
    require_managed(store)
    relative = rel_path(body.get('path'))
    full = store._join(relative)
    backend = store.backend
    with backend.db() as db:
        db.execute('BEGIN IMMEDIATE')
        files, dirs = _tree(db, full)
        if _kind(files, dirs, full) != 'file':
            raise OpError(404, 'File not found')
        if protected(store, full):
            raise OpError(428, 'Diary capture files, month files, the index and AI Memory need If-Match to be replaced')
        _check_version('file', files, dirs, full, body.get('version'))
        now = time.time()
        operation = _capsule(db, store, full, files[full], now, restore_as=_replaced_name(relative, now))
        _commit(backend, db, [], now)
    return {'path': relative, 'trash': operation}


def _transfer(store, body, keep_source):
    require_managed(store)
    source = store._join(rel_path(body.get('path')))
    destination = store._join(rel_path(body.get('destination')))
    if destination == source or destination.startswith(source + '/'):
        raise OpError(403, 'A folder cannot be moved or copied into itself')
    overwrite = body.get('overwrite', False) is True
    backend = store.backend
    with backend.db() as db:
        db.execute('BEGIN IMMEDIATE')
        files, dirs = _tree(db, source)
        kind = _kind(files, dirs, source)
        if kind is None:
            raise OpError(404, 'File or folder not found')
        if keep_source:
            if protected(store, destination) or any(protected(store, destination + p[len(source):]) for p in list(files) + dirs):
                raise OpError(403, 'Diary capture files, month files, the index and AI Memory are protected')
            if 'version' in body:
                _check_version(kind, files, dirs, source, body.get('version'))
        else:
            _refuse_protected(store, files, dirs, source)
            _check_version(kind, files, dirs, source, body.get('version'))
            if protected(store, destination) or (kind == 'dir' and any(protected(store, destination + p[len(source):]) for p in list(files) + dirs)):
                raise OpError(403, 'Diary capture files, month files, the index and AI Memory are protected')
        _bounded(files, dirs)
        if kind == 'file' and not destination.lower().endswith('.md'):
            raise OpError(403, 'Only Markdown file names are allowed')
        if not _parent_exists(db, destination):
            raise OpError(409, 'Destination folder does not exist')
        dest_files, dest_dirs = _tree(db, destination)
        dest_kind = _kind(dest_files, dest_dirs, destination)
        now = time.time()
        replaced = False
        if dest_kind is not None:
            if not overwrite:
                raise OpError(412, 'Destination exists; send Overwrite: T with its ETag to replace it')
            if dest_kind != 'file' or kind != 'file':
                raise OpError(409, 'Replacing folders is not supported; delete the destination first')
            _check_version(dest_kind, dest_files, dest_dirs, destination, body.get('destinationVersion'), 'destination')
            _capsule(db, store, destination, dest_files[destination], now)
            db.execute('DELETE FROM files WHERE path=?', (destination,))
            replaced = True
        touched = []
        for path, row in files.items():
            target = destination + path[len(source):]
            if keep_source:
                data = bytes(row['data'])
                db.execute('INSERT INTO files VALUES (?,?,?,?)', (target, data, digest(data), now))
            else:
                db.execute('UPDATE files SET path=?, updated=? WHERE path=?', (target, now, path))
                touched.append(path)
            touched.append(target)
        for path in dirs:
            target = destination + path[len(source):]
            if keep_source:
                db.execute('INSERT OR IGNORE INTO directories VALUES (?)', (target,))
            else:
                db.execute('UPDATE directories SET path=? WHERE path=?', (target, path))
        if not keep_source:
            _keep_parent(db, source)
        _commit(backend, db, touched, now)
    return {'path': body['path'].strip('/'), 'destination': body['destination'].strip('/'), 'replaced': replaced}


def move(store, body):
    return _transfer(store, body, keep_source=False)


def copy(store, body):
    return _transfer(store, body, keep_source=True)


def operate(store, body):
    op = body.get('op') if isinstance(body, dict) else None
    handlers = {'stat': lambda: stat(store, body.get('path')), 'delete': lambda: delete(store, body),
                'move': lambda: move(store, body), 'copy': lambda: copy(store, body),
                'preserve': lambda: preserve(store, body)}
    if op not in handlers:
        raise OpError(400, 'Choose stat, delete, move, copy or preserve')
    return handlers[op]()
