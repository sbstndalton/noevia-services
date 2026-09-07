"""Tenant-relative Markdown files, guarded writes, and ephemeral browser corpora."""
import hashlib
import re
from fastapi import HTTPException

MAX_FILE = 512 * 1024
MAX_TOTAL = 12 * 1024 * 1024
MAX_FILES = 500


def safe_path(path, directory=False):
    if not isinstance(path, str) or len(path) > 500 or '\\' in path or '\x00' in path:
        raise HTTPException(400, 'Invalid path')
    if directory and path == '':
        return path
    if any(not part or part in ('.', '..') or part.startswith('.') for part in path.split('/')):
        raise HTTPException(400, 'Choose a path inside the diary folder')
    if not directory and not path.lower().endswith('.md'):
        raise HTTPException(400, 'Only Markdown files are supported')
    return path


def version(text):
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()


def file_list(store, path=''):
    safe_path(path, True)
    entries = store.backend.list_dir(store._join(path))
    out = []
    for item in entries:
        name = item.get('name', '')
        if not name or '/' in name or name.startswith('.'):
            continue
        rel = '/'.join(p for p in (path, name) if p)
        is_dir = bool(item.get('is_dir'))
        if is_dir or name.lower().endswith('.md'):
            out.append({'path': rel, 'name': name, 'isDir': is_dir})
        if len(out) > MAX_FILES:
            raise HTTPException(413, 'This folder has too many files; choose a subfolder')
    return out


def file_read(store, path):
    safe_path(path)
    text, _ = store.backend.get_text(store._join(path))
    if text is not None and len(text.encode()) > MAX_FILE:
        raise HTTPException(413, 'File exceeds the 512 KiB editor limit')
    return {'path': path, 'content': text, 'version': version(text)}


def file_write(store, body):
    path = safe_path(body.get('path'))
    content = body.get('content')
    if not isinstance(content, str) or len(content.encode()) > MAX_FILE or 'version' not in body:
        raise HTTPException(400, 'Content and base version required; maximum 512 KiB')
    full = store._join(path)
    with store._write_lock:
        current, etag = store.backend.get_text(full)
        if version(current) != body['version']:
            raise HTTPException(409, 'File changed elsewhere. Reopen it before saving; your draft has been kept.')
        if current == content:
            return {'path': path, 'content': content, 'version': version(content)}
        if current is not None and not etag:
            raise HTTPException(409, 'Storage did not return a version; refusing an unguarded overwrite')
        store.journal.mark_dirty(full)
        ok, _, status = store.backend.put(full, content.encode(), if_match=etag)
        if not ok:
            raise HTTPException(409 if status == 412 else 502, 'Storage write failed or conflicted; your draft has been kept')
    return {'path': path, 'content': content, 'version': version(content)}


def reference_text(files, query=''):
    words = set(re.findall(r'\w{3,}', query.lower()))
    ordered = sorted(files.items(), key=lambda item: -sum(w in item[1].lower() for w in words))
    return '\n\n'.join(f'--- {p} ---\n{t[:4000]}' for p, t in ordered[:8])[:16000]


def memory_text(store):
    """Explicit memory area, bounded; treat contents as reference, not code."""
    texts = {}
    try:
        for folder in ('', 'memory', 'Memory', 'context', 'Context'):
            try:
                entries = file_list(store, folder)
            except Exception:
                continue
            for entry in entries[:40]:
                if entry['isDir']:
                    continue
                if folder == '' and entry['name'].lower() not in ('memory.md', 'context.md', 'instructions.md'):
                    continue
                row = file_read(store, entry['path'])
                if row['content']:
                    texts[entry['path']] = row['content'][:4000]
                if len(texts) >= 8:
                    return reference_text(texts)
    except Exception:
        pass
    return reference_text(texts)


class MemoryBackend:
    """No disk, remote credentials, or persistent retrieval for local-only turns."""
    def __init__(self, files):
        if not isinstance(files, dict) or len(files) > MAX_FILES:
            raise HTTPException(413, 'Choose a diary folder with at most 500 Markdown files')
        total = 0
        for path, text in files.items():
            safe_path(path)
            if not isinstance(text, str) or len(text.encode()) > MAX_FILE:
                raise HTTPException(413, 'Markdown file exceeds 512 KiB')
            total += len(text.encode())
        if total > MAX_TOTAL:
            raise HTTPException(413, 'Local diary exceeds 12 MiB; choose a smaller folder')
        self.original = dict(files)
        self.files = dict(files)

    def get_text(self, path):
        text = self.files.get(path)
        return text, version(text)

    def get(self, path):
        text, tag = self.get_text(path)
        return (None if text is None else text.encode()), tag

    def put(self, path, data, if_match=None, if_none_match='*', **kwargs):
        safe_path(path)
        text, tag = self.get_text(path)
        if (if_match is not None and tag != if_match) or (if_match is None and if_none_match == '*' and text is not None):
            return False, None, 412
        self.files[path] = data.decode()
        return True, version(self.files[path]), 204 if text is not None else 201

    def list_dir(self, path):
        prefix = path.strip('/') + '/' if path.strip('/') else ''
        children = {}
        for key in self.files:
            if key.startswith(prefix):
                name = key[len(prefix):].split('/')[0]
                children[name] = {'name': name, 'path': prefix + name, 'is_dir': '/' in key[len(prefix):]}
        return list(children.values())

    def exists(self, path):
        return path in self.files

    def close(self):
        pass
