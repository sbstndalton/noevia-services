"""Shadow comparison for the Diary Rust routes: the same synthetic tenant operations through the Python path and
the Rust path, compared reply by reply (status, headers, body bytes) and then storage by storage.

This module does no network or process work itself. The caller passes two `send` functions (one per path) and
two storage roots (one per path, both starting from the same state). Nothing here reads or sets a Diary switch,
and nothing reads a real tenant: `load_ops` refuses any tenant id outside the synthetic prefix.

Volatile values are not compared: the headers in VOLATILE_HEADERS and the internal x-noevia-diary-* headers, the
JSON fields the operation lists in `mask`, and the time columns in TIME_COLUMNS of a SQLite database. Everything
else is compared exactly. The owner decides whether that is the right definition of "a difference" (plan R3).

R9 (Diary core: answering and indexing in Rust) adds three things, all opt-in per call:
- An event-stream reply (`text/event-stream`, the exchange's progress stream) is compared as the list of its
  parsed `data:` events, keep-alive comments left out: the Rust frames carry the same JSON values with another key
  order and spacing (the two processes serialise differently), which is not a difference.
- `normalise_uuids`: every UUID in a reply body, a file or a database value is replaced by its order of first
  appearance on that side (`<uuid:1>`, ...). An exchange makes fresh uuid4 ids (xids, journal ids) on each path,
  so two correct runs differ only there; the replacement keeps "the same id in the same places" comparable.
- `load_vec`: a callable that loads the vector extension into a read-only connection, so a vector table is
  compared by its rows (chunk id, vector bytes) instead of failing to open; its internal shadow tables, whose
  layout is the extension's business, are left out.
"""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

SYNTHETIC_PREFIX = '00000000-0000-4000-8000-'
VERSION = 1
METHODS = frozenset({'GET', 'PUT', 'POST'})
# The routes the forwarder knows (rust_forward.SUPPORTED and WRITE_ROUTES); a new route must be added here on purpose.
PATHS = frozenset({
    '/api/health', '/api/months', '/api/day', '/api/files', '/api/storage-status', '/api/workspace-export',
    '/api/file', '/api/directory', '/api/entries/append', '/api/entries/edit', '/api/storage-backup',
    # R9 answering (DIARY_RUST_ANSWER): the exchange, JSON or event stream.
    '/v1/chat/completions',
})
VOLATILE_HEADERS = frozenset({'date', 'server', 'connection', 'keep-alive', 'transfer-encoding', 'content-length'})
INTERNAL_HEADER_PREFIX = 'x-noevia-diary-'
# Columns the store fills from the clock; two correct runs differ in them, so they are left out of the digest.
TIME_COLUMNS = frozenset({'updated', 'last_success', 'retry_at', 'created_at', 'applied_at',
                          # chunks.updated_at (retrieval index rows): datetime('now') at insert.
                          'updated_at'})
# SQLite sidecar files: their content is covered by the logical digest of the main database file.
SQLITE_SIDECARS = ('-wal', '-shm', '-journal')
_UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
_ANY_UUID = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
_VEC_TABLE = re.compile(r'CREATE\s+VIRTUAL\s+TABLE\b.*\bUSING\s+vec0\b', re.I | re.S)


class UuidNumbering:
    """Replaces each distinct UUID by `<uuid:n>`, n its order of first appearance (one instance per side)."""

    def __init__(self):
        self.seen = {}

    def text(self, value):
        def number(m):
            key = m.group(0).lower()
            if key not in self.seen:
                self.seen[key] = len(self.seen) + 1
            return '<uuid:%d>' % self.seen[key]
        return _ANY_UUID.sub(number, value)

    def value(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, dict):
            return {self.text(k): self.value(v) for k, v in value.items()}
        return value
_OP_KEYS = frozenset({'name', 'method', 'path', 'query', 'body', 'mask'})


class OpsError(ValueError):
    pass


def is_synthetic_tenant(tenant):
    return isinstance(tenant, str) and bool(_UUID.fullmatch(tenant)) and tenant.startswith(SYNTHETIC_PREFIX)


def load_ops(path):
    """Read and check an operation list. Refuses a non-synthetic tenant, an unknown route or method, duplicate names."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('version') != VERSION:
        raise OpsError('unsupported operation list version')
    tenant = data.get('tenant')
    if not is_synthetic_tenant(tenant):
        raise OpsError('operation list tenant must be a synthetic id (prefix %s)' % SYNTHETIC_PREFIX)
    ops = data.get('ops')
    if not isinstance(ops, list) or not ops:
        raise OpsError('operation list has no operations')
    names = set()
    for op in ops:
        if not isinstance(op, dict) or not set(op) <= _OP_KEYS:
            raise OpsError('operation has unknown keys: %r' % (op,))
        name, method, route = op.get('name'), op.get('method'), op.get('path')
        if not isinstance(name, str) or name in names:
            raise OpsError('operation names must be unique strings')
        names.add(name)
        if method not in METHODS or route not in PATHS:
            raise OpsError('operation %s: unknown method or path' % name)
        if method == 'GET' and 'body' in op:
            raise OpsError('operation %s: a GET carries no body' % name)
        if not isinstance(op.get('mask', []), list):
            raise OpsError('operation %s: mask must be a list of keys' % name)
    return data


def stream_events(body):
    """The parsed `data:` events of an event-stream body, in order (comments and blank lines left out). A data line
    that is not JSON is kept as its text, so a broken frame is still a difference."""
    out = []
    for line in body.decode('utf-8', 'replace').splitlines():
        if not line.startswith('data:'):
            continue
        raw = line[5:].strip()
        try:
            out.append(json.loads(raw))
        except ValueError:
            out.append({'<unparsed>': raw})
    return out


def canonical_reply(status, headers, body, mask=(), numbering=None):
    """A comparable form of one reply. `headers` is a dict or a list of (name, value) pairs; `body` is bytes.
    `numbering` (a UuidNumbering) replaces UUIDs in the body; an event stream compares as its parsed events."""
    pairs = headers.items() if isinstance(headers, dict) else headers
    kept = []
    for name, value in pairs:
        key = name.decode() if isinstance(name, bytes) else str(name)
        key = key.lower()
        if key in VOLATILE_HEADERS or key.startswith(INTERNAL_HEADER_PREFIX):
            continue
        val = value.decode() if isinstance(value, bytes) else str(value)
        kept.append([key, val])
    kept.sort()
    content_type = next((v for k, v in kept if k == 'content-type'), '')
    if content_type.startswith('text/event-stream'):
        events = stream_events(body)
        if numbering is not None:
            events = numbering.value(events)
        body = json.dumps(_mask_value(events, set(mask)), sort_keys=True, separators=(',', ':')).encode()
    else:
        if numbering is not None:
            body = numbering.text(body.decode('utf-8', 'replace')).encode()
        if mask:
            body = _mask_json(body, set(mask))
    return {'status': int(status), 'headers': kept,
            'body_len': len(body), 'body_sha256': hashlib.sha256(body).hexdigest()}


def _mask_value(value, keys):
    if isinstance(value, dict):
        return {k: ('<masked>' if k in keys else _mask_value(v, keys)) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_value(v, keys) for v in value]
    return value


def _mask_json(body, keys):
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    return json.dumps(_mask_value(value, keys), sort_keys=True, separators=(',', ':')).encode()


def diff_replies(left, right):
    """Names of the fields that differ between two canonical replies ([] when identical)."""
    out = []
    if left['status'] != right['status']:
        out.append('status')
    if left['headers'] != right['headers']:
        out.append('headers')
    if left['body_sha256'] != right['body_sha256'] or left['body_len'] != right['body_len']:
        out.append('body')
    return out


def _sqlite_digest(path, numbering=None, load_vec=None):
    try:
        db = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
    except sqlite3.Error as exc:
        return {'kind': 'sqlite', 'error': 'open: %s' % exc}
    try:
        if load_vec is not None:
            load_vec(db)
        master = list(db.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"))
        vec_tables = {name for name, sql in master if load_vec is not None and sql and _VEC_TABLE.search(sql)}
        tables = {}
        for name, sql in master:
            if name.startswith('sqlite_') or any(name.startswith(v + '_') for v in vec_tables):
                continue  # the vector extension's own storage: compared through its table's rows
            try:
                cols = [row[1] for row in db.execute('PRAGMA table_info("%s")' % name)]
                keep = [i for i, c in enumerate(cols) if c not in TIME_COLUMNS]
                rows = [tuple(_plain(row[i], numbering) for i in keep) for row in db.execute('SELECT * FROM "%s"' % name)]
                rows.sort(key=repr)
                tables[name] = {'schema': sql, 'columns': [cols[i] for i in keep], 'rows': rows}
            except sqlite3.Error as exc:
                tables[name] = {'schema': sql, 'error': str(exc)}
        blob = json.dumps(tables, sort_keys=True, default=_plain_repr).encode()
        return {'kind': 'sqlite', 'sha256': hashlib.sha256(blob).hexdigest(), 'tables': sorted(tables)}
    except sqlite3.Error as exc:
        return {'kind': 'sqlite', 'error': str(exc)}
    finally:
        db.close()


def _plain(value, numbering=None):
    if isinstance(value, bytes):
        return {'bytes_sha256': hashlib.sha256(value).hexdigest(), 'len': len(value)}
    if numbering is not None and isinstance(value, str):
        return numbering.text(value)
    return value


def _plain_repr(value):
    return repr(value)


def storage_digest(root, normalise_uuids=False, load_vec=None):
    """{relative path: digest} for every file under `root`. Plain files by bytes; SQLite files logically.
    `normalise_uuids` numbers UUIDs (one numbering for the whole root, files in path order); `load_vec` reads
    vector tables (see the module docstring)."""
    root = Path(root)
    numbering = UuidNumbering() if normalise_uuids else None
    out = {}
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.endswith(SQLITE_SIDECARS):
            continue
        key = numbering.text(rel) if numbering is not None else rel
        if rel.endswith('.db') or path.read_bytes()[:16] == b'SQLite format 3\x00':
            out[key] = _sqlite_digest(path, numbering, load_vec)
        else:
            data = path.read_bytes()
            if numbering is not None:
                data = numbering.text(data.decode('utf-8', 'replace')).encode()
            out[key] = {'kind': 'file', 'sha256': hashlib.sha256(data).hexdigest()}
    return out


def diff_storage(left, right):
    """Differences between two storage digests: [{'path', 'reason'}]. A missing path on one side is a difference."""
    out = []
    for rel in sorted(set(left) | set(right)):
        if rel not in right:
            out.append({'path': rel, 'reason': 'only-python'})
        elif rel not in left:
            out.append({'path': rel, 'reason': 'only-rust'})
        elif left[rel] != right[rel]:
            out.append({'path': rel, 'reason': 'content'})
    return out


def run_ops(ops, python_send, rust_send, normalise_uuids=False):
    """Run each operation through both paths, in order, and compare the replies.

    `python_send(op)` and `rust_send(op)` return {'status', 'headers', 'body', 'served'}; `served` is 'rust' when the
    Rust process answered, and 'python' when the Rust path declined or was not reached (a fallback, counted, not a
    difference). The reply that a client would see is the Python one.
    """
    report = {'tenant': ops['tenant'], 'operations': [], 'differences': 0, 'rust_served': 0, 'fallbacks': 0}
    left_numbers = UuidNumbering() if normalise_uuids else None
    right_numbers = UuidNumbering() if normalise_uuids else None
    for op in ops['ops']:
        mask = tuple(op.get('mask', ()))
        left_reply = python_send(op)
        right_reply = rust_send(op)
        left = canonical_reply(left_reply['status'], left_reply['headers'], left_reply['body'], mask, left_numbers)
        right = canonical_reply(right_reply['status'], right_reply['headers'], right_reply['body'], mask, right_numbers)
        diffs = diff_replies(left, right)
        served = right_reply.get('served', 'python')
        report['operations'].append({'name': op['name'], 'method': op['method'], 'path': op['path'],
                                     'status': left['status'], 'served': served, 'differences': diffs})
        report['differences'] += 1 if diffs else 0
        report['rust_served'] += 1 if served == 'rust' else 0
        report['fallbacks'] += 1 if served != 'rust' else 0
    return report


def compare(ops, python_send, rust_send, python_root, rust_root, normalise_uuids=False, load_vec=None):
    """run_ops plus the storage comparison of the two roots after the operations. Returns the full report."""
    report = run_ops(ops, python_send, rust_send, normalise_uuids)
    storage = diff_storage(storage_digest(python_root, normalise_uuids, load_vec),
                           storage_digest(rust_root, normalise_uuids, load_vec))
    report['storage_differences'] = storage
    report['differences'] += len(storage)
    return report


def summarise(report):
    """Plain-text counts for the owner's daily check: one line per operation that differs, then totals."""
    lines = []
    for op in report['operations']:
        if op['differences']:
            lines.append('DIFF %-32s %s %s: %s' % (op['name'], op['method'], op['path'], ','.join(op['differences'])))
    for item in report.get('storage_differences', []):
        lines.append('DIFF storage %s: %s' % (item['path'], item['reason']))
    lines.append('operations=%d differences=%d rust_served=%d fallbacks=%d' % (
        len(report['operations']), report['differences'], report['rust_served'], report['fallbacks']))
    return '\n'.join(lines)
