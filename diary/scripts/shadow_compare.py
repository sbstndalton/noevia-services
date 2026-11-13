"""Shadow comparison for the Diary Rust routes: the same synthetic tenant operations through the Python path and
the Rust path, compared reply by reply (status, headers, body bytes) and then storage by storage.

This module does no network or process work itself. The caller passes two `send` functions (one per path) and
two storage roots (one per path, both starting from the same state). Nothing here reads or sets a Diary switch,
and nothing reads a real tenant: `load_ops` refuses any tenant id outside the synthetic prefix.

Volatile values are not compared: the headers in VOLATILE_HEADERS and the internal x-noevia-diary-* headers, the
JSON fields the operation lists in `mask`, and the time columns in TIME_COLUMNS of a SQLite database. Everything
else is compared exactly. The owner decides whether that is the right definition of "a difference" (plan R3).
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
})
VOLATILE_HEADERS = frozenset({'date', 'server', 'connection', 'keep-alive', 'transfer-encoding', 'content-length'})
INTERNAL_HEADER_PREFIX = 'x-noevia-diary-'
# Columns the store fills from the clock; two correct runs differ in them, so they are left out of the digest.
TIME_COLUMNS = frozenset({'updated', 'last_success', 'retry_at', 'created_at', 'applied_at'})
# SQLite sidecar files: their content is covered by the logical digest of the main database file.
SQLITE_SIDECARS = ('-wal', '-shm', '-journal')
_UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
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


def canonical_reply(status, headers, body, mask=()):
    """A comparable form of one reply. `headers` is a dict or a list of (name, value) pairs; `body` is bytes."""
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
    if mask:
        body = _mask_json(body, set(mask))
    return {'status': int(status), 'headers': kept,
            'body_len': len(body), 'body_sha256': hashlib.sha256(body).hexdigest()}


def _mask_json(body, keys):
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    def walk(node):
        if isinstance(node, dict):
            return {k: ('<masked>' if k in keys else walk(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return json.dumps(walk(value), sort_keys=True, separators=(',', ':')).encode()


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


def _sqlite_digest(path):
    try:
        db = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
    except sqlite3.Error as exc:
        return {'kind': 'sqlite', 'error': 'open: %s' % exc}
    try:
        tables = {}
        for name, sql in db.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"):
            if name.startswith('sqlite_'):
                continue
            try:
                cols = [row[1] for row in db.execute('PRAGMA table_info("%s")' % name)]
                keep = [i for i, c in enumerate(cols) if c not in TIME_COLUMNS]
                rows = [tuple(_plain(row[i]) for i in keep) for row in db.execute('SELECT * FROM "%s"' % name)]
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


def _plain(value):
    if isinstance(value, bytes):
        return {'bytes_sha256': hashlib.sha256(value).hexdigest(), 'len': len(value)}
    return value


def _plain_repr(value):
    return repr(value)


def storage_digest(root):
    """{relative path: digest} for every file under `root`. Plain files by bytes; SQLite files logically."""
    root = Path(root)
    out = {}
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.endswith(SQLITE_SIDECARS):
            continue
        if rel.endswith('.db') or path.read_bytes()[:16] == b'SQLite format 3\x00':
            out[rel] = _sqlite_digest(path)
        else:
            out[rel] = {'kind': 'file', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
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


def run_ops(ops, python_send, rust_send):
    """Run each operation through both paths, in order, and compare the replies.

    `python_send(op)` and `rust_send(op)` return {'status', 'headers', 'body', 'served'}; `served` is 'rust' when the
    Rust process answered, and 'python' when the Rust path declined or was not reached (a fallback, counted, not a
    difference). The reply that a client would see is the Python one.
    """
    report = {'tenant': ops['tenant'], 'operations': [], 'differences': 0, 'rust_served': 0, 'fallbacks': 0}
    for op in ops['ops']:
        mask = tuple(op.get('mask', ()))
        left_reply = python_send(op)
        right_reply = rust_send(op)
        left = canonical_reply(left_reply['status'], left_reply['headers'], left_reply['body'], mask)
        right = canonical_reply(right_reply['status'], right_reply['headers'], right_reply['body'], mask)
        diffs = diff_replies(left, right)
        served = right_reply.get('served', 'python')
        report['operations'].append({'name': op['name'], 'method': op['method'], 'path': op['path'],
                                     'status': left['status'], 'served': served, 'differences': diffs})
        report['differences'] += 1 if diffs else 0
        report['rust_served'] += 1 if served == 'rust' else 0
        report['fallbacks'] += 1 if served != 'rust' else 0
    return report


def compare(ops, python_send, rust_send, python_root, rust_root):
    """run_ops plus the storage comparison of the two roots after the operations. Returns the full report."""
    report = run_ops(ops, python_send, rust_send)
    storage = diff_storage(storage_digest(python_root), storage_digest(rust_root))
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
