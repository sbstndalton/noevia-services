"""Shadow comparison for the Diary Rust routes: the same synthetic tenant operations through the Python path and
the Rust path, compared reply by reply (status, headers, body bytes) and then storage by storage.

This module does no network or process work itself. The caller passes two `send` functions (one per path) and
two storage roots (one per path, both starting from the same state). Nothing here reads or sets a Diary switch,
and nothing reads a real tenant: `load_ops` refuses any tenant id outside the synthetic prefix.

Volatile values are not compared: the headers in VOLATILE_HEADERS and the internal x-noevia-diary-* headers, the
JSON fields the operation lists in `mask`, and the time columns of a SQLite database (TIME_COLUMNS in any table,
TIME_COLUMN_PAIRS as (table, column)). Everything else is compared exactly. The owner decides whether that is the
right definition of "a difference" (plan R3).

R9 (Diary core: answering and indexing in Rust) adds the following.
- An event-stream reply (content type `text/event-stream`, the exchange's progress stream) is compared as its list of
  parsed events, for every caller; it is not opt-in. R3's routes return no event stream, so R3 is unaffected. The
  framing follows SSE: events end at a blank line, the `data:` lines of one event are joined with a newline before
  the JSON is parsed, `event:` and `id:` are part of the compared event, and only `:` comments and `retry:` are
  dropped. A data payload that is not JSON is kept as text, so it stays a difference, and so does a stream that ends
  inside an event. The Rust frames carry the same JSON values with another key order and spacing, which is not a
  difference.
- `normalise_uuids` (opt-in): a UUID in a reply body, a file or a database value is replaced by its order of first
  appearance (`<uuid:1>`, ...) on that side, keyed on the exact text (case included), on the raw bytes. An exchange
  makes fresh uuid4 ids (xids, journal ids) on each path, so two correct runs differ only there. The numbering is
  one per side for the replies (in operation order) and then the storage, so "answered id A, stored id B" differs.
  JSON is canonicalised (parsed, masked, dumped with sorted keys) before it is numbered, and files and rows are
  numbered in an order that does not depend on the fresh ids. Ids in `keep_uuids` and the operation list's tenant
  are never replaced, so a leak of one of them, or a wrong existing id, stays visible. Rows that differ only in fresh
  ids cannot be ordered by anything else, so two such rows may still number in different orders (a false difference).
- `load_vec` (opt-in): a callable that loads the vector extension into a read-only connection, so a vector table is
  compared by its rows (chunk id, vector bytes) instead of failing to open; the extension's own shadow tables are
  left out. With `load_vec` a database that cannot be read (loader failure, a table read error) is reported as a
  difference (`unreadable`) instead of as two equal error digests.
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
TIME_COLUMNS = frozenset({'updated', 'last_success', 'retry_at', 'created_at', 'applied_at'})
# (table, column) pairs filled from the clock only in that table; the same column elsewhere is compared.
# chunks.updated_at (retrieval index rows): datetime('now') at insert.
TIME_COLUMN_PAIRS = frozenset({('chunks', 'updated_at')})
# SQLite sidecar files: their content is covered by the logical digest of the main database file.
SQLITE_SIDECARS = ('-wal', '-shm', '-journal')
_UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
_ANY_UUID = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
_ANY_UUID_RAW = re.compile(rb'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
# sqlite-vec's own storage for a vec0 table `v` (the table's rows are compared through the table itself).
_VEC_SHADOW_SUFFIX = r'_(?:info|chunks|rowids|auxiliary|vector_chunks\d+|metadatachunks\d+|metadatatext\d+)'
_VEC_TABLE = re.compile(r'CREATE\s+VIRTUAL\s+TABLE\b.*\bUSING\s+vec0\b', re.I | re.S)


class UuidNumbering:
    """Replaces each distinct UUID by `<uuid:n>`, n its order of first appearance (one instance per side). The key is
    the exact text, case included. The ids in `keep` are left as they are and take no number."""

    def __init__(self, keep=()):
        self.keep = frozenset(keep)
        self.seen = {}

    def _number(self, key):
        if key in self.keep:
            return key
        if key not in self.seen:
            self.seen[key] = len(self.seen) + 1
        return '<uuid:%d>' % self.seen[key]

    def text(self, value):
        return _ANY_UUID.sub(lambda m: self._number(m.group(0)), value)

    def raw(self, data):
        """The same on bytes, without decoding them (a lossy decode would make different bytes equal)."""
        return _ANY_UUID_RAW.sub(lambda m: self._number(m.group(0).decode('ascii')).encode('ascii'), data)


def _blank_text(value):
    return _ANY_UUID.sub('<uuid>', value)


def _blank_raw(data):
    return _ANY_UUID_RAW.sub(b'<uuid>', data)


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


def _sse_text(raw):
    return raw.decode('utf-8', 'surrogateescape')


def stream_events(body):
    """The events of an event-stream body, in order, as dicts: `data` (the JSON value of the joined data lines) or
    `data_text` (when that is not JSON, so a broken frame is still a difference), `event`, `id` and `fields` (other
    field names) when present, `incomplete` when the stream ends inside the event. Lines split on a newline only (a
    trailing carriage return is dropped), events on blank lines; `:` comments and `retry:` are dropped."""
    out = []
    lines = body.split(b'\n')
    ended_cleanly = lines[-1] == b''
    if ended_cleanly:
        lines.pop()
    data, other = [], []

    def flush(complete):
        nonlocal data, other
        if data or other:
            event = {}
            if data:
                joined = b'\n'.join(data)
                try:
                    event['data'] = json.loads(joined)
                except ValueError:  # includes a payload that is not UTF-8
                    event['data_text'] = _sse_text(joined)
            repeated = {}
            for name, value in other:
                repeated.setdefault(name, []).append(value)
            for name in ('event', 'id'):
                if name in repeated:
                    values = repeated.pop(name)
                    event[name] = values[0] if len(values) == 1 else values
            if repeated:
                event['fields'] = [[n, v] for n, vs in sorted(repeated.items()) for v in vs]
            if not complete:
                event['incomplete'] = True
            out.append(event)
        data, other = [], []

    for line in lines:
        if line.endswith(b'\r'):
            line = line[:-1]
        if line == b'':
            flush(True)
            continue
        if line.startswith(b':'):
            continue
        name, sep, value = line.partition(b':')
        if sep and value.startswith(b' '):
            value = value[1:]
        if name == b'retry':
            continue
        if name == b'data':
            data.append(value)
        else:
            other.append((_sse_text(name), _sse_text(value)))
    flush(False)
    return out


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def canonical_reply(status, headers, body, mask=(), numbering=None):
    """A comparable form of one reply. `headers` is a dict or a list of (name, value) pairs; `body` is bytes.
    An event stream compares as its parsed events. With `numbering` (a UuidNumbering) a JSON body is canonicalised
    (parsed, masked, dumped with sorted keys) and then its UUIDs are numbered; any other body is numbered raw.
    Without `numbering` a body is compared as sent (after masking, if a mask is given)."""
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
        keys = set(mask)
        for event in events:
            if 'data' in event:
                event['data'] = _mask_value(event['data'], keys)
        body = _canonical_json(events)
        if numbering is not None:
            body = numbering.raw(body)
    elif numbering is not None:
        try:
            value = json.loads(body)
        except ValueError:
            body = numbering.raw(body)
        else:
            body = numbering.raw(_canonical_json(_mask_value(value, set(mask))))
    elif mask:
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


def _unreadable(message, strict):
    out = {'kind': 'sqlite', 'error': message}
    if strict:
        out['unreadable'] = True
    return out


def _is_vec_shadow(name, vec_tables):
    return any(re.fullmatch(re.escape(v) + _VEC_SHADOW_SUFFIX, name) for v in vec_tables)


def _sqlite_digest(path, numbering=None, load_vec=None):
    strict = load_vec is not None  # then a database that cannot be read is a difference, not a matching error text
    try:
        db = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
    except sqlite3.Error as exc:
        return _unreadable('open: %s' % exc, strict)
    try:
        if load_vec is not None:
            try:
                load_vec(db)
            except Exception as exc:  # any loader failure: the vector tables cannot be compared
                return _unreadable('load_vec: %s: %s' % (type(exc).__name__, exc), strict)
        master = list(db.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"))
        vec_tables = {name for name, sql in master if load_vec is not None and sql and _VEC_TABLE.search(sql)}
        tables = {}
        unreadable = False
        for name, sql in master:
            if name.startswith('sqlite_') or _is_vec_shadow(name, vec_tables):
                continue  # the vector extension's own storage: compared through its table's rows
            try:
                cols = [row[1] for row in db.execute('PRAGMA table_info("%s")' % name)]
                keep = [i for i, c in enumerate(cols) if c not in TIME_COLUMNS and (name, c) not in TIME_COLUMN_PAIRS]
                rows = [tuple(_plain(row[i]) for i in keep) for row in db.execute('SELECT * FROM "%s"' % name)]
                if numbering is None:
                    rows.sort(key=repr)
                else:
                    # Order by the rows with their UUIDs blanked, so the order does not depend on the fresh ids,
                    # and only then number them.
                    rows.sort(key=lambda r: (repr(_blank_row(r)), repr(r)))
                    rows = [tuple(numbering.text(v) if isinstance(v, str) else v for v in r) for r in rows]
                tables[name] = {'schema': sql, 'columns': [cols[i] for i in keep], 'rows': rows}
            except sqlite3.Error as exc:
                unreadable = True
                tables[name] = {'schema': sql, 'error': str(exc)}
        blob = json.dumps(tables, sort_keys=True, default=_plain_repr).encode()
        out = {'kind': 'sqlite', 'sha256': hashlib.sha256(blob).hexdigest(), 'tables': sorted(tables)}
        if strict and unreadable:
            out['unreadable'] = True
        return out
    except sqlite3.Error as exc:
        return _unreadable(str(exc), strict)
    finally:
        db.close()


def _plain(value):
    if isinstance(value, bytes):
        return {'bytes_sha256': hashlib.sha256(value).hexdigest(), 'len': len(value)}
    return value


def _blank_row(row):
    return tuple(_blank_text(v) if isinstance(v, str) else v for v in row)


def _plain_repr(value):
    return repr(value)


def _is_sqlite_file(rel, data):
    return rel.endswith('.db') or data[:16] == b'SQLite format 3\x00'


def storage_digest(root, normalise_uuids=False, load_vec=None, keep_uuids=(), numbering=None):
    """{relative path: digest} for every file under `root`. Plain files by bytes; SQLite files logically.
    `normalise_uuids` numbers UUIDs (see the module docstring) except `keep_uuids`; `numbering` is the side's
    UuidNumbering when the caller already used it for the replies (it then also implies normalise_uuids);
    `load_vec` reads vector tables."""
    root = Path(root)
    if numbering is None and normalise_uuids:
        numbering = UuidNumbering(keep_uuids)
    files = []
    for path in root.rglob('*'):
        if path.is_file() and not path.relative_to(root).as_posix().endswith(SQLITE_SIDECARS):
            files.append((path.relative_to(root).as_posix(), path))
    if numbering is None:
        order = sorted(files, key=lambda item: item[0])
    else:
        # Number in an order that does not depend on the fresh ids: the path and the content with UUIDs blanked
        # (a SQLite file by its own digest under a scratch numbering), the raw path last.
        blanked = {rel: _blank_text(rel) for rel, _ in files}
        shared = {}
        for rel, _ in files:
            shared[blanked[rel]] = shared.get(blanked[rel], 0) + 1

        def content_key(item):
            rel, path = item
            if shared[blanked[rel]] < 2:
                return ''
            data = path.read_bytes()
            if _is_sqlite_file(rel, data):
                d = _sqlite_digest(path, UuidNumbering(numbering.keep), load_vec)
                return 's:' + str(d.get('sha256') or d.get('error'))
            return 'f:' + hashlib.sha256(_blank_raw(data)).hexdigest()
        order = sorted(files, key=lambda item: (blanked[item[0]], content_key(item), item[0]))
    out = {}
    for rel, path in order:
        key = numbering.text(rel) if numbering is not None else rel
        data = path.read_bytes()
        if _is_sqlite_file(rel, data):
            out[key] = _sqlite_digest(path, numbering, load_vec)
        else:
            if numbering is not None:
                data = numbering.raw(data)
            out[key] = {'kind': 'file', 'sha256': hashlib.sha256(data).hexdigest()}
    return out


def diff_storage(left, right):
    """Differences between two storage digests: [{'path', 'reason'}]. A missing path on one side is a difference,
    and so is a database that could not be read (reason `unreadable`, even when both sides failed alike)."""
    out = []
    for rel in sorted(set(left) | set(right)):
        if rel not in right:
            out.append({'path': rel, 'reason': 'only-python'})
        elif rel not in left:
            out.append({'path': rel, 'reason': 'only-rust'})
        elif left[rel].get('unreadable') or right[rel].get('unreadable'):
            out.append({'path': rel, 'reason': 'unreadable'})
        elif left[rel] != right[rel]:
            out.append({'path': rel, 'reason': 'content'})
    return out


def run_ops(ops, python_send, rust_send, normalise_uuids=False, numberings=None, keep_uuids=()):
    """Run each operation through both paths, in order, and compare the replies.

    `python_send(op)` and `rust_send(op)` return {'status', 'headers', 'body', 'served'}; `served` is 'rust' when the
    Rust process answered, and 'python' when the Rust path declined or was not reached (a fallback, counted, not a
    difference). The reply that a client would see is the Python one. `numberings` is a (python, rust) pair of
    UuidNumbering the caller keeps for the storage comparison; without it `normalise_uuids` makes fresh ones, which
    never replace the tenant or `keep_uuids`.
    """
    report = {'tenant': ops['tenant'], 'operations': [], 'differences': 0, 'rust_served': 0, 'fallbacks': 0}
    if numberings is not None:
        left_numbers, right_numbers = numberings
    elif normalise_uuids:
        keep = set(keep_uuids) | {ops['tenant']}
        left_numbers, right_numbers = UuidNumbering(keep), UuidNumbering(keep)
    else:
        left_numbers = right_numbers = None
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


def compare(ops, python_send, rust_send, python_root, rust_root, normalise_uuids=False, load_vec=None,
            keep_uuids=()):
    """run_ops plus the storage comparison of the two roots after the operations. Returns the full report.
    With `normalise_uuids` each side has one numbering across its replies and then its storage. `keep_uuids` (for
    example every UUID in the starting state) and the operation list's tenant are never replaced."""
    numberings = None
    if normalise_uuids:
        keep = set(keep_uuids) | {ops['tenant']}
        numberings = (UuidNumbering(keep), UuidNumbering(keep))
    report = run_ops(ops, python_send, rust_send, numberings=numberings)
    left = storage_digest(python_root, load_vec=load_vec, numbering=numberings[0] if numberings else None)
    right = storage_digest(rust_root, load_vec=load_vec, numbering=numberings[1] if numberings else None)
    storage = diff_storage(left, right)
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
