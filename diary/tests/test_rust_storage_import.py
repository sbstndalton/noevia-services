"""POST /api/storage-import: the real Python handler against the native process, byte for byte.

Synthetic tenants only, with loopback fake WebDAV and fake S3 servers and synthetic local folders.
Each case runs Python first, resets the tenant, then runs the native side on the same source and
compares status, body bytes, the relevant headers and the activated managed database."""
import base64
import fcntl
import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from xml.sax.saxutils import escape

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import agent.app as appmod
from agent import rust_forward, storage_backoff
from agent.config import Config
from agent.diary_migration import SETTING_KEYS
from agent.managed_storage import ManagedCorpusBackend

UID = '12345678-1234-4234-8234-123456789abc'
LEGACY = '44444444-4444-4444-8444-444444444444'
BIN = os.environ.get('DIARY_RUST_SERVER_BIN', '')
pytestmark = pytest.mark.skipif(not BIN, reason='native server required')
SECRET = 'synthetic-secret-never-logged'
REFLECTED = b'synthetic reflected secret body'


class Faults:
    def __init__(self):
        self.reset()

    def reset(self):
        self.list_status = 0
        self.get_status = 0
        self.fail_after = None  # fail GETs after this many succeeded
        self.mutate_after = None  # change one file after this many GETs
        self.retry_after = None
        self.gets = 0
        self.calls = 0
        self.block = None  # threading.Event the Nth GET waits on
        self.block_at = None
        self.entered = threading.Event()
        self.bad_xml = False


def files_tree(files):
    """{'a/b.md': bytes} -> set of directories."""
    dirs = set()
    for path in files:
        parts = path.split('/')[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add('/'.join(parts[:i]))
    return dirs


def serve(handler):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def make_handler(files, faults, kind):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def answer(self, code, body=b'', ctype='application/xml', extra=()):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def fault(self, status):
            extra = [('Retry-After', faults.retry_after)] if faults.retry_after else []
            if 300 <= status < 400:
                extra.append(('Location', 'http://127.0.0.1:9/elsewhere'))
            self.answer(status, REFLECTED, 'text/plain', extra)

        def read_body(self):
            n = int(self.headers.get('Content-Length') or 0)
            return self.rfile.read(n) if n else b''

        def get_file(self, key):
            faults.gets += 1
            if faults.block is not None and faults.gets == faults.block_at:
                faults.entered.set()
                assert faults.block.wait(15)
            if faults.mutate_after is not None and faults.gets == faults.mutate_after + 1:
                first = sorted(files)[0]
                files[first] = files[first] + b' changed'
            if faults.get_status or (faults.fail_after is not None and faults.gets > faults.fail_after):
                return self.fault(faults.get_status or 503)
            if key in files:
                return self.answer(200, files[key], 'text/markdown')
            return self.answer(404)

        # ---- WebDAV
        def do_PROPFIND(self):
            self.read_body()
            faults.calls += 1
            if faults.list_status:
                return self.fault(faults.list_status)
            path = unquote(urlparse(self.path).path)
            if not path.startswith('/dav/'):
                return self.answer(404)
            rel = path[len('/dav/'):].strip('/')
            dirs = files_tree(files) | {''}
            if rel not in dirs:
                return self.answer(404)
            out = [f'<d:response><d:href>{quote("/dav/" + rel + "/" if rel else "/dav/")}</d:href>'
                   '<d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop>'
                   '<d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>']
            for d in sorted(x for x in dirs if x and (x.rsplit('/', 1)[0] if '/' in x else '') == rel):
                out.append(f'<d:response><d:href>/dav/{quote(d)}/</d:href><d:propstat><d:prop>'
                           '<d:resourcetype><d:collection/></d:resourcetype><d:getetag>"d"</d:getetag></d:prop>'
                           '<d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>')
            for f in sorted(files):
                parent = f.rsplit('/', 1)[0] if '/' in f else ''
                if parent == rel:
                    out.append(f'<d:response><d:href>/dav/{quote(f)}</d:href><d:propstat><d:prop>'
                               '<d:resourcetype/><d:getetag>"e"</d:getetag></d:prop>'
                               '<d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>')
            body = ('<?xml version="1.0"?><d:multistatus xmlns:d="DAV:">' + ''.join(out) + '</d:multistatus>')
            if faults.bad_xml:
                body = '<d:multistatus xmlns:d="DAV:"><d:response>'
            self.answer(207, body.encode())

        def do_GET(self):
            faults.calls += 1
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if kind == 'webdav':
                if not path.startswith('/dav/'):
                    return self.answer(404)
                return self.get_file(path[len('/dav/'):])
            # ---- S3, path style: /bucket?list-type=2... or /bucket/key
            assert path.startswith('/bucket')
            query = parse_qs(parsed.query, keep_blank_values=True)
            if query.get('list-type') == ['2']:
                if faults.list_status:
                    return self.fault(faults.list_status)
                prefix = query.get('prefix', [''])[0]
                kp = getattr(self.server, 'key_prefix', '')
                keys = sorted(kp + f for f in files)
                children, dirs = [], set()
                for key in keys:
                    if not key.startswith(prefix):
                        continue
                    rest = key[len(prefix):]
                    if '/' in rest:
                        dirs.add(prefix + rest.split('/', 1)[0] + '/')
                    else:
                        children.append(key)
                entries = [f'<Contents><Key>{escape(k)}</Key><ETag>"x"</ETag></Contents>' for k in children]
                entries += [f'<CommonPrefixes><Prefix>{escape(d)}</Prefix></CommonPrefixes>' for d in sorted(dirs)]
                token = query.get('continuation-token', [''])[0]
                # Two pages: the first half, then the rest.
                half = len(entries) // 2
                if len(entries) > 1 and not token:
                    page, extra = entries[:half], '<IsTruncated>true</IsTruncated><NextContinuationToken>t1</NextContinuationToken>'
                elif token:
                    page, extra = entries[half:], '<IsTruncated>false</IsTruncated>'
                else:
                    page, extra = entries, '<IsTruncated>false</IsTruncated>'
                body = ('<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                        + extra + ''.join(page) + '</ListBucketResult>')
                if faults.bad_xml:
                    body = '<ListBucketResult><Key>a & b</Key></ListBucketResult>'
                return self.answer(200, body.encode())
            key = path[len('/bucket/'):]
            kp = getattr(self.server, 'key_prefix', '')
            if not key.startswith(kp):
                return self.answer(404)
            return self.get_file(key[len(kp):])
    return Handler


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.delenv('DIARY_LOCAL_VOLUMES', raising=False)
    yield from make_lab(tmp_path, monkeypatch)


def make_lab(tmp_path, monkeypatch):
    monkeypatch.delenv('DIARY_TENANT_KEY', raising=False)
    monkeypatch.setenv('DIARY_LEGACY_USER_ID', LEGACY)
    legacy_root = tmp_path / 'legacy-corpus'
    legacy_root.mkdir()
    cfg = Config({
        'corpus': {'backend': 'local', 'root': '', 'monthly_prefix': 'Monthly', 'entry_layout': 'daily',
                   'entries_prefix': 'Entries', 'index_enabled': False, 'local': {'root': str(legacy_root)}},
        'retrieval': {'db_path': str(tmp_path / 'state' / 'index.db')},
        'llm': {'base_url': 'http://stub', 'chat_model': 'm', 'embed_model': 'e',
                'aux': {'base_url': 'http://stub', 'model': 'a'}},
        'ui': {'auth_token': ''},
    })
    monkeypatch.setattr(appmod, '_base_cfg', cfg)
    monkeypatch.setattr(appmod, 'check_auth', lambda r: True)
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    monkeypatch.setattr(appmod, '_deleted_tenants', {})
    appmod._tenant_states.clear()
    files = {}
    faults = Faults()
    servers = {k: serve(make_handler(files, faults, k)) for k in ('webdav', 's3')}
    port = free_port()
    child_cfg = {
        'port': port, 'control_token': 'ctl', 'forward_secret': 'forward', 'data_root': str(tmp_path / 'state'),
        'legacy_user_id': LEGACY, 'corpus': {},
        'llm': {'main': {'base_url': 'http://127.0.0.1:9', 'setting_key': 'MAIN', 'fallback_setting_key': 'FALLBACK'},
                'aux': {'base_url': 'http://127.0.0.1:9', 'setting_key': 'AUX', 'fallback_setting_key': 'FALLBACK'}},
        'workspace_export': {'impl': 'python', 'binary': '', 'settings': {k: cfg.get('corpus.' + k) for k in SETTING_KEYS}},
        'legacy': rust_forward.legacy_config(cfg),
    }
    stderr = tmp_path / 'native.stderr'
    log = stderr.open('wb')
    env = dict(os.environ)
    proc = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, env=env,
                            cwd=str(tmp_path))
    proc.stdin.write(json.dumps(child_cfg).encode() + b'\n')
    proc.stdin.flush()
    base = f'http://127.0.0.1:{port}'
    for _ in range(300):
        try:
            if httpx.get(base + '/internal/ready', headers={'x-noevia-diary-control': 'ctl'}, timeout=1).status_code == 200:
                break
        except httpx.TransportError:
            pass
        assert proc.poll() is None, stderr.read_text()
        time.sleep(.02)
    else:
        raise AssertionError('native server unavailable')
    lab = Lab(tmp_path, files, faults, servers, base)
    yield lab
    if faults.block is not None:
        faults.block.set()
    proc.terminate()
    proc.wait(timeout=10)
    log.close()
    text = stderr.read_bytes()
    assert SECRET.encode() not in text and REFLECTED not in text
    for s in servers.values():
        s.shutdown()
    for state in appmod._tenant_states.values():
        appmod._close_state(state)
    appmod._tenant_states.clear()


class Lab:
    def __init__(self, tmp, files, faults, servers, base):
        self.tmp, self.files, self.faults, self.servers, self.base = tmp, files, faults, servers, base
        self.client = TestClient(appmod.app, raise_server_exceptions=False)
        self.uid = UID
        self.storage = None
        self.extra = {}

    @property
    def root(self):
        return self.tmp / 'state' / 'users' / self.uid

    def descriptor(self, kind, **over):
        # A fresh login per case: back-off gates are per credential in both processes.
        user = 'synthetic-' + uuid.uuid4().hex[:8]
        if kind == 'webdav':
            d = {'kind': 'webdav', 'baseUrl': f'http://127.0.0.1:{self.servers["webdav"].server_port}/dav/',
                 'username': user, 'secret': SECRET, 'corpusRoot': ''}
        else:
            d = {'kind': 's3', 'baseUrl': f'http://127.0.0.1:{self.servers["s3"].server_port}', 'bucket': 'bucket',
                 'username': user, 'secret': SECRET, 'corpusRoot': '', 'region': 'Eu-West-1'}
        d.update(over)
        self.storage = d
        return d

    def headers(self):
        h = {}
        if self.uid:
            h['x-cowork-user-id'] = self.uid
        if self.storage is not None:
            h['x-cowork-storage'] = base64.urlsafe_b64encode(json.dumps(self.storage).encode()).decode().rstrip('=')
        h.update(self.extra)
        return h

    def reset(self):
        for state in appmod._tenant_states.values():
            appmod._close_state(state)
        appmod._tenant_states.clear()
        storage_backoff._gates.clear()
        root = self.root if self.uid else self.tmp / 'state' / 'users' / LEGACY
        for name in ('managed-diary.db', 'managed-diary.db-wal', 'managed-diary.db-shm', 'managed-diary.active',
                     'managed-index.db', 'index.db'):
            (root / name).unlink(missing_ok=True)

    def warm(self):
        """Python builds and recovers the tenant's legacy state, exactly as on its first request."""
        req = Request({'type': 'http', 'headers': [(k.encode(), v.encode()) for k, v in self.headers().items()]})
        state = appmod._tenant_state(req)
        assert state.recovered
        return state

    def ready(self):
        h = {k.lower(): v for k, v in self.headers().items()}
        return appmod._rust_legacy_ready((self.uid or LEGACY).lower(), lambda n: h.get(n, ''))

    def python(self, body):
        r = self.client.post('/api/storage-import', headers=self.headers(), json=body)
        return r.status_code, r.headers.get('content-type'), r.headers.get('retry-after'), r.content

    def native(self, body, ready=None):
        ready = self.ready() if ready is None else ready
        h = dict(self.headers(), **{'x-noevia-diary-forward': 'forward', 'x-noevia-diary-writer': '1',
                                    'x-noevia-diary-asserted': '0', 'x-noevia-diary-legacy-ready': '1' if ready else '0',
                                    'content-type': 'application/json'})
        r = httpx.post(self.base + '/api/storage-import', headers=h, content=json.dumps(body).encode(), timeout=30)
        return r.status_code, r.headers.get('content-type'), r.headers.get('retry-after'), r.content

    def managed(self):
        root = self.root if self.uid else self.tmp / 'state' / 'users' / LEGACY
        db = sqlite3.connect(root / 'managed-diary.db')
        try:
            meta = dict(db.execute('SELECT key, value FROM meta'))
            out = {
                'files': sorted(db.execute('SELECT path, data, version FROM files')),
                'dirs': sorted(r[0] for r in db.execute('SELECT path FROM directories')),
                'meta': {k: v for k, v in meta.items() if k not in ('snapshot_time', 'due')},
                'marker': (root / 'managed-diary.active').read_text() if (root / 'managed-diary.active').exists() else None,
            }
            if 'snapshot_time' in meta:
                now = float(meta['snapshot_time'])
                assert float(meta['due']) == now + 3.0
                assert {r[0] for r in db.execute('SELECT updated FROM files')} <= {now}
                assert repr(now) == meta['snapshot_time']
            return out
        finally:
            db.close()

    def compare(self, body, *, setup=None, faults=None):
        """Python, then native, from the same starting point. Returns the native result."""
        results, stores = [], []
        for native in (False, True):
            self.reset()
            if setup:
                setup()
            self.faults.reset()
            self.warm()
            if faults:
                faults(self.faults)
            if native:
                assert self.ready(), 'Python did not opt this request in'
            result = self.native(body) if native else self.python(body)
            results.append(result)
            stores.append(self.managed())
        assert results[0] == results[1], results
        assert stores[0] == stores[1]
        assert REFLECTED not in results[1][3] and SECRET.encode() not in results[1][3]
        return results[1], stores[1]


SAMPLE = {'Entries/2026/January/January 2, 2026.md': 'synthetic ü day\n'.encode(),
          'Entries/2026/January/January 3, 2026.md': b'synthetic day two\n',
          'MEMORY.md': b'synthetic memory', 'notes/a b&c.md': b'synthetic note'}


def preview_then_import(lab):
    lab.reset()
    lab.faults.reset()
    lab.warm()
    status, _, _, body = lab.python({})
    assert status == 200
    fp = json.loads(body)['fingerprint']
    return fp


@pytest.mark.parametrize('kind', ['webdav', 's3', 'local', 'legacy-owner'])
def test_preview_and_import_are_byte_identical(lab, kind, monkeypatch):
    lab.files.update(SAMPLE)
    setup = None
    if kind in ('webdav', 's3'):
        lab.descriptor(kind)
    elif kind == 'local':
        def setup():
            corpus = lab.root / 'corpus'
            for path, data in SAMPLE.items():
                (corpus / path).parent.mkdir(parents=True, exist_ok=True)
                (corpus / path).write_bytes(data)
            (corpus / 'empty').mkdir(exist_ok=True)
            (corpus / 'noevia-backups').mkdir(exist_ok=True)
            (corpus / 'noevia-backups' / 'old.md').write_bytes(b'never imported')
    else:
        lab.uid = ''  # the header-less legacy tenant: the base config's local corpus
        for path, data in SAMPLE.items():
            target = Path(appmod._base_cfg.get('corpus.local.root')) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    (status, _, _, body), _ = lab.compare({}, setup=setup)
    assert status == 200
    report = json.loads(body)
    assert report['fileCount'] == len(SAMPLE)
    (status, _, _, body), store = lab.compare({'fingerprint': report['fingerprint']}, setup=setup)
    assert (status, json.loads(body)) == (200, {'imported': True, 'fileCount': len(SAMPLE)})
    assert store['meta']['active'] == '1' and len(store['files']) == len(SAMPLE)


@pytest.mark.parametrize('mode', ['401', '403', '404', '500', '503', '429', '302', 'bad-xml', 'partial'])
@pytest.mark.parametrize('kind', ['webdav', 's3'])
def test_remote_failures(lab, kind, mode):
    lab.files.update(SAMPLE)
    lab.descriptor(kind)

    def faults(f):
        if mode == 'partial':
            f.fail_after = 2
        elif mode == 'bad-xml':
            f.bad_xml = True
        elif mode in ('404',):
            f.list_status = 404
        elif mode in ('401', '403', '302'):
            f.list_status = int(mode)
        else:
            f.get_status = int(mode)
            if mode == '429':
                f.retry_after = '7'
    (status, ctype, retry, body), store = lab.compare({}, faults=faults)
    expected = {'401': 424, '403': 424, '404': 409, '500': 502, '503': 502, '429': 503, '302': 500,
                'bad-xml': 409 if kind == 'webdav' else 500, 'partial': 502}[mode]
    assert status == expected, body
    if mode == '429':
        assert retry == '7' and json.loads(body)['retryAfter'] == 7
    assert store['meta'].get('active') != '1'


def test_login_cooldown_then_retry_header(lab):
    lab.files.update(SAMPLE)
    lab.descriptor('webdav')
    lab.reset()
    lab.warm()
    lab.faults.list_status = 401
    assert lab.native({})[0] == 424
    calls = lab.faults.calls
    lab.faults.list_status = 0
    # Python's own gate is still open here, so force the native path: its own gate refuses
    # without contacting storage, as Python's does after the same 401.
    assert lab.native({}, ready=True)[0] == 424
    assert lab.faults.calls == calls
    lab.extra = {'X-Cowork-Storage-Retry': '1'}
    assert lab.native({}, ready=True)[0] == 200
    assert lab.faults.calls > calls


def test_python_cooldown_keeps_the_import_in_python(lab):
    lab.files.update(SAMPLE)
    lab.descriptor('s3')
    lab.reset()
    lab.warm()
    lab.faults.list_status = 401
    assert lab.python({})[0] == 424
    assert not lab.ready()
    assert lab.native({})[0] == 421


def test_verify_mismatch_and_changed_source(lab):
    lab.files.update(SAMPLE)
    lab.descriptor('webdav')
    fp = preview_then_import(lab)
    n = len(SAMPLE)
    # Changed during the second (verification) snapshot.
    def restore():
        lab.files.clear()
        lab.files.update(SAMPLE)
    (status, _, _, body), store = lab.compare({'fingerprint': fp}, setup=restore,
                                              faults=lambda f: setattr(f, 'mutate_after', n))
    assert status == 409 and b'during verification' in body
    assert store['meta'].get('active') != '1'
    # Changed before the first snapshot: the preview is stale.
    lab.files[sorted(lab.files)[0]] += b' again'
    (status, _, _, body), _ = lab.compare({'fingerprint': fp})
    assert status == 409 and b'fresh import preview' in body


def test_partial_import_then_failure_writes_nothing(lab):
    lab.files.update(SAMPLE)
    lab.descriptor('s3')
    fp = preview_then_import(lab)
    n = len(SAMPLE)
    (status, _, _, _), store = lab.compare({'fingerprint': fp}, faults=lambda f: setattr(f, 'fail_after', n + 1))
    assert status == 502
    assert store['files'] == [] and store['meta'].get('active') != '1' and store['marker'] is None


@pytest.mark.parametrize('body', [{'fingerprint': 'XYZ'}, {'fingerprint': None}, {'fingerprint': 'A' * 64}])
def test_invalid_request(lab, body):
    lab.files.update(SAMPLE)
    lab.descriptor('webdav')
    (status, _, _, payload), _ = lab.compare(body)
    assert (status, payload) == (400, b'{"detail":"Invalid import request"}')


def test_empty_source_and_traversing_root(lab):
    lab.descriptor('webdav')
    (status, _, _, body), _ = lab.compare({})
    assert status == 409 and b'No source files found' in body
    lab.files.update(SAMPLE)
    lab.descriptor('webdav', corpusRoot='../escape')
    (status, _, _, body), _ = lab.compare({})
    assert status == 409


def test_local_symlink_escapes_are_never_followed(lab):
    outside = lab.tmp / 'outside'
    outside.mkdir()
    (outside / 'secret.md').write_bytes(b'outside the corpus')

    def setup():
        corpus = lab.root / 'corpus'
        (corpus / 'in').mkdir(parents=True, exist_ok=True)
        (corpus / 'in' / 'a.md').write_bytes(b'synthetic inside')
        for name, target in (('out-file.md', outside / 'secret.md'), ('out-dir', outside), ('up', Path('..') / '..' / '..')):
            link = corpus / name
            if not link.is_symlink():
                link.symlink_to(target)
        if not (corpus / 'in' / 'alias.md').is_symlink():
            (corpus / 'in' / 'alias.md').symlink_to('a.md')
    (status, _, _, body), _ = lab.compare({}, setup=setup)
    assert status == 200
    report = json.loads(body)
    assert sorted(f['path'] for f in report['files']) == ['in/a.md', 'in/alias.md']
    assert b'outside the corpus' not in body


def test_local_dangling_link_inside_is_python_500(lab):
    def setup():
        corpus = lab.root / 'corpus'
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / 'a.md').write_bytes(b'synthetic')
        if not (corpus / 'gone.md').is_symlink():
            (corpus / 'gone.md').symlink_to(corpus / 'missing.md')
    (status, ctype, _, body), _ = lab.compare({}, setup=setup)
    assert (status, body) == (500, b'Internal Server Error')


def test_activation_after_python_already_active_is_409(lab):
    lab.files.update(SAMPLE)
    lab.descriptor('webdav')
    fp = preview_then_import(lab)
    lab.reset()
    lab.warm()
    assert lab.ready()
    assert json.loads(lab.python({'fingerprint': fp})[3])['imported'] is True
    # The cached legacy state is retired: no native opt-in, and a forced native call refuses.
    assert not lab.ready()
    status, _, _, body = lab.native({'fingerprint': fp}, ready=True)
    assert status == 421  # the marker exists: Python's to answer (409)
    assert lab.python({'fingerprint': fp})[0] == 409


def test_fenced_tenant(lab):
    lab.files.update(SAMPLE)
    lab.descriptor('webdav')
    lab.reset()
    lab.warm()
    appmod._deleted_tenants[UID] = time.monotonic()
    assert not lab.ready()
    assert httpx.post(lab.base + '/internal/fence?tenant=' + UID, headers={'x-noevia-diary-control': 'ctl'}).status_code == 200
    assert lab.python({})[:1] == lab.native({}, ready=True)[:1] == (410,)
    assert lab.python({})[3] == lab.native({}, ready=True)[3] == b'{"detail":"tenant deleted"}'


def test_cancelled_import_settles_before_legacy_writes_resume(lab):
    """LegacyWriteGuard: a Python legacy write waits for the native import's flock, then refuses."""
    lab.files.update(SAMPLE)
    lab.descriptor('webdav')
    fp = preview_then_import(lab)
    lab.reset()
    state = lab.warm()
    lab.faults.reset()
    lab.faults.block = threading.Event()
    lab.faults.block_at = len(SAMPLE) + 2  # inside the verification snapshot
    h = dict(lab.headers(), **{'x-noevia-diary-forward': 'forward', 'x-noevia-diary-writer': '1',
                               'x-noevia-diary-legacy-ready': '1', 'content-type': 'application/json'})
    payload = json.dumps({'fingerprint': fp}).encode()
    url = httpx.URL(lab.base)
    conn = socket.create_connection((url.host, url.port), timeout=5)
    head = ''.join(f'{k}: {v}\r\n' for k, v in h.items())
    conn.sendall(f'POST /api/storage-import HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n{head}\r\n'.encode() + payload)
    assert lab.faults.entered.wait(10)
    conn.close()  # the caller goes away mid-import
    outcome = {}

    def legacy_write():
        try:
            state.backend.put('late.md', b'late synthetic write')
            outcome['result'] = 'written'
        except RuntimeError as exc:
            outcome['result'] = str(exc)
    writer = threading.Thread(target=legacy_write)
    writer.start()
    with (lab.root / 'managed-diary.lock').open('a') as held:
        with pytest.raises(BlockingIOError):
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    time.sleep(.2)
    assert 'result' not in outcome  # blocked on the import's flock
    lab.faults.block.set()
    writer.join(15)
    assert outcome['result'].startswith('Diary moved into the app.')
    assert 'late.md' not in lab.files
    assert lab.managed()['meta']['active'] == '1'


def test_volume_import(tmp_path, monkeypatch):
    """An operator volume (DIARY_LOCAL_VOLUMES) with a prefix: the volume wins over a descriptor,
    its identity is re-checked, and both sides import the same bytes."""
    volume = tmp_path / 'volume'
    (volume / 'Prefix' / 'Entries').mkdir(parents=True)
    (volume / '.noevia-diary-volume').write_text(UID + '\n')
    (volume / 'Prefix' / 'Entries' / 'a.md').write_bytes(b'synthetic volume')
    (volume / 'Prefix' / 'MEMORY.md').write_bytes(b'synthetic memory')
    outside = tmp_path / 'outside.md'
    outside.write_bytes(b'outside')
    (volume / 'Prefix' / 'escape.md').symlink_to(outside)
    monkeypatch.setenv('DIARY_LOCAL_VOLUMES', json.dumps({UID: {'root': str(volume), 'prefix': 'Prefix'}}))
    gen = make_lab(tmp_path, monkeypatch)
    lab_ = next(gen)
    try:
        lab_.descriptor('webdav')
        (status, _, _, body), _ = lab_.compare({})
        assert status == 200
        report = json.loads(body)
        assert sorted(f['path'] for f in report['files']) == ['Entries/a.md', 'MEMORY.md']
        (status, _, _, body), store = lab_.compare({'fingerprint': report['fingerprint']})
        assert json.loads(body)['imported'] is True
        # A replaced identity marker: the volume is unavailable (both refuse alike, 503).
        lab_.reset()
        lab_.warm()
        (volume / '.noevia-diary-volume').write_text('someone-else')
        assert lab_.python({}) == lab_.native({}, ready=True)
        assert lab_.python({})[0] == 503
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


def test_s3_corpus_root_lists_like_python(lab):
    """With a corpus root, Python lists `<root>//` at the top (s3_storage.list_dir): the same
    (empty) answer natively."""
    lab.files.update(SAMPLE)
    lab.servers['s3'].key_prefix = 'pre/'
    lab.descriptor('s3', corpusRoot='pre')
    (status, _, _, body), _ = lab.compare({})
    assert status == 409 and b'No source files found' in body
