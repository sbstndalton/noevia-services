"""R9 main mode (DIARY_MAIN=rust): this app as the child of the Rust Diary process (agent/rust_parent.py,
noevia-rs bins/noevia-diary front.rs).

The unit cases always run. The differential runs the SAME request sequence (seeding writes, native and forwarded
reads, refusals, a replay, tenant deletion) against two real deployments on identical synthetic state: today's
Python main process (DIARY_MAIN unset, no Rust process) and `noevia-diary --main` with this app as its child, and
compares status, content type and body byte for byte. It needs the real binaries (DIARY_R9_SERVER_BIN for
noevia-diary, the tenant-assertion leaf) and skips without them.
"""
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import httpx
import pytest

from agent import rust_forward, rust_parent, tenant_writer
from agent import tenant_assertion as ta
from tests.tenant_binary import needs_binary

ROOT = Path(__file__).resolve().parent.parent
KEY = 'synthetic-tenant-key-r9-main'
TOKEN = 'synthetic-service-token-r9-main'
QA = '3e819f41-6d50-4fb1-8c43-4d5e6f708192'
QB = '4f92a052-7e61-4ac2-9d54-5e6f708192a3'
DAILY_QA = '6b14c274-9083-4ce4-9f76-708192a3b4c5'
BIN = os.environ.get('DIARY_R9_SERVER_BIN', '')
needs_server = pytest.mark.skipif(not (BIN and os.path.isfile(BIN)), reason='noevia-diary not built (set DIARY_R9_SERVER_BIN)')


# ---------------- entrypoint.sh ----------------

def _entrypoint(tmp_path):
    """entrypoint.sh with the noevia-diary path pointed at a stub that prints its argv."""
    stub = tmp_path / 'noevia-diary'
    stub.write_text('#!/bin/sh\necho "rust $*"\n')
    stub.chmod(0o755)
    script = tmp_path / 'entrypoint.sh'
    script.write_text((ROOT / 'entrypoint.sh').read_text().replace('/usr/local/bin/noevia-diary', str(stub)))
    return script


def _run(script, args, main=None):
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin')}
    if main is not None:
        env['DIARY_MAIN'] = main
    return subprocess.run(['sh', str(script), *args], env=env, capture_output=True, text=True, timeout=30).stdout.strip()


def test_the_entrypoint_execs_todays_command_unless_diary_main_is_rust(tmp_path):
    script = _entrypoint(tmp_path)
    cmd = ['echo', 'uvicorn', 'agent.app:app', '--host', '0.0.0.0', '--port', '8010']
    # Unset, empty or another value: today's argv, untouched.
    for main in (None, '', 'python', 'RUST'):
        assert _run(script, cmd, main) == 'uvicorn agent.app:app --host 0.0.0.0 --port 8010'
    # The image command itself (CMD) becomes the Rust parent's child on loopback.
    image_cmd = ['uvicorn', 'agent.app:app', '--host', '0.0.0.0', '--port', '8010']
    assert _run(script, image_cmd, 'rust') == ('rust --main -- uvicorn agent.app:app --host 127.0.0.1 --port 8012 '
                                               '--no-proxy-headers')
    # Operator commands are unchanged in main mode.
    assert _run(script, ['echo', 'something', 'else'], 'rust') == 'something else'


# ---------------- handshake, ParentServer ----------------

def test_the_handshake_must_be_complete():
    good = {'control_token': 't' * 32, 'parent_secret': 's' * 64, 'control_port': 8011}
    assert rust_parent.read_handshake(io.StringIO(json.dumps(good) + '\n')) == {**good, 'embedding_epoch': ''}
    with_epoch = {**good, 'embedding_epoch': 'e' * 32}
    assert rust_parent.read_handshake(io.StringIO(json.dumps(with_epoch) + '\n')) == with_epoch
    for bad in ({**good, 'control_token': ''}, {**good, 'parent_secret': 1}, {**good, 'control_port': 0},
                {**good, 'control_port': 70000}, {k: v for k, v in good.items() if k != 'parent_secret'},
                {**good, 'embedding_epoch': 5}):
        with pytest.raises(ValueError):
            rust_parent.read_handshake(io.StringIO(json.dumps(bad) + '\n'))
    with pytest.raises(ValueError):
        rust_parent.read_handshake(io.StringIO(''))


def test_the_start_up_config_is_retried_until_taken_and_a_refusal_stops_the_start():
    parent = rust_parent.ParentServer('tok', 'sec', 8011)
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers.get('x-noevia-diary-control'), json.loads(request.content)))
        if len(seen) == 1:
            raise httpx.ConnectError('not yet', request=request)
        return httpx.Response(200, text='ok')

    parent.send_config({'a': 1}, transport=httpx.MockTransport(handler))
    assert seen == [('/internal/child/config', 'tok', {'a': 1})] * 2
    with pytest.raises(RuntimeError):
        parent.send_config({'a': 1}, transport=httpx.MockTransport(lambda r: httpx.Response(409)))
    with pytest.raises(RuntimeError):
        parent.send_config({'a': 1}, deadline_s=0.2, transport=httpx.MockTransport(
            lambda r: (_ for _ in ()).throw(httpx.ConnectError('down', request=r))))


def test_the_parent_is_what_the_internal_callers_reach():
    parent = rust_parent.ParentServer('tok', 'sec', 8011)
    assert parent.control() == ('tok', True) and parent.ready and parent.reaping == 0
    assert parent.forward_secret_for(QA) == 'sec' and not parent.answer_only
    saved = rust_forward.current()
    try:
        rust_forward.adopt(parent)
        assert rust_forward.current() is parent
    finally:
        rust_forward.adopt(saved)


# ---------------- ParentMiddleware ----------------

def _call(headers, recovered=lambda uid: uid == QA, written=None):
    seen = {}

    async def app(scope, receive, send):
        seen['headers'] = list(scope['headers'])
        seen['asserted'] = scope.get('state', {}).get('tenant_asserted', False)
        await send({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'text/plain')]})
        await send({'type': 'http.response.body', 'body': b'ok'})

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    mw = rust_parent.ParentMiddleware(app, secret=lambda: 'sec', is_recovered=recovered,
                                      on_written=(written.append if written is not None else None))
    scope = {'type': 'http', 'method': 'GET', 'path': '/api/months', 'headers': headers}
    asyncio.run(mw(scope, receive, send))
    return seen, sent


def test_the_middleware_needs_the_parent_secret():
    for headers in ([], [(b'x-noevia-diary-parent', b'other')], [(b'x-noevia-diary-parent', b'')]):
        seen, sent = _call(headers + [(b'x-noevia-diary-asserted', b'1')])
        assert not seen
        assert sent[0]['status'] == 401 and sent[1]['body'] == b'{"detail":"invalid tenant assertion"}'
    assert rust_parent.ParentMiddleware(None, secret=lambda: '', is_recovered=bool) is not None
    # No secret known yet (before the handshake): nothing is let through.
    seen = {}

    async def app(scope, receive, send):
        seen['called'] = True

    sent = []

    async def send(m):
        sent.append(m)

    asyncio.run(rust_parent.ParentMiddleware(app, secret=lambda: '', is_recovered=bool)(
        {'type': 'http', 'headers': [(b'x-noevia-diary-parent', b'')]}, None, send))
    assert not seen and sent[0]['status'] == 401


def test_the_middleware_takes_the_parents_verdict_strips_internal_headers_and_stamps_tenants():
    base = [(b'x-noevia-diary-parent', b'sec'), (b'authorization', b'Bearer x'), (b'x-noevia-diary-verify', b'1')]
    seen, sent = _call(base + [(b'x-noevia-diary-asserted', b'1'), (b'x-cowork-user-id', QA.upper().encode())])
    assert seen['asserted'] is True
    assert [k for k, _ in seen['headers']] == [b'authorization', b'x-cowork-user-id']
    assert (b'x-noevia-diary-recovered', b'1') in sent[0]['headers']
    # Two tenant headers: the first counts (as for the parent's verifier and Starlette).
    seen, sent = _call(base + [(b'x-noevia-diary-asserted', b'0'), (b'x-cowork-user-id', QB.encode()),
                               (b'x-cowork-user-id', QA.encode())])
    assert seen['asserted'] is False
    assert (b'x-noevia-diary-recovered', b'0') in sent[0]['headers']
    # A native write of the parent: the tenant's state here is marked stale before the request runs.
    told = []
    _call(base + [(b'x-noevia-diary-written', b'1'), (b'x-cowork-user-id', QB.encode())], written=told)
    _call(base + [(b'x-noevia-diary-written', b'1')], written=told)  # no tenant: nothing to mark
    _call(base + [(b'x-cowork-user-id', QA.encode())], written=told)
    assert told == [QB]
    # No tenant, or not a UUID: no stamp.
    for extra in ([], [(b'x-cowork-user-id', b'legacy-user')]):
        _seen, sent = _call(base + extra)
        assert not any(k == b'x-noevia-diary-recovered' for k, _ in sent[0]['headers'])


# ---------------- S3: the writer lock as a lease on the parent's slot ----------------

class FakeParent:
    """tenant_writer's remote, recording every call and refusing two live leases of one tenant."""

    def __init__(self, busy=False, fail=False, release_fails=False):
        self.busy, self.fail, self.release_fails = busy, fail, release_fails
        self.calls, self.live, self.n = [], {}, 0
        self.lock = threading.Lock()

    def acquire(self, uid, wait_s):
        with self.lock:
            self.calls.append(('acquire', uid))
            if self.busy:
                raise tenant_writer.WriterBusy(uid)
            if self.fail:
                raise RuntimeError('parent unreachable')
            assert uid not in self.live.values(), 'two live leases for one tenant'
            self.n += 1
            lease = str(self.n)
            self.live[lease] = uid
            return lease

    def release(self, lease):
        with self.lock:
            self.calls.append(('release', lease))
            self.live.pop(lease)
            if self.release_fails:
                raise RuntimeError('lost')


@pytest.fixture
def leased():
    parent = FakeParent()
    tenant_writer.configure(frozenset({QA}))
    tenant_writer.configure_remote(parent)
    try:
        yield parent
    finally:
        tenant_writer.configure_remote(None)
        tenant_writer.configure(frozenset())


def test_a_writer_tenants_lock_is_one_lease_per_outer_hold(leased):
    assert tenant_writer.lease_of(QA) is None
    with tenant_writer.hold(QA):
        assert tenant_writer.lease_of(QA) == '1' and tenant_writer.held(QA)
        with tenant_writer.hold(QA):  # re-entrant: joins locally
            assert tenant_writer.lease_of(QA) == '1'
        assert tenant_writer.lease_of(QB) is None
    assert tenant_writer.lease_of(QA) is None and not tenant_writer.held(QA)
    with tenant_writer.hold(QB):  # not a writer tenant: no lease
        pass
    assert leased.calls == [('acquire', QA), ('release', '1')] and not leased.live


def test_a_refused_or_failed_lease_holds_nothing(leased):
    for kind in ('busy', 'fail'):
        setattr(leased, kind, True)
        with pytest.raises(tenant_writer.WriterBusy):
            with tenant_writer.hold(QA, deadline_s=0.1):
                pass
        setattr(leased, kind, False)
        assert not tenant_writer.held(QA)
    with tenant_writer.hold(QA, deadline_s=0.1):  # the slot was freed
        pass
    assert not leased.live


def test_writers_in_two_threads_take_the_lease_one_after_the_other(leased):
    inside, order = threading.Event(), []

    def first():
        with tenant_writer.hold(QA):
            order.append('first')
            inside.set()
            time.sleep(0.2)

    t = threading.Thread(target=first)
    t.start()
    inside.wait(5)
    with tenant_writer.hold(QA, deadline_s=5):
        order.append('second')
    t.join(5)
    assert order == ['first', 'second']
    assert [c[0] for c in leased.calls] == ['acquire', 'release', 'acquire', 'release']


def test_a_lost_release_is_logged_and_frees_the_local_slot(leased, caplog):
    leased.release_fails = True
    with tenant_writer.hold(QA):
        pass
    assert 'was not returned' in caplog.text
    leased.release_fails = False
    with tenant_writer.hold(QA, deadline_s=0.1):
        pass


def test_the_async_acquisition_returns_a_lease_granted_after_cancellation(leased):
    gate = threading.Event()
    real = leased.acquire

    def slow(uid, wait_s):
        gate.wait(5)
        return real(uid, wait_s)

    leased.acquire = slow

    async def main():
        task = asyncio.ensure_future(tenant_writer.acquire_async(QA, time.monotonic() + 5))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        for _ in range(100):
            if not leased.live and not tenant_writer._slots:
                return
            await asyncio.sleep(0.02)

    asyncio.run(main())
    assert not leased.live and QA not in tenant_writer._slots
    leased.acquire = real

    async def ok():
        hold = await tenant_writer.acquire_async(QA, time.monotonic() + 5)
        assert tenant_writer.lease_of(QA) == '2'
        hold.release()

    asyncio.run(ok())
    assert not leased.live


def test_the_parent_lease_client():
    seen = []
    other = '5a03b163-8f72-4bd3-8e65-6f708192a3b4'

    def handler(request):
        p = dict(request.url.params)
        seen.append((request.method, request.url.path, p, request.headers.get('x-noevia-diary-control')))
        if request.method == 'POST':
            if p['tenant'] == QA:
                return httpx.Response(200, json={'lease': '7'})
            if p['tenant'] == other:
                raise httpx.ReadTimeout('reply lost', request=request)
            return httpx.Response(503 if p['tenant'] == QB else 403)
        if p.get('lease') == '9':
            return httpx.Response(500)
        return httpx.Response(200 if p.get('lease') == '7' or p.get('holder') else 404)

    leases = rust_parent.ParentLeases('http://127.0.0.1:1', 'tok', transport=httpx.MockTransport(handler),
                                      release_delays=(0.0, 0.0))
    assert leases.acquire(QA, 1.5) == '7'
    post = seen[0]
    assert post[:2] == ('POST', '/internal/lease') and post[3] == 'tok'
    assert post[2]['tenant'] == QA and post[2]['wait_ms'] == '1500' and len(post[2]['holder']) == 32
    # Busy: nothing to return. Refused otherwise, or the reply lost: whatever the holder got is returned.
    for tenant, returned in ((QB, False), (DAILY_QA, True), (other, True)):
        seen.clear()
        with pytest.raises(tenant_writer.WriterBusy):
            leases.acquire(tenant, 0)
        holder = seen[0][2]['holder']
        assert (('DELETE', '/internal/lease', {'holder': holder}, 'tok') in seen) is returned
    leases.release('7')
    leases.release('8')  # 404: nothing live, counts as returned
    with pytest.raises(RuntimeError):
        leases.release('9')  # never confirmed, after the bounded retries
    assert sum(1 for c in seen if c[2].get('lease') == '9') == 2


# ---------------- the differential (real binaries) ----------------

def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _config(root: Path) -> Path:
    import yaml
    cfg = {
        'corpus': {'backend': 'local', 'root': '', 'monthly_prefix': '', 'entry_layout': 'daily',
                   'entries_prefix': 'Entries', 'index_enabled': False, 'month_file_template': '{year}-{month02}.md',
                   'index_file': 'INDEX.md', 'local': {'root': str(root / 'default')}},
        'retrieval': {'db_path': str(root / 'state' / 'index.db')},
        'llm': {'base_url': 'http://127.0.0.1:9/v1', 'chat_model': 'chat-a', 'embed_model': 'e',
                'aux': {'base_url': 'http://127.0.0.1:9/v1', 'model': 'aux-a'}},
        'ui': {'auth_token': TOKEN},
    }
    (root / 'state').mkdir(parents=True, exist_ok=True)
    path = root / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def _env(root: Path, main: bool, writers: bool = False, extra: Optional[dict] = None) -> dict:
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(root), 'TZ': 'UTC',
           'PYTHONPATH': str(ROOT), 'DIARY_CONFIG': str(_config(root)), 'DIARY_TENANT_KEY': KEY,
           'LLM_BASE_URL': 'http://127.0.0.1:9/v1', 'LLM_EMBED_BASE_URL': 'http://127.0.0.1:9/v1'}
    for name in ('TENANT_ASSERTION_BIN', 'TENANT_ASSERTION_IMPL', 'TENANT_ASSERTION_REQUIRED'):
        if os.environ.get(name):
            env[name] = os.environ[name]
    if main:
        env['DIARY_RUST_ROUTES'] = '/api/months,/api/day,/api/files,/api/storage-status,/api/file'
        env['DIARY_RUST_SERVER_BIN'] = BIN
        if writers:
            # S3: QA's writes are the parent's (native), under the slot the child leases.
            env['DIARY_RUST_WRITES'] = QA
    env.update(extra or {})
    return env


class Deployment:
    def __init__(self, root: Path, main: bool, writers: bool = False, extra: Optional[dict] = None):
        self.port = _free_port()
        uvicorn = [sys.executable, '-m', 'uvicorn', 'agent.app:app', '--host', '127.0.0.1']
        if main:
            child = _free_port()
            argv = [BIN, '--main', '--host', '127.0.0.1', '--port', str(self.port), '--control-port', str(_free_port()),
                    '--child-port', str(child), '--', *uvicorn, '--port', str(child), '--no-proxy-headers']
        else:
            argv = [*uvicorn, '--port', str(self.port)]
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.logfile = root / f'server-{"rust" if main else "python"}-{time.monotonic_ns()}.log'
        self.log = open(self.logfile, 'wb')
        self.proc = subprocess.Popen(argv, cwd=ROOT, env=_env(root, main, writers, extra), stdout=self.log,
                                     stderr=subprocess.STDOUT)
        self.client = httpx.Client(base_url=f'http://127.0.0.1:{self.port}', timeout=30, trust_env=False)
        deadline = time.monotonic() + 60
        while True:
            assert self.proc.poll() is None, self.logfile.read_text()
            try:
                if self.client.get('/api/health', headers={'Authorization': f'Bearer {TOKEN}'}).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, self.logfile.read_text()
            time.sleep(0.2)

    def close(self):
        self.client.close()
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                code = self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
                code = None
        else:
            code = self.proc.returncode
        self.log.close()
        return code


def _signed(user, method, path, body=b'', query=b'', nonce=None, content_type=None):
    headers = {'Authorization': f'Bearer {TOKEN}', 'X-Cowork-User-ID': user}
    if content_type:
        headers['Content-Type'] = content_type
    body_hash = ta.sha256_hex(body) if ta.body_is_hashed(content_type or '') else ta.STREAM
    headers[ta.HEADER] = ta.sign(KEY, user, method, path, headers, int(time.time()), nonce or secrets.token_hex(16),
                                 query, body_hash=body_hash)
    return headers


def _sequence(d: Deployment):
    """The requests, each signed afresh for this deployment; returns [(label, status, content type, body)]."""
    out = []

    def go(label, method, path, user=QA, body=b'', query='', content_type=None, headers=None, nonce=None):
        h = headers if headers is not None else _signed(user, method, path, body, query.encode(), nonce, content_type)
        r = d.client.request(method, path + (f'?{query}' if query else ''), headers=h, content=body or None)
        # Wall-clock values differ between two runs; everything else must be byte-equal.
        body = re.sub(rb'"(modified|updated|created_at|at)":\s*[0-9.]+', rb'"\1":0', r.content)
        out.append((label, r.status_code, r.headers.get('content-type'), body))
        return r

    for _ in range(2):
        go('status', 'GET', '/api/storage-status')
        go('months', 'GET', '/api/months')
        go('day', 'GET', '/api/day', query='month=2030-01')
        go('files', 'GET', '/api/files')
        # POST /api/file (a read with a body): a missing file, a bad path, bodies FastAPI refuses.
        for label, body, ctype in (('read missing', b'{"path":"Entries/2030/January/January 5, 2030.md"}', 'application/json'),
                                   ('read no path', b'{}', 'application/json'),
                                   ('read bad path', b'{"path":"../x.md"}', 'application/json'),
                                   ('read path not str', b'{"path":5}', 'application/json'),
                                   ('read list body', b'[]', 'application/json'),
                                   ('read bad json', b'{nope', 'application/json'),
                                   ('read text body', b'path', 'text/plain'),
                                   ('read path null', b'{"path":null}', 'application/json'),
                                   ('read merge-patch', b'{"path":"Entries/x.md"}', 'application/merge-patch+json')):
            go(label, 'POST', '/api/file', body=body, content_type=ctype)
        empty = _signed(QA, 'POST', '/api/file', b'', content_type='application/json')
        go('read empty body', 'POST', '/api/file', headers={**empty, 'Content-Length': '0'})
        no_token = {k: v for k, v in _signed(QA, 'POST', '/api/file', b'{}', content_type='application/json').items()
                    if k != 'Authorization'}
        go('read no token', 'POST', '/api/file', body=b'{}', headers=no_token)
        bad_no_token = {k: v for k, v in _signed(QA, 'POST', '/api/file', b'[]', content_type='application/json').items()
                        if k != 'Authorization'}
        go('read bad body no token', 'POST', '/api/file', body=b'[]', headers=bad_no_token)
    day = '## Sunday, January 6, 2030\n\n### 09:15 — walk\n\n**Me:** QA synthetic walk by the river, “ünïcode”.\n\n'
    path = 'Entries/2030/January/January 6, 2030.md'
    read = go('read new file', 'POST', '/api/file', body=json.dumps({'path': path}).encode(),
              content_type='application/json')
    version = read.json().get('version') if read.status_code == 200 else None
    put = json.dumps({'path': path, 'content': day, 'version': version}).encode()
    go('put', 'PUT', '/api/file', body=put, content_type='application/json')
    for _ in range(2):
        go('months', 'GET', '/api/months')
        go('day', 'GET', '/api/day', query='month=2030-01')
        go('files', 'GET', '/api/files')
        go('read written', 'POST', '/api/file', body=json.dumps({'path': path}).encode(), content_type='application/json')
    go('other tenant months', 'GET', '/api/months', user=QB)
    go('health', 'GET', '/api/health', headers={'Authorization': f'Bearer {TOKEN}'})
    go('no token', 'GET', '/api/months', headers={'X-Cowork-User-ID': QA})
    # An exact replay (same timestamp and nonce: the replay cache's key, as in Python).
    replayed = _signed(QA, 'GET', '/api/months')
    go('nonce first', 'GET', '/api/months', headers=dict(replayed))
    go('nonce replay', 'GET', '/api/months', headers=dict(replayed))
    bad = _signed(QA, 'GET', '/api/months')
    bad[ta.HEADER] = bad[ta.HEADER][:-4] + 'AAAA'
    go('bad assertion', 'GET', '/api/months', headers=bad)
    go('spoofed internal', 'GET', '/api/months',
       headers={**_signed(QA, 'GET', '/api/months'), 'x-noevia-diary-asserted': '1', 'x-noevia-diary-recovered': '1'})
    go('unsigned tenant', 'GET', '/api/months', headers={'Authorization': f'Bearer {TOKEN}', 'X-Cowork-User-ID': QA})
    go('delete', 'DELETE', '/api/internal/tenant')
    go('months after delete', 'GET', '/api/months')
    go('day after delete', 'GET', '/api/day', query='month=2030-01')
    return out


def _reads(d: Deployment, user=QA):
    out = []
    for method, path, query in (('GET', '/api/storage-status', ''), ('GET', '/api/months', ''),
                                ('GET', '/api/day', 'month=2030-01'), ('GET', '/api/files', '')):
        r = d.client.request(method, path + (f'?{query}' if query else ''),
                             headers=_signed(user, method, path, query=query.encode()))
        out.append((path, r.status_code, r.content))
    return out


def _put(d: Deployment, path: str, content: str, user=QA):
    read = d.client.post('/api/file', content=json.dumps({'path': path}).encode(),
                         headers=_signed(user, 'POST', '/api/file', json.dumps({'path': path}).encode(),
                                         content_type='application/json'))
    assert read.status_code == 200, read.text
    body = json.dumps({'path': path, 'content': content, 'version': read.json().get('version')}).encode()
    r = d.client.put('/api/file', content=body,
                     headers=_signed(user, 'PUT', '/api/file', body, content_type='application/json'))
    assert r.status_code == 200, r.text
    return r.json()


@needs_server
@needs_binary
def test_main_mode_answers_exactly_as_the_python_main_process(tmp_path):
    results, logs = {}, {}
    for name, main, writers in (('python', False, False), ('rust', True, False), ('rust-writers', True, True)):
        d = Deployment(tmp_path / name, main, writers)
        try:
            results[name] = _sequence(d)
        finally:
            code = d.close()
        logs[name] = d.logfile.read_text()
        # SIGTERM: both stop after a graceful shutdown (uvicorn re-raises the signal; the Rust parent
        # passes it to its child and exits 0).
        assert code == (0 if main else -signal.SIGTERM), (name, code, logs[name][-4000:])
        assert 'Application shutdown complete' in logs[name], name
    for name in ('rust', 'rust-writers'):
        assert [r[0] for r in results['python']] == [r[0] for r in results[name]]
        for py, rs in zip(results['python'], results[name]):
            assert py == rs, (name, py, rs)
        assert 'child start-up config installed' in logs[name]
        # Some reads were served natively by the parent: the child's access log saw fewer of them.
        seen = [sum(text.count(f'"GET {route}') for route in ('/api/months', '/api/files', '/api/storage-status'))
                for text in (logs['python'], logs[name])]
        assert seen[1] < seen[0], (name, seen)
        reads = [text.count('"POST /api/file') for text in (logs['python'], logs[name])]
        assert reads[1] < reads[0], (name, reads)
    # S3: the writer tenant's save was the parent's (the child never saw the PUT).
    assert '"PUT /api/file' in logs['python'] and '"PUT /api/file' not in logs['rust-writers']
    assert '"PUT /api/file' in logs['rust']
    shutil.rmtree(tmp_path, ignore_errors=True)


@needs_server
@needs_binary
def test_either_parent_starts_on_the_data_the_other_wrote(tmp_path):
    """On-disk compatibility, both directions: Python main → Rust main (writers) → Python main on one data root."""
    root = tmp_path / 'shared'
    day = 'Entries/2030/January/January 6, 2030.md'
    d = Deployment(root, main=False)
    try:
        _reads(d)  # activates the managed tenant
        _put(d, day, '## Sunday, January 6, 2030\n\n### 09:15 — walk\n\n**Me:** QA written by Python.\n\n')
        by_python = _reads(d)
    finally:
        assert d.close() in (0, -signal.SIGTERM)
    d = Deployment(root, main=True, writers=True)
    try:
        assert _reads(d) == by_python
        _put(d, day, '## Sunday, January 6, 2030\n\n### 09:15 — walk\n\n**Me:** QA written by Rust.\n\n')
        _put(d, 'Entries/2030/January/January 7, 2030.md', '## Monday, January 7, 2030\n\n')
        by_rust = _reads(d)
        log = d.logfile
    finally:
        assert d.close() in (0, -signal.SIGTERM)
    assert by_rust != by_python
    assert '"PUT /api/file' not in log.read_text()
    d = Deployment(root, main=False)
    try:
        assert _reads(d) == by_rust
    finally:
        assert d.close() in (0, -signal.SIGTERM)
    shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------- S4: the parent owns the embedding configuration ----------------

def test_the_child_follows_the_parents_embedding_snapshot(tmp_path):
    from agent.config import Config
    from agent.embedding import EmbeddingConfig, EmbeddingConfigError, effective_path_for
    cfg = Config({'retrieval': {'db_path': str(tmp_path / 'index.db')},
                  'llm': {'embed_model': 'qa-embed-a', 'embed_base_url': 'http://10.0.0.9:8080/v1'}})
    path = effective_path_for(cfg)
    assert path == tmp_path / 'embedding-config.effective.json'
    config = EmbeddingConfig(cfg, tmp_path / 'embedding-config.json')
    assert config.active().model == 'qa-embed-a'
    config.follow_parent(path, 'epoch-1')
    # Before the parent's snapshot for this run: embedding off (fail closed), nothing changeable.
    assert config.active().profile == 'unavailable' and config.get()['damaged']
    with pytest.raises(EmbeddingConfigError):
        config.set_next({'model': 'qa-embed-b'})
    snap = {'active': {'base_url': None, 'model': 'qa-embed-b', 'profile': 'nomic-v1'}, 'next': None,
            'previous': {'base_url': 'http://10.0.0.9:8080/v1', 'model': 'qa-embed-a', 'profile': 'nomic-v1'},
            'source': 'saved'}
    path.write_text(json.dumps({**snap, 'epoch': 'epoch-0'}))  # an earlier run's: ignored
    assert config.active().profile == 'unavailable'
    path.write_text(json.dumps({**snap, 'epoch': 'epoch-1'}))
    got = config.get()
    assert got['active']['model'] == 'qa-embed-b' and got['previous']['model'] == 'qa-embed-a'
    assert got['source'] == 'saved' and 'damaged' not in got
    version = got['version']
    assert config.get()['version'] == version  # unchanged file: nothing re-read
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({**snap, 'epoch': 'epoch-1', 'damaged': 'QA refusal', 'next': None,
                               'active': {'base_url': None, 'model': 'qa-embed-b', 'profile': 'unavailable'}}))
    os.replace(tmp, path)
    assert config.active().profile == 'unavailable' and config.get()['damaged'] == 'QA refusal'
    assert config.get()['version'] > version
    # Refusals of seeded addresses are the parent's to make: none here.
    config.refuse_seeded({'active': 'QA'})
    config.release_seeded('active')
    path.write_text('{"broken')
    assert config.active().profile == 'unavailable'  # a torn read keeps the last snapshot


class StubEmbeddings:
    """A synthetic embeddings server: deterministic 8-dimensional vectors from (model, text); it
    records the model of every request."""

    def __init__(self):
        seen = self.seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('content-length') or 0)) or b'{}')
                texts = body.get('input') or []
                texts = [texts] if isinstance(texts, str) else texts
                seen.append((body.get('model'), len(texts)))
                data = []
                for i, text in enumerate(texts):
                    h = hashlib.sha256(f"{body.get('model')}|{text}".encode()).digest()
                    data.append({'index': i, 'embedding': [b / 255.0 + 0.01 for b in h[:8]]})
                out = json.dumps({'data': data}).encode()
                self.send_response(200)
                self.send_header('content-type', 'application/json')
                self.send_header('content-length', str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _embedding(d: Deployment, method='GET', body=None, headers=None):
    h = {'Authorization': f'Bearer {TOKEN}'} if headers is None else headers
    content = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
    return d.client.request(method, '/api/internal/embedding', headers=h, content=content)


def _until(d: Deployment, ready, deadline_s=60.0):
    end = time.monotonic() + deadline_s
    while True:
        r = _embedding(d)
        if r.status_code == 200 and ready(r.json()) and not r.json().get('running'):
            return r.json()
        assert time.monotonic() < end, r.text
        time.sleep(0.3)


def _masked(status_code, payload):
    if isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if k not in ('running', 'last_pass', 'version')}
    return status_code, payload


def _dirty(d: Deployment, user=QA):
    db = d.root / 'state' / 'users' / user / 'managed-index.db'
    with sqlite3.connect(db) as conn:
        return sorted(r[0] for r in conn.execute('SELECT document FROM dirty_documents'))


def _health(d: Deployment, user=QA):
    """GET /api/health with tenant detail (journal counts, retrieval stats): S8 serves it in the
    parent for a recovered managed tenant while the parent owns the embedding configuration."""
    storage = base64.urlsafe_b64encode(json.dumps({'kind': 'local'}).encode()).decode().rstrip('=')
    headers = {'Authorization': f'Bearer {TOKEN}', 'X-Cowork-User-ID': user, 'X-Cowork-Storage': storage}
    headers[ta.HEADER] = ta.sign(KEY, user, 'GET', '/api/health', headers, int(time.time()), secrets.token_hex(16),
                                 b'', body_hash=ta.STREAM)
    r = d.client.get('/api/health', headers=headers)
    return r.status_code, r.content


def _embedding_sequence(d: Deployment, stub: StubEmbeddings):
    out = []

    def norm(payload):
        return json.loads(json.dumps(payload).replace(stub.url, 'STUB'))

    def call(label, method='GET', body=None, headers=None):
        r = _embedding(d, method, body, headers)
        try:
            payload = r.json()
        except ValueError:
            payload = r.text
        out.append((label, *_masked(r.status_code, norm(payload))))
        return r

    call('no token', headers={})
    call('a tenant', headers={'Authorization': f'Bearer {TOKEN}', **{k: v for k, v in _signed(QA, 'GET', '/api/internal/embedding').items() if k != 'Authorization'}})
    _reads(d)
    _put(d, 'Entries/2030/January/January 6, 2030.md',
         '## Sunday, January 6, 2030\n\n### 09:15 — walk\n\n**Me:** QA synthetic walk by the river.\n\n')
    # S6b: the save re-indexed before it replied, whichever process served it.
    out.append(('dirty after save', 200, _dirty(d)))
    out.append(('active ready', *_masked(200, norm(_until(d, lambda s: (s['coverage']['active'] or {}).get('ready'))))))
    _reads(d)  # the child's recovered stamp (a native write voided it)
    out.append(('health with tenant', *_health(d)))
    out.append(('health with tenant again', *_health(d)))
    call('bad action', 'POST', {'action': 'bogus'})
    call('bad json', 'POST', b'{not json')
    call('switch without next', 'POST', {'action': 'switch'})
    call('set next', 'POST', {'action': 'set-next', 'target': {'base_url': stub.url, 'model': 'qa-embed-b', 'profile': 'nomic-v1'}})
    out.append(('next ready', *_masked(200, norm(_until(d, lambda s: (s['coverage']['next'] or {}).get('ready'))))))
    call('switch', 'POST', {'action': 'switch'})
    mark = len(stub.seen)
    _put(d, 'Entries/2030/January/January 7, 2030.md',
         '## Monday, January 7, 2030\n\n### 10:00 — garden\n\n**Me:** QA second note about the garden.\n\n')
    out.append(('dirty after the second save', 200, _dirty(d)))
    # The save after the switch embeds with the new model: the child follows the owner's switch
    # (or, for a writer tenant, the parent re-indexes with the switched configuration).
    models_after = {m for m, _ in stub.seen[mark:]}
    out.append(('models after switch', 200, sorted(models_after)))
    call('finish', 'POST', {'action': 'finish'})
    call('finish again', 'POST', {'action': 'finish'})
    out.append(('final', *_masked(200, norm(_until(d, lambda s: (s['coverage']['active'] or {}).get('ready'))))))
    _reads(d)
    out.append(('health with tenant, final', *_health(d)))
    return out


@needs_server
@needs_binary
def test_the_parent_owning_embeddings_answers_as_the_python_main_process(tmp_path):
    results = {}
    for name, main, writers in (('python', False, False), ('rust-owner', True, False), ('rust-owner-writers', True, True)):
        stub = StubEmbeddings()
        extra = {'LLM_EMBED_BASE_URL': stub.url, 'LLM_EMBED_MODEL': 'qa-embed-a', 'LLM_EMBED_PROFILE': 'nomic-v1'}
        if main:
            extra['DIARY_RUST_EMBEDDING'] = 'on'
            extra['DIARY_RUST_ROUTES'] = '/api/health,/api/months,/api/day,/api/files,/api/storage-status,/api/file'
        d = Deployment(tmp_path / name, main, writers=writers, extra=extra)
        try:
            results[name] = _embedding_sequence(d, stub)
        finally:
            d.close()
            stub.close()
        if main:
            log = d.logfile.read_text()
            assert 'embedding set-next: active nomic-v1/qa-embed-a' in log, log[-3000:]
            # A writer tenant's saves are the parent's (S6b re-indexes them there), not the child's.
            assert ('"PUT /api/file' in log) is not writers, log[-3000:]
            assert 'edit reindex pending' not in log, log[-3000:]
            # S8: the tenant detail was the parent's at least once (3 tenant probes, plus the start-up probe).
            assert log.count('"GET /api/health') < 3, log[-3000:]
            effective = tmp_path / name / 'state' / 'embedding-config.effective.json'
            assert json.loads(effective.read_text())['active']['model'] == 'qa-embed-b'
        saved = json.loads((tmp_path / name / 'state' / 'embedding-config.json').read_text())
        assert saved['active']['model'] == 'qa-embed-b' and saved['previous'] is None
    for name in ('rust-owner', 'rust-owner-writers'):
        assert [r[0] for r in results['python']] == [r[0] for r in results[name]]
        for py, rs in zip(results['python'], results[name]):
            assert py == rs, (name, py, rs)
        after = [r for r in results[name] if r[0] == 'models after switch']
        assert after == [('models after switch', 200, ['qa-embed-b'])], (name, after)
    shutil.rmtree(tmp_path, ignore_errors=True)


@needs_server
@needs_binary
def test_a_refused_seed_address_is_refused_for_the_child_too(tmp_path):
    """A seeded embeddings address off this network (a documentation range: refused without DNS) is refused
    by the owner in memory; the child follows that refusal (embedding off) exactly as Python as main process."""
    statuses = {}
    for name, main in (('python', False), ('rust-owner', True)):
        extra = {'LLM_EMBED_BASE_URL': 'http://203.0.113.9:8080/v1', 'LLM_EMBED_MODEL': 'qa-embed-a'}
        if main:
            extra['DIARY_RUST_EMBEDDING'] = 'on'
        d = Deployment(tmp_path / name, main, extra=extra)
        try:
            statuses[name] = _masked(200, _until(d, lambda s: s.get('damaged'), deadline_s=30))
            if main:
                effective = json.loads((tmp_path / name / 'state' / 'embedding-config.effective.json').read_text())
                assert effective['active']['profile'] == 'unavailable' and 'refused' in effective['damaged']
        finally:
            d.close()
    assert statuses['python'] == statuses['rust-owner']
    assert statuses['python'][1]['active']['profile'] == 'unavailable'
    shutil.rmtree(tmp_path, ignore_errors=True)
