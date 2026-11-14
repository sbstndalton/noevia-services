"""R9 main mode (DIARY_MAIN=rust): this app as the child of the Rust Diary process (agent/rust_parent.py,
noevia-rs bins/noevia-diary front.rs).

The unit cases always run. The differential runs the SAME request sequence (seeding writes, native and forwarded
reads, refusals, a replay, tenant deletion) against two real deployments on identical synthetic state: today's
Python main process (DIARY_MAIN unset, no Rust process) and `noevia-diary --main` with this app as its child, and
compares status, content type and body byte for byte. It needs the real binaries (DIARY_R9_SERVER_BIN for
noevia-diary, the tenant-assertion leaf) and skips without them.
"""
import asyncio
import io
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from agent import rust_forward, rust_parent
from agent import tenant_assertion as ta
from tests.tenant_binary import needs_binary

ROOT = Path(__file__).resolve().parent.parent
KEY = 'synthetic-tenant-key-r9-main'
TOKEN = 'synthetic-service-token-r9-main'
QA = '3e819f41-6d50-4fb1-8c43-4d5e6f708192'
QB = '4f92a052-7e61-4ac2-9d54-5e6f708192a3'
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
    assert _run(script, image_cmd, 'rust') == ('rust --main -- uvicorn agent.app:app --host 127.0.0.1 --port 8012')
    # Operator commands are unchanged in main mode.
    assert _run(script, ['echo', 'something', 'else'], 'rust') == 'something else'


# ---------------- handshake, ParentServer ----------------

def test_the_handshake_must_be_complete():
    good = {'control_token': 't' * 32, 'parent_secret': 's' * 64, 'control_port': 8011}
    assert rust_parent.read_handshake(io.StringIO(json.dumps(good) + '\n')) == good
    for bad in ({**good, 'control_token': ''}, {**good, 'parent_secret': 1}, {**good, 'control_port': 0},
                {**good, 'control_port': 70000}, {k: v for k, v in good.items() if k != 'parent_secret'}):
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

def _call(headers, recovered=lambda uid: uid == QA):
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

    mw = rust_parent.ParentMiddleware(app, secret=lambda: 'sec', is_recovered=recovered)
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
    seen, sent = _call(base + [(b'x-noevia-diary-asserted', b'0'), (b'x-cowork-user-id', QB.encode())])
    assert seen['asserted'] is False
    assert (b'x-noevia-diary-recovered', b'0') in sent[0]['headers']
    # No tenant, or not a UUID: no stamp.
    for extra in ([], [(b'x-cowork-user-id', b'legacy-user')]):
        _seen, sent = _call(base + extra)
        assert not any(k == b'x-noevia-diary-recovered' for k, _ in sent[0]['headers'])


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
    (root / 'state').mkdir(parents=True)
    path = root / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def _env(root: Path, main: bool) -> dict:
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(root), 'TZ': 'UTC',
           'PYTHONPATH': str(ROOT), 'DIARY_CONFIG': str(_config(root)), 'DIARY_TENANT_KEY': KEY,
           'LLM_BASE_URL': 'http://127.0.0.1:9/v1', 'LLM_EMBED_BASE_URL': 'http://127.0.0.1:9/v1'}
    for name in ('TENANT_ASSERTION_BIN', 'TENANT_ASSERTION_IMPL', 'TENANT_ASSERTION_REQUIRED'):
        if os.environ.get(name):
            env[name] = os.environ[name]
    if main:
        env['DIARY_RUST_ROUTES'] = '/api/months,/api/day,/api/files,/api/storage-status'
        env['DIARY_RUST_SERVER_BIN'] = BIN
    return env


class Deployment:
    def __init__(self, root: Path, main: bool):
        self.port = _free_port()
        uvicorn = [sys.executable, '-m', 'uvicorn', 'agent.app:app', '--host', '127.0.0.1']
        if main:
            child = _free_port()
            argv = [BIN, '--main', '--host', '127.0.0.1', '--port', str(self.port), '--control-port', str(_free_port()),
                    '--child-port', str(child), '--', *uvicorn, '--port', str(child)]
        else:
            argv = [*uvicorn, '--port', str(self.port)]
        self.log = open(root / 'server.log', 'wb')
        self.proc = subprocess.Popen(argv, cwd=ROOT, env=_env(root, main), stdout=self.log, stderr=subprocess.STDOUT)
        self.client = httpx.Client(base_url=f'http://127.0.0.1:{self.port}', timeout=30, trust_env=False)
        deadline = time.monotonic() + 60
        while True:
            assert self.proc.poll() is None, (root / 'server.log').read_text()
            try:
                if self.client.get('/api/health', headers={'Authorization': f'Bearer {TOKEN}'}).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, (root / 'server.log').read_text()
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
        out.append((label, r.status_code, r.headers.get('content-type'), r.content))
        return r

    for _ in range(2):
        go('status', 'GET', '/api/storage-status')
        go('months', 'GET', '/api/months')
        go('day', 'GET', '/api/day', query='month=2030-01')
        go('files', 'GET', '/api/files')
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
    go('other tenant months', 'GET', '/api/months', user=QB)
    go('health', 'GET', '/api/health', headers={'Authorization': f'Bearer {TOKEN}'})
    go('no token', 'GET', '/api/months', headers={'X-Cowork-User-ID': QA})
    nonce = secrets.token_hex(16)
    go('nonce first', 'GET', '/api/months', nonce=nonce)
    go('nonce replay', 'GET', '/api/months', nonce=nonce)
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


@needs_server
@needs_binary
def test_main_mode_answers_exactly_as_the_python_main_process(tmp_path):
    results = {}
    for name, main in (('python', False), ('rust', True)):
        d = Deployment(tmp_path / name, main)
        try:
            results[name] = _sequence(d)
        finally:
            code = d.close()
        # SIGTERM: both stop cleanly (uvicorn 0; the Rust parent passes it to its child and exits 0).
        assert code == 0, (name, code, (tmp_path / name / 'server.log').read_text()[-4000:])
    assert [r[0] for r in results['python']] == [r[0] for r in results['rust']]
    for py, rs in zip(results['python'], results['rust']):
        assert py == rs, (py, rs)
    log = (tmp_path / 'rust' / 'server.log').read_text()
    assert 'child start-up config installed' in log
    # Some reads were served natively by the parent: the child's access log saw fewer of them.
    python_log = (tmp_path / 'python' / 'server.log').read_text()
    seen = [sum(text.count(f'"GET {route}') for route in ('/api/months', '/api/files', '/api/storage-status'))
            for text in (python_log, log)]
    assert seen[1] < seen[0], seen
    shutil.rmtree(tmp_path, ignore_errors=True)
