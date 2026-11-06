"""D8 verifier move: the Rust process (noevia-diary assertion.rs) is the only verifier of every
request the forwarder sends it; Python verifies everything it serves itself.

Three groups:
- forwarder unit cases with a scripted loopback child (always run): what is delegated, that a
  decline without ``x-noevia-diary-verified`` (or no answer) is verified here before anything is
  served, that a verified reply consumes the nonce in this process's cache too, and that the
  internal reply headers never reach the client;
- differential cases against the real Python verifier and the real binary: each request runs once
  with Python alone and once through the forwarder (Rust verifies), and the replies are compared
  byte for byte, with the Python leaf call count proving who verified: valid, expired, replayed
  nonce, wrong tenant, missing header, clock skew on both sides, deleted tenant (410), tombstoned
  writer write (410), body tampering, oversize body (413);
- concurrency: one nonce sent many times at once is accepted exactly once across both caches.
Synthetic tenants and keys only.
"""
import asyncio
import hashlib
import json
import secrets
import socket
import threading
import time

import pytest

import agent.app as appmod
from agent import rust_forward
from agent import tenant_assertion as ta
from tests.tenant_binary import needs_binary
from tests.test_rust_forward import (DAILY, KEY, OTHER, TOKEN, both, needs_server, signed, world)  # noqa: F401
from tests.test_rust_forward_d5 import _writer_registry, writers  # noqa: F401
from tests.test_rust_forward_d4b import sign

UNVERIFIED = b'{"detail":"invalid tenant assertion"}'
TOO_LARGE = b'{"detail":"request body too large"}'


@pytest.fixture
def leaf_calls(monkeypatch):
    """Counts Python's own signature checks (the tenant-assertion leaf call)."""
    calls = []
    real = ta._rust_decision

    def counting(payload):
        if payload.get('op') == 'verify':
            calls.append(1)
        return real(payload)
    monkeypatch.setattr(ta, '_rust_decision', counting)
    return calls


def wire(resp):
    return resp.status_code, resp.headers.get('content-type'), resp.content


# ------------------------------------------------------------------ scripted child

class Child:
    """Loopback child answering every request with `reply` (bytes after the status line)."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(8)
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            data = b''
            while b'\r\n\r\n' not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            head = data.split(b'\r\n\r\n', 1)[0]
            self.requests.append({k.lower(): v for k, v in (l.split(b': ', 1) for l in head.split(b'\r\n')[1:])})
            if self.reply is not None:
                conn.sendall(self.reply)
            conn.close()

    def close(self):
        self.listener.close()


class Spy:
    def __init__(self):
        self.calls = []

    async def __call__(self, scope, receive, send):
        self.calls.append(dict(scope.get('state') or {}))
        await send({'type': 'http.response.start', 'status': 200,
                    'headers': [(b'content-length', b'2'), (b'content-type', b'application/json')]})
        await send({'type': 'http.response.body', 'body': b'{}'})


def stack(port):
    """TenantAssertionMiddleware -> ForwardMiddleware -> Spy, with a 'ready' scripted child."""
    server = rust_forward.RustServer(lambda p, t, f: {}, port=port, restart_delays=())
    server.ready, server.forward_secret = True, 'synthetic'
    spy = Spy()
    forward = rust_forward.ForwardMiddleware(spy, server=server, routes={'/api/months': 'GET'})
    return appmod.TenantAssertionMiddleware(forward), forward, spy


def run(app, headers, path='/api/months'):
    scope = {'type': 'http', 'method': 'GET', 'path': path, 'raw_path': path.encode(), 'query_string': b'',
             'headers': [(k.lower().encode(), v.encode()) for k, v in headers.items()], 'state': {}}
    sent = []

    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message):
        sent.append(message)
    asyncio.run(app(scope, receive, send))
    return sent[0]['status'], dict(sent[0]['headers']), b''.join(m.get('body', b'') for m in sent[1:])


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    ta._reset_for_tests()
    yield
    ta._reset_for_tests()


DECLINE = b'HTTP/1.1 421 Misdirected Request\r\nx-noevia-diary-decline: 1\r\ncontent-length: 0\r\n\r\n'
VERIFIED_DECLINE = (b'HTTP/1.1 421 Misdirected Request\r\nx-noevia-diary-decline: 1\r\n'
                    b'x-noevia-diary-verified: 1\r\ncontent-length: 0\r\n\r\n')
VERIFIED_OK = (b'HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nx-noevia-diary-verified: 1\r\n'
               b'content-length: 15\r\n\r\n{"native":true}')


def test_delegated_request_carries_verify_and_python_does_not_check(keyed, leaf_calls):
    child = Child(VERIFIED_OK)
    try:
        app, _, spy = stack(child.port)
        headers = signed(DAILY, 'GET', '/api/months')
        status, reply_headers, body = run(app, headers)
        assert (status, body) == (200, b'{"native":true}')
        assert b'x-noevia-diary-verified' not in reply_headers
        assert child.requests[0][b'x-noevia-diary-verify'] == b'1'
        assert child.requests[0][b'x-noevia-diary-asserted'] == b'0'
        assert leaf_calls == [] and spy.calls == []
        # The nonce Rust accepted is consumed here too: a replay is no longer delegated and is refused.
        assert ta.already_seen(headers[ta.HEADER])
        assert run(app, headers)[0::2] == (401, UNVERIFIED)
        assert len(child.requests) == 1
    finally:
        child.close()


@needs_binary
def test_decline_without_verified_header_is_verified_here(keyed, leaf_calls):
    child = Child(DECLINE)
    try:
        app, _, spy = stack(child.port)
        bad = signed(OTHER, 'GET', '/api/months')
        bad['X-Cowork-User-ID'] = DAILY  # signed for another tenant
        assert run(app, bad)[0::2] == (401, UNVERIFIED)
        assert spy.calls == [] and len(leaf_calls) == 1
        good = signed(DAILY, 'GET', '/api/months')
        assert run(app, good)[0] == 200
        assert spy.calls[-1].get('tenant_asserted') is True and len(leaf_calls) == 2
    finally:
        child.close()


@needs_binary
def test_no_answer_is_verified_here(keyed, leaf_calls):
    child = Child(None)  # closes without replying
    try:
        app, _, spy = stack(child.port)
        expired = signed(DAILY, 'GET', '/api/months', ts=int(time.time()) - 3600)
        assert run(app, expired)[0::2] == (401, UNVERIFIED)
        assert spy.calls == [] and len(leaf_calls) == 1
    finally:
        child.close()


def test_verified_decline_is_served_here_without_a_second_check(keyed, leaf_calls):
    child = Child(VERIFIED_DECLINE)
    try:
        app, _, spy = stack(child.port)
        headers = signed(DAILY, 'GET', '/api/months')
        assert run(app, headers)[0] == 200
        assert spy.calls[-1].get('tenant_asserted') is True and leaf_calls == []
        assert ta.already_seen(headers[ta.HEADER])
    finally:
        child.close()


def test_not_delegated_when_no_forwarder_route_or_unready():
    child = Child(VERIFIED_OK)
    try:
        _, forward, _ = stack(child.port)
        scope = {'type': 'http', 'method': 'GET', 'path': '/api/day', 'headers': [], 'state': {}}
        assert not forward.delegates(scope)
        scope['path'] = '/api/months'
        assert forward.delegates(scope)
        scope['headers'] = [(b'content-length', b'3')]
        assert not forward.delegates(scope), 'a read with a body is served (and verified) by Python'
        scope['headers'] = []
        forward.server.ready = False
        assert not forward.delegates(scope)
    finally:
        child.close()


# ------------------------------------------------------------------ real binary: differential

def months(client, headers):
    return client.get('/api/months', headers=headers)


@needs_binary
@needs_server
def test_differential_read_refusals_and_acceptance(both, leaf_calls):
    client, modes = both
    now = int(time.time())

    def wrong_tenant():
        h = signed(OTHER, 'GET', '/api/months')
        h['X-Cowork-User-ID'] = DAILY
        return h

    def missing():
        h = signed(DAILY, 'GET', '/api/months')
        del h[ta.HEADER]
        return h

    def tampered_query():
        return signed(DAILY, 'GET', '/api/months', query=b'x=1')

    cases = [
        ('valid', lambda: signed(DAILY, 'GET', '/api/months'), 200),
        ('expired', lambda: signed(DAILY, 'GET', '/api/months', ts=now - 3600), 401),
        ('wrong tenant', wrong_tenant, 401),
        ('missing header', missing, 401),
        ('malformed header', lambda: {**signed(DAILY, 'GET', '/api/months'), ta.HEADER: 'v2.1.zz.x'}, 401),
        ('query not signed', tampered_query, 401),
        ('future skew inside window', lambda: signed(DAILY, 'GET', '/api/months', ts=int(time.time()) + 55), 200),
        ('future skew outside window', lambda: signed(DAILY, 'GET', '/api/months', ts=int(time.time()) + 65), 401),
        ('past skew outside window', lambda: signed(DAILY, 'GET', '/api/months', ts=int(time.time()) - 65), 401),
    ]
    modes.python()
    months(client, signed(DAILY, 'GET', '/api/months'))  # Python recovers the tenant
    rows = []
    for name, make, status in cases:
        modes.python()
        leaf_calls.clear()
        p = months(client, make())
        python_checks = len(leaf_calls)
        modes.rust()
        leaf_calls.clear()
        r = months(client, make())
        rows.append((name, p.status_code, r.status_code, python_checks, len(leaf_calls)))
        assert wire(p) == wire(r), name
        assert p.status_code == status, name
        assert b'x-noevia-diary' not in b''.join(k for k, _ in r.headers.raw), name
        # Rust verified it: Python never ran its signature check on the forwarded request.
        assert len(leaf_calls) == 0, (name, rows[-1])
    print('\n'.join(f'{n:30} python={a} rust={b} python_checks={c}/{d}' for n, a, b, c, d in rows))


@needs_binary
@needs_server
def test_differential_replayed_nonce_across_both_caches(both, leaf_calls):
    client, modes = both
    months(client, signed(DAILY, 'GET', '/api/months'))
    for first, then in ((modes.python, modes.rust), (modes.rust, modes.python), (modes.rust, modes.rust)):
        h = signed(DAILY, 'GET', '/api/months')
        first()
        assert months(client, h).status_code == 200
        then()
        resp = months(client, h)
        assert (resp.status_code, resp.content) == (401, UNVERIFIED), (first.__name__, then.__name__)


@needs_binary
@needs_server
def test_differential_deleted_tenant_is_verified_and_answered_by_python(both, leaf_calls):
    client, modes = both
    modes.python()
    assert months(client, signed(OTHER, 'GET', '/api/months')).status_code == 200
    assert client.delete('/api/internal/tenant', headers=signed(OTHER, 'DELETE', '/api/internal/tenant')).status_code == 200
    answers = {}
    for mode in (modes.python, modes.rust):
        mode()
        leaf_calls.clear()
        answers[mode.__name__] = (wire(months(client, signed(OTHER, 'GET', '/api/months'))), len(leaf_calls))
        # An invalid assertion for the deleted tenant is still refused first, as before.
        bad = signed(OTHER, 'GET', '/api/day')
        answers[mode.__name__ + '-bad'] = wire(months(client, bad))
    assert answers['python'] == answers['rust']
    assert answers['rust'][0][0] == 410 and answers['rust'][1] == 1, 'deleted tenants are never delegated'
    assert answers['python-bad'] == answers['rust-bad'] and answers['rust-bad'][2] == UNVERIFIED


@needs_binary
@needs_server
def test_concurrent_replays_are_accepted_once_across_both_processes(both):
    client, modes = both
    months(client, signed(DAILY, 'GET', '/api/months'))
    modes.rust()
    h = signed(DAILY, 'GET', '/api/months')
    results = []
    lock = threading.Lock()

    def go():
        r = months(client, h)
        with lock:
            results.append(r.status_code)
    threads = [threading.Thread(target=go) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert sorted(results) == [200] + [401] * 15


# ------------------------------------------------------------------ writes

def put(client, body, headers):
    return client.put('/api/file', headers=headers, content=body)


@needs_binary
@needs_server
def test_differential_write_tampering_oversize_and_tombstone(writers, leaf_calls):
    client, start, state = writers
    modes = start()
    rows = {}
    for mode in (modes.python, modes.rust):
        mode()
        good = json.dumps({'path': f'AI Memory/verifier-{mode.__name__}.md', 'content': 'synthetic verifier note',
                           'version': None}).encode()
        bad = good.replace(b'synthetic', b'Synthetic')
        leaf_calls.clear()
        headers = sign(DAILY, 'PUT', '/api/file', good, None, b'', 'application/json')
        tampered = wire(put(client, bad, headers))
        accepted = put(client, good, headers)
        replay = wire(put(client, good, headers))
        # Oversize: chunked (no Content-Length), so the forwarder's own bound answers for Rust.
        big = sign(DAILY, 'PUT', '/api/file', b'', None, b'', 'application/json')

        def chunks():
            for _ in range(65):
                yield b' ' * (1024 * 1024)
        oversize = wire(client.put('/api/file', headers=big, content=chunks()))
        rows[mode.__name__] = (tampered, accepted.status_code, replay, oversize, len(leaf_calls))
    p, r = rows['python'], rows['rust']
    assert p[:4] == r[:4]
    assert p[0][0] == 401 and p[0][2] == UNVERIFIED
    assert p[1] == 200 and p[2][2] == UNVERIFIED
    assert p[3][0] == 413 and p[3][2] == TOO_LARGE
    # Rust verified the tampered, the accepted and the oversize write; only the replay (a nonce this
    # process already holds) is kept here and refused by Python's own check.
    assert r[4] == 1 and p[4] >= 3, (p[4], r[4])
    # Tombstoned (deleted within 24 h): never delegated; Python verifies and answers 410.
    assert client.delete('/api/internal/tenant', headers=signed(DAILY, 'DELETE', '/api/internal/tenant')).status_code == 200
    answers = []
    for mode in (modes.python, modes.rust):
        mode()
        answers.append(wire(put(client, good, sign(DAILY, 'PUT', '/api/file', good, None, b'', 'application/json'))))
    assert answers[0] == answers[1] and answers[0][0] == 410
