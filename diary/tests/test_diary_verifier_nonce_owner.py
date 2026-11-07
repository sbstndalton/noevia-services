"""D8 verifier move, review P1 (review-20261107-diary-verifier): one linearizable nonce owner across
the native (Rust-verified) and Python-verified paths, while both processes are healthy.

Every case runs the REAL Python stack (TenantAssertionMiddleware, ForwardMiddleware, the app's
handlers and _tenant_state, the real tenant-assertion leaf for every Python signature check)
through a gating TCP proxy placed in front of the native side, so a request can be held after the
native side has ACCEPTED it and before its verified marker reaches Python, or have its reply lost
while the native process stays up. The native side is:
- `binary`: the real noevia-diary (DIARY_RUST_SERVER_BIN), its own verifier and cache;
- `leafchild`: a loopback child that checks each signature with the same real tenant-assertion leaf
  plus its own replay set, and answers like noevia-diary (200 + verified marker / 401 without).
Asserted per case: exactly one request is actually served (native serves counted at the proxy,
Python serves counted at _tenant_state), the other copy is refused with Python's 401 bytes, and the
reservation/floor state afterwards. Synthetic tenant and key only.
"""
import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

import agent.app as appmod
from agent import rust_forward
from agent import tenant_assertion as ta
from tests.tenant_binary import needs_binary
from tests.test_rust_forward import (DAILY, KEY, TOKEN, Modes, install_forwarder, needs_server, seed,  # noqa: F401
                                     signed, start_server, world)

UNVERIFIED = b'{"detail":"invalid tenant assertion"}'
VERIFIED = b'x-noevia-diary-verified'


def pending():
    return getattr(ta, '_pending_count', lambda: 0)()


def consumed(raw):
    m = ta._ASSERTION_RE.fullmatch(raw)
    return m.group(2) in ta._seen.get(int(m.group(1)), ())


def _read_http(sock):
    data = b''
    while b'\r\n\r\n' not in data:
        chunk = sock.recv(65536)
        if not chunk:
            return data
        data += chunk
    head, body = data.split(b'\r\n\r\n', 1)
    length = 0
    for line in head.split(b'\r\n')[1:]:
        k, _, v = line.partition(b':')
        if k.strip().lower() == b'content-length':
            length = int(v.strip())
    while len(body) < length:
        chunk = sock.recv(65536)
        if not chunk:
            break
        body += chunk
    return head + b'\r\n\r\n' + body


def _close_framing(message):
    head, body = message.split(b'\r\n\r\n', 1)
    lines = [l for l in head.split(b'\r\n') if not l.lower().startswith(b'connection:')]
    return b'\r\n'.join([lines[0], b'connection: close', *lines[1:]]) + b'\r\n\r\n' + body


class GateProxy:
    """Loopback proxy in front of the native side. Per connection, in order, a mode from `modes`
    (default 'pass'): 'pass' relays at once; 'stall' waits for `gate` BEFORE passing the request to the
    native side (delegated, not yet dispatched there); 'hold' gets the native reply, then waits for `gate`
    before relaying it; 'drop' gets the native reply, waits for `gate`, then closes the connection
    without relaying it (a lost reply; the native process stays up)."""

    def __init__(self, upstream_port):
        self.upstream_port, self.modes, self.replies = upstream_port, [], []
        self.gate, self.replied, self.received = threading.Event(), threading.Event(), threading.Event()
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(16)
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            mode = self.modes.pop(0) if self.modes else 'pass'
            threading.Thread(target=self._serve, args=(conn, mode), daemon=True).start()

    def _serve(self, conn, mode):
        try:
            request = _read_http(conn)
            if not request:
                return
            self.received.set()
            if mode == 'stall':
                self.gate.wait(30)
            up = socket.create_connection(('127.0.0.1', self.upstream_port))
            up.sendall(_close_framing(request))
            reply = _read_http(up)
            up.close()
            self.replies.append(reply)
            self.replied.set()
            if mode in ('hold', 'drop'):
                self.gate.wait(30)
            if mode != 'drop':
                conn.sendall(_close_framing(reply))
        finally:
            conn.close()

    def native_serves(self):
        """Replies in which the native side accepted (verified marker) and answered 200."""
        return sum(1 for r in self.replies if r.startswith(b'HTTP/1.1 200') and VERIFIED in r.split(b'\r\n\r\n')[0].lower())

    def close(self):
        self.gate.set()
        self.listener.close()


class LeafChild:
    """Native stand-in: real tenant-assertion leaf signature check + its own replay set."""

    def __init__(self):
        self.seen, self.lock = set(), threading.Lock()
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(16)
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            message = _read_http(conn)
            head, body = message.split(b'\r\n\r\n', 1)
            lines = head.split(b'\r\n')
            method, target = lines[0].split(b' ')[:2]
            raw = [tuple(x.strip() for x in l.split(b':', 1)) for l in lines[1:]]
            headers = Headers(raw=[(k.lower(), v) for k, v in raw])
            path, _, query = target.partition(b'?')
            body_hash = ta.sha256_hex(body) if ta.body_is_hashed(headers.get('content-type', '')) else ta.STREAM
            ok = headers.get('x-noevia-diary-verify') == '1' and ta._check(
                KEY, headers, method.decode(), path.decode('latin-1'), time.time(), query, body_hash) is None
            if ok:
                with self.lock:
                    raw_assertion = headers.get(ta.HEADER)
                    ok = raw_assertion not in self.seen
                    self.seen.add(raw_assertion)
            trusted = headers.get('x-noevia-diary-verify') != '1' and headers.get('x-noevia-diary-asserted') == '1'
            if ok or trusted:
                # Verified here (marker), or verified by Python beforehand (D4 rule: no marker).
                payload = b'{"months":[],"native":true}'
                marker = b'x-noevia-diary-verified: 1\r\n' if ok else b''
                reply = (b'HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n' + marker
                         + b'content-length: %d\r\n\r\n' % len(payload)) + payload
            else:
                reply = (b'HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n'
                         % len(UNVERIFIED)) + UNVERIFIED
            conn.sendall(reply)
        finally:
            conn.close()

    def close(self):
        self.listener.close()


class Rig:
    pass


@pytest.fixture(params=['leafchild', pytest.param('binary', marks=needs_server)])
def rig(request, world, monkeypatch):
    saved = install_forwarder()
    client = TestClient(appmod.app)
    modes = Modes(monkeypatch, None)
    modes.python()
    seed(world, client)
    assert client.get('/api/months', headers=signed(DAILY, 'GET', '/api/months')).status_code == 200  # recovered
    child = None
    if request.param == 'binary':
        server = start_server()
        upstream = server.port
    else:
        child = LeafChild()
        server = rust_forward.RustServer(lambda p, t, f: {}, port=child.port, restart_delays=())
        server.ready, server.forward_secret = True, 'synthetic'
        upstream = child.port
    proxy = GateProxy(upstream)
    server.base_url = f'http://127.0.0.1:{proxy.port}'
    modes.rust(server)
    serves = []
    real_state = appmod._tenant_state

    def counting(req, **kw):
        if req.url.path == '/api/months':
            serves.append(1)
        return real_state(req, **kw)
    monkeypatch.setattr(appmod, '_tenant_state', counting)
    r = Rig()
    r.kind, r.server, r.proxy, r.python_serves, r.monkeypatch = request.param, server, proxy, serves, monkeypatch
    yield r
    proxy.close()
    if child is not None:
        child.close()
    if request.param == 'binary':
        server.stop()
    appmod.app.user_middleware[:] = saved
    appmod.app.middleware_stack = None


async def _empty():
    if False:
        yield b''


def http():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=appmod.app), base_url='http://diary', timeout=60)


async def chunked_copy(c, headers):
    """The identical signed request with empty chunked framing: Transfer-Encoding keeps it on the
    Python path; the body is still empty, so it signs sha256("") exactly like the original."""
    return await c.request('GET', '/api/months', headers=headers, content=_empty())


async def until(event):
    assert await asyncio.to_thread(event.wait, 20)


@needs_binary
def test_native_accepted_then_identical_empty_chunked_copy_on_python_is_refused(rig):
    h = signed(DAILY, 'GET', '/api/months')
    rig.proxy.modes = ['hold']

    async def run():
        async with http() as c:
            a = asyncio.ensure_future(c.get('/api/months', headers=h))
            await until(rig.proxy.replied)  # native has ACCEPTED A; its marker is held
            b = await chunked_copy(c, h)
            rig.proxy.gate.set()
            return await a, b
    a, b = asyncio.run(run())
    assert (a.status_code, b.status_code, b.content) == (200, 401, UNVERIFIED)
    assert len(rig.proxy.replies) == 1, 'B never reached the native side'
    assert rig.proxy.native_serves() == 1 and rig.python_serves == []
    assert consumed(h[ta.HEADER]) and pending() == 0


@needs_binary
def test_delegated_but_not_yet_native_then_python_copy_is_refused(rig):
    """Review's inverse interleaving: A passed the delegation decision, B is checked and admitted by
    Python before A reaches the native verifier."""
    h = signed(DAILY, 'GET', '/api/months')
    rig.proxy.modes = ['stall']

    async def run():
        async with http() as c:
            a = asyncio.ensure_future(c.get('/api/months', headers=h))
            await until(rig.proxy.received)  # A is on its way; the native side has not seen it
            b = await chunked_copy(c, h)
            rig.proxy.gate.set()
            return await a, b
    a, b = asyncio.run(run())
    assert (a.status_code, b.status_code, b.content) == (200, 401, UNVERIFIED)
    assert rig.proxy.native_serves() == 1 and rig.python_serves == []
    assert consumed(h[ta.HEADER]) and pending() == 0


@needs_binary
@pytest.mark.parametrize('order', ['python-admits-first', 'python-checked-then-native-admits'])
def test_reverse_ordering_admits_exactly_one(rig, order):
    h = signed(DAILY, 'GET', '/api/months')
    checked, go = threading.Event(), threading.Event()
    real_check = ta._check

    def gated(key, headers, *a, **kw):
        result = real_check(key, headers, *a, **kw)
        if headers.get('transfer-encoding'):
            checked.set()  # B's real signature check passed; hold it before nonce admission
            assert go.wait(20)
        return result
    rig.monkeypatch.setattr(ta, '_check', gated)

    async def run():
        async with http() as c:
            if order == 'python-admits-first':
                go.set()
                b = await chunked_copy(c, h)
                a = await c.get('/api/months', headers=h)
                return a, b
            b_task = asyncio.ensure_future(chunked_copy(c, h))
            await until(checked)
            a = await c.get('/api/months', headers=h)  # native admits while B is between check and admission
            go.set()
            return a, await b_task
    a, b = asyncio.run(run())
    if order == 'python-admits-first':
        assert (b.status_code, a.status_code, a.content) == (200, 401, UNVERIFIED)
        assert rig.python_serves == [1] and rig.proxy.replies == [], 'A was refused by the owner, never sent'
    else:
        assert (a.status_code, b.status_code, b.content) == (200, 401, UNVERIFIED)
        assert rig.proxy.native_serves() == 1 and rig.python_serves == []
    assert consumed(h[ta.HEADER]) and pending() == 0


@needs_binary
def test_cancellation_before_marker_keeps_the_claim_then_commits(rig):
    h = signed(DAILY, 'GET', '/api/months')
    rig.proxy.modes = ['hold']
    floor = ta._floor

    async def run():
        async with http() as c:
            a = asyncio.ensure_future(c.get('/api/months', headers=h))
            await until(rig.proxy.replied)
            a.cancel()  # the caller goes away before the marker arrives
            with pytest.raises(asyncio.CancelledError):
                await a
            assert pending() == 1, 'a dispatched claim is not released by cancellation'
            b = await chunked_copy(c, h)
            again = await c.get('/api/months', headers=h)
            rig.proxy.gate.set()  # the detached exchange receives the native verdict
            for _ in range(200):
                if pending() == 0:
                    break
                await asyncio.sleep(0.01)
            late = await chunked_copy(c, h)
            return b, again, late
    b, again, late = asyncio.run(run())
    for resp in (b, again, late):
        assert (resp.status_code, resp.content) == (401, UNVERIFIED)
    assert len(rig.proxy.replies) == 1 and rig.python_serves == []
    assert pending() == 0 and consumed(h[ta.HEADER]), 'the late verified answer committed the claim'
    assert ta._floor == floor


@needs_binary
def test_read_reply_lost_while_native_process_is_up_is_not_served_again(rig):
    h = signed(DAILY, 'GET', '/api/months')
    rig.proxy.modes = ['drop']
    if rig.kind == 'binary':
        assert rig.server._proc.poll() is None

    async def run():
        async with http() as c:
            a = asyncio.ensure_future(c.get('/api/months', headers=h))
            await until(rig.proxy.replied)
            b = await chunked_copy(c, h)  # during the unknown window
            rig.proxy.gate.set()  # the reply is lost; the native process is still running
            a = await a
            after = [await chunked_copy(c, h), await c.get('/api/months', headers=h)]
            return a, b, after
    a, b, after = asyncio.run(run())
    assert a.status_code == 503 and a.content == rust_forward.UNCONFIRMED
    assert a.headers.get('retry-after') == '1'
    for resp in (b, *after):
        assert (resp.status_code, resp.content) == (401, UNVERIFIED)
    assert rig.python_serves == [], 'never served a second time here'
    assert len(rig.proxy.replies) == 1
    assert pending() == 1, 'held until the validity horizon, not released'


@needs_binary
def test_lost_child_before_reply_serves_the_same_request_once_and_nothing_else(rig):
    """The accepted exception: the native process EXITED before its reply arrived."""
    if rig.kind == 'binary':
        proc = rig.server._proc
    else:
        class Exited:
            def poll(self):
                return 0

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0
        proc = rig.server._proc = Exited()
    h = signed(DAILY, 'GET', '/api/months')
    rig.proxy.modes = ['drop']

    async def run():
        async with http() as c:
            a = asyncio.ensure_future(c.get('/api/months', headers=h))
            await until(rig.proxy.replied)
            b = await chunked_copy(c, h)
            if rig.kind == 'binary':
                proc.kill()
                proc.wait(10)
            rig.proxy.gate.set()
            a = await a
            return a, b, await chunked_copy(c, h)
    a, b, again = asyncio.run(run())
    assert a.status_code == 200 and rig.python_serves == [1], 'the same request, once'
    for resp in (b, again):
        assert (resp.status_code, resp.content) == (401, UNVERIFIED)
    assert pending() == 0 and consumed(h[ta.HEADER])


@needs_binary
def test_bad_signature_and_capacity_release_without_poisoning_the_floor(rig):
    good = signed(DAILY, 'GET', '/api/months')
    forged = dict(good)
    sig = good[ta.HEADER]
    forged[ta.HEADER] = sig[:-1] + ('A' if sig[-1] != 'A' else 'B')
    floor, lost = ta._floor, set(ta._lost)

    async def run():
        async with http() as c:
            bad = await c.get('/api/months', headers=forged)  # same ts and nonce, invalid signature
            assert pending() == 0, 'an unverified claim is released'
            ok = await c.get('/api/months', headers=good)  # the nonce was never consumed
            # Capacity: with one reservation held, a second request is verified here instead.
            rig.monkeypatch.setattr(ta, '_pending_cap', 1)
            rig.proxy.modes = ['hold']
            rig.proxy.replied.clear()
            held_h = signed(DAILY, 'GET', '/api/months')
            held = asyncio.ensure_future(c.get('/api/months', headers=held_h))
            await until(rig.proxy.replied)
            other = await c.get('/api/months', headers=signed(DAILY, 'GET', '/api/months'))
            rig.proxy.gate.set()
            return bad, ok, other, await held
    bad, ok, other, held = asyncio.run(run())
    assert (bad.status_code, bad.content) == (401, UNVERIFIED)
    assert ok.status_code == 200 and consumed(good[ta.HEADER])
    # Full table: `other` was not delegated; Python verified it and forwarded it as asserted (D4
    # rule), so the native side served it WITHOUT verifying it (no marker).
    assert other.status_code == 200 and held.status_code == 200
    assert rig.proxy.native_serves() == 2, 'native verified only `good` and `held`'
    assert sum(r.startswith(b'HTTP/1.1 200') for r in rig.proxy.replies) == 3
    assert pending() == 0 and ta._floor == floor and set(ta._lost) == lost


# ------------------------------------------------------------------ the owner table itself

def test_reservations_are_exclusive_owned_bounded_and_never_raise_the_floor(monkeypatch):
    ta._reset_for_tests()
    now = 1_000_000.0
    raw = ta.sign(KEY, DAILY, 'GET', '/api/months', {}, int(now), 'a' * 32)
    claim = ta.reserve(raw, now=now)
    assert claim is not None and ta.reserve(raw, now=now) is None
    # Every other admission of the nonce is refused while it is held; the owner's own is not.
    assert ta._remember_nonce('a' * 32, int(now), now) == 'replayed'
    stale = ta.Claim(claim.key)
    ta.release(stale)
    ta.commit(stale, now=now)
    assert ta.holds(claim), 'a foreign token neither releases nor commits'
    assert ta._remember_nonce('a' * 32, int(now), now, owner=claim) is None
    assert not ta.holds(claim)
    ta.release(claim)  # late callback after the owner consumed it: no effect on the committed nonce
    assert ta.reserve(raw, now=now) is None and ta._remember_nonce('a' * 32, int(now), now) == 'replayed'
    # Outside the clock window: never reserved.
    assert ta.reserve(ta.sign(KEY, DAILY, 'GET', '/', {}, int(now) + 61, 'b' * 32), now=now) is None
    # Bounded, and unverified claims (future-dated too) never move the floor or `_lost`.
    monkeypatch.setattr(ta, '_pending_cap', 3)
    claims = [ta.reserve(ta.sign(KEY, DAILY, 'GET', '/', {}, int(now) + 50, f'{i:032x}'), now=now) for i in range(4)]
    assert claims[3] is None and all(claims[:3])
    floor, lost = ta._floor, set(ta._lost)
    # Held to the validity horizon (ts + SKEW_S + RESTART_SKEW_S), then gone without trace.
    assert ta._pending_count() == 3
    later = now + 50 + ta.SKEW_S + ta.RESTART_SKEW_S + 1
    assert ta.reserve(ta.sign(KEY, DAILY, 'GET', '/', {}, int(later), 'c' * 32), now=later) is not None
    assert ta._pending_count() == 1
    assert ta._floor == floor and set(ta._lost) == lost
    ta._reset_for_tests()
