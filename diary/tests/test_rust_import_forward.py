"""POST /api/storage-import forwarding: the legacy-ready opt-in, the header-less legacy tenant, and
settling (a timeout or cancellation never kills the child under its writer and flock)."""
import asyncio
import pytest
from agent import rust_forward, tenant_writer
from tests.test_rust_forward_d5 import DAILY, FakeChild, forwarder, asgi, put

LEGACY = '33333333-3333-4333-8333-333333333333'


def _sent_headers(child):
    assert child.requests
    return child.requests[-1][0]


@pytest.mark.parametrize('ready', [True, False])
def test_import_carries_the_legacy_ready_answer(monkeypatch, ready):
    async def run():
        child = FakeChild('native')
        _, spy, mw = forwarder(child.port, monkeypatch)
        seen = []
        mw.legacy_ready = lambda uid, header: seen.append((uid, header('x-cowork-storage'))) or ready
        headers, body = put(body=b'{}', extra=[(b'x-cowork-storage', b'synthetic')])
        try:
            await asgi(mw, headers, [body], method='POST', path='/api/storage-import')
            sent = _sent_headers(child)
            assert sent[b'x-noevia-diary-legacy-ready'] == (b'1' if ready else b'0')
            assert sent[b'x-noevia-diary-writer'] == b'1'
            assert seen == [(DAILY, 'synthetic')]
            assert spy.calls == []
        finally:
            child.close()
    asyncio.run(run())


def test_other_writes_carry_no_legacy_ready_header(monkeypatch):
    async def run():
        child = FakeChild('native')
        _, _, mw = forwarder(child.port, monkeypatch)
        mw.legacy_ready = lambda uid, header: True
        headers, body = put()
        try:
            await asgi(mw, headers, [body])
            assert b'x-noevia-diary-legacy-ready' not in _sent_headers(child)
        finally:
            child.close()
    asyncio.run(run())


def test_a_client_cannot_supply_the_legacy_ready_header(monkeypatch):
    async def run():
        child = FakeChild('native')
        _, _, mw = forwarder(child.port, monkeypatch)
        headers, body = put(body=b'{}', extra=[(b'x-noevia-diary-legacy-ready', b'1')])
        try:
            await asgi(mw, headers, [body], method='POST', path='/api/storage-import')
            assert _sent_headers(child)[b'x-noevia-diary-legacy-ready'] == b'0'
        finally:
            child.close()
    asyncio.run(run())


def test_header_less_legacy_tenant_imports_under_its_writer_lock(monkeypatch):
    async def run():
        monkeypatch.setenv('DIARY_LEGACY_USER_ID', LEGACY)
        child = FakeChild('native')
        _, spy, mw = forwarder(child.port, monkeypatch, writers=frozenset({LEGACY}))
        seen = []
        mw.legacy_ready = lambda uid, header: seen.append(uid) or True
        body = b'{}'
        headers = [(b'content-type', b'application/json'), (b'content-length', str(len(body)).encode())]
        try:
            await asgi(mw, headers, [body], method='POST', path='/api/storage-import')
            assert seen == [LEGACY]
            assert _sent_headers(child)[b'x-noevia-diary-legacy-ready'] == b'1'
            # Other header-less writes still never move.
            await asgi(mw, headers, [b'{"path":"a.md","content":"x","version":null}'])
            assert len(child.requests) == 1 and len(spy.calls) == 1
        finally:
            child.close()
    asyncio.run(run())


@pytest.mark.parametrize('cancel', [False, True])
def test_import_does_not_kill_settling_child(monkeypatch, cancel):
    async def run():
        child = FakeChild('slow')
        server, spy, mw = forwarder(child.port, monkeypatch, deadline=.05 if not cancel else 2)
        lost = []
        monkeypatch.setattr(server, 'lost', lambda: lost.append(True))
        monkeypatch.setattr(tenant_writer, '_writers', frozenset())
        headers, body = put(body=b'{}')
        try:
            task = asyncio.create_task(asgi(mw, headers, [body], method='POST', path='/api/storage-import'))
            if cancel:
                for _ in range(100):
                    if child.requests:
                        break
                    await asyncio.sleep(.005)
                assert child.requests
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                response = await task
                assert response[0] == 503
            assert lost == []
            assert spy.calls == []
            await asyncio.sleep(.4)
            assert child.active == 0
        finally:
            child.close()
    asyncio.run(run())


def test_legacy_config_marks_unreadable_values_unsupported(tmp_path):
    from agent.config import Config
    cfg = Config({'corpus': {'backend': 'WebDAV', 'root': 'Diary', 'webdav': {'base_url': 'http://dav.invalid/', 'username': 'u', 'password': 'p'}}})
    out = rust_forward.legacy_config(cfg)
    assert out['base']['kind'] == 'webdav' and out['corpus_root'] == 'Diary'
    assert out['base']['webdav_password'] == 'p' and out['base']['webdav_timeout_s'] == 60.0
    bad = Config({'corpus': {'backend': 'local', 'webdav': {'username': 7}}})
    assert rust_forward.legacy_config(bad)['base']['kind'] == 'unsupported'
