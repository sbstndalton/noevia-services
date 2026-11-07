"""Backup timeout/cancel must leave the child alive to settle its remote effects."""
import asyncio
import pytest
from agent import tenant_writer
from tests.test_rust_forward_d5 import FakeChild, forwarder, asgi, put


@pytest.mark.parametrize('cancel', [False, True])
def test_backup_does_not_kill_settling_child(monkeypatch, cancel):
    async def run():
        child = FakeChild('slow')
        server, spy, mw = forwarder(child.port, monkeypatch, deadline=.05 if not cancel else 2)
        lost = []
        monkeypatch.setattr(server, 'lost', lambda: lost.append(True))
        monkeypatch.setattr(tenant_writer, '_writers', frozenset())
        headers, body = put(body=b'')
        try:
            task = asyncio.create_task(asgi(mw, headers, [body], method='POST', path='/api/storage-backup'))
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
