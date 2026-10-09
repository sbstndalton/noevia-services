"""TENANT_ASSERTION_IMPL switch: python stays the default and never spawns anything; rust is
AND-composed with Python and fails closed on every fault; the key reaches the child on stdin only.

Fake `tenant-assertion` binaries stand in for the Rust leaf here; the real binary is exercised by
test_tenant_assertion_differential.py when TENANT_ASSERTION_BIN is set. Synthetic keys only."""
import json
import logging
import os
import subprocess
import sys
import time

import pytest

from agent import tenant_assertion as ta
from tests.test_dedicated_storage import volume, A, B  # noqa: F401
from tests.test_tenant_assertion import KEY, signed, keyed  # noqa: F401

SECRET = 'synthetic-storage-secret'


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    ta._reset_for_tests()
    ta._logged.clear()
    for k in ('TENANT_ASSERTION_IMPL', 'TENANT_ASSERTION_BIN'):
        monkeypatch.delenv(k, raising=False)
    yield
    ta._reset_for_tests()
    ta._logged.clear()


def fake(tmp_path, body, name='tenant-assertion'):
    """An executable that records argv/env/stdin to <name>.json, then runs `body` (Python)."""
    path = tmp_path / name
    record = tmp_path / f'{name}.json'
    path.write_text(f'#!{sys.executable}\n'
                    'import json, os, sys\n'
                    'data = sys.stdin.buffer.read()\n'
                    f'open({str(record)!r}, "w").write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ), '
                    '"stdin": data.decode()}))\n' + body)
    path.chmod(0o755)
    return path, record


ACCEPT = 'sys.stdout.write("accept\\n"); sys.exit(0)\n'
REJECT = 'sys.stdout.write("reject: bad signature\\n"); sys.exit(1)\n'


def use(monkeypatch, path):
    monkeypatch.setenv('TENANT_ASSERTION_IMPL', 'rust')
    monkeypatch.setenv('TENANT_ASSERTION_BIN', str(path))


def verify(h, path='/api/day', method='GET', **kw):
    return ta.verify(KEY, h, method, path, **kw)


def test_default_is_python_and_never_spawns(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('spawned under python'))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned under python'))
    assert ta.impl_choice() == 'python'
    assert verify(signed(A, 'GET', '/api/day')) is None
    assert ta.secret_ref_matches(KEY, A, SECRET, ta.storage_secret_ref(KEY, A, SECRET))


def test_unknown_value_is_python_with_one_warning(monkeypatch, caplog):
    monkeypatch.setenv('TENANT_ASSERTION_IMPL', 'fortran')
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned for an unknown value'))
    with caplog.at_level(logging.WARNING, logger=ta.__name__):
        for _ in range(3):
            assert ta.impl_choice() == 'python'
        assert verify(signed(A, 'GET', '/api/day')) is None
    assert sum('TENANT_ASSERTION_IMPL' in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize('value', ['rust', ' RUST ', 'Rust'])
def test_rust_values_select_rust(monkeypatch, value):
    monkeypatch.setenv('TENANT_ASSERTION_IMPL', value)
    assert ta.impl_choice() == 'rust'


def test_rust_accept_and_python_accept_accepts_once(tmp_path, monkeypatch):
    path, record = fake(tmp_path, ACCEPT)
    use(monkeypatch, path)
    h = signed(A, 'POST', '/api/file', {'Content-Type': 'application/json'}, query=b'x=1', body=b'{}')
    assert verify(h, '/api/file', 'POST', query=b'x=1', body_hash=ta.sha256_hex(b'{}')) is None
    sent = json.loads(json.loads(record.read_text())['stdin'])
    assert sent['op'] == 'verify' and sent['user_id'] == A and sent['query_hex'] == b'x=1'.hex()
    assert sent['body_hash'] == ta.sha256_hex(b'{}') and float(sent['now']) > 0
    # The nonce cache still runs after both accept.
    assert verify(h, '/api/file', 'POST', query=b'x=1', body_hash=ta.sha256_hex(b'{}')) == 'replayed'


def test_key_and_headers_travel_on_stdin_only_with_a_minimal_env(tmp_path, monkeypatch):
    path, record = fake(tmp_path, ACCEPT)
    use(monkeypatch, path)
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    monkeypatch.setenv('DIARY_AUTH_TOKEN', 'synthetic-bearer-token')
    assert verify(signed(A, 'GET', '/api/day')) is None
    seen = json.loads(record.read_text())
    assert seen['argv'] == ['check']
    assert set(seen['env']) <= {'PATH', 'PWD', 'SHLVL', '_', 'LC_CTYPE', '__CF_USER_TEXT_ENCODING'}
    assert 'PATH' in seen['env']
    assert KEY not in json.dumps(seen['env']) and 'synthetic-bearer-token' not in json.dumps(seen['env'])
    assert json.loads(seen['stdin'])['key'] == KEY
    assert ta.secret_ref_matches(KEY, A, SECRET, ta.storage_secret_ref(KEY, A, SECRET))
    seen = json.loads(record.read_text())
    assert seen['argv'] == ['check'] and SECRET not in json.dumps(seen['argv']) + json.dumps(seen['env'])
    assert json.loads(seen['stdin']) == {'op': 'secret_ref', 'key': KEY, 'user_id': A, 'secret': SECRET,
                                         'ref': ta.storage_secret_ref(KEY, A, SECRET)}


@pytest.mark.parametrize('body', [
    REJECT,
    'sys.exit(2)\n',
    'sys.exit(3)\n',
    'sys.stdout.write("accept\\n"); sys.exit(1)\n',      # accept text with a refusal exit
    'sys.stdout.write("accepted\\n"); sys.exit(0)\n',    # not exactly "accept"
    'sys.stdout.write("accept\\naccept\\n"); sys.exit(0)\n',
    'sys.stdout.write("accept"); sys.exit(0)\n',         # no newline
    'sys.stdout.write("x" * 100000); sys.exit(0)\n',
    'import os, signal; os.kill(os.getpid(), signal.SIGKILL)\n',
])
def test_any_rust_refusal_or_fault_rejects(tmp_path, monkeypatch, body):
    path, _ = fake(tmp_path, body)
    use(monkeypatch, path)
    h = signed(A, 'GET', '/api/day')
    assert verify(h) in ('rust refused', 'rust unavailable')
    assert not ta.secret_ref_matches(KEY, A, SECRET, ta.storage_secret_ref(KEY, A, SECRET))


def test_missing_or_non_executable_binary_rejects(tmp_path, monkeypatch, caplog):
    use(monkeypatch, tmp_path / 'absent' / 'tenant-assertion')
    with caplog.at_level(logging.WARNING, logger=ta.__name__):
        assert verify(signed(A, 'GET', '/api/day')) == 'rust unavailable'
        assert verify(signed(A, 'GET', '/api/day')) == 'rust unavailable'
    assert sum('not found' in r.getMessage() for r in caplog.records) == 1
    plain = tmp_path / 'plain'
    plain.write_text('not a program')
    use(monkeypatch, plain)
    assert verify(signed(A, 'GET', '/api/day')) == 'rust unavailable'
    monkeypatch.setenv('TENANT_ASSERTION_BIN', 'no-such-tenant-assertion-on-path')
    assert verify(signed(A, 'GET', '/api/day')) == 'rust unavailable'


def test_default_binary_is_the_fixed_image_path(monkeypatch):
    assert ta.RUST_TIMEOUT_S == 0.5
    assert ta.DEFAULT_BIN == '/usr/local/bin/tenant-assertion'
    seen = []
    monkeypatch.setattr(ta.os.path, 'isfile', lambda p: seen.append(p) or False)
    monkeypatch.setattr(ta.shutil, 'which', lambda *a: pytest.fail('PATH lookup for the default binary'))
    assert ta._rust_binary() is None and seen == ['/usr/local/bin/tenant-assertion']


def _pid_gone(pidfile):
    """True once the child is dead AND reaped: an unreaped zombie still answers kill(pid, 0)."""
    try:
        os.kill(int(pidfile.read_text()), 0)
    except ProcessLookupError:
        return True
    return False


def test_timeout_kills_and_reaps_the_child_quickly(tmp_path, monkeypatch):
    pidfile = tmp_path / 'pid'
    path, _ = fake(tmp_path, f'open({str(pidfile)!r}, "w").write(str(os.getpid())); import time; time.sleep(30); sys.stdout.write("accept\\n")\n')
    use(monkeypatch, path)
    t0 = time.monotonic()
    assert verify(signed(A, 'GET', '/api/day')) == 'rust unavailable'
    assert time.monotonic() - t0 < 1.5
    assert _pid_gone(pidfile)


def test_endless_stdout_is_capped_and_the_child_killed(tmp_path, monkeypatch):
    pidfile = tmp_path / 'pid'
    path, _ = fake(tmp_path, f'open({str(pidfile)!r}, "w").write(str(os.getpid()))\n'
                             'sys.stdout.write("accept\\n"); sys.stdout.flush()\n'
                             'while True:\n    sys.stdout.write("x" * 65536); sys.stdout.flush()\n')
    use(monkeypatch, path)
    t0 = time.monotonic()
    assert verify(signed(A, 'GET', '/api/day')) == 'rust unavailable'
    assert time.monotonic() - t0 < 1.5
    assert _pid_gone(pidfile)


def test_stderr_flood_is_discarded_without_blocking(tmp_path, monkeypatch):
    path, _ = fake(tmp_path, 'sys.stderr.write("e" * (4 << 20)); sys.stderr.flush(); sys.stdout.write("accept\\n")\n')
    use(monkeypatch, path)
    assert verify(signed(A, 'GET', '/api/day')) is None


def test_child_that_never_reads_stdin_cannot_block(tmp_path, monkeypatch):
    path, _ = fake(tmp_path, 'sys.stdout.write("accept\\n")\n', name='noread')
    # Replace the recording prologue (which reads stdin) with one that does not.
    path.write_text(f'#!{sys.executable}\nimport sys, time\ntime.sleep(5)\n')
    use(monkeypatch, path)
    big = signed(A, 'GET', '/api/day', {'X-Cowork-Storage': 's' * 200000})
    t0 = time.monotonic()
    assert verify(big) == 'rust unavailable'
    assert time.monotonic() - t0 < 1.5


def test_slow_rust_check_does_not_block_the_event_loop(volume, tmp_path, monkeypatch):  # noqa: F811
    import asyncio
    import httpx
    import agent.app as appmod
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    monkeypatch.setattr(ta, 'RUST_TIMEOUT_S', 3.0)
    path, _ = fake(tmp_path, 'import time; time.sleep(1.5); sys.stdout.write("accept\\n")\n')
    use(monkeypatch, path)

    async def main():
        transport = httpx.ASGITransport(app=appmod.app)
        async with httpx.AsyncClient(transport=transport, base_url='http://diary') as client:
            # Timed from the slow request's launch: a blocked loop would also delay the sleep.
            t0 = time.monotonic()
            slow = asyncio.create_task(client.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status')))
            await asyncio.sleep(0.2)
            health = await client.get('/api/health')
            elapsed = time.monotonic() - t0
            return health.status_code, elapsed, (await slow).status_code

    code, elapsed, slow_code = asyncio.run(main())
    assert code == 200 and elapsed < 1.0, elapsed
    assert slow_code == 200


def test_python_refusal_never_consults_rust(tmp_path, monkeypatch):
    path, record = fake(tmp_path, ACCEPT)
    use(monkeypatch, path)
    h = signed(A, 'GET', '/api/day')
    assert verify({**h, 'X-Cowork-User-ID': B}) == 'bad signature'
    assert verify(h, now=0.0) == 'outside clock window'
    assert not ta.secret_ref_matches(KEY, A, SECRET, 'a' * 32)
    assert not record.exists()


def test_rust_refusal_does_not_reach_the_nonce_cache(tmp_path, monkeypatch):
    path, _ = fake(tmp_path, REJECT)
    use(monkeypatch, path)
    h = signed(A, 'GET', '/api/day')
    assert verify(h) == 'rust refused'
    assert not ta._seen


def test_no_secret_in_logs_on_any_failure(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.DEBUG):
        for body in (REJECT, 'sys.exit(2)\n', 'sys.stdout.write("x"); sys.exit(7)\n'):
            ta._logged.clear()
            path, _ = fake(tmp_path, body, name=f'b{len(caplog.records)}')
            use(monkeypatch, path)
            verify(signed(A, 'GET', '/api/day'))
            ta.secret_ref_matches(KEY, A, SECRET, ta.storage_secret_ref(KEY, A, SECRET))
        use(monkeypatch, tmp_path / 'absent')
        verify(signed(A, 'GET', '/api/day'))
    text = '\n'.join(r.getMessage() for r in caplog.records)
    assert text and KEY not in text and SECRET not in text


def test_middleware_fails_closed_under_rust(keyed, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv('TENANT_ASSERTION_IMPL', 'rust')
    monkeypatch.setenv('TENANT_ASSERTION_BIN', str(tmp_path / 'absent'))
    assert keyed.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status')).status_code == 401
    path, _ = fake(tmp_path, ACCEPT)
    monkeypatch.setenv('TENANT_ASSERTION_BIN', str(path))
    assert keyed.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status')).status_code == 200
    # Rust accepting can never rescue what Python refuses.
    assert keyed.get('/api/storage-status', headers={**signed(B, 'GET', '/api/storage-status'), 'X-Cowork-User-ID': A}).status_code == 401
