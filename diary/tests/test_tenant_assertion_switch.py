"""The Rust leaf is the only signature check (TENANT_ASSERTION_IMPL, retired in #1071): every fault is a refusal, the
key reaches the child on stdin only, and the cheap checks that need no secret (the header shape, the restart guard) and
the nonce cache stay in Python and never reach, or follow, a refused request.

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
    for k in ('TENANT_ASSERTION_IMPL', 'TENANT_ASSERTION_BIN', 'DIARY_TENANT_KEY'):
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
    monkeypatch.setenv('TENANT_ASSERTION_BIN', str(path))


def verify(h, path='/api/day', method='GET', **kw):
    return ta.verify(KEY, h, method, path, **kw)


def test_the_selector_is_gone_and_a_leftover_setting_warns_once_at_start(monkeypatch, caplog):
    assert not hasattr(ta, 'impl_choice') and not hasattr(ta, 'IMPLS')
    for value in (None, '', ' RUST ', 'rust'):
        if value is None:
            monkeypatch.delenv('TENANT_ASSERTION_IMPL', raising=False)
        else:
            monkeypatch.setenv('TENANT_ASSERTION_IMPL', value)
        with caplog.at_level(logging.WARNING, logger=ta.__name__):
            ta.startup_checks()
        assert not caplog.records, value
    for value in ('python', 'fortran', ' Python '):
        monkeypatch.setenv('TENANT_ASSERTION_IMPL', value)
        caplog.clear()
        ta._logged.clear()
        with caplog.at_level(logging.WARNING, logger=ta.__name__):
            for _ in range(3):
                ta.startup_checks()
        assert [r.getMessage() for r in caplog.records] == [f"TENANT_ASSERTION_IMPL={value.strip().lower()!r} is retired; Rust is always used"], value


def test_a_tenant_key_without_a_usable_binary_is_an_error_at_start(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('TENANT_ASSERTION_BIN', str(tmp_path / 'absent'))
    monkeypatch.setattr(ta, 'DEFAULT_BIN', str(tmp_path / 'absent-default'))
    with caplog.at_level(logging.WARNING, logger=ta.__name__):
        ta.startup_checks()
        assert not [r for r in caplog.records if 'not usable' in r.getMessage()], 'no key set: nothing to say'
        monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
        ta.startup_checks()
        ta.startup_checks()
    messages = [r.getMessage() for r in caplog.records if 'not usable' in r.getMessage()]
    assert len(messages) == 1 and KEY not in messages[0]
    path, _ = fake(tmp_path, ACCEPT)
    use(monkeypatch, path)
    ta._logged.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=ta.__name__):
        ta.startup_checks()
    assert not caplog.records


def test_rust_accept_accepts_once(tmp_path, monkeypatch):
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


def test_bin_override_must_be_an_absolute_executable_file(tmp_path, monkeypatch, caplog):
    """#1832: a bare name (PATH lookup), a relative path, a directory or a non-executable file is
    ignored with one warning and the fixed default is used, never the override."""
    good, _ = fake(tmp_path, ACCEPT, name='good-bin')
    fixed, _ = fake(tmp_path, REJECT, name='fixed-default')
    monkeypatch.setattr(ta, 'DEFAULT_BIN', str(fixed))
    bindir = tmp_path / 'onpath'
    bindir.mkdir()
    (bindir / 'tenant-assertion-on-path').write_text(good.read_text())
    (bindir / 'tenant-assertion-on-path').chmod(0o755)
    monkeypatch.setenv('PATH', f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv('TENANT_ASSERTION_IMPL', 'rust')
    plain = tmp_path / 'plain'
    plain.write_text('x')
    plain.chmod(0o644)
    monkeypatch.chdir(tmp_path)
    for bad in ('tenant-assertion-on-path', './good-bin', 'good-bin', str(tmp_path), str(plain)):
        ta._logged.clear()
        monkeypatch.setenv('TENANT_ASSERTION_BIN', bad)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=ta.__name__):
            assert ta._rust_binary() == str(fixed), bad
            assert ta._rust_binary() == str(fixed), bad
            assert verify(signed(A, 'GET', '/api/day')) == 'rust refused', bad  # the default ran, not the override
        assert sum('ignoring it' in r.getMessage() for r in caplog.records) == 1, bad
    monkeypatch.setenv('TENANT_ASSERTION_BIN', str(good))
    assert ta._rust_binary() == str(good)
    assert verify(signed(A, 'GET', '/api/day')) is None


def test_default_binary_is_the_fixed_image_path(monkeypatch):
    assert ta.RUST_TIMEOUT_S == 0.5
    assert ta.DEFAULT_BIN == '/usr/local/bin/tenant-assertion'
    seen = []
    monkeypatch.setattr(ta.os.path, 'isfile', lambda p: seen.append(p) or False)
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


def test_cheap_refusals_never_consult_rust(tmp_path, monkeypatch):
    """No tenant, a malformed or missing assertion and one signed before this process started are refused
    without spawning the child; what only the secret can tell (a bad signature, the clock window) is the leaf's."""
    path, record = fake(tmp_path, ACCEPT)
    use(monkeypatch, path)
    h = signed(A, 'GET', '/api/day')
    assert verify({k: v for k, v in h.items() if k != 'X-Cowork-User-ID'}) == 'missing tenant'
    assert verify({**h, ta.HEADER: 'v2.1.2.3'}) == 'missing or malformed assertion'
    assert verify({**h, ta.HEADER: h[ta.HEADER] + ' '}) == 'missing or malformed assertion'
    assert verify({k: v for k, v in h.items() if k != ta.HEADER}) == 'missing or malformed assertion'
    ta._reset_for_tests(started_at=time.time() + 3600)
    assert verify(h) == 'signed before this process started'
    assert not record.exists()
    # The same shape reaches the leaf, whose answer is final: a wrong tenant or a stale clock is its refusal.
    ta._reset_for_tests()
    path, record = fake(tmp_path, REJECT, name='reject-bin')
    use(monkeypatch, path)
    assert verify({**h, 'X-Cowork-User-ID': B}) == 'rust refused'
    assert verify(h, now=0.0) == 'rust refused'
    assert not ta.secret_ref_matches(KEY, A, SECRET, 'a' * 32)
    assert record.exists()


def test_the_restart_guard_and_the_nonce_cache_still_run_after_the_leaf_accepts(tmp_path, monkeypatch):
    """The leaf is stateless and knows nothing of the process start: it accepts a replay and an assertion signed
    before this process started, and the Python guard and cache refuse them."""
    path, record = fake(tmp_path, ACCEPT)
    use(monkeypatch, path)
    h = signed(A, 'GET', '/api/day')
    assert verify(h) is None
    assert verify(h) == 'replayed'
    old = signed(A, 'GET', '/api/day', ts=int(time.time()) - 30)
    ta._reset_for_tests(started_at=time.time())
    assert verify(old) == 'signed before this process started'
    ta._reset_for_tests()
    assert verify(old) is None
    assert verify(old) == 'replayed'
    assert verify(signed(A, 'GET', '/api/day')) is None


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


def test_middleware_fails_closed_without_the_leaf_and_keeps_the_guards_with_it(keyed, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv('TENANT_ASSERTION_BIN', str(tmp_path / 'absent'))
    assert keyed.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status')).status_code == 401
    path, _ = fake(tmp_path, ACCEPT)
    monkeypatch.setenv('TENANT_ASSERTION_BIN', str(path))
    assert keyed.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status')).status_code == 200
    # An accepting leaf cannot rescue what needs no secret to refuse: a malformed assertion, a replay.
    h = signed(B, 'GET', '/api/storage-status')
    assert keyed.get('/api/storage-status', headers={**h, ta.HEADER: 'v2.garbage'}).status_code == 401
    assert keyed.get('/api/storage-status', headers=h).status_code == 200
    assert keyed.get('/api/storage-status', headers=h).status_code == 401
