"""WORKSPACE_RESTORE_VERIFY_IMPL switch (noevia#982): python stays the default and never spawns
anything; rust runs the Rust leaf first, then Python, requires identical listings and fails closed
on every fault; the archive reaches the child on stdin only, with a minimal environment.

Fake `workspace-restore-verify` binaries stand in for the Rust leaf here; the real binary is
exercised by test_workspace_restore_differential.py when WORKSPACE_RESTORE_VERIFY_BIN is set.
Synthetic archives only."""
import hashlib
import json
import logging
import sys
import time

import pytest

from agent import workspace_restore as wr
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from tests.test_dedicated_storage import B

NOTE = b'# Synthetic switch note\n'


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    wr._logged.clear()
    for k in ('WORKSPACE_RESTORE_VERIFY_IMPL', 'WORKSPACE_RESTORE_VERIFY_BIN'):
        monkeypatch.delenv(k, raising=False)
    yield
    wr._logged.clear()


@pytest.fixture
def bundle(tmp_path):
    backend = ManagedCorpusBackend(tmp_path / 'source', B)
    backend.activate({'raw/note.md': NOTE}, {}, ['empty'])
    return archive(backend, '', {})


def listing(body):
    files, dirs, _ = wr._verify(body)
    return wr.listing_sha256(files, dirs)


def fake(tmp_path, body, name='workspace-restore-verify'):
    """An executable that records argv, env and the stdin digest, then runs `body` (Python)."""
    path = tmp_path / name
    record = tmp_path / f'{name}.json'
    path.write_text(f'#!{sys.executable}\n'
                    'import hashlib, json, os, sys\n'
                    'data = sys.stdin.buffer.read()\n'
                    f'open({str(record)!r}, "w").write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ), '
                    '"stdin_sha256": hashlib.sha256(data).hexdigest(), "stdin_len": len(data)}))\n'
                    + body)
    path.chmod(0o755)
    return path, record


def use(monkeypatch, path):
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_IMPL', 'rust')
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_BIN', str(path))


def test_default_is_python_and_never_spawns(bundle, monkeypatch):
    monkeypatch.setattr(wr.subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned under python'))
    for value in (None, '', '  ', 'python', 'PYTHON'):
        if value is None:
            monkeypatch.delenv('WORKSPACE_RESTORE_VERIFY_IMPL', raising=False)
        else:
            monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_IMPL', value)
        assert wr.impl_choice() == 'python'
        assert sorted(wr.verify(bundle)[0]) == ['raw/note.md']


def test_unknown_value_is_python_with_one_warning(monkeypatch, caplog):
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_IMPL', 'fortran')
    with caplog.at_level(logging.WARNING):
        assert wr.impl_choice() == 'python'
        assert wr.impl_choice() == 'python'
    assert sum('WORKSPACE_RESTORE_VERIFY_IMPL' in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize('value', ['rust', 'RUST', ' rust '])
def test_rust_values_select_rust(monkeypatch, value):
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_IMPL', value)
    assert wr.impl_choice() == 'rust'


def test_rust_accept_with_the_same_listing_accepts(bundle, tmp_path, monkeypatch):
    path, record = fake(tmp_path, f'sys.stdout.write("accept {listing(bundle)}\\n"); sys.exit(0)\n')
    use(monkeypatch, path)
    files, dirs, manifest = wr.verify(bundle)
    assert files == {'raw/note.md': NOTE} and dirs == {'empty', 'raw'}
    seen = json.loads(record.read_text())
    assert seen['argv'] == ['verify']
    assert seen['stdin_sha256'] == hashlib.sha256(bundle).hexdigest() and seen['stdin_len'] == len(bundle)
    # Minimal environment: nothing of the Diary's (tokens, keys) reaches the child.
    assert set(seen['env']) - {'PATH', 'LC_CTYPE', '__CF_USER_TEXT_ENCODING'} == set()


def test_listing_mismatch_fails_closed(bundle, tmp_path, monkeypatch, caplog):
    path, _ = fake(tmp_path, 'sys.stdout.write("accept " + "0" * 64 + "\\n"); sys.exit(0)\n')
    use(monkeypatch, path)
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match='unavailable'):
        wr.verify(bundle)
    assert any('different listings' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('code', sorted(wr.MESSAGES))
def test_rust_refusal_answers_with_pythons_message(bundle, tmp_path, monkeypatch, code):
    path, _ = fake(tmp_path, f'sys.stdout.write("reject: {code}\\n"); sys.exit(1)\n')
    use(monkeypatch, path)
    with pytest.raises(ValueError) as caught:
        wr.verify(bundle)
    assert str(caught.value) == wr.MESSAGES[code]


@pytest.mark.parametrize('body', [
    'sys.exit(0)\n',                                              # accept with no output
    'sys.stdout.write("accept\\n"); sys.exit(0)\n',              # no digest
    'sys.stdout.write("accept " + "A" * 64 + "\\n"); sys.exit(0)\n',  # not lowercase hex
    'sys.stdout.write("reject: invalid\\n"); sys.exit(0)\n',     # refusal with exit 0
    'sys.stdout.write("reject: no_such_code\\n"); sys.exit(1)\n',  # unknown code
    'sys.stdout.write("accept " + "0" * 64 + "\\n"); sys.exit(1)\n',  # accept with exit 1
    'sys.exit(2)\n',
    'sys.exit(3)\n',
    'import os, signal; os.kill(os.getpid(), signal.SIGKILL)\n',
])
def test_every_fault_fails_closed(bundle, tmp_path, monkeypatch, body):
    path, _ = fake(tmp_path, body)
    use(monkeypatch, path)
    with pytest.raises(ValueError, match='unavailable|Invalid workspace archive'):
        wr.verify(bundle)


def test_missing_or_non_executable_binary_fails_closed(bundle, tmp_path, monkeypatch, caplog):
    use(monkeypatch, tmp_path / 'absent')
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match='unavailable'):
        wr.verify(bundle)
    plain = tmp_path / 'plain'
    plain.write_text('not executable')
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_BIN', str(plain))
    with pytest.raises(ValueError, match='unavailable'):
        wr.verify(bundle)
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_BIN', 'no-such-workspace-restore-verify-on-path')
    with pytest.raises(ValueError, match='unavailable'):
        wr.verify(bundle)
    assert any('not found' in r.getMessage() for r in caplog.records)


def test_default_binary_is_the_fixed_image_path(monkeypatch):
    assert wr.DEFAULT_BIN == '/usr/local/bin/workspace-restore-verify'
    seen = []
    monkeypatch.setattr(wr.os.path, 'isfile', lambda p: seen.append(p) or False)
    assert wr._rust_binary() is None
    assert seen == [wr.DEFAULT_BIN]


def test_timeout_kills_the_child(bundle, tmp_path, monkeypatch):
    path, _ = fake(tmp_path, 'import time; time.sleep(30)\n')
    use(monkeypatch, path)
    monkeypatch.setattr(wr, 'RUST_TIMEOUT_S', 0.5)
    started = time.monotonic()
    with pytest.raises(ValueError, match='unavailable'):
        wr.verify(bundle)
    assert time.monotonic() - started < 5


def test_endless_stdout_is_capped_and_the_child_killed(bundle, tmp_path, monkeypatch):
    path, _ = fake(tmp_path, 'while True:\n    sys.stdout.write("x" * 65536); sys.stdout.flush()\n')
    use(monkeypatch, path)
    started = time.monotonic()
    with pytest.raises(ValueError, match='unavailable'):
        wr.verify(bundle)
    assert time.monotonic() - started < 5


def test_child_that_never_reads_stdin_cannot_block(tmp_path, monkeypatch):
    big = b'\0' * (8 * 1024 * 1024)  # far past any pipe buffer
    path = tmp_path / 'deaf'
    path.write_text(f'#!{sys.executable}\nimport sys; sys.stdout.write("reject: invalid\\n"); sys.exit(1)\n')
    path.chmod(0o755)
    use(monkeypatch, path)
    with pytest.raises(ValueError, match='Invalid workspace archive'):
        wr.verify(big)


def test_logs_carry_no_archive_names(tmp_path, monkeypatch, caplog):
    backend = ManagedCorpusBackend(tmp_path / 'secret-src', B)
    backend.activate({'Private-Name-XYZ.md': b'private synthetic'}, {}, [])
    body = archive(backend, '', {})
    for script in ('sys.exit(3)\n', 'sys.stdout.write("accept " + "0" * 64 + "\\n"); sys.exit(0)\n'):
        wr._logged.clear()
        path, _ = fake(tmp_path, script)
        use(monkeypatch, path)
        with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
            wr.verify(body)
    assert 'Private-Name-XYZ' not in caplog.text


def test_import_and_restore_callers_fail_closed_under_rust(bundle, tmp_path, monkeypatch):
    use(monkeypatch, tmp_path / 'absent')
    with pytest.raises(ValueError, match='unavailable'):
        wr.preview(bundle)
    target = tmp_path / 'restored'
    with pytest.raises(ValueError, match='unavailable'):
        wr.restore(bundle, target, hashlib.sha256(bundle).hexdigest())
    assert not target.exists()


def test_verify_bin_override_needs_an_absolute_executable_regular_file(tmp_path, monkeypatch, caplog):
    # #1813: a bare name is never resolved through PATH; a relative path, a directory, a non-executable
    # file and a missing path are all ignored (default used) with ONE warning.
    path, _ = fake(tmp_path, 'print("accept " + "0" * 64)\n')
    bindir = tmp_path / 'bindir'
    bindir.mkdir()
    monkeypatch.setenv('PATH', f'{tmp_path}:{__import__("os").environ["PATH"]}')
    seen = []
    monkeypatch.setattr(wr.os.path, 'isfile', lambda p, _f=wr.os.path.isfile: seen.append(p) or (False if p == wr.DEFAULT_BIN else _f(p)))
    plain = tmp_path / 'plain'
    plain.write_text('x')
    with caplog.at_level(logging.WARNING):
        for bad in (path.name, f'./{path.name}', str(bindir), str(plain), str(tmp_path / 'absent')):
            monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_BIN', bad)
            assert wr._rust_binary() is None, bad  # default is absent here, so nothing is honoured
    assert sum('ignoring it' in r.getMessage() for r in caplog.records) == 1
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_BIN', str(path))
    assert wr._rust_binary() == str(path)


def test_bytes_body_reaches_the_child_without_a_copy(monkeypatch, tmp_path):
    body = b'synthetic archive bytes'
    written = []

    class Proc:
        class stdin:
            write = staticmethod(written.append)
            close = staticmethod(lambda: None)

        class stdout:
            read1 = staticmethod(lambda n: b'')
            close = staticmethod(lambda: None)

        def wait(self, timeout=None):
            return 3

    path, _ = fake(tmp_path, '')
    use(monkeypatch, path)
    monkeypatch.setattr(wr.subprocess, 'Popen', lambda *a, **k: Proc())
    with pytest.raises(ValueError, match='unavailable'):
        wr._rust_verdict(body)
    assert written and written[0] is body
    written.clear()
    with pytest.raises(ValueError, match='unavailable'):
        wr._rust_verdict(bytearray(body))
    assert written[0] == body and type(written[0]) is bytes
