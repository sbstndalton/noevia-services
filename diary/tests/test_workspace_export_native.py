"""Synthetic child-process contract tests; no runtime or customer corpus."""
import hashlib
import json
import struct
import sys
import time

import pytest

from agent import workspace_export_native as native
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive


def frame(content):
    return (native.OUTPUT_MAGIC + struct.pack('>Q', len(content)) + hashlib.sha256(content).digest()
            + content + native.OUTPUT_END)


@pytest.fixture
def child(tmp_path, monkeypatch):
    processes = []
    real_popen = native.subprocess.Popen
    def track(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(native.subprocess, 'Popen', track)
    def install(body):
        path = tmp_path / 'export-child'
        path.write_text('#!' + sys.executable + '\n' + body)
        path.chmod(0o700)
        monkeypatch.setattr(native, 'BINARY', str(path))
        return path
    yield install
    for process in processes:
        assert process.poll() is not None, 'child must be reaped'
        assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))


def test_stream_protocol_and_minimal_environment(child, monkeypatch, tmp_path):
    capture = tmp_path / 'captured'
    monkeypatch.setenv('SECRET_FOR_TEST', 'do-not-inherit')
    child(f"""import os, sys
assert 'SECRET_FOR_TEST' not in os.environ
assert sys.argv[1:] == ['export']
open({str(capture)!r}, 'wb').write(sys.stdin.buffer.read())
sys.stdout.buffer.write({frame(b'synthetic-output')!r})
""")
    result = native.archive({'unicode': 'é'}, ['empty'], [{'path': 'x', 'bytes': 3}], [b'abc'])
    assert result == b'synthetic-output'
    wire = capture.read_bytes()
    assert wire.startswith(native.MAGIC)
    size = struct.unpack('>I', wire[len(native.MAGIC):len(native.MAGIC) + 4])[0]
    start = len(native.MAGIC) + 4
    assert json.loads(wire[start:start + size]) == {
        'settings': {'unicode': 'é'}, 'directories': ['empty'], 'files': [{'path': 'x', 'bytes': 3}]}
    assert wire[start + size:] == b'abc'


@pytest.mark.parametrize('body', [
    "import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(b'partial'); sys.exit(1)",
    "import sys; sys.stdin.buffer.read(); sys.stderr.write('reject: invalid-path\\n'); sys.exit(2)",
    "import sys; sys.stdin.buffer.read(); sys.stderr.write('private-child-detail'); sys.exit(1)",
])
def test_failure_or_truncated_output_never_returns_archive(child, body):
    child(body)
    with pytest.raises(ValueError, match='^' + native.ERROR.replace('.', r'\.') + '$'):
        native.archive({}, [], [], [])


@pytest.mark.parametrize('pipe', ['stdout', 'stderr'])
def test_output_overflow_is_bounded_and_reaped(child, monkeypatch, pipe):
    monkeypatch.setattr(native, 'MAX_OUTPUT', 100)
    child(f"import sys, time; sys.{pipe}.buffer.write(b'x' * 65536); sys.{pipe}.flush(); time.sleep(30)")
    started = time.monotonic()
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.archive({}, [], [], [])
    assert time.monotonic() - started < 5


@pytest.mark.parametrize('nonreading', [False, True])
def test_timeout_including_nonreading_input_reaps_child(child, monkeypatch, nonreading):
    monkeypatch.setattr(native, 'TIMEOUT', .15)
    child('import time; time.sleep(30)')
    started = time.monotonic()
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.archive({}, [], [{'path': 'large', 'bytes': 2 * 1024 * 1024}] if nonreading else [],
                       [b'a' * (2 * 1024 * 1024)] if nonreading else [])
    assert time.monotonic() - started < 5


def test_simultaneous_output_and_input_does_not_deadlock(child):
    child(f"import sys; sys.stdout.buffer.write({frame(b'z' * 200000)!r}); sys.stdout.flush(); sys.stdin.buffer.read()")
    assert len(native.archive({}, [], [], [b'x' * 200000])) == 200000


def test_header_bound_before_spawn(child, monkeypatch):
    child('raise AssertionError("must not start")')
    monkeypatch.setattr(native, 'MAX_HEADER', 10)
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.archive({}, [], [], [])


def test_missing_binary_and_invalid_setting_fail_closed(monkeypatch):
    monkeypatch.setattr(native, 'BINARY', '/definitely/not/an/exporter')
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.archive({}, [], [], [])
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'typo')
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.selected()


def test_managed_snapshot_calls_native_with_ordered_metadata(tmp_path, monkeypatch):
    backend = ManagedCorpusBackend(tmp_path / 'managed', '12345678-1234-1234-1234-123456789abc')
    backend.put('b', b'bbb')
    backend.put('a', b'aa')
    with backend.db() as db:
        db.execute('PRAGMA journal_mode=WAL')
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'rust')
    def fake(settings, directories, files, bodies):
        # Verify the metadata/body order within the caller's read transaction.
        backend.put('a', b'new', if_none_match=None)
        assert settings == {'key': 'value'}
        assert files == [{'path': 'a', 'bytes': 2}, {'path': 'b', 'bytes': 3}]
        assert list(bodies) == [b'aa', b'bbb']
        return b'archive'
    monkeypatch.setattr(native, 'archive', fake)
    assert archive(backend, '', {'key': 'value'}) == b'archive'


def test_python_default_does_not_start_native(tmp_path, monkeypatch):
    monkeypatch.delenv('WORKSPACE_EXPORT_IMPL', raising=False)
    backend = ManagedCorpusBackend(tmp_path / 'managed', '12345678-1234-1234-1234-123456789abc')
    monkeypatch.setattr(native, 'archive', lambda *a: pytest.fail('native selection was not enabled'))
    assert archive(backend, '', {}).startswith(b'PK')


@pytest.mark.parametrize('corrupt', [
    lambda data: data[:-1],
    lambda data: data + b'extra',
    lambda data: b'bad' + data[3:],
    lambda data: data[:len(native.OUTPUT_MAGIC)] + struct.pack('>Q', native.MAX_OUTPUT + 1) + data[len(native.OUTPUT_MAGIC) + 8:],
    lambda data: data[:len(native.OUTPUT_MAGIC) + 8] + b'0' * 32 + data[len(native.OUTPUT_MAGIC) + 40:],
    lambda data: b'',
])
def test_exit_zero_corruption_or_truncation_rejected(child, corrupt):
    payload = corrupt(frame(b'archive'))
    child(f"import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write({payload!r})")
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.archive({}, [], [], [])


def test_success_frame_requires_exit_zero(child):
    child(f"import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write({frame(b'archive')!r}); sys.exit(1)")
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.archive({}, [], [], [])


def test_success_frame_requires_empty_stderr(child, caplog):
    child(f"import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write({frame(b'archive')!r}); sys.stderr.write('private-child-detail')")
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.archive({}, [], [], [])
    assert 'private-child-detail' not in caplog.text


def test_success_frame_requires_eof_and_exit(child, monkeypatch):
    monkeypatch.setattr(native, 'TIMEOUT', .15)
    child(f"import sys, time; sys.stdin.buffer.read(); sys.stdout.buffer.write({frame(b'archive')!r}); sys.stdout.flush(); time.sleep(30)")
    with pytest.raises(ValueError, match='Workspace export could not'):
        native.archive({}, [], [], [])


def test_legacy_export_ignores_native_selection(tmp_path, monkeypatch):
    from agent.local_storage import LocalCorpusBackend
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'rust')
    backend = LocalCorpusBackend(str(tmp_path / 'legacy'))
    backend.put('entry.md', b'synthetic legacy source')
    monkeypatch.setattr(native, 'archive', lambda *a: pytest.fail('legacy must stay Python'))
    assert archive(backend, '', {}).startswith(b'PK')


def test_native_failure_does_not_fall_back_to_python(tmp_path, monkeypatch):
    backend = ManagedCorpusBackend(tmp_path / 'managed', '12345678-1234-1234-1234-123456789abc')
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'rust')
    def fail(*args):
        raise ValueError(native.ERROR)
    monkeypatch.setattr(native, 'archive', fail)
    with pytest.raises(ValueError, match='Workspace export could not'):
        archive(backend, '', {})


def test_caller_cancellation_reaps_child(child):
    class Cancelled(BaseException):
        pass
    child('import sys; sys.stdin.buffer.read()')
    def bodies():
        raise Cancelled()
        yield b''
    with pytest.raises(Cancelled):
        native.archive({}, [], [], bodies())


def test_real_rust_export_and_verifier_interoperate(tmp_path, monkeypatch):
    """Explicit synthetic integration opt-in; executable paths are test-only."""
    import os
    import subprocess
    from tests.test_workspace_export import unpack
    binary = os.environ.get('WORKSPACE_EXPORT_TEST_BINARY')
    verifier = os.environ.get('WORKSPACE_EXPORT_TEST_VERIFIER')
    if not binary or not verifier:
        pytest.skip('set both synthetic Rust integration executable paths')
    monkeypatch.setattr(native, 'BINARY', binary)
    backend = ManagedCorpusBackend(tmp_path / 'managed', '12345678-1234-1234-1234-123456789abc')
    files = {'é/note.md': b'# Synthetic\r\n', 'assets/bin.dat': b'\x00\xff\x89PNG',
             'manifest.json': b'user file named manifest'}
    backend.activate(files, {}, ['empty/nested'])
    settings = {'title': 'Synthetic é', 'nested': {'enabled': True, 'number': 2.5}}
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'python')
    reference = archive(backend, '', settings)
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'rust')
    actual = archive(backend, '', settings)
    with unpack(reference)[0] as before, unpack(actual)[0] as after:
        assert set(before.namelist()) == set(after.namelist())
        for path in before.namelist():
            if path != 'manifest.json':
                assert before.read(path) == after.read(path)
        assert json.loads(before.read('manifest.json')) == json.loads(after.read('manifest.json'))
    result = subprocess.run([verifier, 'verify'], input=actual, capture_output=True, timeout=15,
                            env={'LANG': 'C.UTF-8'}, cwd='/')
    assert result.returncode == 0
    assert result.stdout.startswith(b'accept ') and len(result.stdout) == 72
    empty = ManagedCorpusBackend(tmp_path / 'empty', '12345678-1234-1234-1234-123456789abc')
    empty.activate({}, {}, ['empty'])
    empty_zip = archive(empty, '', {})
    result = subprocess.run([verifier, 'verify'], input=empty_zip, capture_output=True, timeout=15,
                            env={'LANG': 'C.UTF-8'}, cwd='/')
    assert result.returncode == 0


@pytest.mark.parametrize('fails', [False, True])
def test_api_rust_selection_preserves_auth_and_failure_status(tmp_path, monkeypatch, child, fails):
    import threading
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import agent.app as appmod
    backend = ManagedCorpusBackend(tmp_path / 'managed', '12345678-1234-1234-1234-123456789abc')
    backend.activate({'note.md': b'synthetic'}, {})
    state = SimpleNamespace(backend=backend, managed=backend, cfg={},
                            store=SimpleNamespace(_write_lock=threading.RLock(), remote_root=''),
                            journal=SimpleNamespace(pending_count=lambda: 0))
    calls = []
    def resolve(request, *, recover=True):
        assert recover is False
        calls.append('resolved')
        return state
    monkeypatch.setattr(appmod, '_tenant_state', resolve)
    monkeypatch.setattr(appmod, 'corpus_settings', lambda cfg: {})
    monkeypatch.setattr(appmod, 'check_auth', lambda request: True)
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'rust')
    body = "import sys; sys.stdin.buffer.read(); sys.exit(1)" if fails else (
        f"import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write({frame(b'synthetic-archive')!r})")
    child(body)
    client = TestClient(appmod.app)
    reply = client.get('/api/workspace-export')
    assert calls == ['resolved']
    if fails:
        assert reply.status_code == 409
        assert reply.json()['detail'] == native.ERROR
        assert reply.headers['content-type'] == 'application/json'
    else:
        assert reply.status_code == 200
        assert reply.content == b'synthetic-archive'
        assert reply.headers['content-type'] == 'application/zip'
        assert reply.headers['cache-control'] == 'no-store'
    monkeypatch.setattr(appmod, 'check_auth', lambda request: False)
    assert client.get('/api/workspace-export').status_code == 401
    assert calls == ['resolved']
