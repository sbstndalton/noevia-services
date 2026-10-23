"""Synthetic decode-v1 adapter checks, including adversarial child processes."""
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import sys
import time

import pytest

from agent import workspace_decode as wd
from agent import workspace_restore as wr
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from tests.test_dedicated_storage import B

REAL_BIN = os.environ.get('WORKSPACE_RESTORE_VERIFY_BIN')


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    wr._logged.clear()
    monkeypatch.delenv('WORKSPACE_RESTORE_VERIFY_IMPL', raising=False)
    monkeypatch.delenv('WORKSPACE_RESTORE_VERIFY_BIN', raising=False)
    yield
    wr._logged.clear()


@pytest.fixture
def bundle(tmp_path):
    backend = ManagedCorpusBackend(tmp_path / 'source', B)
    backend.activate({'raw/note.md': b'# Synthetic note\r\n', 'unicodé/empty.bin': b''}, {}, ['empty'])
    return archive(backend, '', {'theme': 'synthetic'})


def frame(body, edit=None):
    files, dirs, manifest = wr._verify(body)  # Test reference, never production Rust path.
    metadata = {'version': 1, 'fingerprint': hashlib.sha256(body).hexdigest(),
                'listingSha256': wr.listing_sha256(files, dirs), 'bytes': sum(map(len, files.values())),
                'settings': manifest.get('settings', {}),
                'files': [{'path': p, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for p, data in sorted(files.items())],
                'directories': sorted(dirs)}
    if edit:
        edit(metadata)
    raw = json.dumps(metadata, ensure_ascii=False).encode()
    payload = b''.join(struct.pack('<Q', len(files[row['path']])) + files[row['path']]
                       for row in metadata['files'] if row['path'] in files)
    return wd.MAGIC + struct.pack('<I', len(raw)) + raw + payload + wd.END


def fake(tmp_path, script):
    path = tmp_path / 'fake-decode'
    path.write_text(f'#!{sys.executable}\n' + script)
    path.chmod(0o755)
    return path


def use(monkeypatch, path):
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_IMPL', 'rust')
    monkeypatch.setattr(wd, 'DEFAULT_BIN', str(path))


def emit(tmp_path, payload, suffix=''):
    return fake(tmp_path, 'import os, sys\n'
                'source = sys.stdin.buffer.read()\n'
                f'sys.stdout.buffer.write(bytes.fromhex({payload.hex()!r}))\n'
                f'{suffix}\n')


def test_python_default_and_explicit_rust_never_calls_python_parser(bundle, monkeypatch, tmp_path):
    assert wr.impl_choice() == 'python'
    assert wr.verify(bundle)[0]['raw/note.md'] == b'# Synthetic note\r\n'
    use(monkeypatch, emit(tmp_path, frame(bundle)))
    monkeypatch.setattr(wr, '_verify', lambda _: pytest.fail('Python ZIP parser called'))
    files, dirs, manifest = wr.verify(bundle)
    assert files['raw/note.md'] == b'# Synthetic note\r\n'
    assert files['unicodé/empty.bin'] == b''
    assert dirs == {'empty', 'raw', 'unicodé'}
    assert manifest['settings'] == {'theme': 'synthetic'}
    assert wr.preview(bundle)['fingerprint'] == hashlib.sha256(bundle).hexdigest()


@pytest.mark.parametrize('settings', [None, False, 17, 'str', ['a'], {'nested': True}])
def test_settings_json_value_is_preserved(tmp_path, monkeypatch, settings):
    backend = ManagedCorpusBackend(tmp_path / 'source', B)
    backend.activate({'a.md': b'hello'}, {}, [])
    body = archive(backend, '', settings)
    use(monkeypatch, emit(tmp_path, frame(body)))
    assert wr.preview(body)['settings'] == settings


@pytest.mark.parametrize('mutation', [
    lambda b: b'bad' + b[3:],
    lambda b: b[:len(wd.MAGIC)] + struct.pack('<I', wd.MAX_METADATA + 1) + b[len(wd.MAGIC) + 4:],
    lambda b: b[:-len(wd.END)],
    lambda b: b + b'x',
    lambda b: b[:-1] + b'X',
    lambda b: b[:len(wd.MAGIC) + 4] + b'\xff' + b[len(wd.MAGIC) + 5:],
])
def test_truncated_or_corrupt_frames_fail_closed(bundle, tmp_path, monkeypatch, mutation):
    use(monkeypatch, emit(tmp_path, mutation(frame(bundle))))
    with pytest.raises(wd.DecodeUnavailable):
        wr.verify(bundle)


@pytest.mark.parametrize('edit', [
    lambda m: m.update(version=2),
    lambda m: m.update(fingerprint='0' * 64),
    lambda m: m.update(listingSha256='0' * 64),
    lambda m: m.update(bytes=m['bytes'] + 1),
    lambda m: m.pop('settings'),
    lambda m: m.update(directories=list(reversed(m['directories']))),
    lambda m: m['files'][0].update(sha256='0' * 64),
    lambda m: m['files'][0].update(bytes=m['files'][0]['bytes'] + 1),
    lambda m: m['files'][0].update(path='../escape'),
    lambda m: m['files'].append(dict(m['files'][0])),
    lambda m: m.update(files=[]),
    lambda m: m.update(bytes=True),
])
def test_bad_metadata_refuses_before_preview(bundle, tmp_path, monkeypatch, edit):
    use(monkeypatch, emit(tmp_path, frame(bundle, edit)))
    with pytest.raises(wd.DecodeUnavailable):
        wr.preview(bundle)


def test_child_refusal_and_fault_statuses(bundle, tmp_path, monkeypatch):
    for script, expected in (
        ('import sys\nsys.stdout.write("reject: invalid\\n")\nsys.exit(1)\n', ValueError),
        ('import sys\nsys.stdout.write("reject: invalid\\n")\nsys.exit(0)\n', wd.DecodeUnavailable),
        ('import sys\nsys.stdout.write("reject: unknown\\n")\nsys.exit(1)\n', wd.DecodeUnavailable),
        ('import sys\nsys.exit(3)\n', wd.DecodeUnavailable),
        ('import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n', wd.DecodeUnavailable),
    ):
        use(monkeypatch, fake(tmp_path, script))
        with pytest.raises(expected):
            wr.verify(bundle)


def test_exit_nonzero_even_after_complete_frame(bundle, tmp_path, monkeypatch):
    use(monkeypatch, emit(tmp_path, frame(bundle), 'sys.exit(3)'))
    with pytest.raises(wd.DecodeUnavailable):
        wr.verify(bundle)


def test_missing_binary_is_unavailable(bundle, monkeypatch, tmp_path):
    use(monkeypatch, tmp_path / 'absent')
    with pytest.raises(wd.DecodeUnavailable):
        wr.verify(bundle)


def test_fixed_binary_and_minimal_child_environment(bundle, monkeypatch, tmp_path):
    record = tmp_path / 'record.json'
    payload = frame(bundle)
    script = ('import hashlib, json, os, sys\n'
              'data = sys.stdin.buffer.read()\n'
              f'open({str(record)!r}, "w").write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ), "sha": hashlib.sha256(data).hexdigest()}}))\n'
              f'sys.stdout.buffer.write(bytes.fromhex({payload.hex()!r}))\n')
    use(monkeypatch, fake(tmp_path, script))
    monkeypatch.setenv('SYNTHETIC_SECRET', 'must-not-pass')
    wr.verify(bundle)
    seen = json.loads(record.read_text())
    assert seen['argv'] == ['decode-v1']
    assert seen['sha'] == hashlib.sha256(bundle).hexdigest()
    assert set(seen['env']) - {'PATH', 'LC_CTYPE', '__CF_USER_TEXT_ENCODING'} == set()
    assert seen['env']['PATH'] == '/usr/local/bin:/usr/bin:/bin'


def test_timeout_and_endless_output_reap_owned_child(bundle, monkeypatch, tmp_path):
    use(monkeypatch, fake(tmp_path, 'import time\ntime.sleep(30)\n'))
    monkeypatch.setattr(wd, 'TIMEOUT_S', 0.4)
    started = time.monotonic()
    with pytest.raises(wd.DecodeUnavailable):
        wr.verify(bundle)
    assert time.monotonic() - started < 5
    use(monkeypatch, fake(tmp_path, 'import sys\nwhile True:\n sys.stdout.buffer.write(b"x" * 65536); sys.stdout.flush()\n'))
    monkeypatch.setattr(wd, 'MAX_FRAME', 1024)
    started = time.monotonic()
    with pytest.raises(wd.DecodeUnavailable):
        wr.verify(bundle)
    assert time.monotonic() - started < 5


def test_nonreading_child_and_logs_do_not_expose_source(bundle, monkeypatch, tmp_path, caplog):
    use(monkeypatch, fake(tmp_path, 'import sys\nsys.stdout.write("reject: invalid\\n")\nsys.exit(1)\n'))
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match='Invalid workspace archive'):
        wr.verify(bundle)
    assert 'note.md' not in caplog.text
    assert 'Synthetic note' not in caplog.text


@pytest.mark.skipif(not REAL_BIN, reason='paired Rust binary not supplied')
def test_real_decode_binary(bundle, monkeypatch):
    use(monkeypatch, Path(REAL_BIN))
    assert wr.verify(bundle)[0]['raw/note.md'] == b'# Synthetic note\r\n'


def test_protocol_fault_creates_no_destination_or_managed_receipt(bundle, tmp_path, monkeypatch):
    from agent.workspace_import import apply, prepare
    use(monkeypatch, emit(tmp_path, frame(bundle)[:-len(wd.END)]))
    destination = tmp_path / 'restored'
    with pytest.raises(wd.DecodeUnavailable):
        wr.restore(bundle, destination, hashlib.sha256(bundle).hexdigest())
    assert not destination.exists()
    backend = ManagedCorpusBackend(tmp_path / 'target', B)
    backend.activate({'keep.md': b'keep'}, {}, [])
    with backend.db() as db:
        before = (backend.meta(db, 'generation'),
                  db.execute('SELECT count(*) FROM files').fetchone()[0],
                  db.execute('SELECT count(*) FROM directories').fetchone()[0],
                  db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0],
                  db.execute('SELECT count(*) FROM workspace_index_outbox').fetchone()[0])
    with pytest.raises(wd.DecodeUnavailable):
        prepare(backend, bundle, 'Copy')
    with pytest.raises(wd.DecodeUnavailable):
        apply(backend, bundle, 'Copy', '0' * 64)
    with backend.db() as db:
        after = (backend.meta(db, 'generation'),
                 db.execute('SELECT count(*) FROM files').fetchone()[0],
                 db.execute('SELECT count(*) FROM directories').fetchone()[0],
                 db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0],
                 db.execute('SELECT count(*) FROM workspace_index_outbox').fetchone()[0])
    assert before == after
