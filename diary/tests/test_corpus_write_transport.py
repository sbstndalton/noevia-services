"""Synthetic subprocess boundaries; no storage or model calls."""
import json
import stat
import sys
import pytest
from agent import corpus_write_native as native


def frame(payload):
    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
    return b'DWR1' + len(payload).to_bytes(8, 'big') + payload + b'END1'


def reply():
    return dict(version=1, op='prepare_append', sub_header='', body='')


def binary(tmp_path, monkeypatch, script):
    p = tmp_path / 'synthetic-transform'
    p.write_text('#!' + sys.executable + '\n' + script + '\n')
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(native, 'BINARY', str(p))
    monkeypatch.setattr(native, 'TIMEOUT_S', .3)


def call(**changes):
    fields = dict(time_token='00:00', topic='', me='', assistant='', xid='x')
    fields.update(changes)
    return native.transform('prepare_append', **fields)


@pytest.mark.parametrize('kind', ['stderr', 'duplicate', 'early_stdin', 'utf16'])
def test_success_needs_complete_clean_transport(tmp_path, monkeypatch, kind):
    payload = json.dumps(reply()).encode()
    if kind == 'duplicate':
        payload = b'{"version":2,' + payload[1:]
    if kind == 'utf16':
        payload = payload.decode().encode('utf-16')
    script = 'import os,sys\n'
    script += 'os.close(0)\n' if kind == 'early_stdin' else 'sys.stdin.buffer.read()\n'
    if kind == 'stderr':
        script += 'sys.stderr.write("synthetic diagnostic")\n'
    binary(tmp_path, monkeypatch, script + 'sys.stdout.buffer.write(' + repr(frame(payload)) + ')')
    with pytest.raises(native.NativeWriteUnavailable):
        call(assistant='x' * 400000 if kind == 'early_stdin' else '')


@pytest.mark.parametrize('field', ['current_text', 'current_hash', 'edited_text', 'edited_hash'])
def test_absent_edit_refuses_partial_fields(tmp_path, monkeypatch, field):
    r = dict(version=1, op='inspect_edit', found=False, current_text=None,
             current_hash=None, edited_text=None, edited_hash=None)
    r[field] = 'unexpected'
    binary(tmp_path, monkeypatch, 'import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write(' + repr(frame(r)) + ')')
    with pytest.raises(native.NativeWriteUnavailable):
        native.transform('inspect_edit', current=None, xid='x', new_me='', new_assistant='')


@pytest.mark.parametrize('stage', ['selector', 'nonblocking', 'register', 'read', 'write', 'select', 'close', 'cancel'])
def test_fault_cleanup(tmp_path, monkeypatch, stage):
    binary(tmp_path, monkeypatch, 'import sys,time\nsys.stdout.write("x");sys.stdout.flush()\nsys.stdin.buffer.read()\ntime.sleep(30)')
    children = []
    real_spawn = native.subprocess.Popen
    def spawn(*args, **kwargs):
        child = real_spawn(*args, **kwargs)
        children.append(child)
        return child
    def refuse(*args, **kwargs):
        raise OSError('synthetic fault')
    monkeypatch.setattr(native.subprocess, 'Popen', spawn)
    if stage == 'selector':
        monkeypatch.setattr(native.selectors, 'DefaultSelector', refuse)
    elif stage == 'nonblocking':
        monkeypatch.setattr(native.os, 'set_blocking', refuse)
    elif stage in ('register', 'select', 'close'):
        sel = native.selectors.DefaultSelector()
        monkeypatch.setattr(native.selectors, 'DefaultSelector', lambda: sel)
        original = getattr(sel, stage)
        if stage == 'close':
            def close():
                original()
                refuse()
            monkeypatch.setattr(sel, stage, close)
        else:
            monkeypatch.setattr(sel, stage, refuse)
    elif stage == 'cancel':
        clock = native.time.monotonic
        def cancel():
            if children:
                raise KeyboardInterrupt('synthetic cancellation')
            return clock()
        monkeypatch.setattr(native.time, 'monotonic', cancel)
    else:
        original = getattr(native.os, stage)
        def fault(fd, *args):
            if children:
                stream = children[0].stdout if stage == 'read' else children[0].stdin
                if not stream.closed and fd == stream.fileno():
                    refuse()
            return original(fd, *args)
        monkeypatch.setattr(native.os, stage, fault)
    try:
        with pytest.raises(KeyboardInterrupt if stage == 'cancel' else native.NativeWriteUnavailable):
            call()
        assert len(children) == (0 if stage == 'selector' else 1)
        for child in children:
            assert child.poll() is not None, 'owned child still running'
            assert all(s.closed for s in (child.stdin, child.stdout, child.stderr))
    finally:
        if stage == 'cancel':
            monkeypatch.setattr(native.time, 'monotonic', clock)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
            for stream in (child.stdin, child.stdout, child.stderr):
                stream.close()


def test_transient_write_is_retried(tmp_path, monkeypatch):
    binary(tmp_path, monkeypatch, 'import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write(' + repr(frame(reply())) + ')')
    real = native.os.write
    refused = False
    def write(fd, data):
        nonlocal refused
        if not refused:
            refused = True
            raise BlockingIOError('synthetic not ready')
        return real(fd, data)
    monkeypatch.setattr(native.os, 'write', write)
    assert call() == reply()
    assert refused


@pytest.mark.parametrize('payload', [
    b'{"version":NaN,"op":"prepare_append","sub_header":"","body":""}',
    b'{"version":true,"op":"prepare_append","sub_header":"","body":""}',
    b'{"version":1,"op":"prepare_append","sub_header":"","body":"\\ud800"}',
    b'{"version":1,"op":"prepare_append","sub_header":"","body":"","extra":0}',
])
def test_malformed_success_json_refuses(tmp_path, monkeypatch, payload):
    binary(tmp_path, monkeypatch, 'import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write(' + repr(frame(payload)) + ')')
    with pytest.raises(native.NativeWriteUnavailable):
        call()


def test_present_edit_hash_shape_refuses(tmp_path, monkeypatch):
    r = dict(version=1, op='inspect_edit', found=True, current_text='text',
             current_hash='not-a-hash', edited_text='edited', edited_hash='0' * 64)
    binary(tmp_path, monkeypatch, 'import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write(' + repr(frame(r)) + ')')
    with pytest.raises(native.NativeWriteUnavailable):
        native.transform('inspect_edit', current='text', xid='x', new_me='', new_assistant='')


def test_absent_edit_exact_null_reply_is_valid(tmp_path, monkeypatch):
    r = dict(version=1, op='inspect_edit', found=False, current_text=None,
             current_hash=None, edited_text=None, edited_hash=None)
    binary(tmp_path, monkeypatch, 'import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write(' + repr(frame(r)) + ')')
    assert native.transform('inspect_edit', current=None, xid='x', new_me='', new_assistant='') == r


def test_inherited_pipe_has_bounded_deadline(tmp_path, monkeypatch):
    import time
    script = ('import subprocess,sys\nsys.stdin.buffer.read()\n'
              'subprocess.Popen([sys.executable,"-c","import time;time.sleep(2)"],stdin=subprocess.DEVNULL,close_fds=True)\n')
    binary(tmp_path, monkeypatch, script)
    monkeypatch.setattr(native, 'TIMEOUT_S', .2)
    start = time.monotonic()
    try:
        with pytest.raises(native.NativeWriteUnavailable):
            call()
        assert time.monotonic() - start < 1.5
    finally:
        time.sleep(max(0, 2.3 - (time.monotonic() - start)))
