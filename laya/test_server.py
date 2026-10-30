import unittest
import io
import json
import os
import socket
import struct
import subprocess
import sys
import types
from server import FramedEnginePipe, keep_warm_from_env, touch_weights


def install_synthetic_engine():
    """Only subprocess tests call this; production worker() imports these stubs.

    Exercise the actual retained worker and CLI with no installed ML dependencies,
    real model, provisioning, network access or inference.
    """
    torch = types.ModuleType('torch')
    torch.set_num_threads = lambda value: None
    torch.set_num_interop_threads = lambda value: None
    laya = types.ModuleType('laya')

    class Tokenizer:
        def encode(self, value):
            return list(value)

    class Agent:
        cfg = {'max_len': 512, 'head_max_len': 256}
        tok = Tokenizer()

        def predict(self, state, questions):
            if state == 'fail':
                raise RuntimeError('synthetic private exception contents')
            choice = next(iter(questions['decision']['criteria']))
            scores = {key: int(key == choice) for key in questions['decision']['criteria']}
            return {'answers': {'decision': {'choice': choice, 'probabilities': scores}}}

    def load(path, device):
        assert (path, device) == ('/model', 'cpu')
        print('synthetic library stdout diagnostic', flush=True)
        os.write(1, b'synthetic native stdout diagnostic\n')
        return Agent()

    laya.load = load
    sys.modules.update(torch=torch, laya=laya)


class FramedEngineTests(unittest.TestCase):
    def encoded(self, value):
        data = json.dumps(value).encode()
        return struct.pack('!I', len(data)) + data

    def test_exact_frame_read_and_write(self):
        reader = io.BytesIO(self.encoded(BODY) + self.encoded({'state': 'second'}))
        writer = io.BytesIO()
        pipe = FramedEnginePipe(reader, writer)
        self.assertEqual(pipe.recv(), BODY)
        self.assertEqual(pipe.recv(), {'state': 'second'})
        pipe.send({'ready': True})
        self.assertEqual(writer.getvalue(), self.encoded({'ready': True}))

    def test_short_reads_writes_and_inclusive_bound(self):
        class ShortReader(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 1))

        class ShortWriter(io.BytesIO):
            def write(self, data):
                return super().write(data[:1])

        value = 'x' * 16382  # JSON quotes make the exact 16KiB frame.
        writer = ShortWriter()
        FramedEnginePipe(io.BytesIO(), writer).send(value)
        self.assertEqual(FramedEnginePipe(ShortReader(writer.getvalue()), io.BytesIO()).recv(), value)

    def test_generated_unicode_frames_preserve_values(self):
        # Deterministic property over scalar boundaries and escaped controls.
        for codepoint in (0, 9, 10, 32, 127, 128, 0x7ff, 0x800, 0xd7ff,
                          0xe000, 0xffff, 0x10000, 0x1f680, 0x10ffff):
            for count in (0, 1, 7, 120):
                value = {'state': chr(codepoint) * count}
                writer = io.BytesIO()
                FramedEnginePipe(io.BytesIO(), writer).send(value)
                self.assertEqual(FramedEnginePipe(io.BytesIO(writer.getvalue()), io.BytesIO()).recv(), value)

    def test_frame_bounds_and_truncation(self):
        for size in (0, 16385, 0xffffffff):
            with self.assertRaises(ValueError):
                FramedEnginePipe(io.BytesIO(struct.pack('!I', size)), io.BytesIO()).recv()
        for data in (b'', b'\x00', struct.pack('!I', 3) + b'{}'):
            with self.assertRaises(EOFError):
                FramedEnginePipe(io.BytesIO(data), io.BytesIO()).recv()
        with self.assertRaises(ValueError):
            FramedEnginePipe(io.BytesIO(), io.BytesIO()).send({'x': 'a' * 16384})
        with self.assertRaises(ValueError):
            FramedEnginePipe(io.BytesIO(), io.BytesIO()).send({'x': float('nan')})

    def test_malformed_json_and_unicode_round_trip(self):
        for data in (b'{', b'\xff'):
            with self.assertRaises(ValueError):
                FramedEnginePipe(io.BytesIO(struct.pack('!I', len(data)) + data), io.BytesIO()).recv()
        value = {'state': '\U0001f680\u00e9'}
        writer = io.BytesIO()
        FramedEnginePipe(io.BytesIO(), writer).send(value)
        self.assertEqual(FramedEnginePipe(io.BytesIO(writer.getvalue()), io.BytesIO()).recv(), value)

    def test_poll_after_back_to_back_frames_and_eof(self):
        parent, child = socket.socketpair()
        try:
            with child.makefile('rb', buffering=0) as reader:
                pipe = FramedEnginePipe(reader, io.BytesIO())
                self.assertFalse(pipe.poll(0))
                parent.sendall(self.encoded({'a': 1}) + self.encoded({'b': 2}))
                self.assertTrue(pipe.poll(0.1))
                self.assertEqual(pipe.recv(), {'a': 1})
                self.assertTrue(pipe.poll(0))
                self.assertEqual(pipe.recv(), {'b': 2})
                self.assertFalse(pipe.poll(0))
                parent.shutdown(socket.SHUT_WR)
                self.assertTrue(pipe.poll(0.1))
                with self.assertRaises(EOFError):
                    pipe.recv()
        finally:
            parent.close()
            child.close()

    def start_engine(self):
        code = ('import runpy, sys; from test_server import install_synthetic_engine; '
                'install_synthetic_engine(); sys.argv = ["server.py", "--engine-worker"]; '
                'runpy.run_path("server.py", run_name="__main__")')
        process = subprocess.Popen([sys.executable, '-u', '-c', code],
                                   cwd=os.path.dirname(__file__), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env={**os.environ, 'LAYA_KEEP_WARM_S': '0'})
        self.addCleanup(self.stop_engine, process)
        return process, FramedEnginePipe(process.stdout, process.stdin)

    @staticmethod
    def stop_engine(process):
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=3)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()

    def test_actual_engine_cli_ready_budget_result_and_clean_shutdown(self):
        process, pipe = self.start_engine()
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv(), {'ready': True})
        # Split the real framed input at every byte; reads must not assume a
        # pipe write or an OS read corresponds to one complete frame.
        for byte in self.encoded(BODY):
            process.stdin.write(bytes([byte]))
            process.stdin.flush()
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv(), {'selected': 'continue', 'scores': {'continue': 1, 'verify': 0},
                                      'model': 'convaiinnovations/laya', 'calibrated': False})
        pipe.send({**BODY, 'state': 'x' * 241})
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv(), {'error': 'Input exceeds model context budget'})
        pipe.send({**BODY, 'question': 'x' * 225})
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv(), {'error': 'Input exceeds model context budget'})
        pipe.send(BODY)
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv()['selected'], 'continue')
        process.stdin.close()
        process.wait(timeout=3)
        self.assertEqual(process.returncode, 0)
        diagnostics = process.stderr.read().decode()
        self.assertIn('synthetic library stdout diagnostic', diagnostics)
        self.assertIn('synthetic native stdout diagnostic', diagnostics)

    def test_actual_engine_exception_is_sanitized_and_exits(self):
        process, pipe = self.start_engine()
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv(), {'ready': True})
        pipe.send({**BODY, 'state': 'fail'})
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv(), {'error': 'Decision worker failed'})
        process.wait(timeout=3)
        self.assertNotIn(b'synthetic private exception contents', process.stderr.read())

    def test_actual_engine_oversized_frame_fails_before_reading_payload(self):
        process, pipe = self.start_engine()
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv(), {'ready': True})
        process.stdin.write(struct.pack('!I', 0xffffffff))
        process.stdin.flush()
        self.assertTrue(pipe.poll(3))
        self.assertEqual(pipe.recv(), {'error': 'Decision worker failed'})
        process.wait(timeout=3)

BODY = {'state': 'ok', 'question': 'Next?', 'options': [
    {'id': 'continue', 'label': 'Continue'}, {'id': 'verify', 'label': 'Check'}]}

class KeepWarmTests(unittest.TestCase):
    def test_env_parsing(self):
        logs = []
        self.assertEqual(keep_warm_from_env({}, logs.append), 30.0)
        self.assertEqual(keep_warm_from_env({'LAYA_KEEP_WARM_S': ' '}, logs.append), 30.0)
        self.assertEqual(keep_warm_from_env({'LAYA_KEEP_WARM_S': '0'}, logs.append), 0.0)
        self.assertEqual(keep_warm_from_env({'LAYA_KEEP_WARM_S': '3600'}, logs.append), 3600.0)
        self.assertEqual(logs, [])
        for raw in ['-1', 'x', 'nan', 'inf', '3601']:
            self.assertEqual(keep_warm_from_env({'LAYA_KEEP_WARM_S': raw}, logs.append), 30.0)
        self.assertEqual(len(logs), 5)
        self.assertIn('LAYA_KEEP_WARM_S', logs[0])

    def test_touch_weights_reads_every_module_tensor_only(self):
        touched = []

        class Tensor:
            def __init__(self, name): self.name = name
            def detach(self): return self
            def sum(self): touched.append(self.name)

        class Module:
            def __init__(self, params, buffers): self.p, self.b = params, buffers
            def parameters(self): return iter(self.p)
            def buffers(self): return iter(self.b)

        class NoGrad:
            def __enter__(self): return self
            def __exit__(self, *a): return False

        class Nn: pass
        Nn.Module = Module

        class Torch:
            nn = Nn
            no_grad = NoGrad

        class Agent: pass
        agent = Agent()
        agent.model = Module([Tensor('w1'), Tensor('w2')], [Tensor('b')])
        agent.tok = object()
        agent.cfg = {'max_len': 1}
        self.assertEqual(touch_weights(agent, Torch), 3)
        self.assertEqual(touched, ['w1', 'w2', 'b'])
        touched.clear()
        self.assertEqual(touch_weights(agent, Torch, stop=lambda: len(touched) >= 1), 1)
        self.assertEqual(touched, ['w1'])
        self.assertEqual(touch_weights(Agent(), Torch), 0)


class CliAuthorityTests(unittest.TestCase):
    def test_plain_and_unknown_modes_refuse_before_engine_import(self):
        for args in ((), ('--unknown',), ('--engine-worker', 'extra')):
            result = subprocess.run(
                [sys.executable, 'server.py', *args],
                cwd=os.path.dirname(__file__),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Laya Python HTTP is retired', result.stderr)
            self.assertEqual(result.stdout, b'')


if __name__ == '__main__':
    unittest.main()
