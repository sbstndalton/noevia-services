import unittest
import http.client
import json
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
import io
import struct
import subprocess
import sys
import types
from server import Busy, FramedEnginePipe, Late, Pool, Runtime, decision_timeout_from_env, keep_warm_from_env, touch_weights, max_concurrency_from_env, queue_timeout_from_env, serve, validate


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


def fake_worker(pipe):
    if os.environ.get('LAYA_FAKE_STARTUP_FAIL') == '1':
        return
    pipe.send({'ready': True})
    while True:
        body = pipe.recv()
        if body['state'] == 'hang':
            time.sleep(5)
        elif body['state'] == 'slow':
            # A late answer that differs from the normal one, so a leaked stale reply shows.
            time.sleep(0.3)
            pipe.send({'selected': 'verify', 'scores': {'continue': 0, 'verify': 1},
                       'model': 'fake', 'calibrated': False})
        elif body['state'] == 'slower':
            # #1074: long enough that three requests while it is still running cannot race its answer.
            time.sleep(1.0)
            pipe.send({'selected': 'verify', 'scores': {'continue': 0, 'verify': 1},
                       'model': 'fake', 'calibrated': False})
        elif body['state'] == 'crash':
            os._exit(2)
        elif body['state'] == 'error':
            pipe.send({'error': 'Decision worker failed'})
            return
        else:
            pipe.send({'selected': 'continue', 'scores': {'continue': 1, 'verify': 0},
                       'model': 'fake', 'calibrated': False})


BODY = {'state': 'ok', 'question': 'Next?', 'options': [
    {'id': 'continue', 'label': 'Continue'}, {'id': 'verify', 'label': 'Check'}]}

class ValidationTests(unittest.TestCase):
    def test_valid(self):
        body = {'state':'synthetic','question':'Next?', 'options':[{'id':'continue','label':'Continue'}, {'id':'verify','label':'Check'}]}
        self.assertEqual(validate(body), body)
    def test_invalid(self):
        for body in [None, {}, {'state':'x','question':'?', 'options':[]}, {'state':'x','question':'?', 'options':[{'id':'x','label':'X'}]*2}]:
            with self.assertRaises(ValueError): validate(body)


class TimeoutEnvTests(unittest.TestCase):
    def parse(self, **env):
        logs = []
        return decision_timeout_from_env(env, logs.append), logs

    def test_default_when_unset_or_blank(self):
        self.assertEqual(self.parse(), (1.3, []))
        self.assertEqual(self.parse(LAYA_DECISION_TIMEOUT_S='  '), (1.3, []))

    def test_valid_values_including_bounds(self):
        for raw, want in [('1.8', 1.8), (' 2 ', 2.0), ('0.5', 0.5), ('1.3', 1.3)]:
            self.assertEqual(self.parse(LAYA_DECISION_TIMEOUT_S=raw), (want, []))

    def test_out_of_range_falls_back_with_log(self):
        for raw in ['0.49', '2.01', '0', '-1', '30', 'inf', 'nan']:
            value, logs = self.parse(LAYA_DECISION_TIMEOUT_S=raw)
            self.assertEqual(value, 1.3, raw)
            self.assertEqual(len(logs), 1, raw)
            self.assertIn('LAYA_DECISION_TIMEOUT_S', logs[0])

    def test_garbage_falls_back_with_log(self):
        for raw in ['fast', '1.8s', '1,8', '0x10']:
            value, logs = self.parse(LAYA_DECISION_TIMEOUT_S=raw)
            self.assertEqual(value, 1.3, raw)
            self.assertEqual(len(logs), 1, raw)


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


class WaitingPipe:
    """Test double for the parent end of the worker pipe. poll(t > 0) waits as a real pipe would:
    it reports an answer after `arrives_after` seconds (or never, when None), and sends are recorded
    instead of reaching a worker."""

    def __init__(self, real, arrives_after):
        # arrives_after: seconds into a wait, or 'at_deadline' (the answer lands as the wait ends).
        self.real, self.arrives_after, self.sent = real, arrives_after, []

    def poll(self, timeout=0):
        if timeout and timeout > 0:
            if self.arrives_after == 'at_deadline':
                time.sleep(timeout)
                return True
            if self.arrives_after is not None and self.arrives_after <= timeout:
                time.sleep(self.arrives_after)
                return True
            time.sleep(timeout)
        return False

    def recv(self):
        return {'selected': 'verify', 'scores': {'continue': 0, 'verify': 1}, 'model': 'fake', 'calibrated': False}

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.real.close()


class LateAnswerTests(unittest.TestCase):
    """noevia#1070: a missed deadline keeps the worker and drops its stale answer."""

    def setUp(self):
        self.fatal = []
        self.runtime = Runtime(ctx=mp.get_context('spawn'), worker_target=fake_worker,
                               startup_timeout=2, decision_timeout=0.12,
                               retry_delays=(0, 0.02, 0.04), fatal=self.fatal.append, late_grace=1.0)

    def tearDown(self):
        self.runtime.stop()

    def test_late_answer_is_dropped_and_worker_kept(self):
        first = self.runtime.process.pid
        with self.assertRaises(Late):
            self.runtime.decide({**BODY, 'state': 'slow'})
        self.assertTrue(self.runtime.ready())
        time.sleep(0.4)  # the stale 'verify' answer is now waiting in the pipe
        self.assertEqual(self.runtime.decide(BODY)['selected'], 'continue')
        self.assertEqual(self.runtime.late, 0)
        self.assertEqual(first, self.runtime.process.pid)
        self.assertEqual(self.fatal, [])

    def test_request_behind_a_late_one_waits_within_its_deadline(self):
        self.runtime.decision_timeout = 0.25
        with self.assertRaises(Late):
            self.runtime.decide({**BODY, 'state': 'slow'})  # 0.25 s passed; 0.05 s left
        self.assertEqual(self.runtime.decide(BODY)['selected'], 'continue')
        self.assertEqual(self.runtime.late, 0)

    def test_hung_worker_is_replaced_after_grace(self):
        first = self.runtime.process.pid
        with self.assertRaises(Late):
            self.runtime.decide({**BODY, 'state': 'hang'})
        time.sleep(1.1)
        with self.assertRaises(RuntimeError) as caught:
            self.runtime.decide(BODY)
        self.assertNotIsInstance(caught.exception, Late)
        end = time.monotonic() + 3
        while not self.runtime.ready() and time.monotonic() < end:
            time.sleep(0.01)
        self.assertNotEqual(first, self.runtime.process.pid)
        self.assertEqual(self.runtime.late, 0)
        self.assertEqual(self.runtime.decide(BODY)['selected'], 'continue')

    def test_no_send_once_the_budget_is_spent_waiting_for_an_abandoned_answer(self):
        # #1074: the abandoned answer arrives exactly at the deadline; nothing may be sent after it.
        real = self.runtime.pipe
        fake = WaitingPipe(real, arrives_after='at_deadline')
        self.runtime.pipe = fake
        self.runtime.decision_timeout = 0.2
        self.runtime.late, self.runtime.late_since = 1, time.monotonic()
        try:
            with self.assertRaises(Late):
                self.runtime.decide(BODY)
        finally:
            self.runtime.pipe = real
        self.assertEqual(fake.sent, [])

    def test_a_replaced_worker_does_not_get_a_late_mark(self):
        # #1074: the request's worker was replaced while it waited; the new worker owes no answer.
        real, process = self.runtime.pipe, self.runtime.process
        runtime = self.runtime

        class ReplacedWhileWaiting(WaitingPipe):
            def poll(self, timeout=0):
                if timeout and timeout > 0:
                    runtime.process = None      # the worker this request went to is gone
                return super().poll(timeout)

        fake = ReplacedWhileWaiting(real, arrives_after=None)
        runtime.pipe = fake
        try:
            with self.assertRaises(Late):
                runtime.decide(BODY)
        finally:
            runtime.pipe, runtime.process = real, process
        self.assertEqual(runtime.late, 0)

    def test_arrived_late_answer_after_grace_keeps_the_worker(self):
        first = self.runtime.process.pid
        with self.assertRaises(Late):
            self.runtime.decide({**BODY, 'state': 'slow'})
        time.sleep(1.1)  # past late_grace, but the stale answer arrived long ago
        self.assertEqual(self.runtime.decide(BODY)['selected'], 'continue')
        self.assertEqual(first, self.runtime.process.pid)
        self.assertEqual(self.runtime.late, 0)

    def test_nothing_is_sent_behind_unfinished_abandoned_work(self):
        first = self.runtime.process.pid
        with self.assertRaises(Late):
            self.runtime.decide({**BODY, 'state': 'slower'})
        for _ in range(2):  # still running: 503 without queueing more work
            with self.assertRaises(Late):
                self.runtime.decide({**BODY, 'state': 'slower'})
            self.assertEqual(self.runtime.late, 1)
        time.sleep(1.3)  # the abandoned 1.0 s decision has finished by now (#1074)
        self.assertEqual(self.runtime.decide(BODY)['selected'], 'continue')
        self.assertEqual(first, self.runtime.process.pid)
        self.assertEqual(self.runtime.late, 0)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fatal = []
        self.runtime = Runtime(ctx=mp.get_context('spawn'), worker_target=fake_worker,
                               startup_timeout=2, decision_timeout=0.12,
                               retry_delays=(0, 0.02, 0.04), fatal=self.fatal.append, late_grace=0)

    def tearDown(self):
        self.runtime.stop()

    def wait_ready(self):
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if self.runtime.ready():
                return
            time.sleep(0.01)
        self.fail('replacement did not become ready')

    def test_timeout_recovers_without_replaying_request(self):
        first = self.runtime.process.pid
        with self.assertRaises(RuntimeError):
            self.runtime.decide({**BODY, 'state': 'hang'})
        self.assertFalse(self.runtime.ready())
        with self.assertRaises(RuntimeError):
            self.runtime.decide(BODY)
        self.wait_ready()
        self.assertNotEqual(first, self.runtime.process.pid)
        self.assertEqual(self.runtime.decide(BODY)['selected'], 'continue')
        self.assertEqual(self.fatal, [])

    def test_crash_and_worker_error_recover(self):
        for state in ('crash', 'error'):
            first = self.runtime.process.pid
            with self.assertRaises(RuntimeError):
                self.runtime.decide({**BODY, 'state': state})
            self.wait_ready()
            self.assertNotEqual(first, self.runtime.process.pid)
            self.assertEqual(self.runtime.decide(BODY)['selected'], 'continue')

    def test_idle_worker_death_is_detected_by_health(self):
        first = self.runtime.process
        first.kill()
        first.join(2)
        self.assertFalse(self.runtime.ready())
        self.wait_ready()
        self.assertNotEqual(first.pid, self.runtime.process.pid)
        self.assertEqual(self.runtime.decide(BODY)['selected'], 'continue')

    def test_health_and_client_disconnect(self):
        published = queue.Queue()
        thread = threading.Thread(target=serve, args=(self.runtime, ('127.0.0.1', 0), published.put), daemon=True)
        thread.start()
        server = published.get(timeout=2)
        port = server.server_address[1]
        try:
            def request(method, path, body=None):
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                conn.request(method, path, body=json.dumps(body) if body else None,
                             headers={'Content-Type': 'application/json'})
                response = conn.getresponse()
                status, payload = response.status, json.loads(response.read())
                conn.close()
                return status, payload

            self.assertEqual(request('GET', '/health'), (200, {'ready': True}))
            sock = socket.create_connection(('127.0.0.1', port), timeout=2)
            data = json.dumps(BODY).encode()
            sock.sendall(b'POST /v1/decisions HTTP/1.1\r\nHost: localhost\r\nContent-Length: ' +
                         str(len(data)).encode() + b'\r\n\r\n' + data)
            sock.close()
            # A disconnected client must not make a healthy worker unavailable.
            self.assertEqual(request('POST', '/v1/decisions', BODY)[0], 200)
            self.assertEqual(request('POST', '/v1/decisions', {**BODY, 'state': 'hang'})[0], 503)
            self.assertEqual(request('GET', '/health'), (503, {'ready': False}))
            self.wait_ready()
            self.assertEqual(request('GET', '/health'), (200, {'ready': True}))
        finally:
            server.shutdown()
            thread.join(2)

    def test_shutdown_kills_worker(self):
        process = self.runtime.process
        self.runtime.stop()
        self.assertFalse(process.is_alive())
        self.assertFalse(self.runtime.ready())

    def test_failed_replacement_stops_after_bounded_attempts(self):
        os.environ['LAYA_FAKE_STARTUP_FAIL'] = '1'
        try:
            with self.assertRaises(RuntimeError):
                self.runtime.decide({**BODY, 'state': 'hang'})
            end = time.monotonic() + 3
            while not self.fatal and time.monotonic() < end:
                time.sleep(0.01)
            self.assertEqual(self.fatal, [1])
            self.assertFalse(self.runtime.ready())
        finally:
            os.environ.pop('LAYA_FAKE_STARTUP_FAIL', None)

    def test_stop_during_recovery_does_not_start_another_worker(self):
        os.environ['LAYA_FAKE_STARTUP_FAIL'] = '1'
        try:
            with self.assertRaises(RuntimeError):
                self.runtime.decide({**BODY, 'state': 'hang'})
            self.runtime.stop()
            self.assertFalse(self.runtime.ready())
            self.assertEqual(self.fatal, [])
        finally:
            os.environ.pop('LAYA_FAKE_STARTUP_FAIL', None)


DELAY = 0.4


def delay_worker(pipe):
    """Fake model: state 'sleep:<n>' answers after n seconds, echoing its own state back."""
    pipe.send({'ready': True})
    while True:
        body = pipe.recv()
        time.sleep(float(body['state'].split(':')[1].split()[0]) if body['state'].startswith('sleep:') else 0)
        pipe.send({'selected': body['state'], 'scores': {'continue': 1, 'verify': 0},
                   'model': 'fake', 'calibrated': False})


class ConcurrencyEnvTests(unittest.TestCase):
    def parse(self, **env):
        logs = []
        return max_concurrency_from_env(env, logs.append), logs

    def test_default_one_and_valid(self):
        self.assertEqual(self.parse(), (1, []))
        self.assertEqual(self.parse(LAYA_MAX_CONCURRENCY=' '), (1, []))
        self.assertEqual(self.parse(LAYA_MAX_CONCURRENCY='3'), (3, []))

    def test_garbage_falls_back_with_log(self):
        for raw in ['0', '-1', '5', 'two', '1.5', '']:
            value, logs = self.parse(LAYA_MAX_CONCURRENCY=raw)
            self.assertEqual(value, 1, raw)
            self.assertEqual(len(logs), 0 if raw == '' else 1, raw)


class QueueTimeoutEnvTests(unittest.TestCase):
    def test_parse(self):
        logs = []
        self.assertEqual(queue_timeout_from_env({}, logs.append), 2.0)
        self.assertEqual(queue_timeout_from_env({'LAYA_QUEUE_TIMEOUT_MS': '500'}, logs.append), 0.5)
        self.assertEqual(logs, [])
        for raw in ['-1', 'x', '10001', '1.5']:
            self.assertEqual(queue_timeout_from_env({'LAYA_QUEUE_TIMEOUT_MS': raw}, logs.append), 2.0)
        self.assertEqual(len(logs), 4)


class PoolHttpTests(unittest.TestCase):
    def start(self, size, queue_timeout=5):
        self.pool = Pool.start(size, queue_timeout, ctx=mp.get_context('spawn'), worker_target=delay_worker,
                               startup_timeout=10, decision_timeout=2.0)
        published = queue.Queue()
        self.thread = threading.Thread(target=serve, args=(self.pool, ('127.0.0.1', 0), published.put), daemon=True)
        self.thread.start()
        self.server = published.get(timeout=2)
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(2)
        self.pool.stop()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        conn.request(method, path, body=json.dumps(body) if body else None,
                     headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        out = (response.status, json.loads(response.read()), response.getheader('Retry-After'))
        conn.close()
        return out

    def burst(self, n, state=lambda i: f'sleep:{DELAY}'):
        results = [None] * n

        def run(i):
            results[i] = self.request('POST', '/v1/decisions', {**BODY, 'state': state(i)})
        threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results, time.monotonic() - start

    def test_k1_overlapping_requests_queue_instead_of_503(self):
        self.start(1, queue_timeout=2.0)
        results, took = self.burst(2, lambda i: 'sleep:0.5')
        self.assertEqual([r[0] for r in results], [200, 200])
        self.assertGreaterEqual(took, 1.0)

    def test_default_one_serialises(self):
        self.start(1)
        results, took = self.burst(3)
        self.assertEqual([r[0] for r in results], [200] * 3)
        self.assertGreaterEqual(took, 3 * DELAY)

    def test_concurrency_k_finishes_in_ceil_n_over_k_delays(self):
        self.start(2)
        results, took = self.burst(4)
        self.assertEqual([r[0] for r in results], [200] * 4)
        self.assertGreaterEqual(took, 2 * DELAY)
        self.assertLess(took, 3.5 * DELAY)  # serial would be 4 * DELAY

    def test_no_cross_request_state_leakage(self):
        self.start(3)
        results, _ = self.burst(9, lambda i: f'sleep:0.05 req{i}')
        self.assertEqual([r[1]['selected'] for r in results], [f'sleep:0.05 req{i}' for i in range(9)])

    def test_health_answers_while_inference_busy(self):
        self.start(1)
        busy = threading.Thread(target=self.burst, args=(1,))
        busy.start()
        time.sleep(0.1)
        start = time.monotonic()
        self.assertEqual(self.request('GET', '/health')[:2], (200, {'ready': True}))
        self.assertLess(time.monotonic() - start, DELAY * 0.9)
        busy.join()

    def test_over_capacity_gets_fast_503_with_retry_after(self):
        self.start(1, queue_timeout=0.1)
        out = {}
        busy = threading.Thread(target=lambda: out.update(first=self.request('POST', '/v1/decisions', {**BODY, 'state': 'sleep:0.6'})))
        busy.start()
        time.sleep(0.15)
        start = time.monotonic()
        status, payload, retry = self.request('POST', '/v1/decisions', BODY)
        self.assertEqual((status, retry), (503, '1'))
        self.assertLess(time.monotonic() - start, 0.5)
        busy.join()
        self.assertEqual(out['first'][0], 200)
        self.assertEqual(self.request('POST', '/v1/decisions', BODY)[0], 200)  # recovers

    def test_pool_raises_busy(self):
        self.start(1, queue_timeout=0.05)
        t = threading.Thread(target=self.burst, args=(1, lambda i: 'sleep:0.4'))
        t.start()
        time.sleep(0.1)
        with self.assertRaises(Busy):
            self.pool.decide(BODY)
        t.join()

if __name__ == '__main__': unittest.main()
