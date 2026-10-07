import unittest
import http.client
import json
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
from server import Busy, Pool, Runtime, decision_timeout_from_env, max_concurrency_from_env, queue_timeout_from_env, serve, validate


def fake_worker(pipe):
    if os.environ.get('LAYA_FAKE_STARTUP_FAIL') == '1':
        return
    pipe.send({'ready': True})
    while True:
        body = pipe.recv()
        if body['state'] == 'hang':
            time.sleep(5)
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


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fatal = []
        self.runtime = Runtime(ctx=mp.get_context('spawn'), worker_target=fake_worker,
                               startup_timeout=2, decision_timeout=0.12,
                               retry_delays=(0, 0.02, 0.04), fatal=self.fatal.append)

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
