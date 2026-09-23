import unittest
import http.client
import json
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
from server import Runtime, serve, validate


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

if __name__ == '__main__': unittest.main()
