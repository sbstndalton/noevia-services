"""Private CPU-only typed-decision endpoint. No generation, tools or request logs."""
import json
import multiprocessing as mp
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def validate(body):
    if not isinstance(body, dict) or set(body) != {'state', 'question', 'options'}:
        raise ValueError('Invalid request')
    if not isinstance(body['state'], str) or len(body['state']) > 4000:
        raise ValueError('State too large')
    if not isinstance(body['question'], str) or not 1 <= len(body['question']) <= 500:
        raise ValueError('Invalid question')
    options = body['options']
    if not isinstance(options, list) or not 2 <= len(options) <= 8:
        raise ValueError('Invalid options')
    for o in options:
        if not isinstance(o, dict) or set(o) != {'id', 'label'} or not isinstance(o['id'], str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,31}', o['id']) or not isinstance(o['label'], str) or not 1 <= len(o['label']) <= 120:
            raise ValueError('Invalid option')
    if len({o['id'] for o in options}) != len(options):
        raise ValueError('Duplicate options')
    return body


def worker(pipe):
    try:
        import torch
        import laya
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        agent = laya.load('/model', device='cpu')
        pipe.send({'ready': True})
        while True:
            body = pipe.recv()
            # Reject over-budget inputs rather than let the SDK silently truncate evidence.
            state_budget = agent.cfg['max_len'] - agent.cfg['head_max_len'] - 16
            question = {'type': 'choice', 'instructions': body['question'],
                        'criteria': {o['id']: o['label'] for o in body['options']}}
            if len(agent.tok.encode(body['state'])) > state_budget or len(agent.tok.encode(body['question'] + ' ' + ' '.join(o['id']+' '+o['label'] for o in body['options']))) > agent.cfg['head_max_len'] - 32:
                pipe.send({'error': 'Input exceeds model context budget'})
                continue
            result = agent.predict(body['state'], {'decision': question})['answers']['decision']
            pipe.send({'selected': result['choice'], 'scores': result['probabilities'], 'model': 'convaiinnovations/laya', 'calibrated': False})
    except Exception:
        # Never include request contents or library exceptions in logs/responses.
        try:
            pipe.send({'error': 'Decision worker failed'})
        except (BrokenPipeError, EOFError):
            pass


class Runtime:
    def __init__(self, ctx=None, worker_target=worker, startup_timeout=90,
                 decision_timeout=1.3, retry_delays=(0, 2, 8), fatal=os._exit):
        self.ctx = ctx or mp.get_context('spawn')
        self.worker_target = worker_target
        self.startup_timeout = startup_timeout
        self.decision_timeout = decision_timeout
        self.retry_delays = retry_delays
        self.fatal = fatal
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.process = None
        self.pipe = None
        self.starting = None
        self.recovery = None
        process, pipe = self._start_worker()
        self.process, self.pipe = process, pipe

    def _start_worker(self):
        parent, child = self.ctx.Pipe()
        process = self.ctx.Process(target=self.worker_target, args=(child,), daemon=True)
        try:
            process.start()
            child.close()
            with self.lock:
                self.starting = process
            if self.stopping.is_set() or not parent.poll(self.startup_timeout) or parent.recv() != {'ready': True}:
                raise RuntimeError('Laya startup failed or exceeded deadline')
            with self.lock:
                self.starting = None
            return process, parent
        except (EOFError, OSError, RuntimeError):
            self._stop_process(process, parent)
            raise RuntimeError('Laya startup failed or exceeded deadline') from None
        finally:
            child.close()
            with self.lock:
                if self.starting is process:
                    self.starting = None

    @staticmethod
    def _stop_process(process, pipe):
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
        pipe.close()

    def ready(self):
        with self.lock:
            process = self.process
            ready = not self.stopping.is_set() and process is not None and process.is_alive()
        if process is not None and not ready and not self.stopping.is_set():
            self._replace(process)
        return ready

    def _replace(self, process):
        with self.lock:
            if self.stopping.is_set() or self.process is not process:
                return
            pipe = self.pipe
            self.process = self.pipe = None
            self.recovery = threading.Thread(target=self._recover, args=(process, pipe), daemon=True)
            self.recovery.start()

    def _recover(self, old_process, old_pipe):
        self._stop_process(old_process, old_pipe)
        for delay in self.retry_delays:
            if self.stopping.wait(delay):
                return
            try:
                process, pipe = self._start_worker()
            except RuntimeError:
                continue
            with self.lock:
                if not self.stopping.is_set():
                    self.process, self.pipe = process, pipe
                    return
            self._stop_process(process, pipe)
            return
        if not self.stopping.is_set():
            # The parent must fail so Docker's bounded on-failure policy applies.
            self.fatal(1)

    def stop(self):
        self.stopping.set()
        with self.lock:
            process, pipe, starting, recovery = self.process, self.pipe, self.starting, self.recovery
            self.process = self.pipe = None
        if process is not None:
            self._stop_process(process, pipe)
        if starting is not None and starting is not process and starting.pid is not None:
            starting.terminate()
        if recovery is not None and recovery is not threading.current_thread():
            recovery.join(5)

    def decide(self, body):
        with self.lock:
            process, pipe = self.process, self.pipe
        if process is None:
            raise RuntimeError('Worker unavailable')
        if not process.is_alive():
            self._replace(process)
            raise RuntimeError('Worker unavailable')
        try:
            pipe.send(body)
            if not pipe.poll(self.decision_timeout):
                raise RuntimeError('Decision deadline exceeded')
            result = pipe.recv()
        except (BrokenPipeError, EOFError, OSError, RuntimeError):
            self._replace(process)
            raise RuntimeError('Decision unavailable') from None
        if not isinstance(result, dict) or 'error' in result and result['error'] != 'Input exceeds model context budget':
            self._replace(process)
            raise RuntimeError('Decision unavailable')
        if 'error' in result:
            raise ValueError(result['error'])
        return result


def serve(runtime, address=('0.0.0.0', 8040), on_server=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def reply(self, status, value):
            data = json.dumps(value).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error': 'Unknown route'})
            ready = runtime.ready()
            self.reply(200 if ready else 503, {'ready': ready})

        def do_POST(self):
            if self.path != '/v1/decisions':
                return self.reply(404, {'error': 'Unknown route'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    return self.reply(413, {'error': 'Request too large'})
                body = validate(json.loads(self.rfile.read(length)))
                self.reply(200, runtime.decide(body))
            except (ValueError, TypeError):
                self.reply(422, {'error': 'Invalid or over-budget decision request'})
            except Exception:
                self.reply(503, {'error': 'Decision unavailable'})
    server = HTTPServer(address, Handler)
    try:
        if on_server is not None:
            on_server(server)
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    runtime = Runtime()
    try:
        serve(runtime)
    finally:
        runtime.stop()
