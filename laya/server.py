"""Private CPU-only typed-decision endpoint. No generation, tools or request logs."""
import json
import multiprocessing as mp
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


DEFAULT_KEEP_WARM_S, MAX_KEEP_WARM_S = 30.0, 3600.0


def keep_warm_from_env(env=None, log=None):
    """Idle interval in seconds between weight touches from LAYA_KEEP_WARM_S (0-3600, 0 off).

    noevia#1070: while idle the host swapped part of the worker's weights out, so the first
    decision after a quiet night paid the swap-in and missed its deadline. Unset/blank uses 30.
    """
    env = os.environ if env is None else env
    log = log or (lambda msg: print(msg, flush=True))
    raw = env.get('LAYA_KEEP_WARM_S')
    if raw is None or not raw.strip():
        return DEFAULT_KEEP_WARM_S
    try:
        value = float(raw.strip())
    except ValueError:
        value = float('nan')
    if not 0 <= value <= MAX_KEEP_WARM_S:  # also rejects nan/inf
        log(f'laya: ignoring LAYA_KEEP_WARM_S={raw.strip()[:32]!r}; it must be a number of seconds '
            f'from 0 (off) to {MAX_KEEP_WARM_S:g}. Using {DEFAULT_KEEP_WARM_S:g}.')
        return DEFAULT_KEEP_WARM_S
    return value


def touch_weights(agent, torch, stop=lambda: False):
    """Read every parameter and buffer of the agent's torch modules so the kernel keeps them
    resident. No inference, no request data. Checks `stop()` between tensors so a waiting
    request is picked up promptly; returns the number of tensors touched."""
    count = 0
    with torch.no_grad():
        for value in vars(agent).values():
            if isinstance(value, torch.nn.Module):
                for tensor in list(value.parameters()) + list(value.buffers()):
                    if stop():
                        return count
                    tensor.detach().sum()
                    count += 1
    return count


def worker(pipe):
    try:
        import torch
        import laya
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        agent = laya.load('/model', device='cpu')
        keep_warm = keep_warm_from_env()
        warned = False
        pipe.send({'ready': True})
        while True:
            if keep_warm and not pipe.poll(keep_warm):
                if not touch_weights(agent, torch, stop=lambda: pipe.poll(0)) and not warned and not pipe.poll(0):
                    warned = True
                    print('laya: keep-warm found no model tensors to touch; LAYA_KEEP_WARM_S has no effect.', flush=True)
                continue
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


DEFAULT_DECISION_TIMEOUT_S = 1.3
MIN_DECISION_TIMEOUT_S, MAX_DECISION_TIMEOUT_S = 0.5, 2.0


def decision_timeout_from_env(env=None, log=None):
    """Worker request deadline from LAYA_DECISION_TIMEOUT_S (seconds, 0.5-2.0).

    Unset/blank uses the default. A garbage or out-of-range value is refused with a
    log line naming the variable (never a request body) and the default is used.
    """
    env = os.environ if env is None else env
    log = log or (lambda msg: print(msg, flush=True))
    raw = env.get('LAYA_DECISION_TIMEOUT_S')
    if raw is None or not raw.strip():
        return DEFAULT_DECISION_TIMEOUT_S
    try:
        value = float(raw.strip())
    except ValueError:
        value = float('nan')
    if not MIN_DECISION_TIMEOUT_S <= value <= MAX_DECISION_TIMEOUT_S:  # also rejects nan/inf
        log(f'laya: ignoring LAYA_DECISION_TIMEOUT_S={raw.strip()[:32]!r}; it must be a number of seconds '
            f'from {MIN_DECISION_TIMEOUT_S} to {MAX_DECISION_TIMEOUT_S}. Using {DEFAULT_DECISION_TIMEOUT_S}.')
        return DEFAULT_DECISION_TIMEOUT_S
    return value


class Late(RuntimeError):
    """The worker missed this request's deadline but is kept: its answer will be discarded."""


# A request that missed its deadline usually finishes a moment later (noevia#1070: a cold
# first decision after idle). Killing the worker for it cost a ~17 s reload and left a cold
# replacement, so the worker is kept for this long and its stale answer is dropped. At most one
# abandoned request is ever outstanding: nothing new is sent while it is still running.
DEFAULT_LATE_GRACE_S = 15.0


class Runtime:
    def __init__(self, ctx=None, worker_target=worker, startup_timeout=90,
                 decision_timeout=1.3, retry_delays=(0, 2, 8), fatal=os._exit,
                 late_grace=DEFAULT_LATE_GRACE_S):
        self.ctx = ctx or mp.get_context('spawn')
        self.worker_target = worker_target
        self.startup_timeout = startup_timeout
        self.decision_timeout = decision_timeout
        self.retry_delays = retry_delays
        self.fatal = fatal
        self.late_grace = late_grace
        self.late = 0  # requests sent to the current worker whose answers are still due
        self.late_since = 0.0
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
            self.late = 0
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
            deadline = time.monotonic() + self.decision_timeout
            # Drop stale answers that already arrived, so a worker that finished its late work
            # long ago is never mistaken for a hung one.
            while self.late and pipe.poll(0):
                pipe.recv()
                self.late = max(0, self.late - 1)
            if self.late:
                if time.monotonic() - self.late_since > self.late_grace:
                    raise RuntimeError('Worker stuck on an abandoned decision')
                # Never queue behind unfinished abandoned work: wait for it within this request's
                # own deadline, and if it is still running answer 503 without sending anything.
                if not pipe.poll(max(0.0, deadline - time.monotonic())):
                    raise Late('Worker still finishing an abandoned decision')
                pipe.recv()
                self.late = max(0, self.late - 1)
            pipe.send(body)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not pipe.poll(remaining):
                if self.late_grace <= 0:
                    raise RuntimeError('Decision deadline exceeded')
                self.late_since = time.monotonic()
                self.late = 1
                raise Late('Decision deadline exceeded')
            result = pipe.recv()
        except Late:
            raise
        except (BrokenPipeError, EOFError, OSError, RuntimeError):
            self._replace(process)
            raise RuntimeError('Decision unavailable') from None
        if not isinstance(result, dict) or 'error' in result and result['error'] != 'Input exceeds model context budget':
            self._replace(process)
            raise RuntimeError('Decision unavailable')
        if 'error' in result:
            raise ValueError(result['error'])
        return result


DEFAULT_MAX_CONCURRENCY, MAX_MAX_CONCURRENCY = 1, 4
# Web's decision deadline is at most 2000 ms, so a request queued longer would have been
# abandoned by its caller anyway; this matches the old single-threaded queueing.
DEFAULT_QUEUE_TIMEOUT_S = 2.0
MAX_QUEUE_TIMEOUT_MS = 10000


def queue_timeout_from_env(env=None, log=None):
    """Queue wait in seconds from LAYA_QUEUE_TIMEOUT_MS (0-10000). Default 2000."""
    env = os.environ if env is None else env
    log = log or (lambda msg: print(msg, flush=True))
    raw = env.get('LAYA_QUEUE_TIMEOUT_MS')
    if raw is None or not raw.strip():
        return DEFAULT_QUEUE_TIMEOUT_S
    try:
        value = int(raw.strip())
    except ValueError:
        value = -1
    if not 0 <= value <= MAX_QUEUE_TIMEOUT_MS:
        log(f'laya: ignoring LAYA_QUEUE_TIMEOUT_MS={raw.strip()[:32]!r}; it must be an integer '
            f'from 0 to {MAX_QUEUE_TIMEOUT_MS}. Using {int(DEFAULT_QUEUE_TIMEOUT_S * 1000)}.')
        return DEFAULT_QUEUE_TIMEOUT_S
    return value / 1000


def max_concurrency_from_env(env=None, log=None):
    """Inference slots from LAYA_MAX_CONCURRENCY (integer 1-4). Default 1 keeps one model.

    Each slot is a separate worker process holding its own model copy, so raising it
    multiplies model memory. Garbage or out-of-range values are refused with a log line.
    """
    env = os.environ if env is None else env
    log = log or (lambda msg: print(msg, flush=True))
    raw = env.get('LAYA_MAX_CONCURRENCY')
    if raw is None or not raw.strip():
        return DEFAULT_MAX_CONCURRENCY
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_MAX_CONCURRENCY:
        log(f'laya: ignoring LAYA_MAX_CONCURRENCY={raw.strip()[:32]!r}; it must be an integer '
            f'from 1 to {MAX_MAX_CONCURRENCY}. Using {DEFAULT_MAX_CONCURRENCY}.')
        return DEFAULT_MAX_CONCURRENCY
    return value


class Busy(RuntimeError):
    """All inference slots stayed occupied for the whole queue timeout."""


class Pool:
    """Bounded concurrent decisions over len(runtimes) single-worker Runtimes.

    A semaphore admits at most len(runtimes) in-flight decisions; each holds one idle
    Runtime exclusively, so no worker or pipe is ever shared between requests. Excess
    requests wait up to queue_timeout, then raise Busy. ready() never takes the semaphore.
    """

    def __init__(self, runtimes, queue_timeout=DEFAULT_QUEUE_TIMEOUT_S):
        self.runtimes = list(runtimes)
        self.queue_timeout = queue_timeout
        self.slots = threading.Semaphore(len(self.runtimes))
        self.lock = threading.Lock()
        self.idle = list(self.runtimes)

    @classmethod
    def start(cls, size, queue_timeout=DEFAULT_QUEUE_TIMEOUT_S, **runtime_args):
        runtimes, errors = [], []

        def boot():
            try:
                runtimes.append(Runtime(**runtime_args))
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=boot) for _ in range(size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            for r in runtimes:
                r.stop()
            raise errors[0]
        return cls(runtimes, queue_timeout)

    def ready(self):
        # Ready while any worker can serve; a recovering slot just fails its own request.
        return any([r.ready() for r in self.runtimes])

    def decide(self, body):
        if not self.slots.acquire(timeout=self.queue_timeout):
            raise Busy('Decision service busy')
        try:
            with self.lock:
                # Prefer a worker with no answers still due (#1070), then any ready one.
                runtime = next((r for r in self.idle if r.ready() and not r.late),
                               next((r for r in self.idle if r.ready()), self.idle[0]))
                self.idle.remove(runtime)
            try:
                return runtime.decide(body)
            finally:
                with self.lock:
                    self.idle.append(runtime)
        finally:
            self.slots.release()

    def stop(self):
        for r in self.runtimes:
            r.stop()


def serve(runtime, address=('0.0.0.0', 8040), on_server=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def reply(self, status, value, headers=None):
            data = json.dumps(value).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
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
            except Busy:
                self.reply(503, {'error': 'Decision service busy'}, {'Retry-After': '1'})
            except (ValueError, TypeError):
                self.reply(422, {'error': 'Invalid or over-budget decision request'})
            except Exception:
                self.reply(503, {'error': 'Decision unavailable'})
    class Server(ThreadingHTTPServer):
        request_queue_size = 64  # default 5 drops bursts before a thread accepts
        daemon_threads = True
    server = Server(address, Handler)
    try:
        if on_server is not None:
            on_server(server)
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    runtime = Pool.start(max_concurrency_from_env(), queue_timeout_from_env(), decision_timeout=decision_timeout_from_env())
    try:
        serve(runtime)
    finally:
        runtime.stop()
