"""Retained CPU-only Laya engine worker for the native Rust HTTP front."""
import json
import os
import select
import struct
import sys


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


class FramedEnginePipe:
    """Private native-front transport: big-endian u32 length, then UTF-8 JSON.

    One engine per process, one outstanding request; no HTTP, admission or recovery
    policy lives here. The native parent owns deadlines and terminates stalled I/O.
    Reads are unbuffered so poll() also sees the next frame after recv().
    """
    MAX_FRAME_BYTES = 16384

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    def poll(self, timeout=0):
        return bool(select.select([self.reader], [], [], timeout)[0])

    def _read_exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.reader.read(size - len(data))
            if not chunk:
                raise EOFError('Engine transport closed')
            data.extend(chunk)
        return bytes(data)

    def recv(self):
        size, = struct.unpack('!I', self._read_exact(4))
        if not 0 < size <= self.MAX_FRAME_BYTES:
            raise ValueError('Invalid engine frame size')
        return json.loads(self._read_exact(size))

    def send(self, value):
        data = json.dumps(value, ensure_ascii=True, allow_nan=False).encode('utf-8')
        if not 0 < len(data) <= self.MAX_FRAME_BYTES:
            raise ValueError('Invalid engine frame size')
        data = struct.pack('!I', len(data)) + data
        while data:
            written = self.writer.write(data)
            if not written:
                raise BrokenPipeError('Engine transport closed')
            data = data[written:]
        self.writer.flush()


def engine_worker_stdio():
    """Retained engine only. Reserve stdout before importing third-party engines.

    Redirect fd 1 as well as Python stdout: native-library diagnostics must never
    corrupt the framed channel. Fixed /model and imports remain in worker().
    """
    with os.fdopen(os.dup(1), 'wb', buffering=0) as writer:
        os.dup2(2, 1)
        with os.fdopen(os.dup(0), 'rb', buffering=0) as reader:
            worker(FramedEnginePipe(reader, writer))


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

if __name__ == '__main__':
    if sys.argv[1:] != ['--engine-worker']:
        raise SystemExit('Laya Python HTTP is retired; use noevia-laya')
    engine_worker_stdio()
