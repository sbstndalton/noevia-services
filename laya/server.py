"""Private CPU-only typed-decision endpoint. No generation, tools or request logs."""
import json
import multiprocessing as mp
import os
import re
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
    def __init__(self):
        ctx = mp.get_context('spawn')
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        if not self.pipe.poll(90) or self.pipe.recv() != {'ready': True}:
            self.stop()
            raise RuntimeError('Laya startup failed or exceeded deadline')

    def stop(self):
        self.process.terminate()
        self.process.join(2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(2)

    def decide(self, body):
        if not self.process.is_alive():
            raise RuntimeError('Worker unavailable; restart required')
        self.pipe.send(body)
        if not self.pipe.poll(1.3):
            self.stop()
            raise RuntimeError('Decision deadline exceeded; restart required')
        result = self.pipe.recv()
        if 'error' in result:
            raise ValueError(result['error'])
        return result


def serve(runtime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200 if self.path == '/health' and runtime.process.is_alive() else 503, {'ready': runtime.process.is_alive()})

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
    HTTPServer(('0.0.0.0', 8040), Handler).serve_forever()


if __name__ == '__main__':
    runtime = Runtime()
    try:
        serve(runtime)
    finally:
        runtime.stop()
