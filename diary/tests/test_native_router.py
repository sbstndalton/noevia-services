"""Synthetic HTTP qualification of native router model identity; no Diary corpus."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from agent.llm import LLMClient


def test_native_router_chat_aux_and_embedding_models_share_openai_endpoint():
    calls = []
    class Router(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, body))
            if self.path == '/v1/embeddings':
                result = {'data': [{'index': 1, 'embedding': [0.0, 1.0]}, {'index': 0, 'embedding': [1.0, 0.0]}]}
            else:
                result = {'model': body['model'], 'choices': [{'message': {'content': 'Synthetic response', 'reasoning_content': 'Synthetic reasoning'}}]}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
    router = ThreadingHTTPServer(('127.0.0.1', 0), Router)
    thread = threading.Thread(target=router.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{router.server_port}/v1'
    chat = LLMClient(base, chat_model='synthetic-chat', embed_model='synthetic/embedding:Q8_0', max_retries=1)
    aux = LLMClient(base, chat_model='synthetic-aux', max_retries=1)
    try:
        assert str(chat.chat([{'role': 'user', 'content': 'Synthetic fixture'}])) == 'Synthetic response'
        assert aux.chat([]).reasoning == 'Synthetic reasoning'
        assert chat.embed(['first', 'second']) == [[1.0, 0.0], [0.0, 1.0]]
        assert [(p, b['model']) for p, b in calls] == [
            ('/v1/chat/completions', 'synthetic-chat'),
            ('/v1/chat/completions', 'synthetic-aux'),
            ('/v1/embeddings', 'synthetic/embedding:Q8_0'),
        ]
    finally:
        chat.close()
        aux.close()
        router.shutdown()
        router.server_close()
        thread.join()
