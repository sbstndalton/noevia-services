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


def _server(handler_calls, reply):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            handler_calls.append((self.path, body, self.headers.get('Authorization')))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(reply(self.path, body)).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_embeddings_use_their_own_endpoint_so_the_one_model_engine_keeps_its_chat_model():
    # #697: the engine holds one model; an embedding request there would evict the chat model.
    engine_calls, embed_calls = [], []
    engine, t1 = _server(engine_calls, lambda p, b: {'choices': [{'message': {'content': 'Synthetic response'}}]})
    embed, t2 = _server(embed_calls, lambda p, b: {'data': [{'index': 0, 'embedding': [0.5, 0.5]}]})
    client = LLMClient(f'http://127.0.0.1:{engine.server_port}/v1', api_key='synthetic-key', chat_model='synthetic-chat',
                       embed_model='synthetic-embed', embed_base_url=f'http://127.0.0.1:{embed.server_port}/v1', max_retries=1)
    try:
        assert str(client.chat([{'role': 'user', 'content': 'Synthetic fixture'}])) == 'Synthetic response'
        assert client.embed(['synthetic text']) == [[0.5, 0.5]]
        assert [p for p, _, _ in engine_calls] == ['/v1/chat/completions']
        assert [(p, b['model']) for p, b, _ in embed_calls] == [('/v1/embeddings', 'synthetic-embed')]
        assert engine_calls[0][2] == 'Bearer synthetic-key'
        assert embed_calls[0][2] is None, 'the inference credential is not sent to the embedding server'
    finally:
        client.close()
        for server, thread in ((engine, t1), (embed, t2)):
            server.shutdown()
            server.server_close()
            thread.join()


def test_embed_base_url_comes_from_the_environment(monkeypatch):
    from agent import config as config_mod
    monkeypatch.setenv('LLM_EMBED_BASE_URL', 'http://embed:8080/v1')
    cfg = config_mod.load_config()
    assert cfg.get('llm.embed_base_url') == 'http://embed:8080/v1'
