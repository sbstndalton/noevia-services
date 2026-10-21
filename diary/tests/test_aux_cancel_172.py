"""#172: a deadline-bounded aux call (skip classifier, summarizer, index steps) is cancellable. When its
deadline gives it up the model connection is closed, so the server stops generating instead of holding the
slot until max_tokens. A real local HTTP server stands in for the model server so the disconnect is observed;
no model endpoint is contacted. Synthetic text only."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

TEMPLATES = {
    "skip_classifier": "CLASSIFY {user_message} {assistant_message}",
    "summarizer": "SUMMARIZE {assistant_message}",
    "index_maintenance": "GATE {user_message} {assistant_summary} {current_sections}",
    "index_edit": "EDIT {user_message} {assistant_summary} {current_sections} {today}",
}


class Upstream:
    """Answers CLASSIFY with LOG. SUMMARIZE either stalls after its headers (as a model that keeps
    reasoning does) until the client disconnects, or answers. Streamed requests get SSE, others JSON."""

    def __init__(self, stall_summary):
        self.stall_summary = stall_summary
        self.requests = []
        self.disconnected = threading.Event()
        self.release = threading.Event()
        up = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                up.requests.append(body)
                last = body["messages"][-1]["content"]
                text = "LOG" if last.startswith("CLASSIFY") else "A synthetic summary."
                stall = last.startswith("SUMMARIZE") and up.stall_summary
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    if stall:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        return self._wait()
                    for data in ({"choices": [{"delta": {"content": text}}]},
                                 {"choices": [{"delta": {}, "finish_reason": "stop"}]}):
                        self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                if stall:
                    return self._wait()
                raw = json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _wait(self):
                self.connection.settimeout(0.02)
                end = time.monotonic() + 10
                while not up.release.is_set() and time.monotonic() < end:
                    try:
                        if self.connection.recv(1) == b"":
                            up.disconnected.set()
                            return
                    except socket.timeout:
                        continue
                    except OSError:
                        up.disconnected.set()
                        return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = "http://127.0.0.1:%d/v1" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


class _Store:
    index_enabled = False

    def __init__(self):
        self.saved = []

    def get_standing_sections_text(self, max_chars=0):
        return ""

    def log_exchange(self, **kwargs):
        self.saved.append(kwargs)
        return "xid-synthetic"


@pytest.fixture
def fast(monkeypatch):
    from agent import deadline
    monkeypatch.setattr(deadline, "AUX_STEP_DEADLINE_S", 0.5)
    return deadline


def _client(url):
    from agent.llm import LLMClient
    c = LLMClient(url, chat_model="synthetic-model", timeout_s=10.0, max_retries=1)
    c.reasoning_tokens = 4096
    return c


def _run(upstream, budget):
    from agent.pipeline import LoggingPipeline
    store = _Store()
    client = _client(upstream.url)
    try:
        start = time.monotonic()
        outcome = LoggingPipeline(store, None, client, TEMPLATES).log_exchange(
            "Synthetic question", "Synthetic reply.", now=None, budget=budget)
        return outcome, store, time.monotonic() - start
    finally:
        client.close()


def test_an_abandoned_summarizer_closes_its_model_connection(fast):
    from agent.deadline import ExchangeBudget
    upstream = Upstream(stall_summary=True)
    try:
        outcome, store, took = _run(upstream, ExchangeBudget(total_s=5))
        assert took < 3, "the summarizer is given up at its aux deadline"
        assert outcome.decision == "logged"
        assert store.saved[0]["claude_text"] == "Synthetic reply."  # falls back to the verbatim reply
        assert upstream.disconnected.wait(2), "the model server must see the connection closed"
        summ = [r for r in upstream.requests if r["messages"][-1]["content"].startswith("SUMMARIZE")]
        assert len(summ) == 1 and summ[0]["stream"] is True, "sent once, streamed so it can be cancelled"
    finally:
        upstream.stop()


def test_a_bounded_aux_reply_is_unchanged_by_streaming(fast):
    from agent.deadline import ExchangeBudget
    upstream = Upstream(stall_summary=False)
    try:
        outcome, store, _ = _run(upstream, ExchangeBudget(total_s=5))
        assert outcome.decision == "logged"  # the streamed classifier verdict "LOG" is read as before
        assert store.saved[0]["claude_text"] == "A synthetic summary."
        assert all(r["stream"] is True for r in upstream.requests)
        assert all(r["max_tokens"] == 2048 + 4096 for r in upstream.requests)
        assert not upstream.disconnected.is_set()
    finally:
        upstream.stop()


def test_an_aux_call_without_a_budget_stays_unstreamed():
    upstream = Upstream(stall_summary=False)
    try:
        from agent.pipeline import LoggingPipeline
        client = _client(upstream.url)
        try:
            out = LoggingPipeline(_Store(), None, client, TEMPLATES).summarize("Synthetic reply.")
        finally:
            client.close()
        assert out == "A synthetic summary."
        assert upstream.requests and upstream.requests[0]["stream"] is False
    finally:
        upstream.stop()
