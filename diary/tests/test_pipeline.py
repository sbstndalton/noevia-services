"""Pipeline + app tests with stubbed LLMs and a fake corpus store."""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import agent.app as appmod
from agent.config import Config
from agent.corpus_store import CorpusStore
from agent.journal import Journal
from agent.pipeline import LoggingPipeline, _json_extract
from tests.test_journal_store import FakeWebDAV


class StubAux:
    """Aux LLM stub: classifier says LOG, summarizer echoes third-person prose."""

    def __init__(self, verdict="LOG"):
        self.verdict = verdict

    def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Verdict (LOG or SKIP)" in prompt:
            return self.verdict
        if "biographer" in prompt:
            return "The user shared something personal; the companion responded warmly."
        if "UPDATE or NO" in prompt:
            return "NO"
        return "ok"


class StubMain:
    def chat(self, messages, **kwargs):
        return "A warm, honest reply.\n\n[LOG: ok]"


@pytest.fixture
def store(tmp_path):
    cfg = Config({"corpus": {"webdav": {"remote_root": ""}, "monthly_prefix": "", "index_file": "INDEX.md"}})
    return CorpusStore(cfg, FakeWebDAV(), Journal(tmp_path / "j.db"))


def test_pipeline_logs_substantive_exchange(store):
    pipe = LoggingPipeline(store, StubMain(), StubAux("LOG"), templates={})
    # templates are accessed only via _aux_call; stub bypasses it
    pipe._aux_call = lambda key, sysp, **fields: "LOG" if key == "skip_classifier" else (
        "The user shared something; the companion responded." if key == "summarizer" else "NO"
    )
    outcome = pipe.log_exchange("I felt proud today.", "That's real progress.", now=datetime(2026, 9, 1, 10, 0))
    assert outcome.decision == "logged"
    text, _ = store.read_month(datetime(2026, 9, 1).date())
    assert "**Me:** I felt proud today." in text  # user words verbatim
    assert "companion responded" in text


def test_pipeline_skips_meta_exchange(store):
    pipe = LoggingPipeline(store, StubMain(), StubAux("SKIP"), templates={})
    pipe._aux_call = lambda key, sysp, **fields: "SKIP" if key == "skip_classifier" else "x"
    outcome = pipe.log_exchange("Change the heading format.", "Sure, done.", now=datetime(2026, 9, 1, 10, 0))
    assert outcome.decision == "skipped"
    text, _ = store.read_month(datetime(2026, 9, 1).date())
    assert text is None  # nothing written


def test_classifier_failure_defaults_to_log(store):
    pipe = LoggingPipeline(store, StubMain(), StubAux(), templates={})

    def boom(*a, **k):
        raise RuntimeError("aux down")

    pipe._aux_call = boom
    outcome = pipe.log_exchange("Something real.", "Reply.", now=datetime(2026, 9, 2, 9, 0))
    assert outcome.decision == "logged"  # fail-open


def test_summarizer_failure_logs_verbatim(store):
    pipe = LoggingPipeline(store, StubMain(), StubAux(), templates={})
    calls = {"n": 0}

    def fake_aux(key, sysp, **fields):
        calls["n"] += 1
        if key == "skip_classifier":
            return "LOG"
        raise RuntimeError("summarizer down")

    pipe._aux_call = fake_aux
    outcome = pipe.log_exchange("Entry text.", "Assistant reply text.", now=datetime(2026, 9, 3, 9, 0))
    assert outcome.decision == "logged"
    text, _ = store.read_month(datetime(2026, 9, 3).date())
    assert "**Assistant:** Assistant reply text." in text  # verbatim fallback


def test_json_extract():
    assert _json_extract('{"a": 1}') == '{"a": 1}'
    assert _json_extract('Sure:\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _json_extract('leading prose {"a": {"b": 2}} trailing') == '{"a": {"b": 2}}'


# ---------------- app endpoints ----------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    # tmp config so the lazy AppState never touches the repo's data/ dir
    import json as _json
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_json.dumps({
        "corpus": {"webdav": {"remote_root": ""}, "monthly_prefix": "", "index_file": "INDEX.md"},
        "retrieval": {"db_path": str(tmp_path / "idx.db")},
        "llm": {"base_url": "http://stub", "api_key": "", "chat_model": "m", "embed_model": "e",
                "timeout_s": 5, "max_retries": 1,
                "aux": {"base_url": "http://stub", "api_key": "", "model": "a"}},
        "ui": {"host": "127.0.0.1", "port": 8010},
    }), encoding="utf-8")
    monkeypatch.setenv("DIARY_CONFIG", str(cfg_path))
    # Endpoints are fail-closed: requests must resolve to a tenant. These
    # tests exercise the documented legacy-direct-client path via the
    # DIARY_LEGACY_USER_ID env mapping (one deliberate operator-set UUID).
    monkeypatch.setenv("DIARY_LEGACY_USER_ID", "22222222-2222-4222-8222-222222222222")

    fake = FakeWebDAV()
    appmod.SESSIONS.clear()
    appmod._state = None
    appmod._tenant_states.clear()

    def patched_init(self, cfg_inner):
        from pathlib import Path as _P

        self.cfg = cfg_inner
        self.auth_token = (cfg_inner.get("ui.auth_token") or "").strip()
        self.journal = appmod.Journal(_P(cfg_inner.get("retrieval.db_path")))
        self.dav = fake
        self.llm_main = StubMain()
        self.llm_aux = StubAuxLike()
        self.store = appmod.CorpusStore(cfg_inner, self.dav, self.journal)
        self.retrieval = _StubRetriever()
        self.assembler = appmod.ContextAssembler(self.store, self.retrieval, cfg_inner)
        self.assembler._system_prompt = lambda: "SYSTEM RULES"
        self.pipeline = appmod.LoggingPipeline(self.store, self.llm_main, self.llm_aux, templates={})
        self.pipeline._aux_call = lambda key, sysp, **fields: (
            "LOG" if key == "skip_classifier"
            else ("The companion replied warmly." if key == "summarizer" else "NO")
        )

    monkeypatch.setattr(appmod.AppState, "__init__", patched_init)
    return TestClient(appmod.app)


class StubAuxLike:
    def chat(self, *a, **k):
        return "unused"

    def close(self):
        pass


class _StubRetriever:
    def stats(self):
        return {"chunks": 0, "vec_available": False, "dim": None}

    def reindex_file(self, *a, **k):
        return 0

    def search(self, *a, **k):
        return []

    def close(self):
        pass


def test_chat_endpoint_logs_and_returns_status(client):
    r = client.post("/api/chat", json={"message": "Feeling good about the new agent."})
    assert r.status_code == 200
    data = r.json()
    assert data["reply"] == "A warm, honest reply."  # marker stripped
    assert data["decision"] == "logged"
    assert data["xid"]


def test_chat_endpoint_marker_stripped_not_logged_twice(client):
    r1 = client.post("/api/chat", json={"message": "one"})
    r2 = client.post("/api/chat", json={"message": "two"})
    assert r1.json()["reply"].count("[LOG") == 0
    assert r2.json()["decision"] == "logged"


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "journal_pending" in body and "retrieval" in body


# ---------------- auth + OpenAI-compatible endpoint ----------------


def test_auth_required_when_token_configured(client, monkeypatch):
    st = appmod.get_state()
    monkeypatch.setattr(st, "auth_token", "sekrit", raising=False)
    # no token -> 401 on API and /v1
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/api/health").status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 401
    # wrong token -> 401
    assert client.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    # correct token (either header style) -> 200
    assert client.get("/api/health", headers={"Authorization": "Bearer sekrit"}).status_code == 200
    assert client.get("/api/health", headers={"X-Diary-Token": "sekrit"}).status_code == 200
    # static assets and page shell stay reachable (token lives in localStorage)
    assert client.get("/").status_code == 200


def test_v1_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert any(m["id"] == "diary-companion" for m in data["data"])


def test_missing_tenant_header_fails_closed(client, monkeypatch):
    """A request with no (or a malformed) tenant identity must be rejected,
    never silently served from the process-wide legacy state. (/api/health is
    service-level and deliberately does not require tenant identity.)"""
    monkeypatch.delenv("DIARY_LEGACY_USER_ID", raising=False)
    appmod._tenant_states.clear()
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 400
    assert client.get("/api/day").status_code == 400
    assert client.get("/api/months").status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 400


def test_v1_chat_completions_logs_exchange(client):
    r = client.post("/v1/chat/completions", json={
        "model": "diary-companion",
        "messages": [
            {"role": "system", "content": "irrelevant client system prompt"},
            {"role": "user", "content": "Note from Solair: today was heavy but I got through it."},
        ],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "A warm, honest reply."  # marker stripped
    assert data["diary"]["decision"] == "logged"
    # exchange was actually logged into the corpus via the pipeline
    st = appmod.get_state()
    text, _ = st.store.read_month(__import__("datetime").date.today())
    assert "**Me:** Note from Solair: today was heavy but I got through it." in text


def test_v1_chat_rejects_missing_user_message(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "only system"}]})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
