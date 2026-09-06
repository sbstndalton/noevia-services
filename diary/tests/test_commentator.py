"""Commentator tests — on-demand labeled reflections that never touch the corpus.

The master prompt's verification bar for the commentator:
  - a generated summary is clearly labeled as AI commentary (structural: the
    API response is a separate object from diary text; the prompt marks it),
  - it never gets treated as if the user wrote it (structural: nothing is
    written to corpus, journal, or index — asserted here by inspection).
"""
from __future__ import annotations

from datetime import datetime

import pytest

import agent.app as appmod
from agent.commentator import Commentator, Reflection
from agent.config import Config
from agent.corpus_store import CorpusStore
from agent.journal import Journal
from tests.test_journal_store import FakeWebDAV
from tests.test_retrieval_context import StubLLM


TEMPLATES = {
    "reflection": "REFLECT today={today} focus={focus} standing={standing} retrieved={retrieved}",
    "about_question": "ABOUTQ today={today} question={question} standing={standing} retrieved={retrieved}",
}


class EchoAux:
    """Echoes the prompt so tests can assert exactly what grounded the call."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return messages[-1]["content"].strip()

    def close(self):
        pass


class EmptyRetriever:
    def search(self, query, top_k=8):
        return []

    def stats(self):
        return {"chunks": 0, "vec_available": False, "dim": None}

    def close(self):
        pass


def _commentator(chunks=None, vec_available=True):
    dav = FakeWebDAV()
    store = CorpusStore(Config({"corpus": {"webdav": {"remote_root": ""}, "monthly_prefix": "", "index_file": "INDEX.md"}}), dav, Journal(__import__("pathlib").Path("/tmp") / "c-journal.db"))
    retr = _RetrieverStub(chunks or [], vec_available)
    aux = EchoAux()
    return Commentator(store, retr, aux, TEMPLATES), store, aux


class _RetrieverStub:
    def __init__(self, chunks, vec_available):
        self._chunks = chunks
        self._vec = vec_available
        self.queries = []

    def search(self, query, top_k=8):
        self.queries.append(query)
        return self._chunks

    def stats(self):
        return {"chunks": 0, "vec_available": self._vec, "dim": 64}

    def close(self):
        pass


def test_reflection_is_generated_from_standing_sections_and_chunks():
    c, _, aux = _commentator(chunks=[{"day": "2026-09-01", "header": "09:15 — Morning", "text": "[2026-09-01] …"}])
    r = c.reflect(focus="sleep", today=datetime(2026, 9, 6).date())
    assert r.kind == "reflection"
    assert r.error is None
    prompt = aux.calls[0][0][-1]["content"]
    assert "sleep" in prompt                     # focus present
    assert "2026-09-01" in prompt                # retrieved chunk text present
    assert r.used_chunks == 1


def test_reflection_pulls_query_from_open_questions_when_no_focus():
    c, store, aux = _commentator()
    store.update_standing_sections([{"action": "add", "text": "Will the sleep schedule hold?"}], [], "2026-09-06")
    r = c.reflect(today=datetime(2026, 9, 6).date())
    prompt = aux.calls[0][0][-1]["content"]
    assert "sleep schedule" in prompt            # standing sections grounded the prompt
    # The retrieval query was derived from the standing-section bullets.
    assert c.retrieval.queries and "sleep schedule" in c.retrieval.queries[0]
    assert r.used_chunks == 0


def test_about_question_targets_the_question_in_retrieval():
    c, _, aux = _commentator(chunks=[{"day": "2026-08-30", "header": "21:00 — Evening", "text": "…"}])
    r = c.about_question("Will the sleep schedule hold?", today=datetime(2026, 9, 6).date())
    assert c.retrieval.queries[0] == "Will the sleep schedule hold?"
    assert r.question == "Will the sleep schedule hold?"
    assert r.sources == [{"day": "2026-08-30", "header": "21:00 — Evening"}]


def test_sources_are_distinct_and_ordered():
    chunks = [
        {"day": "2026-09-01", "header": "a", "text": "1"},
        {"day": "2026-09-01", "header": "a", "text": "1-duplicate"},
        {"day": "2026-08-30", "header": "b", "text": "2"},
    ]
    c, _, _ = _commentator(chunks=chunks)
    r = c.reflect(focus="x")
    assert r.sources == [{"day": "2026-09-01", "header": "a"}, {"day": "2026-08-30", "header": "b"}]


def test_model_failure_returns_error_and_touches_nothing():
    c, store, _ = _commentator()

    def boom(messages, **kwargs):
        raise RuntimeError("aux down")

    c.llm_aux = boom
    r = c.reflect(focus="sleep")
    assert r.error and "Nothing was changed" in r.error
    assert r.text == ""
    # Corpus untouched: no journal entries, no index writes.
    assert store.journal.pending_count() == 0


def test_reflection_is_never_persisted_anywhere():
    """The never-log guarantee, asserted structurally: after generating a
    reflection the corpus has no new journal entries and no modified files."""
    c, store, aux = _commentator(chunks=[{"day": "2026-09-01", "header": "a", "text": "t"}])
    dav_files_before = dict(store.backend.files)
    journal_before = store.journal.pending_count()
    r = c.reflect(focus="sleep")
    assert r.text  # a reflection was produced
    assert store.backend.files == dav_files_before
    assert store.journal.pending_count() == journal_before == 0


def test_retrieval_unavailable_flags_degraded():
    c, _, aux = _commentator(vec_available=False)
    r = c.reflect(focus="sleep")
    assert r.degraded is True


# ---------------- endpoint ----------------


class _StubMain:
    def chat(self, messages, **kwargs):
        return "A warm, honest reply.\n\n[LOG: ok]"

    def close(self):
        pass


class _StubAux:
    def chat(self, *a, **k):
        return "unused"

    def close(self):
        pass


class _RecordingCommentator:
    def reflect(self, focus=None):
        return Reflection(kind="reflection", text="REFLECTION TEXT", used_chunks=2,
                          sources=[{"day": "2026-09-01", "header": "09:15 — Morning"}])

    def about_question(self, question):
        return Reflection(kind="about_question", text="Q REFLECTION", question=question, used_chunks=3, sources=[])

    def close(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    from agent.config import Config

    cfg = Config({
        "corpus": {"webdav": {"remote_root": ""}, "monthly_prefix": "", "index_file": "INDEX.md"},
        "retrieval": {"db_path": str(tmp_path / "idx.db")},
        "llm": {"base_url": "http://stub", "api_key": "", "chat_model": "m", "embed_model": "e",
                "timeout_s": 5, "max_retries": 1,
                "aux": {"base_url": "http://stub", "api_key": "", "model": "a"}},
        "ui": {"host": "127.0.0.1", "port": 8010},
    })
    monkeypatch.setattr(appmod, "_base_cfg", cfg)
    monkeypatch.setenv("DIARY_LEGACY_USER_ID", "22222222-2222-4222-8222-222222222222")

    fake = FakeWebDAV()
    appmod.SESSIONS.clear()
    appmod._state = None
    appmod._tenant_states.clear()

    def patched_init(self, cfg_inner):
        from pathlib import Path as _P

        self.cfg = cfg_inner
        self.auth_token = ""
        self.journal = appmod.Journal(_P(cfg_inner.get("retrieval.db_path")))
        self.dav = fake
        self.llm_main = _StubMain()
        self.llm_aux = _StubAux()
        self.store = appmod.CorpusStore(cfg_inner, self.dav, self.journal)
        self.retrieval = _EmptyRetriever()
        self.assembler = appmod.ContextAssembler(self.store, self.retrieval, cfg_inner)
        self.assembler._system_prompt = lambda: "SYSTEM RULES"
        self.pipeline = appmod.LoggingPipeline(self.store, self.llm_main, self.llm_aux, templates={})
        self.pipeline._aux_call = lambda key, sysp, **fields: (
            "LOG" if key == "skip_classifier" else ("summary prose" if key == "summarizer" else "NO")
        )
        self.commentator = _RecordingCommentator()

    monkeypatch.setattr(appmod.AppState, "__init__", patched_init)
    from fastapi.testclient import TestClient

    return TestClient(appmod.app)


class _EmptyRetriever:
    def reindex_file(self, *a, **k):
        return 0

    def search(self, *a, **k):
        return []

    def stats(self):
        return {"chunks": 0, "vec_available": False, "dim": None}

    def close(self):
        pass


def test_reflect_endpoint_returns_labeled_commentary_only(client):
    r = client.post("/api/insights/reflect", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "reflection"
    assert body["text"] == "REFLECTION TEXT"
    assert body["sources"] == [{"day": "2026-09-01", "header": "09:15 — Morning"}]
    # The response is commentary metadata — it is not diary log text.
    assert "today_log" not in body and "log" not in body


def test_about_question_endpoint_requires_question(client):
    assert client.post("/api/insights/about-question", json={"question": ""}).status_code == 400
    r = client.post("/api/insights/about-question", json={"question": "Will the sleep schedule hold?"})
    assert r.status_code == 200
    assert r.json()["kind"] == "about_question"


def test_insights_endpoint_serves_standing_sections(client):
    st = appmod.get_state()
    st.store.update_standing_sections([{"action": "add", "text": "Open thing"}], [{"action": "add", "text": "Key event", "date": "2026-09-01"}], "2026-09-06")
    r = client.get("/api/insights")
    assert r.status_code == 200
    body = r.json()
    assert body["questions"][0]["text"] == "Open thing"
    assert body["questions"][0]["resolved"] is False
    assert body["timeline"][0] == {"date": "2026-09-01", "text": "Key event"}
