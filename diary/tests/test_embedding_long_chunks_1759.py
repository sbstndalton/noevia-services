"""#1759: a Diary chunk (a whole ``###`` subsection) longer than the embeddings server takes in one
batch was refused, under EmbeddingGemma 2 and under nomic. It is now embedded in pieces and nothing is
refused, while a text the server accepts goes out exactly as before.

Synthetic text and a fake llama.cpp embeddings server only. The fake counts one token per word,
times a per-model ratio (EmbeddingGemma 2 counts about 9% more than nomic), and answers an input over
its batch the way llama.cpp does."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import httpx
import pytest

import agent.retrieval as retrieval_mod
from agent import embedding as emb
from agent.config import Config
from agent.embedding_migration import EmbeddingMigration
from agent.llm import InputTooLarge, LLMClient, LLMError
from agent.retrieval import Retriever

_real_load_vec = retrieval_mod._load_vec_extension


def _load_vec(conn):
    try:
        conn.enable_load_extension(True)
    except Exception:  # noqa: BLE001
        pass
    return _real_load_vec(conn)


@pytest.fixture(autouse=True)
def _vec_enabled(monkeypatch):
    monkeypatch.setattr(retrieval_mod, "_load_vec_extension", _load_vec)


def _vec_or_skip(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "probe.db"))
    ok = _load_vec(conn)
    conn.close()
    if not ok:
        pytest.skip("sqlite-vec not loadable here")


BATCH = 2048
RATIO = {"nomic-test": 1.0, "eg2-test": 1.09}
WORDS = ["coffee", "sleep", "river", "garden", "train", "budget", "guitar", "snow"]
EMBED_URL = "http://embed.invalid/v1"


class FakeServer:
    """llama.cpp's /v1/embeddings with ``-ub 2048``: an input over the batch is refused with HTTP 500."""

    def __init__(self, fail_with=None):
        self.requests = []  # (model, [input, ...])
        self.fail_with = fail_with

    @staticmethod
    def tokens(model, text):
        return math.ceil(len(text.split()) * RATIO[model])

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model, inputs = body["model"], body["input"]
        self.requests.append((model, list(inputs)))
        if self.fail_with is not None:
            return self.fail_with
        for text in inputs:
            n = self.tokens(model, text)
            if n > BATCH:
                return httpx.Response(500, json={"error": {"code": 500, "type": "server_error", "message":
                    f"input ({n} tokens) is too large to process. increase the physical batch size (current batch size: {BATCH})"}})
        data = []
        for i, text in enumerate(inputs):
            low = text.lower()
            data.append({"index": i, "embedding": [low.count(w) + 0.01 for w in WORDS]})
        return httpx.Response(200, json={"data": data})

    def client(self, model) -> LLMClient:
        c = LLMClient(base_url="http://engine.invalid/v1", embed_base_url=EMBED_URL, embed_model=model,
                      max_retries=3, retry_backoff_s=0)
        c._embed_client = httpx.Client(base_url=EMBED_URL, transport=httpx.MockTransport(self.handler))
        return c


def long_body(words: int) -> str:
    """Synthetic paragraphs of ``words`` words, with 'river' in every paragraph."""
    filler = ["the", "walk", "was", "long", "and", "quiet", "today", "again", "river", "light"]
    paras, n = [], 0
    while n < words:
        k = min(60, words - n)
        paras.append(" ".join(filler[(n + i) % len(filler)] for i in range(k)) + ".")
        n += k
    return "\n\n".join(paras)


def month(body: str) -> str:
    return ("## Saturday, August 29, 2026\n\n### 09:15 — Coffee\n\n**Me:** Morning coffee by the window.\n\n"
            "## Sunday, August 30, 2026\n\n### 18:40 — River\n\n**Me:** " + body + "\n")


def cfg(tmp_path: Path, model: str, profile: str) -> Config:
    return Config({"llm": {"base_url": "http://engine.invalid/v1", "embed_model": model, "embed_profile": profile,
                           "embed_base_url": EMBED_URL},
                   "retrieval": {"db_path": str(tmp_path / "index.db")}})


def shared_for(tmp_path, server, model, profile):
    config = emb.EmbeddingConfig(cfg(tmp_path, model, profile), tmp_path / "embedding-config.json")
    embedder = emb.Embedder(config)
    embedder._clients[EMBED_URL] = server.client(model)
    return emb.SharedEmbedding(config, embedder)


def chunk_vectors(db: Path):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()


# 1900 words: nomic 1900 tokens (fits), EmbeddingGemma 2 framed "title: none | text: ..." = 1905 words
# -> 2077 tokens, just over the 2048 batch. The shape of the live refusal (2,162 vs 1,989 tokens).
JUST_OVER_FOR_EG2 = 1900


def test_a_too_long_answer_is_a_typed_error_with_the_counts_and_is_not_retried():
    server = FakeServer()
    client = server.client("eg2-test")
    with pytest.raises(InputTooLarge) as exc:
        client.embed(["word " * 2100], model="eg2-test")
    assert exc.value.tokens == math.ceil(2100 * 1.09) and exc.value.limit == BATCH
    assert len(server.requests) == 1, "the same refusal three times is no retry"


def test_any_other_server_error_is_retried_and_never_split_as_before():
    server = FakeServer(fail_with=httpx.Response(500, json={"error": {"message": "out of memory"}}))
    client = server.client("eg2-test")
    with pytest.raises(LLMError) as exc:
        emb.embed_fitting(lambda ts, ks, tis: client.embed(emb.frame("embeddinggemma-2", ts, ks, tis), model="eg2-test"),
                          [long_body(3000)], "document")
    assert not isinstance(exc.value, InputTooLarge)
    assert len(server.requests) == 3 and all(len(inp) == 1 for _m, inp in server.requests)


def test_eg2_subsection_just_over_the_batch_is_indexed_in_pieces_and_searchable(tmp_path):
    _vec_or_skip(tmp_path)
    server = FakeServer()
    shared = shared_for(tmp_path, server, "eg2-test", "embeddinggemma-2")
    r = Retriever(tmp_path / "index.db", None, embedding=emb.EmbeddingView(shared.config, shared.embedder))
    try:
        r.reindex_file("2026-08.md", month(long_body(JUST_OVER_FOR_EG2)))
        assert r.pending_embeddings is False, "a long subsection must not stay pending (refused)"
        assert chunk_vectors(tmp_path / "index.db") == 2
        hits = r.search("river", top_k=1, min_score=0.0)
        assert hits and hits[0]["header"] == "18:40 — River"
        assert "river" in hits[0]["text"] and len(hits[0]["text"].split()) > JUST_OVER_FOR_EG2, "the chunk stays whole"
        sent = [t for _m, inp in server.requests for t in inp]
        refused = [t for t in sent if FakeServer.tokens("eg2-test", t) > BATCH]
        assert len(refused) == 1, "only the first, whole try is over the batch"
        pieces = [t for t in sent if t not in refused and t.startswith("title:") and "Coffee" not in t]
        assert len(pieces) == 2, "2,077 tokens at an 85% margin of 2,048 is two pieces"
        assert all(FakeServer.tokens("eg2-test", p) <= BATCH for p in pieces)
    finally:
        r.close()


def test_migration_to_eg2_refuses_no_long_chunk_and_reaches_ready(tmp_path):
    _vec_or_skip(tmp_path)
    server = FakeServer()
    shared = shared_for(tmp_path, server, "nomic-test", "nomic-v1")
    db = tmp_path / "index.db"
    r = Retriever(db, None, embedding=emb.EmbeddingView(shared.config, shared.embedder))
    mig = EmbeddingMigration(shared, db, duty=0.5, load_vec=_load_vec, sleep=lambda s: None)
    try:
        r.reindex_file("2026-08.md", month(long_body(JUST_OVER_FOR_EG2)))
        assert r.pending_embeddings is False
        mig.set_next({"model": "eg2-test", "profile": "embeddinggemma-2", "base_url": EMBED_URL})
        mig.wait(10)
        st = mig.status()
        assert st["last_pass"]["refused"] == 0, st["last_pass"]
        assert st["coverage"]["next"]["vectors"] == st["coverage"]["next"]["chunks"] == 2
        assert st["coverage"]["next"]["ready"] is True
    finally:
        mig.stop()
        r.close()


def test_nomic_sends_an_accepted_chunk_unchanged_and_no_longer_refuses_a_longer_one(tmp_path):
    _vec_or_skip(tmp_path)
    server = FakeServer()
    llm = server.client("nomic-test")
    # Built without a shared configuration: the client's own model, raw text (nomic-v1), as before #1636.
    r = Retriever(tmp_path / "index.db", llm)
    try:
        fits = month(long_body(JUST_OVER_FOR_EG2))
        r.reindex_file("2026-08.md", fits)
        assert r.pending_embeddings is False
        assert len(server.requests) == 2, "one request per chunk, nothing split ahead of time"
        expected = [row[2] for row in r._parse_chunks(fits)]
        assert [inp for _m, inp in server.requests] == [[t] for t in expected], "raw text, unframed, unsplit"

        server.requests.clear()
        r.reindex_file("2026-09.md", month(long_body(2300)))  # over nomic's batch too: refused before #1759
        assert r.pending_embeddings is False
        assert chunk_vectors(tmp_path / "index.db") == 4
    finally:
        r.close()


def test_a_refused_query_is_raised_and_not_split():
    server = FakeServer()
    client = server.client("eg2-test")
    with pytest.raises(InputTooLarge):
        emb.embed_fitting(lambda ts, ks, tis: client.embed(emb.frame("embeddinggemma-2", ts, ks, tis), model="eg2-test"),
                          [long_body(2500)], "query")
    assert len(server.requests) == 1


def test_split_text_keeps_every_character_and_cuts_at_paragraphs():
    text = long_body(600)
    pieces = emb.split_text(text, 3)
    assert "".join(pieces) == text
    assert len(pieces) == 3 and all(p.endswith("\n\n") for p in pieces[:-1])
