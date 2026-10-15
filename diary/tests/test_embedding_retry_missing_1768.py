"""#1768: the Diary retries chunks without a vector in the ACTIVE slot, without changing the model.
Synthetic diary text and a fake embedder only."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from agent import embedding as emb
from agent import vector_slots as slots
from agent.embedding_migration import EmbeddingMigration
from agent.retrieval import Retriever

from .test_embedding_migration_1636 import (  # noqa: F401  (autouse fixture + helpers)
    FakeEmbedder, MONTH, _load_vec, _vec_enabled, _vec_or_skip, cfg_for)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _setup(tmp_path, **mig_kw):
    """One index with two chunks lacking a vector in the active slot, while the server refuses them."""
    cfg = cfg_for(tmp_path)
    config = emb.EmbeddingConfig(cfg, tmp_path / "embedding-config.json")
    fake = FakeEmbedder(refuse=lambda model, text: not fake.accept_all and ("coffee" in text or "garden" in text))
    fake.accept_all = True
    shared = emb.SharedEmbedding(config, fake)
    path = tmp_path / "index.db"
    r = Retriever(path, SimpleNamespace(embed=None), embedding=emb.EmbeddingView(config, fake))
    r.reindex_file("2026-08.md", MONTH)
    # Two chunks keep their row but have no vector in the active slot: what a refusal leaves behind
    # when the row exists already (EG2 after a migration: the row, and the old model's vector).
    conn = sqlite3.connect(str(path))
    _load_vec(conn)
    ids = [row[0] for row in conn.execute("SELECT id FROM chunks WHERE body LIKE '%coffee%' OR body LIKE '%garden%'")]
    conn.execute("BEGIN IMMEDIATE")
    slots.drop_chunks(conn, ids)
    conn.commit()
    conn.close()
    fake.accept_all = False
    fake.calls.clear()
    clock = Clock()
    mig = EmbeddingMigration(shared, path, duty=0.5, load_vec=_load_vec, sleep=lambda s: None, now=clock, **mig_kw)
    return SimpleNamespace(config=config, fake=fake, path=path, r=r, mig=mig, clock=clock)


def _coverage(svc):
    conn = sqlite3.connect(str(svc.path))
    _load_vec(conn)
    try:
        return slots.coverage(conn, slots.find_slot(conn, svc.config.active()))
    finally:
        conn.close()


def _done(svc):
    svc.mig.stop()
    svc.r.close()


def test_a_chunk_left_without_a_vector_is_embedded_by_the_next_regular_pass(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _setup(tmp_path, cooldown_s=60)
    try:
        assert svc.config.next() is None
        before = _coverage(svc)
        assert before["vectors"] == 1 and before["chunks"] == 3
        svc.mig.kick(); svc.mig.wait(10)  # the start-up pass: refused again, a cooldown starts
        svc.fake.accept_all = True  # the server now takes them (the #1759 splitting fix)
        svc.clock.t += 61
        svc.mig._recheck()  # the regular pass wakes itself; nothing else asks for it
        svc.mig.wait(10)
        assert _coverage(svc) == {"chunks": before["chunks"], "vectors": before["chunks"]}
    finally:
        _done(svc)


def test_start_up_runs_the_pass_even_with_no_next_model(tmp_path, monkeypatch):
    import agent.app as appmod
    kicked = []
    monkeypatch.setattr(appmod.embedding_mod, "shared_for", lambda cfg: SimpleNamespace(config=SimpleNamespace(next=lambda: None)))
    monkeypatch.setattr(appmod.embedding_mod, "guard_seeded_addresses", lambda config: None)
    monkeypatch.setattr(appmod, "embedding_migration", lambda: SimpleNamespace(kick=lambda: kicked.append(1)))
    appmod._start_embedding()
    assert kicked == [1]


def test_a_permanently_refused_chunk_is_not_retried_faster_than_the_cooldown(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _setup(tmp_path, cooldown_s=600, max_cooldown_s=3600)
    try:
        svc.mig.kick(); svc.mig.wait(10)
        first = len(svc.fake.calls)
        assert first > 0
        for _ in range(5):  # kicks inside the cooldown do not touch the server
            svc.clock.t += 100
            svc.mig.kick(); svc.mig.wait(10)
        assert len(svc.fake.calls) == first
        svc.clock.t += 600  # cooldown over: tried again once, then the cooldown doubles
        svc.mig.kick(); svc.mig.wait(10)
        second = len(svc.fake.calls)
        assert second > first
        svc.clock.t += 700  # inside the doubled cooldown (1200 s)
        svc.mig.kick(); svc.mig.wait(10)
        assert len(svc.fake.calls) == second
        svc.clock.t += 600
        svc.mig.kick(); svc.mig.wait(10)
        assert len(svc.fake.calls) > second
    finally:
        _done(svc)


def test_chunks_with_a_vector_are_not_embedded_again_and_reindex_clears_cooldowns(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _setup(tmp_path, cooldown_s=600)
    try:
        svc.fake.accept_all = True
        svc.mig.retry_missing(); svc.mig.wait(10)
        cov = _coverage(svc)
        assert cov["vectors"] == cov["chunks"] > 0
        svc.fake.calls.clear()
        svc.mig.retry_missing(); svc.mig.wait(10)
        svc.clock.t += 10_000
        svc.mig.retry_missing(); svc.mig.wait(10)
        assert svc.fake.calls == []
    finally:
        _done(svc)


def test_the_reindex_action_retries_inside_the_cooldown_and_is_bounded_per_pass(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _setup(tmp_path, cooldown_s=3600, active_budget=1)
    try:
        svc.mig.kick(); svc.mig.wait(10)
        assert len(svc.fake.calls) == 1, "a pass attempts at most active_budget chunks of the active model"
        svc.fake.accept_all = True
        svc.mig.kick(); svc.mig.wait(10)  # the chunk is cooling; the budget goes to another chunk
        assert _coverage(svc)["vectors"] == 2
        svc.mig.retry_missing(); svc.mig.wait(10)
        assert _coverage(svc)["vectors"] == 3
    finally:
        _done(svc)


def test_internal_endpoint_has_a_reindex_action(monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as appmod
    calls = []

    class FakeMig:
        def retry_missing(self):
            calls.append("reindex")
            return {"active": {"model": "m", "profile": "nomic-v1"}}

    monkeypatch.setattr(appmod, "embedding_migration", lambda: FakeMig())
    monkeypatch.setattr(appmod, "get_state", lambda: SimpleNamespace(auth_token="tok"))
    client = TestClient(appmod.app)
    r = client.post("/api/internal/embedding", headers={"Authorization": "Bearer tok"}, json={"action": "reindex"})
    assert r.status_code == 200 and calls == ["reindex"]
