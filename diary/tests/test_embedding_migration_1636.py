"""#1636: embedding profiles, the shared embedding configuration, per-model vector slots and the
no-downtime re-index of the Diary index. Synthetic diary text and a fake embedder only."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.retrieval as retrieval_mod
from agent import embedding as emb
from agent import vector_slots as slots
from agent.config import Config
from agent.embedding_migration import EmbeddingMigration, index_files
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


WORDS = ["coffee", "sleep", "river", "garden", "train", "budget", "guitar", "snow"]


class FakeEmbedder:
    """Keyword vectors; the 'gemma' model writes them reversed (same width, not comparable)."""

    def __init__(self, refuse=None):
        self.calls = []
        self.refuse = refuse

    def embed(self, target, texts, kind="document", titles=None):
        framed = emb.frame(target.profile, texts, kind, titles)
        self.calls.append((target.model, list(framed)))
        if self.refuse and any(self.refuse(target.model, t) for t in framed):
            raise RuntimeError("input is too large to process")
        out = []
        for t in framed:
            low = t.lower()
            v = [low.count(w) + 0.01 for w in WORDS]
            out.append(list(reversed(v)) if "gemma" in target.model else v)
        return out


def day(date_iso: str, title: str, header: str, me: str) -> str:
    return f"## {title}\n\n### {header}\n\n**Me:** {me}\n\n**Claude:** Noted.\n"


MONTH = (day("2026-08-29", "Saturday, August 29, 2026", "09:15 — Coffee", "Morning coffee and more coffee by the window.")
         + "\n" + day("2026-08-30", "Sunday, August 30, 2026", "18:40 — River", "Walked by the river, the river was high.")
         + "\n" + day("2026-08-31", "Monday, August 31, 2026", "21:00 — Garden", "Planted the garden, garden beds ready."))


def cfg_for(tmp_path: Path, **llm) -> Config:
    return Config({"llm": {"base_url": "http://engine.invalid/v1", "embed_model": "nomic-test", **llm},
                   "retrieval": {"db_path": str(tmp_path / "index.db")}})


# ---------------- profiles and configuration ----------------

def test_default_profile_sends_text_unchanged_and_gemma_uses_the_card_prompts():
    texts = ["Plain.", "title: x | text: y", "", "a\nb"]
    for kind in emb.KINDS:
        assert emb.frame("nomic-v1", texts, kind, ["T"] * 4) == texts
    assert emb.frame("embeddinggemma-2", ["why"], "query") == ["task: search result | query: why"]
    assert emb.frame("embeddinggemma-2", ["entry"], "document") == ["title: none | text: entry"]
    assert emb.frame("embeddinggemma-2", ["entry"], "document", ["A | b\nc"]) == ["title: A b c | text: entry"]
    assert emb.frame("embeddinggemma-2", ["x"], "similarity") == ["task: sentence similarity | query: x"]
    with pytest.raises(emb.EmbeddingConfigError):
        emb.frame("nomic-v2", ["x"])
    with pytest.raises(emb.EmbeddingConfigError):
        emb.frame("nomic-v1", ["x"], "passage")


def test_seed_defaults_match_the_pre_1636_settings(tmp_path):
    seed = emb.seed_from_cfg(cfg_for(tmp_path, embed_base_url="http://embed:8080/v1/"))
    assert seed["active"] == emb.Target("http://embed:8080/v1", "nomic-test", "nomic-v1")
    assert seed["next"] is None
    seed = emb.seed_from_cfg(cfg_for(tmp_path, embed_next_model="gemma-test", embed_next_profile="embeddinggemma-2",
                                     embed_next_base_url="http://embed-next:8080/v1"))
    assert seed["next"] == emb.Target("http://embed-next:8080/v1", "gemma-test", "embeddinggemma-2")


def test_targets_are_validated():
    assert emb.normalize_target({"model": "m", "base_url": "http://x:1/v1/"}).base_url == "http://x:1/v1"
    for bad in ({}, {"model": "two words"}, {"model": "m", "profile": "typo"}, {"model": "m", "base_url": "http://u:p@x"},
                {"model": "m", "base_url": "http://x/?q=1"}, {"model": "m", "base_url": "file:///etc"}):
        with pytest.raises(emb.EmbeddingConfigError):
            emb.normalize_target(bad)


def test_runtime_changes_are_saved_and_win_over_the_environment(tmp_path):
    state = tmp_path / "embedding-config.json"
    c = emb.EmbeddingConfig(cfg_for(tmp_path, embed_next_model="gemma-test", embed_next_profile="embeddinggemma-2"), state)
    assert c.source == "environment" and not state.exists()
    c.switch_to_next()
    assert (state.stat().st_mode & 0o777) == 0o600
    again = emb.EmbeddingConfig(cfg_for(tmp_path, embed_model="other"), state)
    assert again.active().model == "gemma-test" and again.previous().model == "nomic-test"
    again.rollback()
    assert again.active().model == "nomic-test"
    again.finish()
    with pytest.raises(emb.EmbeddingConfigError):
        again.finish()


def test_a_damaged_saved_configuration_turns_embedding_off_and_is_left_alone(tmp_path):
    state = tmp_path / "embedding-config.json"
    state.write_text('{"version": 1, "active": {"model": "x", "profile": "nope"}}')
    c = emb.EmbeddingConfig(cfg_for(tmp_path), state)
    assert c.damaged and c.get()["damaged"]
    with pytest.raises(emb.EmbeddingConfigError):
        emb.frame(c.active().profile, ["x"])
    with pytest.raises(emb.EmbeddingConfigError):
        c.set_next({"model": "m"})
    assert '"nope"' in state.read_text()


# ---------------- slots and the re-index ----------------

def _service(tmp_path: Path, refuse=None):
    """A service index plus two tenants, each with its own synthetic month."""
    cfg = cfg_for(tmp_path)
    config = emb.EmbeddingConfig(cfg, tmp_path / "embedding-config.json")
    fake = FakeEmbedder(refuse)
    shared = emb.SharedEmbedding(config, fake)
    paths = {"service": tmp_path / "index.db",
             "a": tmp_path / "users" / "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa" / "index.db",
             "b": tmp_path / "users" / "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb" / "managed-index.db"}
    months = {"service": MONTH,
              "a": day("2026-09-01", "Tuesday, September 1, 2026", "08:00 — Train", "The train was late, train again."),
              "b": day("2026-09-02", "Wednesday, September 2, 2026", "20:00 — Budget", "Budget for October, budget review.")}
    retrievers = {}
    for k, p in paths.items():
        r = Retriever(p, SimpleNamespace(embed=None), embedding=emb.EmbeddingView(config, fake))
        r.reindex_file("2026-08.md", months[k])
        retrievers[k] = r
    mig = EmbeddingMigration(shared, paths["service"], duty=0.5, load_vec=_load_vec, sleep=lambda s: None)
    return SimpleNamespace(config=config, fake=fake, paths=paths, retrievers=retrievers, mig=mig)


def _close(svc):
    svc.mig.stop()
    for r in svc.retrievers.values():
        r.close()


def test_default_profile_keeps_the_legacy_table_an_older_image_reads(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _service(tmp_path)
    try:
        conn = sqlite3.connect(str(svc.paths["service"]))
        assert conn.execute("SELECT value FROM vec_meta WHERE key='dim'").fetchone()[0] == "8"
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
        conn.close()
        assert all(not any("task:" in t or "title:" in t for t in texts) for _m, texts in svc.fake.calls), "raw text on the wire"
        hits = svc.retrievers["service"].search("river", top_k=1, min_score=0.0)
        assert "river" in hits[0]["text"]
    finally:
        _close(svc)


def test_reindex_switch_rollback_finish_without_downtime_and_per_tenant(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _service(tmp_path)
    try:
        r = svc.retrievers["service"]
        gemma = {"model": "gemma-test", "profile": "embeddinggemma-2", "base_url": "http://embed-next:8080/v1"}
        svc.mig.set_next(gemma)
        svc.mig.wait(10)
        st = svc.mig.status()
        assert st["coverage"]["next"]["ready"] and st["coverage"]["next"]["complete"]
        assert st["coverage"]["next"]["files"] == 3
        # Every gemma document went out with the card's document prompt.
        assert all(t.startswith("title: none | text: ") for m, texts in svc.fake.calls if m == "gemma-test" for t in texts)
        # Search still uses the active model until the switch.
        svc.fake.calls.clear()
        assert "garden" in r.search("garden", top_k=1, min_score=0.0)[0]["text"]
        assert svc.fake.calls == [("nomic-test", ["garden"])]

        svc.mig.switch()
        svc.fake.calls.clear()
        assert "garden" in r.search("garden", top_k=1, min_score=0.0)[0]["text"]
        assert svc.fake.calls == [("gemma-test", ["task: search result | query: garden"])]
        # Tenants: each file holds and returns only its own chunks.
        assert "train" in svc.retrievers["a"].search("train", top_k=3, min_score=0.0)[0]["text"]
        assert all("train" not in h["text"] for h in svc.retrievers["b"].search("train", top_k=3, min_score=0.0))

        # New writes after the switch go to the new model's slot.
        r.reindex_file("2026-09.md", day("2026-09-05", "Saturday, September 5, 2026", "07:00 — Snow", "Snow, early snow."))
        assert "snow" in r.search("snow", top_k=1, min_score=0.0)[0]["text"].lower()

        svc.mig.rollback()
        svc.mig.wait(10)  # fills the nomic vectors of the chunk written while gemma was active
        svc.fake.calls.clear()
        assert "snow" in r.search("snow", top_k=1, min_score=0.0)[0]["text"].lower()
        assert svc.fake.calls == [("nomic-test", ["snow"])]

        svc.mig.finish()
        conn = sqlite3.connect(str(svc.paths["a"]))
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'vecmap_%'").fetchone()[0] == 0
        conn.close()
        assert json.loads((tmp_path / "embedding-config.json").read_text())["previous"] is None
    finally:
        _close(svc)


def test_switch_waits_for_the_new_model_and_refused_chunks_do_not_block_the_rest(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _service(tmp_path, refuse=lambda model, text: "gemma" in model and "river" in text and text.startswith("title:"))
    try:
        svc.mig.set_next({"model": "gemma-test", "profile": "embeddinggemma-2"})
        with pytest.raises(emb.EmbeddingConfigError) as early:
            svc.mig.switch()
        assert early.value.code == "notReady"
        svc.mig.wait(10)
        st = svc.mig.status()
        assert st["last_pass"]["refused"] == 1
        nxt, act = st["coverage"]["next"], st["coverage"]["active"]
        assert nxt["vectors"] == act["vectors"] - 1 and not nxt["ready"]
        with pytest.raises(emb.EmbeddingConfigError):
            svc.mig.switch()
        svc.mig.switch(force=True)
        assert svc.config.active().model == "gemma-test"
    finally:
        _close(svc)


def test_a_legacy_index_is_never_searched_with_another_model(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _service(tmp_path)
    _close(svc)
    cfg = cfg_for(tmp_path, embed_model="gemma-test", embed_profile="embeddinggemma-2")
    other = emb.EmbeddingConfig(cfg, None)
    fake = FakeEmbedder()
    r = Retriever(tmp_path / "index.db", SimpleNamespace(embed=None), embedding=emb.EmbeddingView(other, fake))
    try:
        assert r.search("garden", top_k=3, min_score=0.0) == []
        assert fake.calls == [], "no query embedding is spent on a file this model has no vectors in"
    finally:
        r.close()


def test_a_vector_never_outlives_its_text(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _service(tmp_path)
    try:
        svc.mig.set_next({"model": "gemma-test", "profile": "embeddinggemma-2"})
        svc.mig.wait(10)
        r = svc.retrievers["service"]
        # The garden subsection is rewritten: its chunk (and every model's vector for it) is replaced.
        r.reindex_file("2026-08.md", MONTH.replace("Planted the garden, garden beds ready.", "Budget talk, budget again."))
        conn = sqlite3.connect(str(svc.paths["service"]))
        _load_vec(conn)
        slot = slots.find_slot(conn, emb.Target(None, "gemma-test", "embeddinggemma-2"))
        cov = slots.coverage(conn, slot)
        conn.close()
        assert cov["vectors"] == cov["chunks"] - 1, "the rewritten chunk has no gemma vector until the pass fills it"
        svc.mig.kick()
        svc.mig.wait(10)
        svc.mig.switch()
        assert all("garden" not in h["text"] for h in r.search("garden", top_k=5, min_score=0.0))
    finally:
        _close(svc)


def test_index_files_skip_deleted_tenants_and_never_create_files(tmp_path):
    (tmp_path / "users" / "keep").mkdir(parents=True)
    (tmp_path / "users" / "keep" / "index.db").write_bytes(b"")
    (tmp_path / "users" / "gone").mkdir(parents=True)
    (tmp_path / "users" / "gone" / "index.db").write_bytes(b"")
    files = index_files(tmp_path / "index.db", skip_user=lambda u: u == "gone")
    assert files == [tmp_path / "users" / "keep" / "index.db"]
    assert not (tmp_path / "index.db").exists()


# ---------------- the internal endpoint ----------------

def test_internal_endpoint_is_service_only(monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as appmod

    calls = []

    class FakeMig:
        def status(self):
            return {"active": {"model": "nomic-test", "profile": "nomic-v1", "base_url": None}}

        def switch(self, force=False):
            calls.append(("switch", force))
            raise emb.EmbeddingConfigError("not yet", "notReady", 409)

        def set_next(self, target):
            calls.append(("set-next", target))
            return {"active": {"model": "nomic-test", "profile": "nomic-v1"}, "next": target}

    monkeypatch.setattr(appmod, "embedding_migration", lambda: FakeMig())
    state = SimpleNamespace(auth_token="tok")
    monkeypatch.setattr(appmod, "get_state", lambda: state)
    client = TestClient(appmod.app)
    auth = {"Authorization": "Bearer tok"}
    assert client.get("/api/internal/embedding").status_code == 401
    assert client.get("/api/internal/embedding", headers=auth).json()["active"]["model"] == "nomic-test"
    tenant = {**auth, "X-Cowork-User-ID": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}
    assert client.post("/api/internal/embedding", headers=tenant, json={"action": "switch"}).status_code in (400, 401)
    assert client.post("/api/internal/embedding", headers=auth, json={"action": "nuke"}).status_code == 400
    r = client.post("/api/internal/embedding", headers=auth, json={"action": "switch"})
    assert r.status_code == 409 and r.json()["code"] == "notReady"
    target = {"model": "gemma-test", "profile": "embeddinggemma-2", "base_url": "http://embed-next:8080/v1"}
    assert client.post("/api/internal/embedding", headers=auth, json={"action": "set-next", "target": target}).status_code == 200
    assert calls == [("switch", False), ("set-next", target)]
    # Open (LAN-only) mode: nobody may change the model.
    state.auth_token = ""
    assert client.post("/api/internal/embedding", json={"action": "switch"}).status_code == 403
