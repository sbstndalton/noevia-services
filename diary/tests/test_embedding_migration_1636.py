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


def test_a_forgotten_next_profile_is_warned_about_with_no_change_of_behaviour(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="agent.embedding"):
        seed = emb.seed_from_cfg(cfg_for(tmp_path, embed_next_model="embeddinggemma-2"))
    assert seed["next"] == emb.Target(None, "embeddinggemma-2", "nomic-v1")
    warned = [r.getMessage() for r in caplog.records]
    assert len(warned) == 1 and "LLM_EMBED_NEXT_MODEL=embeddinggemma-2" in warned[0] and "LLM_EMBED_NEXT_PROFILE" in warned[0]
    caplog.clear()
    with caplog.at_level("WARNING", logger="agent.embedding"):
        emb.seed_from_cfg(cfg_for(tmp_path, embed_next_model="embeddinggemma-2", embed_next_profile="embeddinggemma-2"))
        emb.seed_from_cfg(cfg_for(tmp_path))
    assert caplog.records == [], "no warning when the profile is set, or when there is no next model"


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


def _slot_models(path):
    conn = sqlite3.connect(str(path))
    try:
        return sorted(json.loads(v)["model"] for (v,) in conn.execute("SELECT value FROM vec_meta WHERE key LIKE 'slot:%'"))
    finally:
        conn.close()


def test_a_switch_is_for_the_next_that_was_checked(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _service(tmp_path)
    try:
        # Same model on another endpoint: the switch awaits a parity check before it changes anything.
        elsewhere = emb.Target("http://other:8080/v1", "nomic-test", "nomic-v1")
        svc.config.set_next(elsewhere)
        swapped = []
        original = svc.fake.embed

        def embed_then_change_next(target, texts, kind="document", titles=None):
            if not swapped:  # an administrator picks another model while the parity check runs
                swapped.append(svc.config.set_next({"model": "gemma-test", "profile": "embeddinggemma-2"}))
            return original(target, texts, kind, titles)

        svc.fake.embed = embed_then_change_next
        with pytest.raises(emb.EmbeddingConfigError) as refused:
            svc.mig.switch()
        assert refused.value.code == "nextChanged" and refused.value.status == 409
        assert svc.config.active().base_url is None, "the endpoint did not move"
        assert svc.config.next().model == "gemma-test", "the newer choice is untouched"
        assert svc.config.previous() is None
        with pytest.raises(emb.EmbeddingConfigError) as stale:
            svc.config.switch_to_next(expected=elsewhere)
        assert stale.value.code == "nextChanged"
    finally:
        _close(svc)


def test_switching_drops_the_rollback_target_it_displaces_and_a_rollback_drops_a_partly_built_next(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _service(tmp_path)
    gemma = {"model": "gemma-test", "profile": "embeddinggemma-2"}
    gemma_b = {"model": "gemma-test-b", "profile": "embeddinggemma-2"}
    try:
        svc.mig.set_next(gemma)
        svc.mig.wait(10)
        svc.mig.switch()  # active gemma-test, previous nomic-test
        assert _slot_models(svc.paths["service"]) == ["gemma-test", "nomic-test"]

        # A rollback leaves no partly built next behind.
        svc.mig.set_next(gemma_b)
        svc.mig.wait(10)
        assert _slot_models(svc.paths["service"]) == ["gemma-test", "gemma-test-b", "nomic-test"]
        svc.mig.rollback()  # active nomic-test, previous gemma-test, next abandoned
        svc.mig.wait(10)
        assert svc.config.active().model == "nomic-test" and svc.config.next() is None
        for path in svc.paths.values():
            assert _slot_models(path) == ["gemma-test", "nomic-test"], path

        # Going forward again displaces gemma-test (the older previous): its vectors go.
        svc.mig.set_next(gemma_b)
        svc.mig.wait(10)
        svc.mig.switch()  # active gemma-test-b, previous nomic-test
        svc.mig.wait(10)
        assert svc.config.previous().model == "nomic-test"
        for path in svc.paths.values():
            assert _slot_models(path) == ["gemma-test-b", "nomic-test"], path
        # The kept rollback target still serves at once.
        svc.mig.rollback()
        svc.mig.wait(10)
        svc.fake.calls.clear()
        assert "garden" in svc.retrievers["service"].search("garden", top_k=1, min_score=0.0)[0]["text"]
        assert svc.fake.calls == [("nomic-test", ["garden"])]
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
    refused_tenant = client.post("/api/internal/embedding", headers=tenant, json={"action": "switch"})
    assert refused_tenant.status_code == 400
    assert refused_tenant.json() == {"error": "service-wide endpoint: no tenant"}
    assert calls == [], "a tenant-scoped call never reaches the migration"
    assert client.post("/api/internal/embedding", headers=auth, json={"action": "nuke"}).status_code == 400
    r = client.post("/api/internal/embedding", headers=auth, json={"action": "switch"})
    assert r.status_code == 409 and r.json()["code"] == "notReady"
    target = {"model": "gemma-test", "profile": "embeddinggemma-2", "base_url": "http://embed-next:8080/v1"}
    assert client.post("/api/internal/embedding", headers=auth, json={"action": "set-next", "target": target}).status_code == 200
    assert calls == [("switch", False), ("set-next", target)]
    # Open (LAN-only) mode: nobody may change the model.
    state.auth_token = ""
    assert client.post("/api/internal/embedding", json={"action": "switch"}).status_code == 403


# ---------------- where an embeddings server may be ----------------

def _dns(table):
    def resolve(host):
        if host not in table:
            raise OSError(f"cannot resolve {host}")
        return table[host]
    return resolve


def test_the_address_check_is_positive_and_only_local_addresses_pass():
    ok = ["http://127.0.0.1:8080/v1", "http://10.1.2.3/v1", "http://192.168.0.5:8081", "http://172.31.255.255/", "http://100.64.1.1/",
          "http://[::1]:8080/v1", "http://[fd00::5]/", "http://[::ffff:127.0.0.1]/", "https://127.0.0.1/v1"]
    for url in ok:
        assert emb.check_embedding_address(url).ok, url
    refused = ["http://8.8.8.8/v1", "http://[2001:4860:4860::8888]/v1", "http://[::ffff:8.8.8.8]/", "http://0.0.0.0:8080/v1", "http://0.1.2.3/",
               "http://169.254.169.254/latest", "http://[fe80::1]/", "http://224.0.0.1/", "http://239.255.255.250/", "http://240.0.0.1/",
               "http://255.255.255.255/", "http://192.0.2.10/", "http://[::]/", "http://[::ffff:0.0.0.0]/", "http://[ff02::1]/",
               "http://[::8.8.8.8]/", "http://[64:ff9b::a00:1]/", "http://[2001:db8::1]/",
               "ftp://127.0.0.1/", "http://user:pw@127.0.0.1/", "http://127.0.0.1:notaport/", "file:///etc/passwd", "http:///", "", "embed:8080"]
    for url in refused:
        r = emb.check_embedding_address(url)
        assert not r.ok and r.reason and not r.transient, url


def test_names_are_judged_by_every_answer_and_an_unresolvable_one_is_not_a_public_one():
    resolver = _dns({"embed": ["10.0.0.6"], "mixed": ["10.0.0.8", "93.184.216.34"], "public.example": ["93.184.216.34"],
                     "linklocal": ["169.254.169.254"], "mapped": ["::ffff:8.8.8.8"], "empty": [], "private.example.": ["192.168.1.20"]})
    assert emb.check_embedding_address("http://embed:8080/v1", resolver).ok
    assert emb.check_embedding_address("http://private.example./v1", resolver).ok, "a trailing dot is a name like another"
    for name in ("mixed", "public.example", "linklocal", "mapped"):
        r = emb.check_embedding_address(f"http://{name}/v1", resolver)
        assert not r.ok and not r.transient, name
    for name in ("empty", "nowhere"):
        r = emb.check_embedding_address(f"http://{name}/v1", resolver)
        assert not r.ok and r.transient, name


def _env_config(tmp_path, **llm):
    return emb.EmbeddingConfig(cfg_for(tmp_path, **llm), None)


class _Net:
    """A fake resolver for the start-up guard: fails `failing` times, then answers."""

    def __init__(self, failing=0, answers=None):
        self.failing = failing
        self.answers = answers or {"embed": ["10.0.0.6"], "embed-next": ["10.0.0.7"]}
        self.slept, self.scheduled = [], []

    def resolve(self, host):
        if self.failing > 0 or host not in self.answers:
            self.failing -= 1
            raise OSError(f"temporary failure resolving {host}")
        return self.answers[host]

    def guard(self, config):
        emb.guard_seeded_addresses(config, check=lambda url: emb.check_embedding_address(url, self.resolve),
                                   sleep=self.slept.append, schedule=lambda fn, s: self.scheduled.append((fn, s)))


def test_a_resolver_that_is_not_up_yet_is_retried_and_the_seeded_address_used_once_it_answers(tmp_path):
    config = _env_config(tmp_path, embed_base_url="http://embed:8080/v1")
    net = _Net(failing=2)
    net.guard(config)
    assert net.slept == [1.0, 2.0] and net.scheduled == []
    assert config.active().profile == "nomic-v1" and config.active().base_url == "http://embed:8080/v1"
    assert config.damaged is None


def test_a_name_unresolvable_for_30_seconds_is_held_back_rechecked_and_used_again(tmp_path):
    config = _env_config(tmp_path, embed_base_url="http://embed:8080/v1", embed_next_model="gemma-test",
                         embed_next_profile="embeddinggemma-2", embed_next_base_url="http://embed-next:8080/v1")
    net = _Net(failing=10_000)
    net.guard(config)
    assert sum(net.slept) == 30.0
    assert config.active().profile == "unavailable" and config.next() is None
    assert "not used yet" in config.damaged
    assert [s for _fn, s in net.scheduled] == [60.0, 60.0]
    with pytest.raises(emb.EmbeddingConfigError) as locked:
        config.set_next({"model": "m"})
    assert locked.value.code == "badState"
    net.failing = 0  # the network comes up
    for fn, _s in net.scheduled[:]:
        net.scheduled.pop(0)
        fn()
    assert config.active().profile == "nomic-v1" and config.next().base_url == "http://embed-next:8080/v1"
    assert config.damaged is None and net.scheduled == [] and config.source == "environment"
    config.set_next({"model": "other"})  # changes work again


def test_a_name_that_later_resolves_to_a_public_address_stays_refused(tmp_path):
    config = _env_config(tmp_path, embed_base_url="http://embed:8080/v1")
    net = _Net(failing=10_000)
    net.guard(config)
    assert config.active().profile == "unavailable"
    net.failing = 0
    net.answers["embed"] = ["93.184.216.34"]
    fn, _s = net.scheduled.pop(0)
    fn()
    assert config.active().profile == "unavailable" and "refused" in config.damaged and net.scheduled == []
    config.release_seeded("active")
    assert config.active().profile == "unavailable", "a permanent refusal cannot be released"


def test_a_public_seeded_address_is_logged_and_not_used_without_crashing(tmp_path, caplog):
    config = _env_config(tmp_path, embed_base_url="http://[2001:4860:4860::8888]/v1", embed_next_model="gemma-test",
                         embed_next_profile="embeddinggemma-2", embed_next_base_url="http://127.0.0.1:8081/v1")
    with caplog.at_level("ERROR", logger="agent.embedding"):
        emb.guard_seeded_addresses(config)
    assert config.active().profile == "unavailable"
    with pytest.raises(emb.EmbeddingConfigError):
        emb.frame(config.active().profile, ["x"])
    assert config.next().base_url == "http://127.0.0.1:8081/v1", "a private next is left alone"
    assert any("LLM_EMBED_BASE_URL" in r.getMessage() and "refused" in r.getMessage() for r in caplog.records)
    assert config.source == "environment"


def test_an_administrators_choice_while_next_is_held_back_is_kept(tmp_path):
    config = _env_config(tmp_path, embed_next_model="gemma-test", embed_next_profile="embeddinggemma-2",
                         embed_next_base_url="http://embed-next:8080/v1")
    net = _Net(failing=10_000)
    net.guard(config)
    assert config.next() is None
    config.set_next({"model": "chosen", "base_url": "http://127.0.0.1:9000/v1"})
    net.failing = 0
    fn, _s = net.scheduled.pop(0)
    fn()
    assert config.next().model == "chosen"


def test_a_saved_configuration_is_not_second_guessed(tmp_path):
    state = tmp_path / "embedding-config.json"
    state.write_text(json.dumps({"version": 1, "active": {"model": "m", "profile": "nomic-v1", "base_url": "http://8.8.8.8/v1"}, "next": None, "previous": None}))
    config = emb.EmbeddingConfig(cfg_for(tmp_path), state)
    emb.guard_seeded_addresses(config, check=lambda url: pytest.fail("checked a saved value"))
    assert config.active().profile == "nomic-v1"


# ---------------- a fill does not write after the configuration changed ----------------

class _ProxyConn:
    """A connection whose BEGIN IMMEDIATE first runs `before_begin` (the configuration changes while the
    embeddings were being computed, after the fill's first check)."""

    def __init__(self, conn, before_begin):
        self._conn, self._before = conn, before_begin

    def execute(self, sql, *args):
        if str(sql).strip().upper().startswith("BEGIN IMMEDIATE"):
            self._before()
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_a_fill_rechecks_the_configuration_inside_its_write_transaction(tmp_path):
    _vec_or_skip(tmp_path)
    svc = _service(tmp_path)
    try:
        gemma = emb.Target(None, "gemma-test", "embeddinggemma-2")
        svc.config.set_next(gemma)
        version = svc.config.version
        real_open = svc.mig._open
        svc.mig._open = lambda path: (lambda c: c and _ProxyConn(c, svc.config.clear_next))(real_open(path))
        result = svc.mig._fill(svc.paths["service"], gemma, version)
        assert result.get("aborted") is True
        assert _slot_models(svc.paths["service"]) == ["nomic-test"], "nothing was written for the abandoned next"
    finally:
        _close(svc)
