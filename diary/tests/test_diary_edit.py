"""Diary editor tests — guarded edit-by-xid through journal + retrieval supersede.

Covers the three guarantees the master prompt asks for:
  1. the corpus file updates correctly (and only the targeted exchange changes),
  2. journal replay after a simulated crash applies the edit idempotently,
  3. retrieval no longer serves the superseded chunk next to the new one.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime

import pytest

import agent.app as appmod
from agent.config import Config
from agent.corpus import parse_diary, render_exchange, replace_exchange_text
from agent.corpus_store import CorpusError, CorpusStore
from agent.journal import Journal
from agent.retrieval import Retriever
from tests.test_journal_store import FakeWebDAV
from tests.test_retrieval_context import StubLLM

DAY = date(2026, 9, 1)


def _month_text() -> str:
    xid = "11111111-1111-4111-8111-111111111111"
    return (
        "## Tuesday, September 1, 2026\n\n"
        "### 09:15 — Morning plan\n\n"
        + render_exchange("I want to fix my sleep schedule.", "The user set a sleep goal.", xid)
        + "\n"
        + render_exchange("Also starting a run log.", "The user began tracking runs.", "22222222-2222-4222-8222-222222222222")
    )


# ---------------- format-level replacement ----------------


def test_replace_rewrites_target_and_preserves_the_rest():
    text = _month_text()
    out = replace_exchange_text(text, "11111111-1111-4111-8111-111111111111", "Fixed my words.", "New assistant line.")
    assert out is not None
    assert "**Me:** Fixed my words." in out
    assert "**Assistant:** New assistant line." in out
    # Sibling exchange untouched.
    assert "**Me:** Also starting a run log." in out
    assert "The user began tracking runs." in out
    # Structural context preserved.
    assert "### 09:15 — Morning plan" in out
    assert "## Tuesday, September 1, 2026" in out
    # The edited exchange keeps its marker (replay-safe, re-editable).
    assert "<!-- xid:11111111-1111-4111-8111-111111111111 -->" in out
    assert "<!-- xid:22222222-2222-4222-8222-222222222222 -->" in out


def test_replace_parses_back_to_exactly_two_exchanges():
    out = replace_exchange_text(_month_text(), "11111111-1111-4111-8111-111111111111", "a", "b")
    days = parse_diary(out)
    assert len(days[0].subsections[0].exchanges) == 2
    ex = days[0].subsections[0].exchanges[0]
    assert (ex.me, ex.claude, ex.xid) == ("a", "b", "11111111-1111-4111-8111-111111111111")


def test_replace_survives_marker_inside_claude_layout():
    """A hand-mangled file where the marker sits right after the Me: block
    (no assistant section) still targets the right exchange."""
    xid = "33333333-3333-4333-8333-333333333333"
    text = (
        "### 10:00 — t\n\n"
        f"**Me:** only me here\n\n<!-- xid:{xid} -->\n\n"
        "**Me:** next exchange\n\n**Assistant:** reply\n\n<!-- xid:44444444-4444-4444-8444-444444444444 -->\n"
    )
    out = replace_exchange_text(text, xid, "rewritten", "")
    assert out is not None
    assert "**Me:** rewritten" in out
    assert "**Me:** next exchange" in out  # sibling intact
    assert "**Assistant:** reply" in out


def test_replace_unknown_or_malformed_returns_none():
    assert replace_exchange_text(_month_text(), "99999999-9999-4999-8999-999999999999", "a", "b") is None
    # Marker orphaned from any Me: opener — refuse rather than guess.
    assert replace_exchange_text("### t\n\njust a marker <!-- xid:11111111-1111-4111-8111-111111111111 -->\n",
                                 "11111111-1111-4111-8111-111111111111", "a", "b") is None


def test_replace_empty_assistant_drops_assistant_block():
    xid = "11111111-1111-4111-8111-111111111111"
    out = replace_exchange_text(_month_text(), xid, "just me", "")
    days = parse_diary(out)
    ex = days[0].subsections[0].exchanges[0]
    assert ex.me == "just me"
    assert ex.claude == ""


# ---------------- store-level: journaled, guarded, idempotent ----------------


def _store(tmp_path, dav=None, layout="monthly"):
    cfg = Config({
        "corpus": {
            "webdav": {"remote_root": ""},
            "monthly_prefix": "",
            "index_file": "INDEX.md",
            "entry_layout": layout,
            "month_file_template": "{year}-{month02}.md",
        }
    })
    return CorpusStore(cfg, dav or FakeWebDAV(), Journal(tmp_path / "j.db"))


def test_edit_exchange_updates_corpus_and_returns_location(tmp_path):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "Morning plan", "I want to fix my sleep.", "The user set a goal.", now=datetime(2026, 9, 1, 9, 15))
    path, day_iso = st.edit_exchange(xid, "Edited words.", "Edited assistant.", month="2026-09")
    assert day_iso == DAY.isoformat()
    text, _ = st.read_month(DAY)
    assert "**Me:** Edited words." in text
    assert "I want to fix my sleep." not in text
    assert path == st.document_path(DAY)


def test_edit_is_idempotent_on_replay(tmp_path):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "t", "original", "reply", now=datetime(2026, 9, 1, 9, 0))
    st.edit_exchange(xid, "edited", "reply2", month="2026-09")
    # Simulate a replay (crash between apply and mark_applied): re-run the applier
    # over the journal. The mutate must report "no change" the second time.
    st.apply_pending()
    text, _ = st.read_month(DAY)
    assert text.count("**Me:** edited") == 1


def test_edit_replays_after_simulated_crash_before_apply(tmp_path):
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "t", "original", "reply", now=datetime(2026, 9, 1, 9, 0))
    # Crash right after enqueue, before any apply: journal has the intent only.
    st.journal.enqueue("exchange_edit", {"xid": xid, "new_me": "crash-proof", "new_claude": "r", "month": "2026-09"})
    # "Reboot": fresh store over the same journal + backend state.
    st2 = CorpusStore(st.cfg, st.backend, Journal(tmp_path / "j.db"))
    st2.apply_pending()
    text, _ = st2.read_month(DAY)
    assert text.count("**Me:** crash-proof") == 1


def test_edit_survives_write_failure_mid_apply(tmp_path):
    """Crash AFTER the PUT lands but before mark_applied: replay must not duplicate."""
    dav = FakeWebDAV()
    st = _store(tmp_path, dav)
    xid = st.log_exchange(DAY, "t", "original", "reply", now=datetime(2026, 9, 1, 9, 0))
    st.edit_exchange(xid, "edited", "r", month="2026-09")
    # Force a replay of an already-applied edit (as a crash-before-mark would).
    st.journal.enqueue("exchange_edit", {"xid": xid, "new_me": "edited", "new_claude": "r", "month": "2026-09"})
    st.apply_pending()
    text, _ = st.read_month(DAY)
    assert text.count("**Me:** edited") == 1
    assert dav.put  # backend was exercised


def test_edit_unknown_xid_fails_fast_without_journal_entry(tmp_path):
    st = _store(tmp_path)
    with pytest.raises(CorpusError):
        st.edit_exchange("99999999-9999-4999-8999-999999999999", "x", "y", month="2026-09")
    assert st.journal.pending_count() == 0


def test_edit_daily_layout_finds_the_day_file(tmp_path):
    st = _store(tmp_path, layout="daily")
    xid = st.log_exchange(DAY, "t", "original", "reply", now=datetime(2026, 9, 1, 9, 0))
    path, _ = st.edit_exchange(xid, "daily edit", "r", month="2026-09")
    assert path == st.daily_path(DAY)
    text, _ = st.backend.get_text(path)
    assert "**Me:** daily edit" in text


# ---------------- retrieval supersede ----------------


def test_retrieval_drops_superseded_chunk_after_edit(tmp_path):
    st = _store(tmp_path)
    llm = StubLLM()
    retr = Retriever(tmp_path / "idx.db", llm)
    xid = st.log_exchange(DAY, "Morning plan", "I want to fix my sleep schedule badly.", "The user set a sleep goal.", now=datetime(2026, 9, 1, 9, 15))
    path = st.document_path(DAY)
    month_text, _ = st.read_month(DAY)
    retr.reindex_file(path, month_text)
    before = retr._conn.execute("SELECT COUNT(*) FROM chunks WHERE file = ?", (path,)).fetchone()[0]
    assert before >= 1

    st.edit_exchange(xid, "I fixed my sleep schedule completely.", "The user solved sleep.", month="2026-09")
    month_text2, _ = st.read_month(DAY)
    retr.reindex_file(path, month_text2)

    # The old chunk's hash is gone from the index: the corrected text is the
    # only searchable version of that exchange.
    old_hash = hashlib.sha256(
        f"[{DAY.isoformat()}] ## Tuesday, September 1, 2026\n### 09:15 — Morning plan\n**Me:** I want to fix my sleep schedule badly.\n\n**Assistant:** The user set a sleep goal.".encode()
    ).hexdigest()
    remaining = retr._conn.execute("SELECT content_hash FROM chunks WHERE file = ?", (path,)).fetchall()
    assert all(r["content_hash"] != old_hash for r in remaining)
    retr.close()


# ---------------- endpoint ----------------


class _StubAux:
    def chat(self, *a, **k):
        return "unused"

    def close(self):
        pass


class _StubMain:
    def chat(self, messages, **kwargs):
        return "A warm, honest reply.\n\n[LOG: ok]"


@pytest.fixture
def client(tmp_path, monkeypatch):
    # AppState pulls its config from the module-level _base_cfg, which is
    # loaded from DIARY_CONFIG at import time — before any monkeypatch.setenv
    # can run. Patch the module attribute directly so the app under test uses
    # this tmp config (monthly layout) instead of the repo's default file.
    from agent.config import Config as _Cfg

    cfg = _Cfg({
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
    _FakeRetriever.calls = []

    def patched_init(self, cfg_inner):
        from pathlib import Path as _P

        self.cfg = cfg_inner
        self.auth_token = (cfg_inner.get("ui.auth_token") or "").strip()
        self.journal = appmod.Journal(_P(cfg_inner.get("retrieval.db_path")))
        self.dav = fake
        self.llm_main = _StubMain()
        self.llm_aux = _StubAux()
        self.store = appmod.CorpusStore(cfg_inner, self.dav, self.journal)
        self.retrieval = _FakeRetriever()
        self.assembler = appmod.ContextAssembler(self.store, self.retrieval, cfg_inner)
        self.assembler._system_prompt = lambda: "SYSTEM RULES"
        self.pipeline = appmod.LoggingPipeline(self.store, self.llm_main, self.llm_aux, templates={})
        self.pipeline._aux_call = lambda key, sysp, **fields: (
            "LOG" if key == "skip_classifier" else ("summary prose" if key == "summarizer" else "NO")
        )

    monkeypatch.setattr(appmod.AppState, "__init__", patched_init)
    from fastapi.testclient import TestClient

    return TestClient(appmod.app)


class _FakeRetriever:
    """Records reindex calls so tests can assert the endpoint reindexed.

    The call log is class-level: each AppState (global and per-tenant) builds
    its own retriever instance, but the assertions want everything recorded
    regardless of which instance served the request.
    """

    calls: list = []

    def reindex_file(self, file_name, month_text, day_of_month=None):
        type(self).calls.append((file_name, month_text))
        return 0

    def search(self, *a, **k):
        return []

    def stats(self):
        return {"chunks": 0, "vec_available": False, "dim": None}

    def close(self):
        pass


def test_edit_endpoint_updates_corpus_and_reindexes(client):
    r = client.post("/api/chat", json={"message": "Feeling good about the new agent."})
    assert r.status_code == 200
    logged_xid = r.json()["xid"]
    st = appmod.get_state()

    r2 = client.post("/api/entries/edit", json={"xid": logged_xid, "me": "Corrected words.", "assistant": "Summary prose.", "month": datetime.now().strftime("%Y-%m")})
    assert r2.status_code == 200
    body = r2.json()
    assert body["ok"] is True
    # Monthly layout anchors documents to the month's first day.
    assert body["day"].startswith(datetime.now().strftime("%Y-%m"))

    text, _ = st.store.read_month(datetime.now().date())
    assert "**Me:** Corrected words." in text
    assert "Feeling good about the new agent." not in text
    # Retrieval was refreshed with the edited document text (the chat round
    # trip also triggers a background reindex; either way the edited text is
    # the latest thing indexed).
    assert _FakeRetriever.calls, "endpoint never reindexed after the edit"
    assert any("Corrected words." in (t or "") for _, t in _FakeRetriever.calls)


def test_edit_endpoint_validates_input(client):
    assert client.post("/api/entries/edit", json={"xid": "not an xid!", "me": "x"}).status_code == 400
    assert client.post("/api/entries/edit", json={"xid": "11111111-1111-4111-8111-111111111111", "me": ""}).status_code == 400
    r = client.post("/api/entries/edit", json={"xid": "99999999-9999-4999-8999-999999999999", "me": "x"})
    assert r.status_code == 404
