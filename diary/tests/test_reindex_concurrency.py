"""Reindex vs. journal/edit concurrency (#858, #859). Synthetic text only."""
from __future__ import annotations

import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

import agent.app as appmod
from agent.journal import Journal
from agent.retrieval import Retriever
from tests.test_diary_edit import DAY, _store
from tests.test_retrieval_context import DAY_TEXT, StubLLM

import agent.retrieval as retrieval_mod

_real_load_vec = retrieval_mod._load_vec_extension


@pytest.fixture(autouse=True)
def _vec_enabled(monkeypatch):
    """Exercise the embedding path: this venv's sqlite3 refuses extensions
    unless loading is enabled on the connection first (test-only)."""
    def load(conn):
        try:
            conn.enable_load_extension(True)
        except Exception:  # noqa: BLE001 — builds without extension support
            pass
        return _real_load_vec(conn)
    monkeypatch.setattr(retrieval_mod, "_load_vec_extension", load)


TWO_CHUNKS = DAY_TEXT + """
### 18:40 — Evening walk

**Me:** Walked by the river with a synthetic friend.

**Claude:** The user took an evening walk.
"""


class SlowLLM(StubLLM):
    """Blocks inside one embed call until released (a slow or cold embed server)."""

    def __init__(self, block_on_call):
        super().__init__()
        self.block_on_call = block_on_call
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed(self, texts, model=None):
        if self.calls + 1 == self.block_on_call:
            self.entered.set()
            assert self.release.wait(30), "test never released the embedder"
        return super().embed(texts, model=model)


@pytest.mark.parametrize("scenario", ["new_file", "edited_file"])
def test_journal_writes_succeed_while_reindex_waits_on_embeddings(tmp_path, scenario):
    """#858: reindex used to open its delete/insert transaction before
    embedding, so a slow embed server held index.db's write lock and the
    journal (same file, second connection) failed with 'database is locked'."""
    db = tmp_path / "index.db"
    llm = SlowLLM(block_on_call=2 if scenario == "new_file" else 1)
    retriever = Retriever(db, llm)
    if not retriever.vec_available:
        pytest.skip("sqlite-vec not available in this environment")
    if scenario == "edited_file":
        llm.block_on_call = 3  # after the first full index (two chunks)
        assert retriever.reindex_file("2026-08.md", TWO_CHUNKS) == 2
        text = TWO_CHUNKS.replace("synthetic friend", "synthetic neighbour")  # one stale row to delete
    else:
        text = TWO_CHUNKS
    journal = Journal(db)
    journal._conn.execute("PRAGMA busy_timeout = 1000")  # fail fast if the lock is held
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("n", retriever.reindex_file("2026-08.md", text)))
    worker.start()
    try:
        assert llm.entered.wait(10), "reindex never reached the embedder"
        jid = journal.enqueue("exchange", {"xid": "synthetic", "text": "durable while embedding"})
        journal.mark_dirty("2026-08.md")
        journal.clear_dirty("2026-08.md")
        assert journal.pending_count() == 1 and not journal.is_applied(jid)
    finally:
        llm.release.set()
        worker.join(30)
    assert result["n"] == (2 if scenario == "new_file" else 1)
    bodies = [r[0] for r in retriever._conn.execute("SELECT body FROM chunks WHERE file='2026-08.md'")]
    assert len(bodies) == 2
    if scenario == "edited_file":
        assert all("synthetic friend" not in b for b in bodies)
        assert any("synthetic neighbour" in b for b in bodies)
    assert retriever.search("evening walk by the river", min_score=-1)
    journal.close()
    retriever.close()


def test_failed_embed_during_apply_keeps_retry_semantics(tmp_path):
    """The split plan/apply keeps the old contract: a failed embed inserts no
    dedupe row and reports pending_embeddings so _reindex_dirty keeps the
    document dirty."""

    class FailSecond(StubLLM):
        def embed(self, texts, model=None):
            if self.calls == 1:
                self.calls += 1
                raise RuntimeError("synthetic embed outage")
            return super().embed(texts, model=model)

    retriever = Retriever(tmp_path / "index.db", FailSecond())
    if not retriever.vec_available:
        pytest.skip("sqlite-vec not available in this environment")
    assert retriever.reindex_file("2026-08.md", TWO_CHUNKS) == 1
    assert retriever.pending_embeddings is True
    assert retriever.stats()["chunks"] == 1
    assert retriever.reindex_file("2026-08.md", TWO_CHUNKS) == 1
    assert retriever.pending_embeddings is False
    assert retriever.stats()["chunks"] == 2
    retriever.close()


def test_background_reindex_does_not_restore_pre_edit_text(tmp_path):
    """#859: _reindex_today read the document without the store write lock;
    an edit saved and reindexed (dirty flag cleared) while it was embedding
    was then overwritten in the index with the pre-edit text."""
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "Morning plan", "Synthetic pre-edit words about tea.", "reply", now=datetime(2026, 9, 1, 9, 15))
    ns = SimpleNamespace(store=st, journal=st.journal)

    class EditWhileEmbedding(StubLLM):
        fired = False

        def embed(self, texts, model=None):
            if not self.fired and "pre-edit words" in texts[0]:
                type(self).fired = True
                st.edit_exchange(xid, "Synthetic corrected words about coffee.", "reply", month="2026-09")
                appmod._reindex_dirty(ns)
                assert st.journal.dirty_documents() == []
            return super().embed(texts, model=model)

    ns.retrieval = Retriever(tmp_path / "index.db", EditWhileEmbedding())
    if not ns.retrieval.vec_available:
        pytest.skip("sqlite-vec not available in this environment")
    appmod._reindex_today(ns, DAY)
    assert EditWhileEmbedding.fired, "the edit never ran during the background reindex"
    bodies = [r[0] for r in ns.retrieval._conn.execute("SELECT body FROM chunks")]
    assert bodies and all("pre-edit words" not in b for b in bodies)
    assert any("corrected words about coffee" in b for b in bodies)
    ns.retrieval.close()


def test_background_reindex_indexes_unchanged_document(tmp_path):
    st = _store(tmp_path)
    st.log_exchange(DAY, "Morning plan", "Synthetic note about a bicycle.", "reply", now=datetime(2026, 9, 1, 9, 15))
    ns = SimpleNamespace(store=st, journal=st.journal, retrieval=Retriever(tmp_path / "index.db", StubLLM()))
    appmod._reindex_today(ns, DAY)
    bodies = [r[0] for r in ns.retrieval._conn.execute("SELECT body FROM chunks")]
    assert any("bicycle" in b for b in bodies)
    ns.retrieval.close()
