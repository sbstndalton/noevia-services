from datetime import date

import pytest

from agent.config import Config
from agent.corpus_store import CorpusStore
from agent.journal import Journal
from agent.local_storage import LocalCorpusBackend
from agent.storage import create_backend


def test_local_backend_create_update_and_conflict(tmp_path):
    backend = LocalCorpusBackend(str(tmp_path / "corpus"))
    ok, version1, status = backend.put("notes/a.md", b"one")
    assert (ok, status) == (True, 201)
    assert backend.get_text("notes/a.md") == ("one", version1)

    ok, _, status = backend.put("notes/a.md", b"stale", if_match='"stale"')
    assert (ok, status) == (False, 412)
    ok, version2, status = backend.put("notes/a.md", b"two", if_match=version1)
    assert (ok, status) == (True, 204)
    assert version2 != version1
    assert backend.get_text("notes/a.md")[0] == "two"
    assert not list((tmp_path / "corpus" / "notes").glob("*.tmp"))


def test_local_backend_rejects_escape_and_external_symlink(tmp_path):
    backend = LocalCorpusBackend(str(tmp_path / "corpus"))
    with pytest.raises(ValueError):
        backend.get_text("../outside.md")
    outside = tmp_path / "outside"
    outside.mkdir()
    (backend.root / "external").symlink_to(outside, target_is_directory=True)
    assert backend.list_dir("") == []


def test_local_backend_lists_direct_children(tmp_path):
    backend = LocalCorpusBackend(str(tmp_path / "corpus"))
    backend.put("2026-09.md", b"month")
    (backend.root / "nested").mkdir()
    entries = backend.list_dir("")
    assert [(item["name"], item["is_dir"]) for item in entries] == [
        ("2026-09.md", False),
        ("nested", True),
    ]


def test_journal_replay_with_local_backend(tmp_path):
    backend = LocalCorpusBackend(str(tmp_path / "corpus"))
    cfg = Config({"corpus": {"root": "", "monthly_prefix": "", "index_file": "INDEX.md"}})
    journal = Journal(tmp_path / "index.db")
    store = CorpusStore(cfg, backend, journal)
    store.log_exchange(date(2026, 9, 3), "Local", "hello", "saved")
    store.apply_pending()
    text, _ = store.read_month(date(2026, 9, 3))
    assert text.count("**Me:** hello") == 1
    assert "**Assistant:** saved" in text
    assert journal.pending_count() == 0


def test_backend_factory_defaults_to_local(tmp_path):
    cfg = Config({"corpus": {"local": {"root": str(tmp_path)}}})
    assert isinstance(create_backend(cfg), LocalCorpusBackend)


def test_daily_layout_reads_writes_and_lists_months(tmp_path):
    backend = LocalCorpusBackend(str(tmp_path / "corpus"))
    cfg = Config({
        "corpus": {
            "root": "",
            "entry_layout": "daily",
            "entries_prefix": "Entries",
            "index_enabled": False,
        }
    })
    store = CorpusStore(cfg, backend, Journal(tmp_path / "daily.db"))

    store.log_exchange(date(2026, 8, 31), "Evening", "August note", "Saved")
    store.log_exchange(date(2026, 9, 1), "Morning", "First September note", "Saved")
    store.log_exchange(date(2026, 9, 4), "Later", "Fourth September note", "Saved")

    assert store.daily_path(date(2026, 9, 4)) == "Entries/2026/September/September 4, 2026.md"
    assert backend.exists(store.daily_path(date(2026, 9, 4)))
    september = store.read_month_text(2026, 9)
    assert "First September note" in september
    assert "Fourth September note" in september
    assert "August note" not in september
    assert "xid" not in september
    assert [item["id"] for item in store.list_months()] == ["2026-08", "2026-09"]
    assert "Fourth September note" in store.get_day_text(date(2026, 9, 4))
    assert not backend.exists("INDEX.md")
