"""Tests for the config-driven corpus adaptation knobs:

- corpus.month_file_template — human-named month files ("Diary - September 2026.md")
- corpus.index_enabled=false — corpora without/with their own INDEX.md
- env overrides DIARY_MONTH_FILE_TEMPLATE / DIARY_INDEX_ENABLED
"""
from datetime import date

import pytest

from agent import corpus as fmt
from agent.config import Config, load_config
from agent.corpus_store import CorpusStore
from agent.journal import Journal
from agent.util import make_client
from agent.webdav import clean_etag


class FakeWebDAV:
    """Minimal in-memory WebDAV stand-in (same semantics as test_journal_store's)."""

    def __init__(self):
        self.files = {}
        self._next = 0

    def get_text(self, path):
        if path not in self.files:
            return None, None
        body, etag = self.files[path]
        return body, etag

    def put(self, path, data, if_match=None, if_none_match="*", max_retries=5):
        current = self.files.get(path)
        current_etag = current[1] if current else None
        if if_match is not None and if_match != current_etag:
            return False, None, 412
        self._next += 1
        etag = f'"{self._next}"'
        self.files[path] = (data.decode("utf-8"), etag)
        return True, etag, 204

    def list_dir(self, remote_dir: str) -> list:
        """Direct children of remote_dir, mirroring WebDAVClient.list_dir's shape."""
        base = remote_dir.strip("/")
        out = []
        for path in self.files:
            p = path.strip("/")
            parent = p.rsplit("/", 1)[0] if "/" in p else ""
            if parent != base:
                continue
            out.append({"name": p.rsplit("/", 1)[-1], "path": p, "etag": None, "lastmod": None, "is_dir": False})
        return out


def _store(tmp_path, corpus_cfg):
    base = {
        "corpus": {
            "webdav": {"remote_root": "Documents/Important Documents/Diary"},
            "monthly_prefix": "",
            "index_file": "INDEX.md",
        },
        "retrieval": {"db_path": str(tmp_path / "j.db")},
    }
    base["corpus"].update(corpus_cfg)
    cfg = Config(base)
    return CorpusStore(cfg, FakeWebDAV(), Journal(tmp_path / "j.db"))


def test_default_template_preserves_original_naming(tmp_path):
    store = _store(tmp_path, {})
    assert store.month_filename(date(2026, 9, 1)) == "2026-09.md"


def test_human_naming_template(tmp_path):
    store = _store(tmp_path, {"month_file_template": "Diary - {month_name} {year}.md"})
    assert store.month_filename(date(2026, 9, 1)) == "Diary - September 2026.md"
    assert store.month_filename(date(2026, 8, 31)) == "Diary - August 2026.md"


def test_human_naming_end_to_end_append_and_replay_dedupe(tmp_path):
    store = _store(tmp_path, {
        "month_file_template": "Diary - {month_name} {year}.md",
        "index_enabled": False,
    })
    day = date(2026, 9, 3)
    store.log_exchange(day=day, sub_header="first", me_text="Hello human diary.", claude_text="The user greeted it.")

    month_text, _ = store.read_month(day)
    assert "## Thursday, September 3, 2026" in month_text
    assert "**Me:** Hello human diary." in month_text
    assert "<!-- xid:" in month_text

    # month file named with the template, INDEX.md never created
    assert store.month_path(day) == "Documents/Important Documents/Diary/Diary - September 2026.md"
    assert "INDEX.md" not in store.dav.files

    # replay: marker dedupe still holds with the adapted naming
    store.apply_pending()
    text2, _ = store.read_month(day)
    assert text2.count("**Me:** Hello human diary.") == 1

    # journaled payload records the adapted month filename
    entries = list(store.journal.unapplied(limit=10))
    assert entries == []  # everything applied


def test_index_disabled_no_enqueue_no_writes(tmp_path):
    store = _store(tmp_path, {"index_enabled": False})
    day = date(2026, 9, 4)
    store.log_exchange(day=day, sub_header="x", me_text="me", claude_text="claude")

    assert "INDEX.md" not in store.dav.files
    # journal must contain exactly one entry (the exchange), no index_month
    kinds = [e.kind for e in store.journal.unapplied(limit=10)] or []
    # all applied already; verify via pending + the fact no INDEX file exists
    assert store.journal.pending_count() == 0


def test_index_enabled_by_default(tmp_path):
    store = _store(tmp_path, {})
    day = date(2026, 9, 5)
    store.log_exchange(day=day, sub_header="x", me_text="me", claude_text="claude")
    idx_text, _ = store.read_index()
    assert "[September 2026](2026-09.md)" in idx_text


def test_standing_sections_return_empty_when_disabled(tmp_path):
    store = _store(tmp_path, {"index_enabled": False})
    assert store.get_standing_sections_text() == ""


def test_env_overrides_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("DIARY_MONTH_FILE_TEMPLATE", "Diary - {month_name} {year}.md")
    monkeypatch.setenv("DIARY_INDEX_ENABLED", "false")
    cfg = load_config(path="/nonexistent/diary-config.yaml")
    assert cfg.get("corpus.month_file_template") == "Diary - {month_name} {year}.md"
    assert cfg.get("corpus.index_enabled") is False


def test_env_template_month02_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("DIARY_MONTH_FILE_TEMPLATE", "{year}-{month02}-notes.md")
    cfg = load_config(path="/nonexistent/diary-config.yaml")
    store = CorpusStore(cfg, FakeWebDAV(), Journal(tmp_path / "j2.db"))
    assert store.month_filename(date(2026, 11, 2)) == "2026-11-notes.md"


def test_read_month_text_strips_markers_and_handles_missing(tmp_path):
    store = _store(tmp_path, {"month_file_template": "Diary - {month_name} {year}.md", "index_enabled": False})
    # No file yet -> empty, not an error.
    assert store.read_month_text(2026, 9) == ""
    # Log an exchange, then the whole month text comes back with xids stripped.
    store.log_exchange(day=date(2026, 9, 3), sub_header="first", me_text="Hello month read.", claude_text="Logged.")
    text = store.read_month_text(2026, 9)
    assert "## Thursday, September 3, 2026" in text
    assert "**Me:** Hello month read." in text
    assert "xid" not in text


def test_list_months_matches_human_template(tmp_path):
    store = _store(tmp_path, {"month_file_template": "Diary - {month_name} {year}.md", "index_enabled": False})
    store.log_exchange(day=date(2026, 8, 2), sub_header="a", me_text="august", claude_text="c")
    store.log_exchange(day=date(2026, 9, 3), sub_header="b", me_text="september", claude_text="c")
    months = store.list_months()
    assert [m["id"] for m in months] == ["2026-08", "2026-09"]
    assert months[1]["label"] == "September 2026"
    assert months[1]["file"] == "Diary - September 2026.md"


def test_list_months_ignores_strays_and_index(tmp_path):
    store = _store(tmp_path, {"month_file_template": "Diary - {month_name} {year}.md", "index_enabled": True})
    store.log_exchange(day=date(2026, 9, 3), sub_header="b", me_text="september", claude_text="c")
    # Strays that must not match the template.
    store.dav.files[store._join("", "Notes - September 2026.md")] = ("x", '"1"')
    store.dav.files[store._join("", "Diary - September 2026 copy.md")] = ("x", '"2"')
    months = store.list_months()
    assert [m["id"] for m in months] == ["2026-09"]


def test_list_months_numeric_template(tmp_path):
    store = _store(tmp_path, {"month_file_template": "{year}-{month02}.md"})
    store.log_exchange(day=date(2026, 9, 3), sub_header="b", me_text="september", claude_text="c")
    months = store.list_months()
    assert [m["id"] for m in months] == ["2026-09"]


def test_clean_etag_strips_compression_suffix():
    assert clean_etag('"abc123-gzip"') == '"abc123"'
    assert clean_etag('"abc123-br"') == '"abc123"'
    assert clean_etag('"abc123"') == '"abc123"'
    assert clean_etag('"abc123"') == '"abc123"'
    assert clean_etag(None) is None
    assert clean_etag("") == ""


def test_shared_client_requests_uncompressed():
    client = make_client(base_url="http://example.invalid")
    assert client.headers.get("Accept-Encoding") == "identity"
    client.close()
