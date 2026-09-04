"""Integration-style tests for CorpusStore against a fake WebDAV backend.

The fake mirrors the Nextcloud semantics the real client relies on:
GET returns bytes + ETag, PUT honors If-Match with 412 on mismatch.
"""
from datetime import date

import pytest

from agent import corpus as fmt
from agent.config import Config
from agent.corpus_store import CorpusStore
from agent.journal import Journal


class FakeWebDAV:
    """In-memory WebDAV stand-in with ETag + If-Match semantics."""

    def __init__(self):
        self.files = {}
        self._next = 0
        self.get_calls = 0
        self.conflicts_injected = 0

    def get_text(self, path):
        self.get_calls += 1
        if path not in self.files:
            return None, None
        body, etag = self.files[path]
        return body, etag

    def put(self, path, data, if_match=None, if_none_match="*", max_retries=5):
        # Simulate a concurrent writer landing just before our PUT: bumps the ETag
        # and prepends a concurrent edit, so our If-Match precondition fails (412).
        if self.conflicts_injected > 0:
            self.conflicts_injected -= 1
            current = self.files.get(path)
            base = current[0] if current else ""
            self._next += 1
            # concurrent edit lands as its own line, preserving existing structure
            self.files[path] = (f"(concurrent edit)\n{base}", f'"{self._next}"')
        current = self.files.get(path)
        current_etag = current[1] if current else None
        if if_match is not None and if_match != current_etag:
            return False, None, 412
        self._next += 1
        etag = f'"{self._next}"'
        self.files[path] = (data.decode("utf-8"), etag)
        return True, etag, 204


@pytest.fixture
def store(tmp_path):
    cfg = Config({"corpus": {"webdav": {"remote_root": "Diary"}, "monthly_prefix": "", "index_file": "INDEX.md"}})
    dav = FakeWebDAV()
    journal = Journal(tmp_path / "j.db")
    return CorpusStore(cfg, dav, journal)


def test_log_exchange_creates_month_index_and_is_idempotent(store):
    day = date(2026, 9, 1)
    store.log_exchange(day=day, sub_header="first", me_text="Hello diary.", claude_text="The user opened the diary.", now=None)

    month_text, _ = store.read_month(day)
    assert "## Tuesday, September 1, 2026" in month_text
    assert "**Me:** Hello diary." in month_text
    assert "The user opened the diary." in month_text

    # month registered in INDEX.md
    index_text, _ = store.read_index()
    assert "[September 2026](2026-09.md)" in index_text

    # replay every unapplied-again entry (simulated crash before mark_applied)
    # by re-running appliers on the same journal payload: content must NOT duplicate
    for entry in store.journal.unapplied():
        pass  # none unapplied — already applied

    pending_before = store.journal.pending_count()
    store.apply_pending()
    assert store.journal.pending_count() == pending_before == 0

    month_text2, _ = store.read_month(day)
    assert month_text2.count("**Me:** Hello diary.") == 1


def test_replay_after_simulated_crash_does_not_duplicate(store):
    day = date(2026, 9, 2)
    # enqueue but do not apply (simulates crash after journal write, before apply)
    xid = store.new_xid()
    body = fmt.render_exchange("me", "claude", xid)
    store.journal.enqueue("exchange", {
        "xid": xid, "day": day.isoformat(), "sub_header": "### 09:00",
        "body": body, "month": store.month_filename(day),
        "month_label": store.month_label(day),
    })
    assert store.journal.pending_count() == 1

    # first apply writes the file
    store.apply_pending()
    text1, _ = store.read_month(day)
    assert xid in text1

    # crash scenario: entry unapplied again (marker present in file) -> replay skips
    store.journal.enqueue("exchange", {
        "xid": xid, "day": day.isoformat(), "sub_header": "### 09:00",
        "body": body, "month": store.month_filename(day),
        "month_label": store.month_label(day),
    })
    store.apply_pending()
    text2, _ = store.read_month(day)
    assert text2.count("**Me:** me") == 1  # marker dedupe prevented duplication


def test_conflict_retry_merges_on_fresh_content(store):
    day = date(2026, 9, 3)
    # seed the month file with a first, non-conflicting exchange
    store.log_exchange(day=day, sub_header="first", me_text="seed", claude_text="logged")
    # next write hits a simulated concurrent edit: first PUT gets 412, then the
    # store re-GETs fresh content (including the concurrent edit) and succeeds.
    store.dav.conflicts_injected = 1
    store.log_exchange(day=day, sub_header="second", me_text="me text", claude_text="claude text")
    text, _ = store.read_month(day)
    assert "**Me:** me text" in text
    assert "(concurrent edit)" in text  # concurrent edit preserved, not clobbered
    assert text.index("(concurrent edit)") < text.index("**Me:** me text")


def test_standing_section_updates_and_dedupe(store):
    day = date(2026, 9, 4)
    jid1 = store.update_standing_sections(
        [{"action": "add", "text": "New open question?"}],
        [{"action": "add", "date": day.isoformat(), "text": "Key event"}],
        day.isoformat(),
    )
    assert jid1 is not None
    idx_text, _ = store.read_index()
    assert "- [ ] New open question?" in idx_text
    assert f"- **{day.isoformat()}** — Key event" in idx_text

    # identical edit op applied again -> no change (bullet equality dedupe)
    before, _ = store.read_index()
    store.update_standing_sections(
        [{"action": "add", "text": "New open question?"}], [], day.isoformat()
    )
    after, _ = store.read_index()
    assert before == after


def test_get_day_text_returns_only_today(store):
    d1, d2 = date(2026, 9, 5), date(2026, 9, 6)
    store.log_exchange(day=d1, sub_header="day one", me_text="first day", claude_text="logged")
    store.log_exchange(day=d2, sub_header="day two", me_text="second day", claude_text="logged")
    today_text = store.get_day_text(d2)
    assert "second day" in today_text
    assert "first day" not in today_text
    assert "<!-- xid:" not in today_text  # markers stripped for model context


def test_monthly_rollover(store):
    august = date(2026, 8, 31)
    september = date(2026, 9, 1)
    store.log_exchange(day=august, sub_header="aug", me_text="aug entry", claude_text="logged")
    store.log_exchange(day=september, sub_header="sep", me_text="sep entry", claude_text="logged")
    aug_text, _ = store.read_month(august)
    sep_text, _ = store.read_month(september)
    assert "aug entry" in aug_text and "sep entry" not in aug_text
    assert "sep entry" in sep_text and "aug entry" not in sep_text
    idx_text, _ = store.read_index()
    assert "[August 2026](2026-08.md)" in idx_text
    assert "[September 2026](2026-09.md)" in idx_text
