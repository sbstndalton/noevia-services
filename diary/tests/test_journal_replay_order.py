"""Regression: journal replay must order by rowid (insertion order), not created_at.

A local clock step-back (e.g. NTP correction) can give an older-inserted entry a
created_at timestamp *earlier* than an entry inserted before it. If replay orders by
created_at, an older exchange_edit can replay after — and overwrite — a newer one.
rowid is monotonic insertion order regardless of wall-clock jumps, so ordering by it
alone is correct; created_at is kept only for display.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from agent.journal import Journal


def test_replay_orders_by_rowid_even_with_reversed_created_at():
    with tempfile.TemporaryDirectory() as tmp:
        journal = Journal(Path(tmp) / "journal.sqlite3")
        first = journal.enqueue("exchange", {"text": "first, inserted first"})
        second = journal.enqueue("exchange", {"text": "second, inserted second"})

        # Simulate a clock step-back: the second (later-inserted) entry ends up with an
        # earlier created_at than the first.
        with journal._conn:
            journal._conn.execute(
                "UPDATE journal SET created_at = '2020-01-01T00:00:00' WHERE id = ?", (second,)
            )
            journal._conn.execute(
                "UPDATE journal SET created_at = '2030-01-01T00:00:00' WHERE id = ?", (first,)
            )

        order = [entry.id for entry in journal.unapplied()]
        assert order == [first, second], (
            "replay must follow insertion order (rowid), not the (now reversed) created_at"
        )


def test_unapplied_still_reports_created_at_for_display():
    with tempfile.TemporaryDirectory() as tmp:
        journal = Journal(Path(tmp) / "journal.sqlite3")
        jid = journal.enqueue("exchange", {"text": "hello"})
        [entry] = journal.unapplied()
        assert entry.id == jid
        assert entry.created_at  # still populated, just not used for ordering
