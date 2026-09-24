"""Regression for #177: mark_failed must be @synchronized like every other
write method on Journal, since the underlying sqlite connection is shared
across threads with check_same_thread=False.

Before the fix, concurrent mark_failed() + append() calls could race on the
shared connection and raise (or, in the worst case, corrupt the journal
table). This drives both methods hard from separate threads and asserts
nothing raises and every write is durably recorded.
"""
import threading

from agent.journal import Journal


def test_concurrent_mark_failed_and_enqueue_do_not_raise(tmp_path):
    journal = Journal(tmp_path / "journal.sqlite3")
    jid = journal.enqueue("exchange", {"seed": True})

    errors = []
    ITERATIONS = 200

    def hammer_mark_failed():
        try:
            for i in range(ITERATIONS):
                journal.mark_failed(jid, f"attempt {i}")
        except Exception as exc:  # pragma: no cover - failure path under test
            errors.append(exc)

    def hammer_enqueue():
        try:
            for i in range(ITERATIONS):
                journal.enqueue("exchange", {"i": i})
        except Exception as exc:  # pragma: no cover - failure path under test
            errors.append(exc)

    threads = [threading.Thread(target=hammer_mark_failed) for _ in range(3)] + \
        [threading.Thread(target=hammer_enqueue) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors

    entries = journal.unapplied(limit=10_000)
    # 1 seed + 3 threads * ITERATIONS enqueues
    assert len(entries) == 1 + 3 * ITERATIONS
    seeded = next(e for e in entries if e.id == jid)
    assert seeded.attempts == 3 * ITERATIONS
    assert seeded.last_error is not None
