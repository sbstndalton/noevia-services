"""A failure while starting a benchmark must release the slot (#874 item 4): otherwise the state
stays 'starting' and every later benchmark is refused until the service restarts."""
from __future__ import annotations

import pytest

from app import bench, db


@pytest.fixture(autouse=True)
def _idle(monkeypatch):
    db.init()
    bench._STATE = bench.JobState()
    monkeypatch.setattr(db, "prompts_by_ids", lambda ids: [{"name": "p", "body": "{}"}])
    monkeypatch.setattr(bench, "_resolve_endpoint", lambda backend: ("http://x", ""))
    yield
    bench._STATE = bench.JobState()


def test_db_failure_creating_the_run_releases_the_slot(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "bench_create_run", boom)
    ok, err = bench.start(backend="b", aliases=["a"], prompt_ids=[1], reps=1)
    assert not ok and "database is locked" in err
    assert not bench._STATE.active

    # ...and the next attempt is not refused as "already running"
    monkeypatch.undo()
    monkeypatch.setattr(db, "prompts_by_ids", lambda ids: [{"name": "p", "body": "{}"}])
    monkeypatch.setattr(bench, "_resolve_endpoint", lambda backend: ("http://x", ""))
    started = []
    monkeypatch.setattr(bench, "_run", lambda *a: started.append(a))
    ok, err = bench.start(backend="b", aliases=["a"], prompt_ids=[1], reps=1)
    assert ok, err
    bench._THREAD.join(2)
    assert started


def test_thread_start_failure_releases_the_slot_and_closes_the_run_row(monkeypatch):
    finished = []
    monkeypatch.setattr(db, "bench_create_run", lambda *a, **kw: 41)
    monkeypatch.setattr(db, "bench_finish_run", lambda rid, status, ts, note="": finished.append((rid, status)))

    class _NoThread:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")
    monkeypatch.setattr(bench.threading, "Thread", _NoThread)
    ok, err = bench.start(backend="b", aliases=["a"], prompt_ids=[1], reps=1)
    assert not ok and "can't start new thread" in err
    assert not bench._STATE.active
    assert finished == [(41, "error")]


def test_resolve_endpoint_raising_releases_the_slot(monkeypatch):
    def boom(backend):
        raise OSError("docker socket gone")
    monkeypatch.setattr(bench, "_resolve_endpoint", boom)
    ok, err = bench.start(backend="b", aliases=["a"], prompt_ids=[1], reps=1)
    assert not ok and not bench._STATE.active


def test_sweep_start_failure_releases_the_slot_too(monkeypatch):
    from app import ini
    monkeypatch.setattr(ini, "get_section", lambda a: {"model": "m/x.gguf"})
    monkeypatch.setattr(bench, "sweep_args_for_section", lambda a: [])

    def boom(*a, **kw):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "bench_create_run", boom)
    ok, err = bench.start_sweep(backend="b", aliases=["a"])
    assert not ok and not bench._STATE.active
