"""#2612: with an R9 switch set, the answer/index shadow outcome lines logged at INFO reach a stream
(the image configures no Python logging, so before this only WARNING lines were printed). Off by
default and then inert; never configured twice; left alone when logging is already configured.
Synthetic data only."""
import contextlib
import io
import logging

import pytest

from agent import rust_answer, rust_index


@pytest.fixture
def outcome_loggers(monkeypatch):
    """Clean outcome loggers and switches; restored after."""
    loggers = [logging.getLogger(n) for n in rust_answer.OUTCOME_LOGGERS]
    saved = [(lg, lg.handlers[:], lg.level) for lg in loggers]
    for lg in loggers:
        lg.handlers[:] = []
        lg.setLevel(logging.NOTSET)
    monkeypatch.delenv("DIARY_RUST_ANSWER", raising=False)
    monkeypatch.delenv("DIARY_RUST_INDEX", raising=False)
    yield loggers
    for lg, handlers, level in saved:
        lg.handlers[:] = handlers
        lg.setLevel(level)


@contextlib.contextmanager
def no_root_handlers():
    """The root logger without handlers, as in the Diary image. pytest's logging plugin attaches its own
    root handlers per test phase, so this runs inside the test body and puts them back after."""
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers[:] = []
    try:
        yield
    finally:
        root.handlers[:] = saved


def test_off_by_default_adds_nothing(outcome_loggers):
    with no_root_handlers():
        assert rust_answer.configure_outcome_logging(io.StringIO()) is False
    assert all(not lg.handlers for lg in outcome_loggers)


@pytest.mark.parametrize("key,value", [("DIARY_RUST_ANSWER", "shadow"), ("DIARY_RUST_ANSWER", "on"),
                                       ("DIARY_RUST_INDEX", "shadow"), ("DIARY_RUST_INDEX", "on")])
def test_switch_set_makes_info_outcomes_visible_once(outcome_loggers, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    out = io.StringIO()
    with no_root_handlers():
        assert rust_answer.configure_outcome_logging(out) is True
        assert rust_answer.configure_outcome_logging(out) is False  # idempotent: one handler per logger
        rust_answer.log.info("Diary answer shadow: context matches (%d messages)", 3)
        rust_index.log.info("Diary index shadow: plan matches (%d chunks)", 2)
    assert out.getvalue().splitlines() == [
        "INFO diary.rust.answer: Diary answer shadow: context matches (3 messages)",
        "INFO diary.rust.index: Diary index shadow: plan matches (2 chunks)"]
    assert all(len(lg.handlers) == 1 for lg in outcome_loggers)


def test_unknown_value_stays_off(outcome_loggers, monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "maybe")
    with no_root_handlers():
        assert rust_answer.configure_outcome_logging(io.StringIO()) is False
    assert all(not lg.handlers for lg in outcome_loggers)


def test_configured_logging_is_left_alone(outcome_loggers, monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    with no_root_handlers():
        logging.getLogger().addHandler(logging.NullHandler())
        assert logging.getLogger().handlers
        assert rust_answer.configure_outcome_logging(io.StringIO()) is False
    assert all(not lg.handlers for lg in outcome_loggers)
