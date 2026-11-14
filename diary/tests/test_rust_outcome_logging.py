"""#2612: with an R9 switch set, the answer/index shadow outcome lines logged at INFO reach a stream
(the image configures no Python logging, so before this only WARNING lines were printed). Off by
default and then inert; never configured twice; left alone when logging is already configured.
Synthetic data only."""
import io
import logging

import pytest

from agent import rust_answer, rust_index


@pytest.fixture
def bare_logging(monkeypatch):
    """No root handler (as in the Diary image) and clean outcome loggers; everything restored after."""
    root = logging.getLogger()
    saved_root = root.handlers[:]
    loggers = [logging.getLogger(n) for n in rust_answer.OUTCOME_LOGGERS]
    saved = [(lg, lg.handlers[:], lg.level) for lg in loggers]
    root.handlers[:] = []
    for lg in loggers:
        lg.handlers[:] = []
        lg.setLevel(logging.NOTSET)
    monkeypatch.delenv("DIARY_RUST_ANSWER", raising=False)
    monkeypatch.delenv("DIARY_RUST_INDEX", raising=False)
    yield loggers
    root.handlers[:] = saved_root
    for lg, handlers, level in saved:
        lg.handlers[:] = handlers
        lg.setLevel(level)


def test_off_by_default_adds_nothing(bare_logging):
    assert rust_answer.configure_outcome_logging(io.StringIO()) is False
    assert all(not lg.handlers for lg in bare_logging)


@pytest.mark.parametrize("key,value", [("DIARY_RUST_ANSWER", "shadow"), ("DIARY_RUST_ANSWER", "on"),
                                       ("DIARY_RUST_INDEX", "shadow"), ("DIARY_RUST_INDEX", "on")])
def test_switch_set_makes_info_outcomes_visible_once(bare_logging, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    out = io.StringIO()
    assert rust_answer.configure_outcome_logging(out) is True
    assert rust_answer.configure_outcome_logging(out) is False  # idempotent: one handler per logger
    rust_answer.log.info("Diary answer shadow: context matches (%d messages)", 3)
    rust_index.log.info("Diary index shadow: plan matches (%d chunks)", 2)
    lines = out.getvalue().splitlines()
    assert lines == ["INFO diary.rust.answer: Diary answer shadow: context matches (3 messages)",
                     "INFO diary.rust.index: Diary index shadow: plan matches (2 chunks)"]
    assert all(len(lg.handlers) == 1 for lg in bare_logging)


def test_unknown_value_stays_off(bare_logging, monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "maybe")
    assert rust_answer.configure_outcome_logging(io.StringIO()) is False


def test_configured_logging_is_left_alone(bare_logging, monkeypatch):
    monkeypatch.setenv("DIARY_RUST_ANSWER", "shadow")
    logging.getLogger().addHandler(logging.NullHandler())
    assert rust_answer.configure_outcome_logging(io.StringIO()) is False
    assert all(not lg.handlers for lg in bare_logging)
