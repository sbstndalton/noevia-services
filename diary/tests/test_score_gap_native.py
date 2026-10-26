"""Synthetic native adapter transport tests; no Diary, models or storage."""
import os
from pathlib import Path

import pytest

from agent import score_gap, score_gap_native as native
from tests.test_score_gap import _build, _row, _loaded


@pytest.fixture(autouse=True)
def native_mode(monkeypatch):
    monkeypatch.setenv("DIARY_SCORE_GAP_IMPL", "rust")


def fake(tmp_path, monkeypatch, code):
    path = tmp_path / "child"
    path.write_text("#!/usr/bin/python3\n" + code)
    path.chmod(0o700)
    monkeypatch.setattr(native, "BINARY", str(path))
    monkeypatch.setattr(native, "TIMEOUT", 0.15)
    return path


def run():
    diagnostics = {}
    ranked = [(9, {"score": 0.9}), (3, {"score": 0.1}), (7, {"score": None})]
    result = score_gap.cut(ranked, 0.08, diagnostics=diagnostics)
    return ranked, result, diagnostics


def test_native_success_owns_partition_and_receives_scores_only(tmp_path, monkeypatch):
    fake(tmp_path, monkeypatch, """import sys,os
data=sys.stdin.buffer.read()
assert data == b'diary-score-gap-v1\\t0.08\\t0.9,0.1,null\\n'
assert 'DIARY_SCORE_GAP_IMPL' not in os.environ
sys.stdout.write('diary-score-gap-v1\\t0.9\\t0,2\\t1\\n')
""")
    ranked, result, diagnostics = run()
    assert result == ([ranked[0], ranked[2]], [ranked[1]], 0.9)
    assert diagnostics == {"implementation": "rust", "status": "ok"}


@pytest.mark.parametrize("reply", [
    b"", b"diary-score-gap-v2\t0.9\t0,2\t1\n",
    b"diary-score-gap-v1\t0.9\t0,2\t1", b"diary-score-gap-v1\t0.9\t0,2\t1\r\n",
    b"diary-score-gap-v1\t0.9\t0,2\t1\nextra", b"diary-score-gap-v1\t0.9\t0,2\t1\n\n",
    b"diary-score-gap-v1\t0.8\t0,2\t1\n", b"diary-score-gap-v1\tnan\t0,2\t1\n",
    b"diary-score-gap-v1\t0.9\t0,2\t2\n", b"diary-score-gap-v1\t0.9\t2,0\t1\n",
    b"diary-score-gap-v1\t0.9\t00,2\t1\n", b"diary-score-gap-v1\t0.9\t0\t1\n",
    b"diary-score-gap-v1\t0.9\t0,3\t1\n", b"\xff", b"x" * 32769,
])
def test_malformed_output_retains_all(tmp_path, monkeypatch, reply):
    fake(tmp_path, monkeypatch, f"import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write({reply!r})\n")
    ranked, result, diagnostics = run()
    assert result == (ranked, [], 0.9)
    assert diagnostics["status"] == "unavailable"


@pytest.mark.parametrize("code", [
    "import sys\nsys.stdin.buffer.read()\nsys.exit(2)\n",
    "import time\ntime.sleep(5)\n",
    "import sys\nsys.stderr.write('x'*129)\nsys.stderr.flush()\nsys.stdin.buffer.read()\n",
    "import sys\nsys.stdin.buffer.read()\nsys.stderr.write('warning')\nsys.stdout.write('diary-score-gap-v1\\t0.9\\t0,2\\t1\\n')\n",
    "import os\nos.close(0)\n",
    "import os,time\nif os.fork() == 0: time.sleep(5)\n",
])
def test_child_failures_retain_all(tmp_path, monkeypatch, code):
    fake(tmp_path, monkeypatch, code)
    ranked, result, diagnostics = run()
    assert result == (ranked, [], 0.9)
    assert diagnostics["status"] == "unavailable"


def test_missing_binary_and_count_bound(monkeypatch):
    monkeypatch.setattr(native, "BINARY", "/nonexistent/score-gap")
    ranked, result, diagnostics = run()
    assert result == (ranked, [], 0.9)
    assert diagnostics["status"] == "unavailable"
    assert native.partition([None] * 4097, None, None) is None


def test_unavailable_trace_truthful(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "BINARY", "/nonexistent/score-gap")
    rows = [_row("a", "2030-01-10", 0.9, "hive"), _row("b", "2030-01-11", 0.4, "honey")]
    data, _ = _build(rows, "what kind of hive?", score_gap=0.08)
    assert _loaded(data) == ["a", "b"]
    assert data["score_gap"] == {"gap": 0.08, "applied": False, "best": 0.9, "removed": [],
                                 "implementation": "rust", "status": "unavailable"}


def test_python_default_preserves_trace(monkeypatch):
    monkeypatch.delenv("DIARY_SCORE_GAP_IMPL")
    ranked, result, diagnostics = run()
    assert result == ([ranked[0], ranked[2]], [ranked[1]], 0.9)
    assert diagnostics == {}


def test_actual_cli_differential(monkeypatch):
    binary = os.environ.get("SCORE_GAP_TEST_BINARY")
    if not binary:
        pytest.skip("Set SCORE_GAP_TEST_BINARY/VECTORS for the paired CLI differential")
    monkeypatch.setattr(native, "BINARY", binary)
    vectors = Path(os.environ["SCORE_GAP_TEST_VECTORS"]).read_text().splitlines()[1:]
    assert len(vectors) == 352
    for line in vectors:
        gap, values, kept, removed, best = line.split("\t")
        parse = lambda v: None if v == "missing" else float(v)
        rows = [(i * 3 + 1, {"score": parse(v)}) for i, v in enumerate(values.split(","))] if values else []
        monkeypatch.setenv("DIARY_SCORE_GAP_IMPL", "python")
        expected = score_gap.cut(rows, parse(gap))
        monkeypatch.setenv("DIARY_SCORE_GAP_IMPL", "rust")
        diagnostics = {}
        actual = score_gap.cut(rows, parse(gap), diagnostics=diagnostics)
        assert diagnostics["status"] == "ok", line
        assert actual == expected, line


def test_actual_cli_preserves_context_focus_policy(monkeypatch):
    binary = os.environ.get("SCORE_GAP_TEST_BINARY")
    if not binary:
        pytest.skip("Set SCORE_GAP_TEST_BINARY for paired CLI context validation")
    monkeypatch.setattr(native, "BINARY", binary)
    from tests.test_score_gap import (
        test_unfocused_question_loads_only_candidates_near_the_best,
        test_time_anchored_question_measures_the_gap_from_the_best_entry_before_the_anchor,
        test_an_entry_about_the_asked_person_is_never_cut_for_its_score,
    )
    test_unfocused_question_loads_only_candidates_near_the_best()
    test_time_anchored_question_measures_the_gap_from_the_best_entry_before_the_anchor()
    test_an_entry_about_the_asked_person_is_never_cut_for_its_score(
        "What does Orla Kilbride grow?", ["best", "other", "needle"], [])
    test_an_entry_about_the_asked_person_is_never_cut_for_its_score(
        "What does Orla Kilbrid grow?", ["other"], ["best", "needle"])
    rows = [_row("best", "2030-01-10", 0.9, "Orla Kilbride grows peas."),
            _row("tail", "2030-01-11", 0.5, "Orla Kilbride grows beans.")]
    data, _ = _build(rows, "What does Orla Kilbride grow?", score_gap=0.08, entity_focus=False)
    assert _loaded(data) == ["best"]
    assert data["score_gap"]["applied"] is True
    assert data["score_gap"]["status"] == "ok"

def test_busy_transport_keeps_retrieval_available():
    for _ in range(4):
        assert native._SLOTS.acquire(blocking=False)
    try:
        assert native.partition([0.9], 0.08, 0.9) is None
    finally:
        for _ in range(4):
            native._SLOTS.release()


def test_normalizes_scores_before_transport(tmp_path, monkeypatch):
    fake(tmp_path, monkeypatch, """import sys
assert sys.stdin.buffer.read() == b'diary-score-gap-v1\\tnull\\t1.0,null,null,null,0.5\\n'
sys.stdout.write('diary-score-gap-v1\\t1\\t0,1,2,3,4\\t\\n')
""")
    rows = [(i, {"score": s, "text": "PRIVATE"}) for i, s in enumerate([True, "bad", float("nan"), float("inf"), "0.5"])]
    assert score_gap.cut(rows, None) == (rows, [], 1.0)


def test_disabled_guard_rejects_native_removals(tmp_path, monkeypatch):
    fake(tmp_path, monkeypatch, "import sys\nsys.stdin.buffer.read()\nsys.stdout.write('diary-score-gap-v1\\t0.9\\t0,2\\t1\\n')\n")
    rows = [(9, {"score": 0.9}), (3, {"score": 0.1}), (7, {"score": None})]
    diagnostics = {}
    assert score_gap.cut(rows, None, diagnostics=diagnostics) == (rows, [], 0.9)
    assert diagnostics["status"] == "unavailable"
