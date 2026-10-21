"""Relative score cut (agent/score_gap.py, services#114) and the offline precision replay.

Synthetic candidate lists only: no real Diary, model, embedding or storage.
"""
from __future__ import annotations

from datetime import date

import pytest

from agent import score_gap
from agent.config import Config
from agent.context import ContextAssembler
from agent.retrieval_trace import RetrievalTrace
from gate import replay_precision


class _Store:
    def get_day_text(self, day, max_chars=None):
        return ""

    def get_standing_sections_text(self, max_chars=None):
        return ""


class _Rows:
    """Returns fixed synthetic rows in score order, like Retriever.search."""

    def __init__(self, rows):
        self.rows = rows

    def search(self, query, top_k=8, min_score=0.30, with_ids=False, diag=None):
        if diag is not None:
            diag["status"] = "ok"
        return [dict(r) for r in self.rows if r["score"] >= min_score][:top_k]


def _row(rid, day, score, body, header="10:00 — Note"):
    return {"id": rid, "day": day, "header": header, "score": score,
            "text": f"[{day}] Day\n### {header}\n**Me:** {body}\n\n**Assistant:** Noted."}


def _build(rows, message, **retrieval):
    cfg = Config({"context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
                  "retrieval": {"top_k": 8, "min_score": 0.30, "max_context_tokens": 5200, **retrieval},
                  "prompts": {"system_path": ""}})
    asm = ContextAssembler(_Store(), _Rows(rows), cfg)
    asm._system_prompt = lambda: "SYSTEM"
    trace = RetrievalTrace()
    messages = asm.build(date(2030, 3, 7), message, trace=trace)
    return trace.data, messages[1]["content"]


def _loaded(data):
    return [e["id"] for e in data["loaded"]]


# ---------------- parsing ----------------


@pytest.mark.parametrize("value, expected", [
    (0.08, 0.08), ("0.12", 0.12), (1, 1.0), (0.001, 0.001),
    (None, None), (False, None), ("off", None), ("None", None), (" false ", None), ("", None), (0, None), ("0", None),
])
def test_parse_gap(value, expected):
    assert score_gap.parse_gap(value) == expected


@pytest.mark.parametrize("value", ["wide", float("nan"), -0.1, 1.5, True, [0.1], float("inf")])
def test_parse_gap_invalid_values_leave_the_cut_off(value):
    assert score_gap.SCORE_GAP is None
    assert score_gap.parse_gap(value) is None


# ---------------- the cut itself ----------------


def test_cut_keeps_a_lower_scored_needle_inside_the_gap_and_drops_the_tail():
    ranked = list(enumerate([
        {"id": "near-topic", "score": 0.80},
        {"id": "needle", "score": 0.725},      # 0.075 below the best: inside 0.08
        {"id": "tie", "score": 0.72},          # exactly 0.08 below: kept
        {"id": "tail-1", "score": 0.7199},
        {"id": "tail-2", "score": 0.60},
        {"id": "no-score", "score": None},     # unknown score: never cut
    ], 1))
    kept, removed, best = score_gap.cut(ranked, 0.08)
    assert best == 0.80
    assert [r["id"] for _, r in kept] == ["near-topic", "needle", "tie", "no-score"]
    assert [(rank, r["id"]) for rank, r in removed] == [(4, "tail-1"), (5, "tail-2")]


def test_cut_off_or_empty_changes_nothing():
    ranked = list(enumerate([{"score": 0.9}, {"score": 0.1}], 1))
    assert score_gap.cut(ranked, None) == (ranked, [], 0.9)
    assert score_gap.cut([], 0.08) == ([], [], None)
    nan = [(1, {"score": float("nan")})]
    assert score_gap.cut(nan, 0.08) == (nan, [], None)


# ---------------- in the context assembler ----------------


def test_unfocused_question_loads_only_candidates_near_the_best():
    rows = [_row("a", "2030-01-10", 0.80, "the hive by the wall"),
            _row("needle", "2030-01-12", 0.74, "the hive is a top-bar hive"),
            _row("c", "2030-01-14", 0.72, "bees were calm today"),
            _row("d", "2030-01-15", 0.66, "honey jars to label"),
            _row("e", "2030-01-16", 0.55, "a walk by the canal")]
    data, context = _build(rows, "what kind of hive do i keep?", score_gap=0.08)
    assert data["focus"]["checked"] is False
    assert _loaded(data) == ["a", "needle", "c"]
    assert [c["id"] for c in data["candidates"]] == ["a", "needle", "c"]
    gap = data["score_gap"]
    assert gap["gap"] == 0.08 and gap["applied"] is True and gap["best"] == 0.80
    assert [(r["id"], r["vector_rank"]) for r in gap["removed"]] == [("d", 4), ("e", 5)]
    assert "a walk by the canal" not in context and "top-bar" in context


def test_time_anchored_question_measures_the_gap_from_the_best_entry_before_the_anchor():
    # The best vector hit is dated after the anchor. Measured from it, the needle (0.12 below) would be cut;
    # the focus removes it first, so the gap is measured from the best entry the question can use.
    rows = [_row("later", "2030-02-20", 0.90, "the hive was replaced with a skep"),
            _row("needle", "2030-02-05", 0.78, "the hive is a top-bar hive"),
            _row("near", "2030-02-01", 0.72, "checked the hive roof"),
            _row("old", "2030-01-10", 0.65, "bought jars for honey")]
    data, context = _build(rows, "As of February 11, 2030, what had I written down about the hive?", score_gap=0.08)
    assert data["focus"]["time_anchor"] is True
    assert [r["reason"] for r in data["focus"]["removed"]] == ["after_time_anchor"]
    assert _loaded(data) == ["needle", "near"]
    assert data["score_gap"]["applied"] is True and data["score_gap"]["best"] == 0.78
    assert [r["id"] for r in data["score_gap"]["removed"]] == ["old"]
    assert "skep" not in context and "top-bar" in context


@pytest.mark.parametrize("question, loaded, by_name", [
    ("What does Orla Kilbride grow?", ["best", "other", "needle"], []),  # the name in full
    # A typo: no full exact hit. services#145 leaves the near spelling out for its name, never for its score.
    ("What does Orla Kilbrid grow?", ["other"], ["best", "needle"]),
])
def test_an_entry_about_the_asked_person_is_never_cut_for_its_score(question, loaded, by_name):
    rows = [_row("best", "2030-01-10", 0.85, "Orla Kilbride grows sweet peas."),
            _row("other", "2030-01-11", 0.80, "Gardening club met in the hall."),
            _row("needle", "2030-01-20", 0.60, "Orla Kilbride said the sweet peas failed this year.")]
    data, context = _build(rows, question, score_gap=0.08)
    assert data["focus"]["entities"] == 1
    assert data["score_gap"]["applied"] is False and data["score_gap"]["removed"] == []
    assert _loaded(data) == loaded
    assert [(r["id"], r["reason"]) for r in data["focus"]["removed"]] == [(i, "near_name_unmatched") for i in by_name]
    if by_name:
        assert 'ask whether "Orla Kilbride" was meant' in context and "sweet peas" not in context


def test_by_default_nothing_is_cut():
    rows = [_row(str(n), f"2030-01-1{n}", round(0.9 - n * 0.1, 2), f"note {n}") for n in range(6)]
    data, _ = _build(rows, "what did i note?")  # no score_gap setting at all
    assert len(_loaded(data)) == 6
    assert data["score_gap"] == {"gap": None, "applied": False, "best": 0.9, "removed": []}


def test_shipped_config_leaves_the_cut_off():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "config.yaml").read_text())
    assert score_gap.parse_gap(cfg["retrieval"]["score_gap"]) is None


def test_score_gap_off_restores_the_fixed_top_k():
    rows = [_row(str(n), f"2030-01-1{n}", round(0.9 - n * 0.05, 2), f"note {n}") for n in range(6)]
    data, _ = _build(rows, "what did i note?", score_gap="off")
    assert len(_loaded(data)) == 6
    assert data["score_gap"] == {"gap": None, "applied": False, "best": 0.9, "removed": []}
    wider, _ = _build(rows, "what did i note?", score_gap=0.2)
    assert len(_loaded(wider)) == 5


def test_trace_score_gap_carries_refs_and_scores_only():
    rows = [_row("a", "2030-01-10", 0.80, "SECRETBODYWORD one"),
            _row("b", "2030-01-11", 0.50, "SECRETBODYWORD two", header="11:30 — SECRETHEADER")]
    data, _ = _build(rows, "anything about secretbodyword?", score_gap=0.08)
    removed = data["score_gap"]["removed"]
    assert removed == [{"vector_rank": 2, "id": "b", "day": "2030-01-11", "time": "11:30", "score": 0.50}]
    assert "SECRET" not in repr(data)


# ---------------- offline replay of gate run 1 (synthetic) ----------------


def test_replay_of_run_1_candidates_keeps_every_needle_and_lifts_precision():
    result = replay_precision.replay()
    base = result["run 1 (no focus, no gap)"]["all"]
    assert base["queries"] == 37 and base["precision@8"] < 0.40 and base["recall@8"] == 1.0
    for label, fams in result.items():
        for name, m in fams.items():
            assert m["recall@8"] == 1.0, (label, name)  # no setting may buy precision with a needle
            assert m["loaded_recall"] == 1.0, (label, name)
    gap_only = result["score gap 0.08 only"]["all"]
    assert gap_only["precision@8"] > 2 * base["precision@8"]
    default = result["default settings (focus on, gap off)"]["all"]
    focus_gap = result["focus + score gap 0.08"]["all"]
    assert default["precision@8"] >= 0.40 and focus_gap["precision@8"] == default["precision@8"]
