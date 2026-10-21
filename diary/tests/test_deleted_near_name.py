"""services#145: after a deletion nobody names the asked person in full; a near-spelled person is left out.

Offline only: the recorded synthetic candidates of gate run 1 (fixtures/precision_replay_114.json) are replayed
through the real ContextAssembler, and every needle row's loaded set is compared with the one recorded on main
before this change (fixtures/precision_replay_145_main_loaded.json). No model, embedding or service call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import entity_focus as ef
from agent.config import Config
from agent.context import ContextAssembler
from agent.retrieval_trace import RetrievalTrace
from gate import replay_precision as rp

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DATA = json.loads(rp.FIXTURE.read_text(encoding="utf-8"))
QUERIES = {q["id"]: q for q in DATA["queries"]}
MAIN = json.loads((FIXTURES / "precision_replay_145_main_loaded.json").read_text(encoding="utf-8"))
# Deletion rows: the asked needle was deleted; the recorded candidates hold a near-spelled distractor.
DELETED = {"Q-needle-10": ("Caspian Juxon", "Caspian Juxton", "E022"),
           "Q-needle-25": ("Fenella Rookwood", "Fenella Rockwood", "E005"),
           "Q-needle-38": ("Soren Oakeshott", "Soren Okeshott", "E152")}


def _context(query, text=None, earlier=()):
    """(loaded entry ids, note lines, trace) for one recorded query at default settings."""
    cfg = Config({"context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
                  "retrieval": {"top_k": rp.K, "min_score": 0.30, "max_context_tokens": 5200},
                  "prompts": {"system_path": ""}})
    asm = ContextAssembler(rp._Store(), rp._Recorded(rp._rows(DATA, query)), cfg)
    trace = RetrievalTrace()
    turns = [{"role": "user", "content": t} for t in earlier]
    kept, _, notes = asm._retrieve_focused(text or query["text"], turns, trace)
    return [r["id"] for r in kept], notes, trace.data


@pytest.mark.parametrize("qid", sorted(DELETED))
def test_deleted_person_context_leaves_out_the_near_spelling_and_says_so(qid):
    asked, near, distractor = DELETED[qid]
    assert MAIN["loaded"][qid][0] == distractor  # on main the distractor was loaded first (the #145 failure)
    loaded, notes, trace = _context(QUERIES[qid])
    assert distractor not in loaded
    assert not any(near.split()[1] in DATA["entries"][i]["body"] for i in loaded)
    removed = {r["id"]: r for r in trace["focus"]["removed"]}
    assert removed[distractor]["vector_rank"] == 1 and removed[distractor]["reason"] == "near_name_unmatched"
    note = " ".join(notes)
    assert f'The diary has no entry naming "{asked}".' in note
    assert f'"{near}"' in note and "say the diary does not record" in note and "was meant" in note


def test_every_other_row_loads_exactly_what_main_loaded():
    # services#160 updated the snapshot for Q-needle-35 only: C04 (another person sharing the asked given name,
    # at a sentence start) is dropped as other_person; no gold entry is lost.
    changed = {}
    for qid, query in QUERIES.items():
        loaded, _, _ = _context(query)
        if qid in DELETED:
            kept = [i for i in MAIN["loaded"][qid] if i != DELETED[qid][2]]
            assert loaded[:len(kept)] == kept and len(loaded) <= len(MAIN["loaded"][qid]), qid  # next one backfills
        elif loaded != MAIN["loaded"][qid]:
            changed[qid] = (MAIN["loaded"][qid], loaded)
    assert changed == {}
    assert len(QUERIES) == 40


def test_precision_and_recall_hold():
    default = rp.replay()["default settings (focus on, gap off)"]["all"]
    assert default["precision@8"] >= 0.91 and default["precision@8"] >= MAIN["metrics"]["precision@8"]
    assert default["recall@8"] >= MAIN["metrics"]["recall@8"] == 1.0


def test_follow_up_naming_the_near_spelled_person_still_loads_that_entry():
    q = QUERIES["Q-needle-00"]
    assert "E106" in [c for c, _ in q["candidates"]]  # Orla Killbride's entry, a recorded candidate
    loaded, notes, _ = _context(q, "And what about Orla Killbride?", earlier=[q["text"]])
    assert "E106" in loaded and "The diary has no entry naming" not in " ".join(notes)


# ---------------- the rule on its own ----------------

def _row(text, score=0.8, day="2030-01-10"):
    return {"id": text[:8], "day": day, "header": "09:00 — Note", "text": text, "score": score}


def _reasons(question, rows, earlier=()):
    res = ef.focus_candidates(question, rows, earlier=earlier)
    return [rank for rank, _ in res.kept], {rank: why for rank, _, why in res.removed}, res


def test_rule_drops_only_same_given_name_with_near_surname():
    rows = [_row("Had coffee with Caspian Juxton by the map library."),
            _row("Met Caspian Pennick at the archive.", 0.79),
            _row("Called Casper Juxon about the shift.", 0.78),  # near given name: not the contract
            _row("Inked the legend for the chart.", 0.70)]
    kept, removed, res = _reasons("Which shift does Caspian Juxon cover?", rows)
    assert removed == {1: "near_name_unmatched"} and kept == [2, 3, 4]
    assert ef.context_notes(res) == [
        '(Asked about "Caspian Juxon". The diary has no entry naming "Caspian Juxon". Entries naming'
        ' "Caspian Juxton", a near spelling with the same first name, were left out as a different person:'
        ' say the diary does not record this about "Caspian Juxon", do not give the details of "Caspian'
        ' Juxton" as the answer, and ask whether "Caspian Juxton" was meant.)']


def test_no_effect_when_a_full_name_candidate_exists():
    rows = [_row("Had coffee with Caspian Juxton by the map library."),
            _row("Caspian Juxon covers the middle watch.", 0.75)]
    kept, removed, res = _reasons("Which shift does Caspian Juxon cover?", rows)
    assert removed == {1: "near_name"} and kept == [2]  # the existing #116 drop, unchanged
    assert "The diary has no entry naming" not in " ".join(ef.context_notes(res))


def test_single_word_name_is_unchanged():
    rows = [_row("Juxton covers the forenoon watch."), _row("Had coffee with Caspian Juxton.", 0.7)]
    kept, removed, res = _reasons("Which shift does Juxon cover?", rows)
    assert removed == {} and kept == [1, 2] and not res.absent


def test_a_part_of_the_asked_name_keeps_the_entry():
    rows = [_row("Had coffee with Caspian Juxton; later, Juxon rang about the shift.")]
    kept, removed, _ = _reasons("Which shift does Caspian Juxon cover?", rows)
    assert removed == {} and kept == [1]  # "Juxon" alone may be the asked person


def test_relational_question_is_unchanged():
    rows = [_row("Had coffee with Caspian Juxton by the map library.")]
    kept, removed, _ = _reasons("What is Caspian Juxon's sister called?", rows)
    assert removed == {} and kept == [1]


def test_carried_follow_up_after_a_deletion_also_abstains():
    rows = [_row("Had coffee with Fenella Rockwood; their boat is the Dunlin.")]
    kept, removed, res = _reasons("And her boat?", rows, earlier=["Where does Fenella Rookwood sail?"])
    assert res.carried and removed == {1: "near_name_unmatched"} and kept == []
    assert '"Fenella Rockwood"' in ef.context_notes(res)[0]


@pytest.mark.parametrize("question", [
    'Where did "Caspian-Juxon" (Juxon!!) go?',
    "Where did Ca$pian Ju.x.on go?",
    "Where did O'Neil McJuxon-Smythe go?",
    "Where did Caspian " + "Juxon" + "n" * 4000 + " go?",
    "Where did " + " ".join(["Caspian"] * 300) + " Juxon go?",
])
def test_hostile_names_do_not_raise(question):
    rows = [_row("Had coffee with Caspian Juxton. " + "Caspian " * 200), _row("x" * 20000, 0.5), _row("", 0.4),
            {"id": None, "day": "not a day", "header": None, "text": None, "score": "nan"}]
    res = ef.focus_candidates(question, rows, limit=16)
    assert len(res.kept) + len(res.removed) == len(rows)
    assert all(isinstance(line, str) for line in ef.context_notes(res))
