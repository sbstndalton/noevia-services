"""services#148 item 1: a surname written with a trailing "s" ("Alder Brannocks boat", "the Brannocks") or a
possessive ("Brannock's", "Brannock’s") names the asked person; it is not a near spelling.

Offline only: synthetic rows through the real entity-focus check. No model, embedding or service call.
"""
from __future__ import annotations

import pytest

from agent import entity_focus as ef

ASK = "Where does Alder Brannock keep the boat?"


def _row(text, score=0.8, day="2030-01-10"):
    return {"id": text[:8], "day": day, "header": "09:00 — Note", "text": text, "score": score}


def _focus(question, rows, earlier=()):
    res = ef.focus_candidates(question, rows, earlier=earlier)
    return [rank for rank, _ in res.kept], {rank: why for rank, _, why in res.removed}, res


def _offers(res, spelling):
    """Does any note or label offer ``spelling`` as a different spelling of the asked name?"""
    notes = " ".join(ef.context_notes(res))
    labels = [s for s in res.similar.values() if spelling in s[1]]
    absent = [s for spellings in res.absent.values() for s in spellings if spelling in s]
    return spelling in notes or bool(labels) or bool(absent)


@pytest.mark.parametrize("text", [
    "Took Alder Brannocks boat out past the breakwater.",       # possessive without an apostrophe
    "Took Alder Brannock's boat out past the breakwater.",      # straight apostrophe
    "Took Alder Brannock’s boat out past the breakwater.",  # curly apostrophe
    "Took ALDER BRANNOCKS boat out past the breakwater.",       # case-insensitive
    "Supper with Alder Brannocks and the rest of the crew.",    # plural-like surname after the given name
])
def test_the_entry_is_the_asked_person_on_its_own(text):
    rows = [_row(text), _row("Inked the legend for the chart.", 0.5)]
    kept, removed, res = _focus(ASK, rows)
    assert 1 in kept and 1 not in removed
    assert res.matched == res.entities  # it names the asked person in full: no near_name_unmatched
    assert not res.absent and 1 not in res.similar
    assert not _offers(res, "Brannocks")
    assert "The diary has no entry naming" not in " ".join(ef.context_notes(res))


@pytest.mark.parametrize("text", [
    "Took Alder Brannocks boat out past the breakwater.",
    "Took Alder Brannock’s boat out past the breakwater.",
])
def test_the_entry_counts_as_a_full_exact_hit_so_a_near_spelling_is_still_dropped(text):
    rows = [_row("Had coffee with Alder Brennock by the quay."), _row(text, 0.78)]
    kept, removed, res = _focus(ASK, rows)
    assert removed == {1: "near_name"} and kept == [2]  # the existing #116 drop; no near_name_unmatched
    assert not _offers(res, "Brannocks") and not res.absent


def test_the_plural_form_alongside_the_given_name_is_kept():
    rows = [_row("Alder came round; the Brannocks had the boat out.")]
    kept, removed, res = _focus(ASK, rows)
    assert kept == [1] and removed == {}
    assert 1 not in res.similar and not _offers(res, "Brannocks")


def test_the_plural_surname_on_its_own_is_not_labelled_a_near_spelling():
    rows = [_row("Dinner with the Brannocks at the harbour.")]
    kept, removed, res = _focus(ASK, rows)
    assert kept == [1] and removed == {}
    assert 1 not in res.similar and not _offers(res, "Brannocks")


# ---------------- unchanged behaviour ----------------

@pytest.mark.parametrize("near", ["Alder Brennock", "Alder Branock"])
def test_a_genuine_near_spelling_is_still_dropped_with_the_note(near):
    rows = [_row(f"Had coffee with {near} by the quay."), _row("Inked the legend for the chart.", 0.7)]
    kept, removed, res = _focus(ASK, rows)
    assert removed == {1: "near_name_unmatched"} and kept == [2]
    assert f'"{near}"' in ef.context_notes(res)[0]


def test_dropping_a_trailing_s_from_the_asked_name_is_still_a_near_spelling():
    rows = [_row("Had coffee with Rhys Evan by the quay."), _row("Inked the legend for the chart.", 0.7)]
    kept, removed, res = _focus("Where does Rhys Evans keep the boat?", rows)
    assert removed == {1: "near_name_unmatched"} and kept == [2]
    assert '"Rhys Evan"' in ef.context_notes(res)[0]


def test_a_different_given_name_with_the_plural_surname_is_another_person():
    rows = [_row("Had coffee with Wren Brannocks at the quay."),
            _row("Alder Brannock keeps the boat at the north quay.", 0.79)]
    kept, removed, res = _focus(ASK, rows)
    assert kept == [2] and removed == {1: "other_person"}


def test_a_one_word_name_is_unchanged():
    rows = [_row("Brannocks rang about the mooring.")]
    kept, removed, res = _focus("Where does Brannock keep the boat?", rows)
    assert kept == [1] and removed == {}


def test_a_relational_question_is_unchanged():
    rows = [_row("Had coffee with Alder Brennock by the quay.")]
    kept, removed, _ = _focus("What is Alder Brannock's sister called?", rows)
    assert removed == {} and kept == [1]


# ---------------- hostile inputs ----------------

@pytest.mark.parametrize("question", [
    "Where did Alder go?",
    "Where did Alder S go?",
    "Where did Alder Ss go?",
    "Where did Alder ' go?",
    "Where did Alder ’s go?",
    "Where did Alder " + "Brannock" * 500 + " go?",
    "Where did Alder Brannock’’s‘ go?!?",
])
@pytest.mark.parametrize("text", [
    "Saw Alder S and Alder Ss and Alder s at the quay.",
    "Alder ' and Alder ’s and Alder '' went out.",
    "Alder " + "Brannock" * 500 + "s rang.",
    "Alder Brannock’’s‘ boat — «Alder Brannocks».",
    "",
])
def test_hostile_names_do_not_raise(question, text):
    rows = [_row(text), {"id": None, "day": "x", "header": None, "text": None, "score": "nan"}]
    res = ef.focus_candidates(question, rows, limit=16)
    assert len(res.kept) + len(res.removed) == len(rows)
    assert all(isinstance(line, str) for line in ef.context_notes(res))


@pytest.mark.parametrize("word", ["", "s", "S", "'", "’s", "ss"])
def test_the_surname_helper_takes_any_word(word):
    entity = ef.asked_entities("Where does Alder Brannock keep the boat?")[0]
    assert ef._same_given_near_surname(f"Alder {word} rang.", entity) in (None, f"Alder {word}")
