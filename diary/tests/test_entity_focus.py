"""services#116: near-name precision, the exact-entity context and the deletion-abstain context.

Synthetic only: the gate's own fixture generator (diary/gate/gen_fixture.py) plus small invented cases, a
character-trigram stub embedder (similar spellings embed close together, which is how the failure looks with
a real embedder) and no model or inference call. Precision and recall use the gate's own scorer and
thresholds (diary/gate/score.py, thresholds.json) on the opt-in retrieval trace.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import corpus as fmt
from agent import entity_focus as ef
from agent.config import Config
from agent.context import ContextAssembler
from agent.retrieval import Retriever
from agent.retrieval_trace import RetrievalTrace
from tests.test_retrieval_context import FakeStore

GATE = Path(__file__).resolve().parent.parent / "gate"
sys.path.insert(0, str(GATE))

import gen_fixture  # noqa: E402
import score  # noqa: E402

THRESHOLDS = json.loads((GATE / "thresholds.json").read_text())
K = THRESHOLDS["candidate_retrieval"]["k"]
SEED, ANCHOR, DAYS = 20261020, date(2030, 1, 6), 60
BEGIN, END = "=== RETRIEVED PAST ENTRIES", "=== END RETRIEVED PAST ENTRIES ==="


class TrigramLLM:
    """Deterministic embedder: hashed character trigrams of each word, L2-normalised. No model."""

    DIM = 256

    def embed(self, texts, model=None):
        out = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                word = f"^{word}$"
                for i in range(len(word) - 2):
                    h = int.from_bytes(hashlib.blake2b(word[i:i + 3].encode(), digest_size=4).digest(), "big")
                    vec[h % self.DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def chat(self, *a, **k):  # pragma: no cover
        raise AssertionError("no inference in these tests")


def _cfg(focus=True):
    return Config({
        "context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
        "retrieval": {"top_k": 8, "min_score": 0.30, "max_context_tokens": 5200, "entity_focus": focus},
        "prompts": {"system_path": ""},
    })


def _assembler(retriever, focus=True):
    asm = ContextAssembler(FakeStore(), retriever, _cfg(focus))
    asm._system_prompt = lambda: "SYSTEM"
    return asm


def _retrieved_block(messages) -> str:
    content = messages[1]["content"]
    return content[content.index(BEGIN):content.index(END)]


def _index_files(retriever, files):
    for path, text in sorted(files.items()):
        retriever.reindex_file(path, text)
    if not retriever.vec_available:
        pytest.skip("sqlite-vec not available in this environment")


def _apply(files, entries, mutations):
    """The gate's scripted mutations, applied to the synthetic files as the API would write them."""
    files = dict(files)
    path_of = {e["xid"]: e["path"] for e in entries}
    for m in mutations:
        if m["op"] == "edit":
            new = fmt.replace_exchange_text(files[path_of[m["xid"]]], m["xid"], m["me"], m["assistant"])
            assert new is not None
            files[path_of[m["xid"]]] = new
        else:
            files[m["path"]] = files[m["path"]].rstrip("\n") + "\n\n" + m["block"] + "\n"
    return files


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    files, entries, queries, mutations, _ = gen_fixture.build(SEED, ANCHOR, DAYS)
    files = _apply(files, entries, mutations)
    retriever = Retriever(tmp_path_factory.mktemp("focus") / "idx.db", TrigramLLM())
    _index_files(retriever, files)
    index = {(e["day"], e["header"][:5]): e["id"] for e in entries}
    assert len(index) == len(entries)  # day + clock time names each synthetic entry
    return SimpleNamespace(retriever=retriever, index=index, queries=queries,
                           by_id={q["id"]: q for q in queries}, entries={e["id"]: e for e in entries})


def ask(world, text, day, turns=None, focus=True):
    trace = RetrievalTrace()
    messages = _assembler(world.retriever, focus).build(day, text, session_turns=turns, trace=trace)
    ids = lambda rows: [world.index.get((r["day"], r["time"]), "unmapped") for r in rows]  # noqa: E731
    return SimpleNamespace(messages=messages, block=_retrieved_block(messages), trace=trace.data,
                           candidates=ids(trace.data["candidates"]), loaded=ids(trace.data["loaded"]))


def _day(q):
    return date.fromisoformat(q["clock"][:10])


def metrics(world, focus, categories=None):
    rec, prec, ctx = [], [], []
    for q in world.queries:
        if not q["relevant_entry_ids"] or (categories and q["category"] not in categories):
            continue
        got = ask(world, q["text"], _day(q), focus=focus)
        sets, top = q["relevant_entry_ids"], got.candidates[:K]
        rec.append(score.best_recall(sets, top))
        prec.append(score.precision(sets, top))
        ctx.append(score.best_recall(sets, got.loaded))
    return {"queries": len(rec), "recall": score.mean(rec), "precision": score.mean(prec), "context_recall": score.mean(ctx)}


def _kinds(world, ids):
    return {world.entries[i]["kind"] if i in world.entries else i for i in ids}


def _norm(text):
    return re.sub(r"\s+", " ", text.lower())


# ---------------- the lexical check ----------------


@pytest.mark.parametrize("message, entities", [
    ("What is Alder Brannock's favourite tea?", ["Alder Brannock"]),
    ("What is Alder Brannock’s favourite tea?", ["Alder Brannock"]),
    ("And what about Alder Brennock?", ["Alder Brennock"]),
    ("Which three places did I survey during the Hollin Moor survey?", ["Hollin Moor"]),
    ("What did Tobiah give me?", ["Tobiah"]),
    ("Alder Brannock came by. How was I feeling?", ["Alder Brannock"]),
    ("Did Bryony and Caspian Dunmore meet on Monday in March?", ["Bryony", "Caspian Dunmore"]),
    ("Which ferry crossing does Hesper Gilchrist take on Thursdays?", ["Hesper Gilchrist"]),
    ("Which dentist did I see in February?", []),
    ("Tell me about the entries", []),
    ("Remind me what I wrote yesterday.", []),
    ("SECRETQUERYPHRASE about the entries", []),
    ("", []),
])
def test_asked_entities(message, entities):
    assert ef.asked_entities(message) == entities


def test_asked_entities_are_capped_and_unique():
    message = "Did Anna, anna, Bea, Cleo, Dora and Esme come?"
    assert ef.asked_entities(message) == ["Anna", "Bea", "Cleo", "Dora"][:ef.MAX_ENTITIES]


@pytest.mark.parametrize("message, anchor", [
    ("As of January 20, 2030, what had I written down as Alder Brannock's tea?", (date(2030, 1, 20), True)),
    ("as at 3rd February 2030 what was it?", (date(2030, 2, 3), True)),
    ("What did I note before 2030-02-10?", (date(2030, 2, 10), False)),
    ("Up to Sept 9, 2030 who visited?", (date(2030, 9, 9), True)),
    ("As of today, how am I?", None),
    ("As of February 30, 2030, what then?", None),  # not a date: no anchor rather than an error
    ("What happened on January 20, 2030?", None),
])
def test_time_anchor(message, anchor):
    assert ef.time_anchor(message) == anchor


def test_every_fixture_near_name_pair_is_near_and_names_differ():
    for real, near in gen_fixture.SURNAMES:
        assert ef.near_word(real, near) and ef.near_word(near, real), (real, near)
        assert ef.near_name(f"Had coffee with Alder {near} today.", f"Alder {real}") == f"Alder {near}"
        assert not ef.names_exactly(f"Had coffee with Alder {near}.", f"Alder {real}")
    assert ef.near_name("Alder Pennick came round.", "Alder Brannock") is None  # same first name, other person
    assert ef.near_name("the brennock was cold", "Brannock") is None  # a lower-case word is not a name
    assert ef.near_word("Ivo", "Iva") and not ef.near_word("Ivo", "Ida")  # short names: one edit at most
    assert not ef.near_word("Al", "Ab")  # under three letters: never


def _row(day, header, text, score=0.5):
    return {"day": day, "header": header, "text": text, "score": score}


def test_focus_keeps_exact_entries_and_says_why_others_left():
    rows = [
        _row("2030-01-10", "09:00 — Chat with Alder Brennock", "Alder Brennock likes nettle tea.", 0.9),
        _row("2030-01-08", "12:00 — Bumped into Alder Brannock", "Alder Brannock likes smoked lapsang.", 0.8),
        _row("2030-01-09", "19:00 — Studio day", "Inked the coastal chart.", 0.7),
        _row("2030-01-30", "22:00 — Correction", "Correction: Alder Brannock's tea is white peony.", 0.6),
    ]
    res = ef.focus_candidates("What is Alder Brannock's favourite tea?", rows)
    assert res.applied and [rank for rank, _ in res.kept] == [2, 4]
    assert [(rank, reason) for rank, _, reason in res.removed] == [(1, "near_name"), (3, "entity_absent")]
    assert res.corrected_by == {2: "2030-01-30"}  # the earlier entry is superseded by the correction
    assert "dated before the correction of 2030-01-30" in ef.entry_label(res, 2)
    assert ef.entry_label(res, 4) == ""
    notes = ef.context_notes(res)
    assert notes and '"Alder Brannock"' in notes[0] and "similar spelling were left out" in notes[0]

    anchored = ef.focus_candidates("As of January 20, 2030, what was Alder Brannock's tea?", rows)
    assert [rank for rank, _ in anchored.kept] == [2] and (4, rows[3], "after_time_anchor") in anchored.removed
    assert anchored.corrected_by == {}
    assert "Time anchor" in ef.context_notes(anchored)[0]


def test_focus_without_an_exact_entry_keeps_everything_and_labels_near_names():
    rows = [_row("2030-01-10", "09:00 — Chat with Alder Brennock", "Alder Brennock likes nettle tea."),
            _row("2030-01-09", "19:00 — Studio day", "Inked the coastal chart.")]
    res = ef.focus_candidates("What is Alder Brannock's favourite tea?", rows)
    assert not res.applied and [rank for rank, _ in res.kept] == [1, 2] and res.removed == []
    assert ef.entry_label(res, 1) == '; names "Alder Brennock", not "Alder Brannock": a different person'
    assert "No retrieved entry names it exactly" in ef.context_notes(res)[0]
    # A misspelled question still finds the entry: no exact match means nothing is dropped.
    assert [rank for rank, _ in ef.focus_candidates("What does Alder Brenock drink?", rows).kept] == [1, 2]


def test_focus_limit_is_recorded_as_a_removal():
    rows = [_row("2030-01-0%d" % (i + 1), "09:00 — Tobiah", f"Tobiah visit {i}") for i in range(5)]
    res = ef.focus_candidates("What did Tobiah give me?", rows, limit=3)
    assert len(res.kept) == 3 and [(r, why) for r, _, why in res.removed] == [(4, "candidate_limit"), (5, "candidate_limit")]


# ---------------- retrieval precision on the gate's near-name fixture ----------------


def test_near_name_precision_meets_the_gate_without_losing_recall(world):
    before, after = metrics(world, focus=False), metrics(world, focus=True)
    cand, ctx = THRESHOLDS["candidate_retrieval"], THRESHOLDS["context_load"]
    assert before["queries"] == after["queries"] > 40
    # The stub reproduces the gate failure: full recall, near-name distractors fill the top k.
    assert before["recall"] >= cand["recall_at_k_min"] and before["precision"] < cand["precision_at_k_min"]
    assert after["precision"] >= cand["precision_at_k_min"]
    assert after["recall"] >= max(before["recall"], cand["recall_at_k_min"])
    assert after["context_recall"] >= max(before["context_recall"], ctx["recall_min"])
    needles = metrics(world, focus=True, categories={"needle", "paraphrase", "temporal"})
    assert needles["precision"] == 1.0 and needles["recall"] == 1.0


def test_trace_accounts_for_every_vector_candidate(world):
    for q in world.queries:
        got = ask(world, q["text"], _day(q))
        focus, cands = got.trace["focus"], got.trace["candidates"]
        if not ef.analyse(q["text"]).active:
            assert focus == {"checked": False} and got.trace["params"]["candidate_k"] == 2 * 8
            continue
        assert focus["checked"] and got.trace["params"]["candidate_k"] == 4 * 8
        assert len(cands) <= 2 * 8
        assert focus["vector_candidates"] == len(cands) + len(focus["removed"])
        ranks = sorted([c["vector_rank"] for c in cands] + [r["vector_rank"] for r in focus["removed"]])
        assert ranks == list(range(1, focus["vector_candidates"] + 1))
        assert [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
        assert [c["vector_rank"] for c in cands] == sorted(c["vector_rank"] for c in cands)  # never reordered
        assert {r["id"] for r in focus["removed"]}.isdisjoint(e["id"] for e in got.trace["loaded"])
        assert got.trace["rerank"] == {"present": False}
        for r in focus["removed"]:
            assert set(r) <= {"vector_rank", "id", "day", "time", "score", "reason"}
            assert r["reason"] in {"near_name", "entity_absent", "after_time_anchor", "candidate_limit"}
        blob = json.dumps(got.trace)
        for name in ef.asked_entities(q["text"]):
            assert name not in blob  # counts and refs only: never the asked names
        for fact in q["expected_facts"] + q["forbidden_facts"]:
            assert fact not in blob


def test_trace_off_gives_the_same_context(world):
    ids = ("Q-needle-00", "Q-needle-10", "Q-multi-0", "Q-ambig-0", "Q-unans-4")
    picked = [world.by_id[i] for i in ids] + [q for q in world.queries if q["category"] in ("temporal", "correction")]
    for q in picked:
        asm = _assembler(world.retriever)
        assert asm.build(_day(q), q["text"]) == asm.build(_day(q), q["text"], trace=RetrievalTrace()), q["id"]


def test_a_message_without_names_or_dates_is_unchanged(world):
    for text in ("how have the survey days felt lately?", "Which film did I see at the cinema on my birthday?"):
        on, off = ask(world, text, date(2030, 3, 7)), ask(world, text, date(2030, 3, 7), focus=False)
        assert on.messages == off.messages and on.trace["candidates"] == off.trace["candidates"]
        assert on.trace["focus"] == {"checked": False} and "Asked about" not in on.block


# ---------------- the answer-assembly context ----------------


def test_needle_context_holds_only_the_asked_person(world):
    seen = 0
    for q in world.queries:
        if q["category"] not in ("needle", "paraphrase"):
            continue
        holder = ef.asked_entities(q["text"])[0]
        got = ask(world, q["text"], _day(q))
        block = _norm(got.block)
        assert _kinds(world, got.loaded) <= {"needle", "restatement"}, q["id"]
        assert all(_norm(f) in block for f in q["expected_facts"]), q["id"]
        assert not any(_norm(f) in block for f in q["forbidden_facts"]), q["id"]
        assert f'(asked about "{holder.lower()}". every entry below names it exactly.' in block
        # Without the focus the same context mixes in the near-name distractor (the #116 failure).
        assert "distractor" in _kinds(world, ask(world, q["text"], _day(q), focus=False).loaded), q["id"]
        seen += 1
    assert seen >= 30


def test_followup_about_the_near_name_gets_that_person_only(world):
    seen = 0
    for q in world.queries:
        if "followup" not in q:
            continue
        f = q["followup"]
        said = q["expected_facts"][0] if q["expected_facts"] else "nothing"
        turns = [{"role": "user", "content": q["text"]}, {"role": "companion", "content": f"Your note says {said}."}]
        got = ask(world, f["text"], _day(q), turns=turns)
        block = _norm(got.block)
        assert score.best_recall(f["relevant_entry_ids"], got.loaded) == 1.0, q["id"]
        assert all(_norm(x) in block for x in f["expected_facts"]), q["id"]
        assert not any(_norm(x) in block for x in f["forbidden_facts"]), q["id"]
        assert _kinds(world, got.loaded) == {"distractor"}, q["id"]
        seen += 1
    assert seen >= 4


def test_correction_context_marks_the_superseded_entry(world):
    seen = 0
    for q in world.queries:
        if q["category"] != "correction":
            continue
        got = ask(world, q["text"], _day(q))
        correction = next(i for i in got.loaded if world.entries[i]["kind"] == "correction")
        cday = world.entries[correction]["day"]
        assert all(_norm(f) in _norm(got.block) for f in q["expected_facts"])
        stale = [line for line in got.block.splitlines() if line.startswith("--- [") and f"correction of {cday}" in line]
        originals = [i for i in got.loaded if world.entries[i]["kind"] == "needle"]
        assert len(stale) == len(originals) >= 1, q["id"]
        assert _kinds(world, got.loaded) <= {"needle", "correction", "restatement"}
        seen += 1
    assert seen == 4


def test_time_anchored_context_leaves_out_the_later_correction(world):
    seen = 0
    for q in world.queries:
        if q["category"] != "temporal":
            continue
        got = ask(world, q["text"], _day(q))
        block = _norm(got.block)
        assert all(_norm(f) in block for f in q["expected_facts"])
        assert not any(_norm(f) in block for f in q["forbidden_facts"]), q["id"]
        assert "time anchor" in block and _kinds(world, got.loaded) == {"needle"}
        assert any(r["reason"] == "after_time_anchor" for r in got.trace["focus"]["removed"])
        seen += 1
    assert seen == 2


def test_deleted_entry_context_gives_nothing_to_answer_from(world):
    seen = 0
    for q in world.queries:
        if q["category"] != "deletion":
            continue
        got = ask(world, q["text"], _day(q))
        block = _norm(got.block)
        assert q["expect_abstain"] and q["forbidden_facts"]
        assert not any(_norm(f) in block for f in q["forbidden_facts"]), q["id"]  # neither old nor near-name value
        assert "withdrew this note" in block and _kinds(world, got.loaded) == {"needle"}
        assert any(r["reason"] == "near_name" for r in got.trace["focus"]["removed"])
        # Without the focus the near-name value is in the context: the #116 deleted-entry failure.
        unfocused = _norm(ask(world, q["text"], _day(q), focus=False).block)
        assert any(_norm(f) in unfocused for f in q["forbidden_facts"])
        seen += 1
    assert seen == 3


NEAR_ONLY = """## Monday, January 14, 2030

### 08:10 — Chat with Alder Brennock

**Me:** Had coffee with Alder Brennock near the boatyard. Funny coincidence: their favourite tea is nettle and mint.

**Assistant:** Marit had coffee with Alder Brennock, whose favourite tea is nettle and mint.

### 13:20 — Studio day

**Me:** Spent the session inking the legend for the coastal chart. Bright and cold.

**Assistant:** Marit spent the session inking the legend for the coastal chart.
"""


def test_entry_removed_outright_context_separates_the_near_name(tmp_path):
    """The asked person's entry is gone entirely: nothing names them, so nothing is dropped, but the
    near-name entry is labelled as a different person and the block says nothing names the asked one."""
    r = Retriever(tmp_path / "idx.db", TrigramLLM())
    _index_files(r, {"Entries/2030/January/January 14, 2030.md": NEAR_ONLY})
    question = "What is Alder Brannock's favourite tea?"
    trace = RetrievalTrace()
    block = _retrieved_block(_assembler(r).build(date(2030, 3, 7), question, trace=trace))
    assert '(Asked about "Alder Brannock". No retrieved entry names it exactly.' in block
    assert "say the diary does not record it" in block
    header = next(line for line in block.splitlines() if "Chat with Alder Brennock" in line)
    assert header.endswith('; names "Alder Brennock", not "Alder Brannock": a different person) ---')
    assert trace.data["focus"]["near_name_labelled"] == 1 and not trace.data["focus"]["applied"]
    off = RetrievalTrace()
    plain = _retrieved_block(_assembler(r, focus=False).build(date(2030, 3, 7), question, trace=off))
    # Recall kept: without an exact match nothing is dropped, the same entries load as with no focus.
    assert [e["id"] for e in trace.data["loaded"]] == [e["id"] for e in off.data["loaded"]] != []
    assert "a different person" not in plain and "Asked about" not in plain


def test_notes_and_labels_are_charged_to_the_budget(world):
    q = next(x for x in world.queries if x["category"] == "correction")
    got = ask(world, q["text"], _day(q))
    budget = got.trace["budget"]
    assert budget["used_chars"] <= budget["max_context_chars"]
    assert budget["used_chars"] > sum(e["chars"] for e in got.trace["loaded"])  # the opening note is charged
    assert any("correction of" in line for line in got.block.splitlines())


def test_system_prompt_states_the_name_correction_and_abstain_rules():
    text = (Path(__file__).resolve().parent.parent / "config" / "prompts" / "system.md").read_text()
    for rule in ("Different names are different people", "The latest record wins", "Withdrawn means not recorded",
                 "Abstain plainly", "Time-anchored questions", "Shared first names"):
        assert rule in text


@pytest.mark.parametrize("value", [False, "false", "0", "off", "no"])
def test_entity_focus_switch_turns_the_check_off(world, value):
    q = world.by_id["Q-needle-00"]
    asm = ContextAssembler(FakeStore(), world.retriever, _cfg(value))
    asm._system_prompt = lambda: "SYSTEM"
    trace = RetrievalTrace()
    asm.build(_day(q), q["text"], trace=trace)
    assert trace.data["focus"] == {"checked": False} and trace.data["params"]["candidate_k"] == 16
