"""services#173: an elliptical follow-up about a different name ("And what about Tamsin Fenimore?") becomes the
earlier question asked about that name, for retrieval and as the final user message. The earlier exchange is
kept and labelled as being about the other person, and so is today's log when it holds that exchange.
Same-person and non-elliptical messages are unchanged. Synthetic names and facts only; no model call."""
from __future__ import annotations

from datetime import date

import pytest

from agent import follow_up
from agent.config import Config
from agent.context import ContextAssembler

BEGIN, END = "=== RETRIEVED PAST ENTRIES", "=== END RETRIEVED PAST ENTRIES ==="
TODAY = date(2030, 3, 7)

ENTRIES = [
    {"day": "2030-02-27", "header": "12:27 — Chat with Tamsin Fennimore",
     "text": "**Me:** Tamsin Fennimore told me their map projection is the Winkel tripel."},
    {"day": "2030-02-23", "header": "19:26 — Chat with Tamsin Fenimore",
     "text": "**Me:** Had coffee with Tamsin Fenimore. Their map projection is the Eckert IV."},
    {"day": "2030-01-08", "header": "09:24 — Chat with Orla Brannock",
     "text": "**Me:** Orla Brannock uses a 5-inch maksutov."},
    {"day": "2030-01-12", "header": "18:27 — Chat with Orla Brennock",
     "text": "**Me:** Orla Brennock uses a 12-inch truss dobsonian."},
]


class Store:
    index_enabled = False

    def __init__(self, today=""):
        self.today = today

    def get_day_text(self, day, max_chars=None):
        return self.today if day == TODAY else ""

    def get_standing_sections_text(self, max_chars=None):
        return ""


class Retriever:
    """Entries whose header shares a word with the query score higher; every query is recorded."""

    def __init__(self):
        self.queries = []

    def search(self, query, top_k=8, min_score=0.3, with_ids=False, diag=None):
        self.queries.append(query)
        words = {w.strip("?.,").lower().removesuffix("'s") for w in query.split()}
        rows = []
        for i, e in enumerate(ENTRIES):
            hit = sum(1 for w in e["header"].lower().split() if w in words)
            rows.append({**e, "id": i, "score": round(0.6 + 0.05 * hit, 4)})
        rows.sort(key=lambda r: -r["score"])
        return rows[:top_k]


@pytest.fixture(autouse=True)
def _no_memory(monkeypatch):
    import agent.workspace_files as wf
    monkeypatch.setattr(wf, "memory_text", lambda store: "")


def _build(message, turns, today=""):
    cfg = Config({"context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
                  "retrieval": {"top_k": 8, "min_score": 0.3, "max_context_tokens": 5200, "entity_focus": True},
                  "prompts": {"system_path": ""}})
    retr = Retriever()
    asm = ContextAssembler(Store(today), retr, cfg)
    asm._system_prompt = lambda: "SYSTEM"
    return asm.build(TODAY, message, session_turns=turns), retr.queries


def _turns(question, answer):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


FENNIMORE = _turns("Which map projection does Tamsin Fennimore prefer?",
                   "According to your entry from February 27, 2030, Tamsin Fennimore's map projection is the Winkel tripel.")
BRANNOCK = _turns("What kind of telescope does Orla Brannock use?",
                  "Orla Brannock uses a 5-inch maksutov, according to your entry from January 8, 2030.")


@pytest.mark.parametrize("turns,message,standalone,earlier,earlier_value", [
    (FENNIMORE, "And what about Tamsin Fenimore?", "Which map projection does Tamsin Fenimore prefer?",
     "Tamsin Fennimore", "Winkel tripel"),
    (FENNIMORE, "What about Tamsin Fenimore?", "Which map projection does Tamsin Fenimore prefer?",
     "Tamsin Fennimore", "Winkel tripel"),
    (BRANNOCK, "And Orla Brennock?", "What kind of telescope does Orla Brennock use?",
     "Orla Brannock", "maksutov"),
    (BRANNOCK, "how about Orla Brennock", "What kind of telescope does Orla Brennock use?",
     "Orla Brannock", "maksutov"),
])
def test_a_near_name_follow_up_becomes_a_standalone_question(turns, message, standalone, earlier, earlier_value):
    messages, queries = _build(message, turns)
    assert messages[-1] == {"role": "user", "content": standalone}
    assert queries == [standalone], "retrieval searches the standalone question alone"
    block = messages[1]["content"]
    retrieved = block[block.index(BEGIN):block.index(END)]
    assert earlier_value not in retrieved and earlier not in retrieved
    # The earlier exchange is replayed, not dropped, and labelled as about the other person.
    assert turns[1]["content"] in block
    assert f'the earlier exchange above was about "{earlier}"' in block
    assert "different people" in block
    assert block.index(turns[1]["content"]) < block.index("(Note: the earlier exchange")


def test_todays_log_holding_the_earlier_exchange_is_labelled():
    today = "### 12:00 — Chat\n**Me:** %s\n\n**Assistant:** %s\n" % (FENNIMORE[0]["content"], FENNIMORE[1]["content"])
    messages, _ = _build("And what about Tamsin Fenimore?", FENNIMORE, today=today)
    block = messages[1]["content"]
    note = ("(Today's log may include the earlier exchange about \"Tamsin Fennimore\". It is about "
            "\"Tamsin Fennimore\", not \"Tamsin Fenimore\".)")
    assert note in block and block.index(note) < block.index("### 12:00")
    # A today's log that does not mention the earlier person gets no label.
    messages, _ = _build("And what about Tamsin Fenimore?", FENNIMORE, today="### 08:00 — Walk\nA quiet walk.\n")
    assert "Today's log may include" not in messages[1]["content"]


@pytest.mark.parametrize("message", ["And what about Tamsin Fennimore?", "And what about Tamsin?",
                                     "What about Fennimore?"])
def test_a_same_person_follow_up_is_unchanged(message):
    messages, queries = _build(message, FENNIMORE)
    assert messages[-1] == {"role": "user", "content": message}
    assert queries == [FENNIMORE[0]["content"] + "\n" + message]
    assert "(Note: the earlier exchange" not in messages[1]["content"]


@pytest.mark.parametrize("message,turns", [
    ("Is that written in my diary, or are you inferring it?", FENNIMORE),
    ("And what about her boat?", FENNIMORE),
    ("And what about Tamsin Fenimore's boat?", FENNIMORE),
    ("Which map projection does Tamsin Fenimore prefer?", FENNIMORE),
    ("And what about Tamsin Fenimore?", None),
    ("And what about Tamsin Fenimore?", _turns("Did Tamsin Fennimore meet Orla Brannock?", "Synthetic answer.")),
    ("And what about Tamsin Fenimore?", _turns("How was my week?", "Synthetic answer.")),
])
def test_other_messages_are_unchanged(message, turns):
    messages, _ = _build(message, turns)
    assert messages[-1] == {"role": "user", "content": message}
    assert "(Note: the earlier exchange" not in messages[1]["content"]
    assert "Today's log may include" not in messages[1]["content"]


def test_an_unanswered_earlier_question_is_not_used():
    # #1505: only answered exchanges are replayed, so a question without a reply resolves nothing.
    turns = [{"role": "user", "content": "Which map projection does Tamsin Fennimore prefer?"}]
    messages, _ = _build("And what about Tamsin Fenimore?", turns)
    assert messages[-1]["content"] == "And what about Tamsin Fenimore?"


def test_a_different_name_is_labelled_as_a_different_name_not_a_near_spelling():
    messages, _ = _build("And what about Kestrel Ivensey?", BRANNOCK)
    assert messages[-1]["content"] == "What kind of telescope does Kestrel Ivensey use?"
    block = messages[1]["content"]
    assert '"Kestrel Ivensey", a different name.' in block
    messages, _ = _build("And what about Orla Brennock?", BRANNOCK)
    assert '"Orla Brennock", a near spelling of that name.' in messages[1]["content"]
