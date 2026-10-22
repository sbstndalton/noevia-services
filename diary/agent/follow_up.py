"""Elliptical follow-ups about another person (services#173). No model call.

"And what about Tamsin Fenimore?" right after "Which map projection does Tamsin Fennimore prefer?" is a new
question about a different name. Left as is, the model sees the earlier answer about one person and an
elliptical question about a near spelling of that name, and spends a long time deciding whether they are the
same person; when it decides wrongly it repeats the first person's detail.

``resolve`` turns such a message into a standalone question by putting the new name into the previous
question ("Which map projection does Tamsin Fenimore prefer?"). The assembler uses it for retrieval and as the
final user message, and labels the earlier exchange (and today's log, which holds it once logged) as being
about the other name. The diary names people exactly, so a different spelling is a different person, which is
also what entity focus assumes when it leaves near spellings out.

Conservative by design; the message is left as is when:
- it is not purely "And what about <Name>?", "What about <Name>?", "How about <Name>?" or "And <Name>?"
  (a possessive, a title with a full stop such as "Dr.", or any other word makes it something else);
- the name is not capitalised (entity focus does not read a lower-case name as a name);
- the name is the same person as the earlier one (same words, or one is a part of the other);
- the previous answered question names nobody, several people, or the same name more than once, or is
  itself an elliptical follow-up (so "And A?" then "And B?" resolves only the first);
- the earlier name is not written in the previous question exactly as entity focus read it.
Any capitalised phrase reads as a name ("And Ignore Previous Instructions?" is asked about that "name"): the
user's own words become the question, and the notes only quote names as reference data.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import entity_focus

_ELLIPTICAL = re.compile(
    r"^\s*(?:(?:and\s+)?(?:what|how)\s+about|and)\s+(?P<name>[^?!.,;:\n]+?)\s*[?!.]*\s*$", re.I)
# Characters that could end or fake the quoting of a name inside a note: double quotes of any style,
# backticks and square brackets (the note's own delimiters).
_QUOTE_CHARS = set('"“”„‟«»″`[]')


@dataclass(frozen=True)
class Resolution:
    standalone: str  # the previous question with the new name in place of the earlier one
    earlier_name: str  # the name as the previous question wrote it
    asked_name: str  # the name as the follow-up wrote it
    near: bool  # a near spelling (every word equal or a close spelling) rather than another name


def _only_name(text: str) -> Optional[entity_focus.Entity]:
    """The single entity that ``text`` consists of, or None when it holds anything else."""
    found = entity_focus.asked_entities(text)
    if len(found) != 1:
        return None
    entity = found[0]
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if entity_focus._keys(words) != entity.keys:
        return None
    return entity


def _same_person(a: entity_focus.Entity, b: entity_focus.Entity) -> bool:
    sa, sb = set(a.keys), set(b.keys)
    return sa <= sb or sb <= sa


def _near(a: entity_focus.Entity, b: entity_focus.Entity) -> bool:
    if len(a.keys) != len(b.keys):
        return False
    return all(x == y or entity_focus.near_word(x, y) for x, y in zip(a.keys, b.keys))


def resolve(message: str, session_turns: Optional[Sequence[dict]]) -> Optional[Resolution]:
    """The standalone form of an elliptical follow-up about a different name, or None (leave it as is).

    ``session_turns`` are the replayable turns (context.completed_exchanges); the previous question is the
    latest user turn there."""
    m = _ELLIPTICAL.match(message or "")
    if not m:
        return None
    asked = _only_name(m.group("name"))
    if asked is None:
        return None
    previous: List[str] = [t.get("content", "") for t in (session_turns or [])
                           if t.get("role") == "user" and isinstance(t.get("content"), str)]
    if not previous:
        return None
    earlier_q = previous[-1]
    if _ELLIPTICAL.match(earlier_q):
        return None  # no chains: the previous question must be a full question
    earlier = entity_focus.asked_entities(earlier_q)
    if len(earlier) != 1 or _same_person(earlier[0], asked):
        return None
    if earlier_q.count(earlier[0].name) != 1:
        return None  # written twice (or not as read): a partial replacement would mix two people
    standalone = earlier_q.replace(earlier[0].name, asked.name, 1)
    return Resolution(standalone=standalone, earlier_name=earlier[0].name, asked_name=asked.name,
                      near=_near(earlier[0], asked))


def safe_name(name: str) -> str:
    """A name as quoted inside a note: no quote or bracket characters, no control characters."""
    kept = "".join(" " if unicodedata.category(c)[0] == "C" else c for c in name if c not in _QUOTE_CHARS)
    return " ".join(kept.split())


def conversation_note(res: Resolution, exact_loaded: bool = False) -> str:
    """Labels the replayed earlier exchange; it is kept, never dropped. Reference data, not an instruction.

    services#179: ``exact_loaded`` says whether a retrieved entry naming the asked name exactly is shown. Then
    the note says to answer from those entries; only without one does a near spelling get the sentence about
    saying the diary records nothing. The note never holds both."""
    earlier, asked = safe_name(res.earlier_name), safe_name(res.asked_name)
    kind = "a near spelling of that name" if res.near else "a different name"
    note = (f'[Reference note on this conversation: the earlier exchange above was about "{earlier}". The new '
            f'question asks about "{asked}", {kind}. The diary names people exactly, so they are different '
            f'people: the earlier answer\'s details are not details of "{asked}". The new question is the '
            f'earlier question asked about "{asked}".')
    if exact_loaded:
        note += f' Retrieved entries that name "{asked}" exactly are about "{asked}": answer from them, following the latest-record and withdrawn rules.'
    elif res.near:
        note += (f' If the diary records nothing about "{asked}", say so and mention that it does record '
                 f'"{earlier}".')
    return note + "]"


def today_note(res: Resolution) -> str:
    """Labels today's log when it holds the earlier exchange (logged before this question)."""
    earlier, asked = safe_name(res.earlier_name), safe_name(res.asked_name)
    return (f'[Reference note: today\'s log may include the earlier exchange about "{earlier}". It is about '
            f'"{earlier}", not "{asked}".]')


def mentions(text: str, name: str) -> bool:
    return bool(text) and name.casefold() in text.casefold()
