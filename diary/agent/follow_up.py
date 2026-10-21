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

Unchanged: any message that is not purely "And what about <Name>?", "What about <Name>?", "How about
<Name>?" or "And <Name>?"; a name that is the same person as the earlier one (same words, or one is a
part of the other, such as a given name alone); and a previous question that names nobody or several people.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import entity_focus

_ELLIPTICAL = re.compile(
    r"^\s*(?:(?:and\s+)?(?:what|how)\s+about|and)\s+(?P<name>[^?!.,;:\n]+?)\s*[?!.]*\s*$", re.I)


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
    earlier = entity_focus.asked_entities(earlier_q)
    if len(earlier) != 1 or _same_person(earlier[0], asked):
        return None
    if earlier[0].name not in earlier_q:
        return None
    standalone = earlier_q.replace(earlier[0].name, asked.name, 1)
    if standalone == earlier_q or _ELLIPTICAL.match(standalone):
        return None
    return Resolution(standalone=standalone, earlier_name=earlier[0].name, asked_name=asked.name,
                      near=_near(earlier[0], asked))


def conversation_note(res: Resolution) -> str:
    """Labels the replayed earlier exchange; it is kept, never dropped."""
    kind = "a near spelling of that name" if res.near else "a different name"
    return (f'(Note: the earlier exchange above was about "{res.earlier_name}". The new question asks about '
            f'"{res.asked_name}", {kind}. The diary names people exactly, so they are different people: do not '
            f'give the earlier answer\'s details as details of "{res.asked_name}". The new question is the '
            f'earlier question asked about "{res.asked_name}".)')


def today_note(res: Resolution) -> str:
    """Labels today's log when it holds the earlier exchange (logged before this question)."""
    return (f'(Today\'s log may include the earlier exchange about "{res.earlier_name}". It is about '
            f'"{res.earlier_name}", not "{res.asked_name}".)')


def mentions(text: str, name: str) -> bool:
    return bool(text) and name.casefold() in text.casefold()
