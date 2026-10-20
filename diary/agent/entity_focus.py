"""Entity focus: a cheap lexical check on retrieved candidates (services#116).

Vector similarity cannot tell "Alder Brannock" from "Alder Brennock". Entries about a similarly spelled
person score as high as the entry about the person asked about, fill the loaded context, and the answer
then presents one person's detail as competing with another's. This module needs no model call:

- Asked entities: capitalised name phrases in the user's current message. Question words, pronouns,
  common verbs, months and weekdays are not names, and a single capitalised word that only starts a
  sentence is ignored. At most MAX_ENTITIES are kept.
- Exact: a candidate whose text names an asked entity as a whole phrase (case-insensitive).
- Near-name: a candidate that names only a similar but different name: the same number of words, each
  equal or within a small edit distance, at least one differing and capitalised.
- Time anchor: "as of <date>" (also "as at", "up to", "until", "before") drops candidates dated after it.

focus_candidates(): when at least one candidate names an asked entity exactly, only those candidates are
kept (higher precision; the entries about the asked entity are exactly the ones naming it). Otherwise every
candidate is kept (no loss of recall, e.g. for a misspelled name) and near-name candidates are labelled so
the context can say they are about someone else. Score order is never changed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

MAX_ENTITIES = 4

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_ABBR["sept"] = 9
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Capitalised words that are not names. Lower case; compared after possessive stripping.
_NOT_NAMES = set(MONTHS) | set(_WEEKDAYS) | {d + "s" for d in _WEEKDAYS} | set(_MONTH_ABBR) | {
    "i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself", "we", "we're", "we've", "us", "our", "you",
    "you're", "your", "he", "she", "it", "it's", "its", "they", "they're", "them", "his", "her", "their",
    "what", "what's", "which", "who", "who's", "whom", "whose", "where", "where's", "when", "why", "how",
    "how's", "and", "or", "but", "so", "nor", "yet", "the", "a", "an", "as", "at", "of", "in", "on", "to",
    "for", "from", "with", "by", "about", "after", "before", "since", "until", "during", "between", "into",
    "do", "does", "did", "is", "are", "was", "were", "be", "been", "am", "can", "could", "would", "should",
    "will", "shall", "may", "might", "must", "have", "has", "had", "not", "no", "yes", "if", "then", "than",
    "this", "that", "these", "those", "there", "here", "any", "anything", "some", "something", "everything",
    "all", "each", "every", "just", "still", "again", "also", "last", "next", "today", "tonight",
    "yesterday", "tomorrow", "ok", "okay", "hi", "hello", "hey", "thanks", "thank", "please", "dear",
    "diary", "note", "notes", "entry", "entries", "tell", "remind", "remember", "recall", "show", "give",
    "list", "find", "summarise", "summarize", "describe", "explain", "let", "let's", "say", "said",
}

_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['\-](?=[^\W\d_]))*")
_POSSESSIVE = re.compile(r"(?:'s|')$", re.I)
_SENTENCE_END = set(".!?:;\n\"(")


def _norm(text: str) -> str:
    return (text or "").replace("’", "'").replace("‘", "'")


def _words(text: str) -> List[Tuple[str, int, int, bool]]:
    """(word without possessive, start, end, had a possessive) for each word of ``text``."""
    out = []
    for m in _WORD.finditer(text):
        w = m.group(0)
        stripped = _POSSESSIVE.sub("", w) if len(w) > 2 else w
        out.append((stripped or w, m.start(), m.end(), bool(stripped) and stripped != w))
    return out


def _sentence_start(text: str, pos: int) -> bool:
    i = pos - 1
    while i >= 0 and text[i] in " \t\r*_":
        i -= 1
    return i < 0 or text[i] in _SENTENCE_END


def asked_entities(message: str) -> List[str]:
    """Capitalised name phrases in ``message``, in order, case-insensitively unique."""
    text = _norm(message)
    runs: List[List[Tuple[str, int]]] = []  # [(word, start)]
    current: List[Tuple[str, int]] = []
    prev_end, prev_possessive = None, False
    for word, start, end, possessive in _words(text):
        joined = prev_end is not None and text[prev_end:start].strip(" \t") == "" and not prev_possessive
        name_like = word[:1].isupper() and len(word) > 1 and word.lower() not in _NOT_NAMES
        if name_like and current and joined:
            current.append((word, start))
        else:
            if current:
                runs.append(current)
            current = [(word, start)] if name_like else []
        prev_end, prev_possessive = end, possessive
    if current:
        runs.append(current)
    out, seen = [], set()
    for run in runs:
        if len(run) == 1 and _sentence_start(text, run[0][1]):
            continue  # a capitalised word that only starts a sentence says nothing
        phrase = " ".join(w for w, _ in run)
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            out.append(phrase)
        if len(out) >= MAX_ENTITIES:
            break
    return out


# ---------------- time anchor ----------------

_DAY_MONTH_YEAR = r"(?P<d1>\d{1,2})(?:st|nd|rd|th)?\s+(?P<m1>[A-Za-z]{3,9})\.?,?\s+(?P<y1>\d{4})"
_MONTH_DAY_YEAR = r"(?P<m2>[A-Za-z]{3,9})\.?\s+(?P<d2>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y2>\d{4})"
_ISO = r"(?P<iso>\d{4}-\d{2}-\d{2})"
_ANCHOR = re.compile(r"\b(?P<kw>as\s+of|as\s+at|up\s+to|until|till|before)\s+(?:the\s+)?(?:"
                     + _ISO + "|" + _MONTH_DAY_YEAR + "|" + _DAY_MONTH_YEAR + r")\b", re.I)


def _month_number(name: str) -> Optional[int]:
    name = name.lower()
    if name in MONTHS:
        return MONTHS.index(name) + 1
    return _MONTH_ABBR.get(name)


def time_anchor(message: str) -> Optional[Tuple[date, bool]]:
    """(cutoff day, inclusive) for the first "as of <date>"-style anchor in ``message``, else None."""
    for m in _ANCHOR.finditer(_norm(message)):
        try:
            if m.group("iso"):
                day = date.fromisoformat(m.group("iso"))
            elif m.group("m2"):
                month = _month_number(m.group("m2"))
                day = date(int(m.group("y2")), month, int(m.group("d2"))) if month else None
            else:
                month = _month_number(m.group("m1"))
                day = date(int(m.group("y1")), month, int(m.group("d1"))) if month else None
        except ValueError:
            day = None
        if day is not None:
            return day, m.group("kw").lower() != "before"
    return None


# ---------------- matching ----------------

def _distance(a: str, b: str, limit: int) -> int:
    """Optimal-string-alignment edit distance, stopping early once it exceeds ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def near_word(a: str, b: str) -> bool:
    """True when two different words are close spellings: one edit for short words, two from six letters."""
    a, b = a.lower(), b.lower()
    if a == b or min(len(a), len(b)) < 3:
        return False
    limit = 1 if min(len(a), len(b)) <= 5 else 2
    return _distance(a, b, limit) <= limit


def _phrase_re(entity: str) -> "re.Pattern":
    return re.compile(r"(?<![^\W_])" + r"\s+".join(re.escape(w) for w in entity.split()) + r"(?![^\W_])", re.I)


def names_exactly(text: str, entity: str) -> bool:
    return bool(_phrase_re(_norm(entity)).search(_norm(text)))


def near_name(text: str, entity: str) -> Optional[str]:
    """The first similar but different name for ``entity`` in ``text`` (None if there is none)."""
    target = [w.lower() for w in _norm(entity).split()]
    text = _norm(text)
    words = _words(text)
    n = len(target)
    for i in range(len(words) - n + 1):
        window = words[i:i + n]
        if any(text[window[k][2]:window[k + 1][1]].strip(" \t") or window[k][3] for k in range(n - 1)):
            continue  # one phrase only: spaces between the words, no possessive inside
        differs = False
        for (word, _, _, _), want in zip(window, target):
            if word.lower() == want:
                continue
            if word[:1].isupper() and near_word(word, want):
                differs = True
                continue
            break
        else:
            if differs:
                return " ".join(w for w, _, _, _ in window)
    return None


_CORRECTION = re.compile(r"\bcorrect(?:ion|ed|ing)\b", re.I)


@dataclass
class FocusResult:
    entities: List[str] = field(default_factory=list)
    anchor: Optional[Tuple[date, bool]] = None
    applied: bool = False  # some candidate named an asked entity exactly: only those were kept
    kept: List[Tuple[int, dict]] = field(default_factory=list)  # (vector rank, row), score order
    removed: List[Tuple[int, dict, str]] = field(default_factory=list)  # (vector rank, row, reason)
    near: Dict[int, Tuple[str, str]] = field(default_factory=dict)  # vector rank -> (name in entry, asked entity)
    corrected_by: Dict[int, str] = field(default_factory=dict)  # vector rank -> day of a later correction

    @property
    def active(self) -> bool:
        """The message names an entity or a time anchor, so the candidates were checked."""
        return bool(self.entities or self.anchor)


def analyse(message: str) -> FocusResult:
    return FocusResult(entities=asked_entities(message), anchor=time_anchor(message))


def focus_candidates(message: str, rows: List[dict], limit: Optional[int] = None) -> FocusResult:
    """Check vector candidates (``rows`` in score order) against ``message``; keep at most ``limit``.

    Removal reasons: ``after_time_anchor``, ``near_name`` (names only a similar, different name),
    ``entity_absent`` (names no asked entity while others do) and ``candidate_limit`` (beyond ``limit``).
    Kept rows stay in score order."""
    res = analyse(message)
    ranked = list(enumerate(rows, 1))
    if res.anchor is not None:
        cutoff, inclusive = res.anchor
        survivors = []
        for rank, row in ranked:
            try:
                day = date.fromisoformat(str(row.get("day")))
            except ValueError:
                day = None
            if day is not None and (day > cutoff or (day == cutoff and not inclusive)):
                res.removed.append((rank, row, "after_time_anchor"))
            else:
                survivors.append((rank, row))
        ranked = survivors
    if res.entities:
        exact = [(rank, row) for rank, row in ranked if _names_any(row, res.entities)]
        if exact:
            res.applied = True
            keep = {rank for rank, _ in exact}
            for rank, row in ranked:
                if rank not in keep:
                    res.removed.append((rank, row, "near_name" if _near(row, res.entities) else "entity_absent"))
            ranked = exact
        else:
            for rank, row in ranked:
                hit = _near(row, res.entities)
                if hit:
                    res.near[rank] = hit
    if limit is not None and len(ranked) > limit:
        res.removed.extend((rank, row, "candidate_limit") for rank, row in ranked[limit:])
        ranked = ranked[:limit]
        res.near = {rank: hit for rank, hit in res.near.items() if rank in {r for r, _ in ranked}}
    res.kept = ranked
    if res.applied:
        # Entries about the same entity dated before an explicit correction of it: the correction wins.
        corrections = [str(row.get("day")) for _, row in ranked if _CORRECTION.search(_row_text(row))]
        if corrections:
            latest = max(corrections)
            for rank, row in ranked:
                if str(row.get("day")) < latest:
                    res.corrected_by[rank] = latest
    res.removed.sort(key=lambda r: r[0])
    return res


def _row_text(row: dict) -> str:
    return f"{row.get('header') or ''}\n{row.get('text') or ''}"


def _names_any(row: dict, entities: List[str]) -> bool:
    text = _row_text(row)
    return any(names_exactly(text, e) for e in entities)


def _near(row: dict, entities: List[str]) -> Optional[Tuple[str, str]]:
    if _names_any(row, entities):
        return None
    text = _row_text(row)
    for e in entities:
        hit = near_name(text, e)
        if hit:
            return hit, e
    return None


def context_notes(res: FocusResult) -> List[str]:
    """Lines that open the retrieved-entries block when the message names an entity or a time anchor."""
    lines = []
    if res.anchor is not None:
        cutoff, inclusive = res.anchor
        when, hidden = ("on", "after") if inclusive else ("before", "on or after")
        lines.append(f"(Time anchor: the message asks about the diary as it stood {when} {cutoff.isoformat()}. "
                     f"Entries dated {hidden} it are not shown.)")
    if res.entities:
        names = ", ".join(f'"{e}"' for e in res.entities)
        single = len(res.entities) == 1
        if res.applied:
            line = (f"(Asked about {names}. Every entry below names {'it' if single else 'one of them'} exactly. "
                    "A differently spelled name is a different person")
            if any(reason == "near_name" for _, _, reason in res.removed):
                line += "; entries naming only a similar spelling were left out"
            lines.append(line + ".)")
        else:
            lines.append(f"(Asked about {names}. No retrieved entry names {'it' if single else 'any of them'} exactly. "
                         "An entry about a differently spelled name is about a different person: do not answer from it. "
                         "If no entry records what was asked, say the diary does not record it.)")
    return lines


def entry_label(res: Optional[FocusResult], vector_rank: int) -> str:
    """Suffix for one retrieved entry's header line ('' when there is nothing to say)."""
    if res is None:
        return ""
    if vector_rank in res.near:
        found, asked = res.near[vector_rank]
        return f'; names "{found}", not "{asked}": a different person'
    if vector_rank in res.corrected_by:
        return f"; dated before the correction of {res.corrected_by[vector_rank]}, which wins where they disagree"
    return ""
