"""Entity focus: a cheap lexical check on retrieved candidates (services#116).

Vector similarity cannot tell "Alder Brannock" from "Alder Brennock". Entries about a similarly spelled
person score as high as the entry about the person asked about, and the answer then presents one person's
detail as competing with another's. This module needs no model call.

- Asked entities: capitalised name phrases in the user's message (question words, pronouns, common verbs,
  months and weekdays are not names; a single capitalised word that only starts a sentence is ignored;
  titles such as "Dr" or "Dr." are not part of the name). A follow-up that names nobody ("and her
  email?") carries the entities of the previous question in the same conversation.
- Each candidate is classified per asked entity. Names compare case-, accent-, title- and
  punctuation-insensitively:
    match    names the entity in full;
    partial  names only part of a multi-word name on its own ("Alder called", "Dr. Brannock");
    near     names a similar but different spelling (every word equal or within a small edit distance);
    variant  names a possible short form ("Al" for "Alder"): never treated as a different person;
    shared   names a different full name that shares a word with the entity.
- Only confirmed near-name hits are dropped: a near hit is dropped when some other candidate names the
  same entity in full, so the diary does name that person and the near spelling is someone else. With no
  full match for an entity nothing is dropped (a typo or a nickname in the question must not hide the entry);
  similar names are labelled "may be a different person; check the name" instead.
- Time anchor: an explicitly framed question ("As of <date>, ...?", "As at <date>", or a question opening
  "Before/Until/Up to <date>, ...?") drops candidates dated after it. A date in a statement does not.
- Supersession: an earlier entry about the same entity is labelled superseded only by an explicit
  correction of that entity that names the corrected value ("..., not <old value>") found in the earlier
  entry, or, without one, shares the corrected attribute's words.
Score order is never changed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

MAX_ENTITIES = 4

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_ABBR["sept"] = 9
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TITLES = {"dr", "mr", "mrs", "ms", "mx", "miss", "prof", "professor", "sir", "dame", "rev", "revd", "fr",
          "lord", "lady", "aunt", "auntie", "uncle", "cousin", "nan", "nana", "grandma", "grandpa"}

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
    "compare", "check", "ask", "see", "look", "help", "write", "read", "add", "update", "search",
}

_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['\-](?=[^\W\d_]))*")
_POSSESSIVE = re.compile(r"(?:'s|')$", re.I)
_SENTENCE_END = set(".!?:;\n\"(")


def _norm(text: str) -> str:
    return (text or "").replace("’", "'").replace("‘", "'")


def _key(word: str) -> str:
    """Comparison form of one name word: lower case, no accents, no apostrophes."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c))
    return plain.lower().replace("'", "")


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


def _cap_runs(text: str, stop: bool) -> List[List[Tuple[str, int]]]:
    """Runs of capitalised words joined only by spaces (or "." after a title). ``stop`` splits at non-names."""
    runs: List[List[Tuple[str, int]]] = []
    current: List[Tuple[str, int]] = []
    prev_end, prev_possessive, prev_title = None, False, False
    for word, start, end, possessive in _words(text):
        gap = text[prev_end:start].strip(" \t") if prev_end is not None else None
        joined = gap is not None and not prev_possessive and (gap == "" or (prev_title and gap == "."))
        name_like = word[:1].isupper() and (len(word) > 1 or not stop) and not (stop and word.lower() in _NOT_NAMES)
        if name_like and current and joined:
            current.append((word, start))
        else:
            if current:
                runs.append(current)
            current = [(word, start)] if name_like else []
        prev_end, prev_possessive, prev_title = end, possessive, _key(word) in TITLES
    if current:
        runs.append(current)
    return runs


def _keys(words: Sequence[str]) -> Tuple[str, ...]:
    """Name words without titles; hyphenated names split into their parts."""
    out = []
    for w in words:
        for part in w.split("-"):
            k = _key(part)
            if k and k not in TITLES:
                out.append(k)
    return tuple(out)


@dataclass(frozen=True)
class Entity:
    name: str  # as the user wrote it (titles included)
    keys: Tuple[str, ...]  # comparison words: no titles, accents, apostrophes or case


def asked_entities(message: str) -> List[Entity]:
    """Capitalised name phrases in ``message``, in order, unique by comparison form."""
    text = _norm(message)
    out, seen = [], set()
    for run in _cap_runs(text, stop=True):
        if len(run) == 1 and _sentence_start(text, run[0][1]):
            continue  # a capitalised word that only starts a sentence says nothing
        keys = _keys([w for w, _ in run])
        if not keys or keys in seen:
            continue
        seen.add(keys)
        out.append(Entity(" ".join(w for w, _ in run), keys))
        if len(out) >= MAX_ENTITIES:
            break
    return out


_FOLLOW_UP = re.compile(r"^\s*(?:and|also|plus|so|then|what about|how about)\b|\b(?:he|she|they|him|her|them|his|hers|"
                        r"their|theirs|that person)\b", re.I)


def carried_entities(message: str, earlier: Sequence[str]) -> List[Entity]:
    """For a follow-up that names nobody ("and her email?"): the entities of the latest earlier user message
    of the same conversation that named any. A message that is not worded as a follow-up carries nothing."""
    if asked_entities(message) or not _FOLLOW_UP.search(_norm(message)):
        return []
    for text in reversed(list(earlier or [])):
        found = asked_entities(text)
        if found:
            return found
    return []


# ---------------- time anchor ----------------

_DAY_MONTH_YEAR = r"(?P<d1>\d{1,2})(?:st|nd|rd|th)?\s+(?P<m1>[A-Za-z]{3,9})\.?,?\s+(?P<y1>\d{4})"
_MONTH_DAY_YEAR = r"(?P<m2>[A-Za-z]{3,9})\.?\s+(?P<d2>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y2>\d{4})"
_ISO = r"(?P<iso>\d{4}-\d{2}-\d{2})"
_ANCHOR = re.compile(r"\b(?P<kw>as\s+of|as\s+at|up\s+to|until|till|before)\s+(?:the\s+)?(?:"
                     + _ISO + "|" + _MONTH_DAY_YEAR + "|" + _DAY_MONTH_YEAR + r")\b", re.I)
_OPENING_ONLY = {"up to", "until", "till", "before"}


def _month_number(name: str) -> Optional[int]:
    name = name.lower()
    if name in MONTHS:
        return MONTHS.index(name) + 1
    return _MONTH_ABBR.get(name)


def _ends_sentence(text: str, i: int) -> bool:
    if text[i] in "!?\n":
        return True
    if text[i] != ".":
        return False
    before = re.search(r"([^\W\d_]+)$", text[:i])
    return not (before and _key(before.group(1)) in TITLES | set(_MONTH_ABBR))  # "Dr. Patel", "Sept. 9"


def _question_sentence(text: str, start: int, end: int) -> Tuple[bool, bool]:
    """(the sentence holding text[start:end] is a question, the match opens that sentence)."""
    i = start - 1
    while i >= 0 and not _ends_sentence(text, i):
        i -= 1
    opens = text[i + 1:start].strip(" \t\"'(*_") == ""
    j = end
    while j < len(text) and not _ends_sentence(text, j):
        j += 1
    return j < len(text) and text[j] == "?", opens


def time_anchor(message: str) -> Optional[Tuple[date, bool]]:
    """(cutoff day, inclusive) when a question is explicitly framed by a date, else None.

    "as of" / "as at" anywhere in a question; "before", "until", "till", "up to" only when they open the
    question ("Before 3 March 2030, what ...?"). A date in a statement ("Sam is away until ...") is not an
    anchor."""
    text = _norm(message)
    for m in _ANCHOR.finditer(text):
        kw = re.sub(r"\s+", " ", m.group("kw").lower())
        question, opens = _question_sentence(text, m.start(), m.end())
        if not question or (kw in _OPENING_ONLY and not opens):
            continue
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
            return day, kw != "before"
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
    a, b = _key(a), _key(b)
    if a == b or min(len(a), len(b)) < 3:
        return False
    limit = 1 if min(len(a), len(b)) <= 5 else 2
    return _distance(a, b, limit) <= limit


def _relation(word: str, want: str) -> Optional[str]:
    if word == want:
        return "eq"
    if near_word(word, want):
        return "near"
    if min(len(word), len(want)) >= 2 and (want.startswith(word) or word.startswith(want)):
        return "variant"  # a possible short form: "Al" / "Alder", "Alex" / "Alexander"
    return None


def _subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    it = iter(long)
    return bool(short) and all(word in it for word in short)


ABOUT = ("match", "partial")  # an entry about the entity (full name, or first name or surname alone)
_RANK = {"match": 5, "partial": 4, "near": 3, "variant": 2, "shared": 1}


def classify(text: str, entity: Entity) -> Tuple[Optional[str], Optional[str]]:
    """(kind, the name as written) of the strongest mention of ``entity`` in ``text``; kind is one of match,
    partial (a first name or surname on its own), near, variant, shared or None (module docstring)."""
    e, n = entity.keys, len(entity.keys)
    best: Tuple[Optional[str], Optional[str]] = (None, None)
    for run in _cap_runs(_norm(text), stop=False):
        words = [w for w, _ in run]
        keys = _keys(words)
        if not keys:
            continue
        shown = " ".join(words)
        if any(keys[i:i + n] == e for i in range(len(keys) - n + 1)):
            return "match", shown
        kinds = []
        if len(keys) < n and _subsequence(keys, e):
            kinds.append("partial")  # "Alder called", "Dr. Brannock": part of a longer name on its own
        windows = [keys[i:i + n] for i in range(len(keys) - n + 1)]
        if n >= 2 and len(keys) == 1:  # a lone word: compared with the first name and the surname
            rel = [_relation(keys[0], e[0]), _relation(keys[0], e[-1])]
            if "near" in rel:
                kinds.append("near")
            elif "variant" in rel:
                kinds.append("variant")
        for window in windows:
            rel = [_relation(a, b) for a, b in zip(window, e)]
            if all(rel):
                kinds.append("near" if all(r in ("eq", "near") for r in rel) else "variant")
            elif n >= 2 and "eq" in rel:
                kinds.append("shared")  # "Alder Pennick" for "Alder Brannock": another person
        for kind in kinds:
            if best[0] is None or _RANK[kind] > _RANK[best[0]]:
                best = (kind, shown)
    return best


@dataclass
class FocusResult:
    entities: List[Entity] = field(default_factory=list)
    carried: bool = False  # the entities come from the previous question (a name-less follow-up)
    anchor: Optional[Tuple[date, bool]] = None
    matched: List[Entity] = field(default_factory=list)  # entities some candidate names in full
    kept: List[Tuple[int, dict]] = field(default_factory=list)  # (vector rank, row), score order
    removed: List[Tuple[int, dict, str]] = field(default_factory=list)  # (vector rank, row, reason)
    similar: Dict[int, Tuple[str, str, str]] = field(default_factory=dict)  # rank -> (kind, name in entry, asked)
    corrected_by: Dict[int, str] = field(default_factory=dict)  # vector rank -> day of a later correction

    @property
    def active(self) -> bool:
        """The message (or the follow-up it continues) names an entity, or it has a time anchor."""
        return bool(self.entities or self.anchor)

    @property
    def applied(self) -> bool:
        return bool(self.matched)


def analyse(message: str, earlier: Sequence[str] = ()) -> FocusResult:
    entities = asked_entities(message)
    carried = [] if entities else carried_entities(message, earlier)
    return FocusResult(entities=entities or carried, carried=bool(carried), anchor=time_anchor(message))


def _row_text(row: dict) -> str:
    return f"{row.get('header') or ''}\n{row.get('text') or ''}"


def focus_candidates(message: str, rows: List[dict], limit: Optional[int] = None,
                     earlier: Sequence[str] = ()) -> FocusResult:
    """Check vector candidates (``rows`` in score order) against ``message``; keep at most ``limit``.

    Removal reasons: ``after_time_anchor``, ``near_name`` (names only a near spelling of an entity that
    another candidate names exactly) and ``candidate_limit`` (beyond ``limit``). Kept rows stay in order."""
    res = analyse(message, earlier)
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
        kinds = {rank: [classify(_row_text(row), e) for e in res.entities] for rank, row in ranked}
        res.matched = [e for i, e in enumerate(res.entities) if any(k[i][0] == "match" for k in kinds.values())]
        survivors = []
        for rank, row in ranked:
            found = kinds[rank]
            if any(kind in ABOUT for kind, _ in found):
                survivors.append((rank, row))
                continue
            confirmed = [i for i, (kind, _) in enumerate(found) if kind == "near" and res.entities[i] in res.matched]
            if confirmed:
                res.removed.append((rank, row, "near_name"))
                continue
            survivors.append((rank, row))
            similar = [(_RANK[kind], kind, shown, res.entities[i].name) for i, (kind, shown) in enumerate(found) if kind]
            if similar:
                res.similar[rank] = max(similar)[1:]
        ranked = survivors
    if limit is not None and len(ranked) > limit:
        res.removed.extend((rank, row, "candidate_limit") for rank, row in ranked[limit:])
        ranked = ranked[:limit]
        kept_ranks = {rank for rank, _ in ranked}
        res.similar = {rank: s for rank, s in res.similar.items() if rank in kept_ranks}
    res.kept = ranked
    res.corrected_by = _superseded(ranked, res.matched)
    res.removed.sort(key=lambda r: r[0])
    return res


# ---------------- explicit corrections ----------------

_CUE = re.compile(r"\bcorrection\b|\bI\s+(?:had\s+)?(?:misheard|misremembered|misspoke|misread)\b"
                  r"|\bI\s+(?:got|had)\s+(?:it|that|this)\s+wrong\b"
                  r"|\bto\s+correct\s+(?:my|an|the|that)\s+(?:earlier\s+|previous\s+)?(?:note|entry)\b", re.I)
_OLD_VALUE = re.compile(r"\b(?:not|instead\s+of|rather\s+than)\s+(?:the\s+|a\s+|an\s+)?([^.,;:!?\n]{2,60})", re.I)
_SENTENCE = re.compile(r"[^.!?\n]+")
_CUE_WORDS = {"correction", "correct", "corrected", "actually", "misheard", "misremembered", "wrong", "earlier",
              "previous", "note", "notes", "entry", "really", "instead", "rather", "after", "told", "said"}
_TIME = re.compile(r"^\s*(\d{1,2}:\d{2})")


def _order(row: dict) -> Tuple[str, str]:
    m = _TIME.match(str(row.get("header") or ""))
    return str(row.get("day") or ""), (m.group(1).zfill(5) if m else "")


def _plain(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", _key(_norm(text)))) + " "


def _content_words(sentences: Sequence[str], entity: Entity) -> set:
    words = set()
    for s in sentences:
        for w in re.findall(r"[a-z]+", _key(_norm(s))):
            if len(w) >= 3 and w not in _NOT_NAMES and w not in _CUE_WORDS and w not in entity.keys:
                words.add(w)
    return words


def _entity_sentences(text: str, entity: Entity) -> List[str]:
    return [s for s in _SENTENCE.findall(_norm(text)) if classify(s, entity)[0] in ABOUT]


def _superseded(ranked: List[Tuple[int, dict]], matched: Sequence[Entity]) -> Dict[int, str]:
    """Vector rank -> day of the explicit correction that supersedes that entry (see the module docstring)."""
    out: Dict[int, Tuple[Tuple[str, str], str]] = {}
    for _, fix in ranked:
        header, body = str(fix.get("header") or ""), str(fix.get("text") or "")
        for entity in matched:
            said = _entity_sentences(body, entity)
            if not said or not (_CUE.search(header) or any(_CUE.search(s) for s in said)):
                continue
            old = [_plain(m.group(1)) for s in said for m in _OLD_VALUE.finditer(s)]
            old = [v for v in old if v.strip()]
            attribute = _content_words(said, entity)
            for rank, row in ranked:
                if row is fix or _order(row) >= _order(fix):
                    continue
                earlier = _row_text(row)
                if classify(earlier, entity)[0] not in ABOUT:
                    continue  # only an entry about the same entity
                if old:
                    hit = any(v in _plain(earlier) for v in old)  # it holds the value the correction replaced
                else:
                    hit = len(attribute & _content_words([earlier], entity)) >= 2  # the same attribute's words
                if hit and (rank not in out or _order(fix) > out[rank][0]):
                    out[rank] = (_order(fix), str(fix.get("day")))
    return {rank: day for rank, (_, day) in out.items()}


# ---------------- what the context says ----------------

def _names(entities: Sequence[Entity]) -> str:
    return ", ".join(f'"{e.name}"' for e in entities)


def context_notes(res: FocusResult) -> List[str]:
    """Lines that open the retrieved-entries block when the message names an entity or a time anchor."""
    lines = []
    if res.anchor is not None:
        cutoff, inclusive = res.anchor
        when, hidden = ("on", "after") if inclusive else ("before", "on or after")
        lines.append(f"(Time anchor: the question asks about the diary as it stood {when} {cutoff.isoformat()}. "
                     f"Entries dated {hidden} it are not shown.)")
    if res.entities:
        line = f"(Asked about {_names(res.entities)}"
        line += ", carried from the earlier question in this conversation." if res.carried else "."
        unmatched = [e for e in res.entities if e not in res.matched]
        if res.matched:
            line += (f" Entries naming {_names(res.matched)}, or only the first name or surname, are about "
                     f"{'that person' if len(res.matched) == 1 else 'those people'}.")
            if any(reason == "near_name" for _, _, reason in res.removed):
                line += (" Entries naming only a near spelling of that name were left out: the diary names the"
                         " person exactly, so the near spelling is someone else.")
        if unmatched:
            line += (f" No retrieved entry names {_names(unmatched)} exactly. An entry with a similar name may be a"
                     " different person, or the name may be spelled differently: check the name before relying"
                     " on it, and if no entry records what was asked, say so.")
        lines.append(line + ")")
    return lines


def entry_label(res: Optional[FocusResult], vector_rank: int) -> str:
    """Suffix for one retrieved entry's header line ('' when there is nothing to say)."""
    if res is None:
        return ""
    if vector_rank in res.similar:
        kind, found, asked = res.similar[vector_rank]
        if kind == "shared":
            return f'; names "{found}", who shares a name with "{asked}": may be a different person'
        return f'; names "{found}", similar to "{asked}": may be a different person, check the name'
    if vector_rank in res.corrected_by:
        return f"; dated before the correction of {res.corrected_by[vector_rank]}, which wins where they disagree"
    return ""
