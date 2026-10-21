"""Entity focus: a cheap lexical check on retrieved candidates (services#116).

Vector similarity cannot tell "Alder Brannock" from "Alder Brennock". Entries about a similarly spelled
person score as high as the entry about the person asked about, and the answer then presents one person's
detail as competing with another's. This module needs no model call.

- Asked entities: capitalised name phrases in the user's message (question words, pronouns, common verbs,
  months and weekdays are not names; a single capitalised word that only starts a sentence is ignored;
  titles such as "Dr" or "Dr." are not part of the name). A follow-up that names nobody ("and her
  email?") carries the entities of the previous question in the same conversation.
- Each candidate is classified per asked entity. Names compare case-, accent-, title- and
  punctuation-insensitively; "**Me:**"-style role labels and function words ("We", "So") are never names:
    match     names the entity in full;
    extended  names it with a middle name or initial, or in reverse order ("Alder J. Brannock");
    partial   names only part of a multi-word name on its own ("Alder called", "Dr. Brannock");
    near      names a similar but different spelling (every word equal or within a small edit distance);
    variant   names a possible short form or initial ("Al", "A. Brannock"): never a different person;
    shared    names a different full name that shares a word with the entity ("Alder Pennick").
- An entry "could refer" to an asked entity when it is match, extended, partial or variant for it, is a
  near spelling of or shares a name with one that no hit names in full, or spells it in lower case. Such
  entries are never dropped for a name. Only when some hit names an asked entity in full (a "full exact hit"):
    near_name     a near spelling of that entity: the diary names the person, so this is someone else;
    other_person  (the asked entity is a person: a full name, or a title and a name) the entry could refer
                  to no asked entity and confidently names another person in its body: a title and a
                  name, or a full name that is not sentence-initial and does not look like a place,
                  organisation or work. Single words ("Marit", "Mum", "Leeds") identify nobody;
    below_margin  the entry could refer to no asked entity and scores more than MARGIN below the best
                  full exact hit of a two-or-more-word or titled asked name, while every asked entity has a
                  full exact hit (a deliberate precision/recall trade-off, see the README).
  With no full exact hit nothing is dropped for a name (a typo or a nickname in the question must not
  hide the entry); similar names are labelled "may be a different person, check the name" instead.
  A relational question ("Alder Brannock's sister-in-law", "the wife of Alder Brannock") drops nothing for
  not naming the asked entity.
- Accepted recall limits: an entry about the asked person that never names them (a pronoun only, a
  nickname unrelated to the name such as "Bob" for "Robert") is dropped when it names another person, or
  scores more than MARGIN below the best exact hit.
- Time anchor: an explicitly framed question ("As of <date>, ...?", "As at <date>", or a question opening
  "Before/Until/Up to <date>, ...?") drops candidates dated after it. A date in a statement does not.
- Supersession: an earlier entry about the same entity is labelled superseded only by an explicit
  correction of that entity that names the corrected value ("..., not <old value>") found in the earlier
  entry, or, without one, shares the corrected attribute's words.
Score order is never changed.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

MAX_ENTITIES = 4
# below_margin: an entry naming no asked entity is dropped when it scores more than this below the best full
# exact-name hit. A deliberate precision/recall trade-off (README); retrieval.entity_focus_margin overrides it.
MARGIN = 0.05
_EPSILON = 1e-9  # scores are rounded to 4 decimals: a gap of exactly MARGIN is not "more than" it

log = logging.getLogger(__name__)
_WARNED: set = set()


def parse_margin(value) -> float:
    """retrieval.entity_focus_margin as a finite number in [0, 1]; anything else (None, text, nan, a bool, a
    negative or a value above 1) logs one warning per value and gives MARGIN. Never raises."""
    try:
        margin = None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        margin = None
    if margin is not None and math.isfinite(margin) and 0.0 <= margin <= 1.0:
        return margin
    key = repr(value)[:40]
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning("retrieval.entity_focus_margin %s is not a number in [0, 1]; using %s", key, MARGIN)
    return MARGIN

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
# What may stand before a sentence: an end mark, a bracket, an ellipsis, a dash or a bullet; list markers,
# quotes and emphasis around the first word are skipped ("- Visited Leeds", "1) Called Mum", "> Saw Marit").
_SENTENCE_END = set(".!?:;\n\"()[]…—–•“”«»")
_LEAD = set(" \t\r*_->'")


def _norm(text: str) -> str:
    return (text or "").replace("’", "'").replace("‘", "'")


# Letters that Unicode decomposition does not fold to ASCII ("Søren" = "Soren", "Strauß" = "Strauss").
_FOLD = str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "æ": "ae", "œ": "oe", "ß": "ss", "ı": "i"})


def _key(word: str) -> str:
    """Comparison form of one name word: lower case, no accents, no apostrophes."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", word.lower().translate(_FOLD))
                    if not unicodedata.combining(c))
    return plain.replace("'", "")


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
    while i >= 0 and text[i] in _LEAD:
        i -= 1
    return i < 0 or text[i] in _SENTENCE_END


def _cap_runs(text: str, stop: bool) -> List[List[Tuple[str, int]]]:
    """Runs of capitalised words joined only by spaces (or "." after a title or an initial). ``stop`` splits
    at non-names."""
    runs: List[List[Tuple[str, int]]] = []
    current: List[Tuple[str, int]] = []
    prev_end, prev_possessive, prev_title, prev_initial = None, False, False, False
    for word, start, end, possessive in _words(text):
        gap = text[prev_end:start].strip(" \t") if prev_end is not None else None
        dotted = prev_title or prev_initial  # "Dr. Patel", "Alder J. Brannock"
        joined = gap is not None and not prev_possessive and (gap == "" or (dotted and gap == "."))
        initial = len(word) == 1 and word.isupper() and word != "I" and text[end:end + 1] == "."
        name_like = (word[:1].isupper() and (len(word) > 1 or not stop or initial)
                     and not (stop and not initial and word.lower() in _NOT_NAMES))
        if name_like and current and joined:
            current.append((word, start))
        else:
            if current:
                runs.append(current)
            current = [(word, start)] if name_like else []
        prev_end, prev_possessive, prev_title = end, possessive, _key(word) in TITLES
        prev_initial = len(word) == 1 and word.isupper()
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
    if (len(want) == 1 and word.startswith(want)) or (len(word) == 1 and want.startswith(word)):
        return "variant"  # an initial: "J" for "John", "A. Brannock" for "Alder Brannock"
    if word in _NOT_NAMES or word in _ROLE_WORDS:
        return None  # "We", "Me", "So", a role label: never a spelling or short form of a name
    if near_word(word, want):
        return "near"
    if min(len(word), len(want)) >= 2 and (want.startswith(word) or word.startswith(want)):
        return "variant"  # a possible short form: "Al" / "Alder", "Alex" / "Alexander"
    return None


def _subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    it = iter(long)
    return bool(short) and all(word in it for word in short)


# Kinds that make an entry about the entity: the full name; the full name with a middle name or initial, or in
# reverse order ("Alder J Brannock", "Brannock Alder"); or a first name or surname on its own.
ABOUT = ("match", "extended", "partial")
_RANK = {"match": 6, "extended": 5, "partial": 4, "near": 3, "variant": 2, "shared": 1}
_ROLE_WORDS = {"assistant", "claude", "companion", "user"}
_ROLE_LABEL = re.compile(r"\*\*\s*(?:Me|Assistant|Claude|Companion|User)\s*:\s*\*\*", re.I)


def _clean(text: str) -> str:
    """Text to read names from: no "**Me:**" / "**Assistant:**" role labels (each starts a new sentence)."""
    return _ROLE_LABEL.sub("\n", _norm(text))


def classify(text: str, entity: Entity) -> Tuple[Optional[str], Optional[str]]:
    """(kind, the name as written) of the strongest mention of ``entity`` in ``text``; kind is one of match,
    extended, partial, near, variant, shared or None (see the module docstring)."""
    e, n = entity.keys, len(entity.keys)
    best: Tuple[Optional[str], Optional[str]] = (None, None)
    for run in _cap_runs(_clean(text), stop=False):
        words = [w for w, _ in run]
        keys = _keys(words)
        if not keys:
            continue
        shown = " ".join(words)
        if any(keys[i:i + n] == e for i in range(len(keys) - n + 1)):
            return "match", shown
        kinds = []
        if n >= 2 and len(keys) >= n and (_subsequence(e, keys) or set(e) <= set(keys)):
            kinds.append("extended")  # "Alder John Brannock", "Alder J. Brannock", "Brannock Alder"
        if len(keys) < n and _subsequence(keys, e):
            kinds.append("partial")  # "Alder called", "Dr. Brannock": part of a longer name on its own
        windows = [keys[i:i + n] for i in range(len(keys) - n + 1)]
        if n >= 2 and len(keys) == 1 and len(keys[0]) > 1:  # a lone word (not a lone letter) vs each name
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


# ---------------- other identified people ----------------

# First and last words that make a capitalised phrase a place, organisation or work rather than a person.
_NOT_PERSON_FIRST = {
    "blue", "red", "green", "black", "white", "grey", "gray", "yellow", "orange", "purple", "pink", "brown",
    "golden", "silver", "new", "old", "north", "south", "east", "west", "northern", "southern", "eastern",
    "western", "upper", "lower", "great", "little", "big", "royal", "national", "united", "grand", "central",
    "saint", "mount", "lake", "peak", "port", "fort", "cape", "the", "morning", "evening", "weekly", "daily",
    "monthly", "annual", "happy", "merry", "good", "best", "first", "second", "third", "last", "final",
}
_NOT_PERSON_LAST = {
    "moor", "cove", "bay", "point", "hill", "hills", "street", "st", "road", "rd", "lane", "avenue", "way", "lock",
    "mill", "wood", "woods", "fell", "end", "cross", "quay", "flats", "sands", "rise", "hollow", "fold", "gap",
    "brow", "steps", "strand", "reach", "park", "square", "bridge", "river", "lake", "island", "beach", "harbour",
    "harbor", "market", "library", "cafe", "station", "school", "college", "university", "hospital", "church",
    "chapel", "hall", "house", "office", "centre", "center", "museum", "gallery", "shop", "club", "green",
    "common", "heath", "vale", "valley", "head", "ness", "marsh", "field", "fields", "farm", "court", "gardens",
    "garden", "place", "close", "crescent", "terrace", "drive", "city", "town", "village", "county", "survey",
    "project", "council", "company", "ltd", "inc", "group", "team", "crossing", "kingdom", "states", "sea",
    "ocean", "mountain", "mountains", "forest", "castle", "abbey", "cathedral", "airport", "port", "pier",
    "theatre", "cinema", "hotel", "inn", "pub", "bar", "restaurant", "kitchen", "bakery", "studio", "works",
    "district", "region", "province", "state", "republic", "borough", "ward", "estate", "trust", "society",
    "association", "foundation", "institute", "bank", "agency", "service", "services", "limited", "plc", "llc",
    "press", "books", "records", "festival", "show", "series", "band", "orchestra", "choir", "league", "cup",
    "fc", "rovers", "trip", "pages", "review", "time", "day", "week", "notes", "plan", "list", "meeting",
    "men", "women", "man", "woman", "people", "kids", "children", "boys", "girls", "everyone", "family",
}


def _person_keys(words: Sequence[str]) -> Optional[Tuple[str, ...]]:
    """The comparison words of a capitalised phrase that confidently identifies a person: a title and a name
    ("Dr. Patel"), or a two- to four-word full name ("Bryony Castellow") that does not look like a place,
    organisation or work. A single word ("Marit", "Leeds", "Mum") identifies nobody for certain."""
    if not words:
        return None
    titled = _key(words[0]) in TITLES
    keys = _keys(words)
    if not keys or any(k in _NOT_NAMES or len(k) < 2 for k in keys):
        return None
    if any(len(w) > 1 and w.isupper() for w in words):
        return None  # an acronym or numeral ("Eckert IV", "NHS Trust"): not a confident person name
    if titled:
        return keys if keys[-1] not in _NOT_PERSON_LAST else None  # not "Mr Men"
    if 2 <= len(keys) <= 4 and keys[0] not in _NOT_PERSON_FIRST and keys[-1] not in _NOT_PERSON_LAST:
        return keys
    return None


_HEADER_LINE = re.compile(r"^\s*(?:\[\d{4}-\d{2}-\d{2}\]\s*)?#+\s.*$", re.M)


def _body(row: dict) -> str:
    """An entry's own words: the text without its header lines or role labels (headers are often title case)."""
    return _clean(_HEADER_LINE.sub("\n", str(row.get("text") or "")))


def identified_people(text: str) -> List[Tuple[str, ...]]:
    """The people ``text`` confidently names (see _person_keys), as comparison words.

    A run that opens a sentence (after list markers, quotes, dashes and brackets) never counts unless it
    starts with a title: "Visited Leeds", "Mum Visited Leeds", "- Called Mum" identify nobody, and so does a
    full name opening a sentence ("Bryony Castellow came"): dropping too little is the safe side. Relation
    words ("Mum", "Dad") are never part of a person name."""
    text = _clean(text)
    out = []
    for run in _cap_runs(text, stop=False):
        words = [w for w, _ in run]
        if _sentence_start(text, run[0][1]):
            while words and _key(words[0]) in _NOT_NAMES:
                words = words[1:]  # "Then Dr Patel said": the opening function word goes, the title stays
            if not words or _key(words[0]) not in TITLES:
                continue
        if any(_key(w) in RELATIONS and _key(w) not in TITLES for w in words):  # "Mum", not "Aunt Peg"
            continue
        keys = _person_keys(words)
        if keys and keys not in out:
            out.append(keys)
    return out


def is_person(entity: Entity) -> bool:
    """The asked entity is a person by the same test (a full name, or a title and a name)."""
    return _person_keys(entity.name.split()) is not None


RELATIONS = {
    "sister", "brother", "mother", "father", "mum", "mom", "mam", "dad", "wife", "husband", "partner", "spouse",
    "son", "daughter", "child", "children", "kid", "kids", "baby", "friend", "boyfriend", "girlfriend", "fiance",
    "fiancee", "boss", "manager", "colleague", "coworker", "neighbour", "neighbor", "cousin", "aunt", "uncle",
    "niece", "nephew", "grandmother", "grandfather", "grandma", "grandpa", "granddaughter", "grandson", "nan",
    "family", "parent", "parents", "sibling", "siblings", "stepmother", "stepfather", "stepson", "stepdaughter",
    "flatmate", "housemate", "roommate", "landlord", "landlady", "carer", "doctor", "therapist", "teacher",
    "student", "mentor", "ex", "twin", "in-laws", "relative", "relatives",
}


def relational(message: str, entities: Sequence[Entity]) -> bool:
    """The question asks about someone by their relation to an asked entity ("Alder Brannock's sister",
    "the sister-in-law of Alder Brannock"): that person's entries need not name the asked one, so nothing
    is dropped for not naming them."""
    text = _norm(message)
    for e in entities:
        name = r"\.?\s+".join(re.escape(w) for w in e.name.split())
        for m in re.finditer(name + r"(?:'s|')\s+((?:[^\W\d_]+\s+)?[^\W\d_-]+(?:-[^\W\d_]+)*)", text, re.I):
            words = [part.lower() for w in m.group(1).split() for part in [w, *w.split("-")]]  # "sister-in-law"
            if any(w in RELATIONS or w.rstrip("s") in RELATIONS for w in words):
                return True
        for m in re.finditer(r"\b([^\W\d_]+(?:-[^\W\d_]+)*)\s+of\s+(?:the\s+)?" + name + r"\b", text, re.I):
            parts = [m.group(1).lower(), *m.group(1).lower().split("-")]  # "the sister of Alder Brannock"
            if any(w in RELATIONS or w.rstrip("s") in RELATIONS for w in parts):
                return True
    return False


def could_refer(found: Sequence[Tuple[Optional[str], Optional[str]]], entities: Sequence[Entity],
                matched: Sequence[Entity], text: str) -> bool:
    """The entry may be about an asked entity, so no name-based drop may touch it: it names one in any
    form (full, extended, partial, near or variant; near only stays when no hit names the entity in full,
    because near_name runs first), shares a name with one that no hit names in full (a nickname case), or
    spells an asked name in lower case ("with alder brannock", "saw marit"; one word needs 3 letters)."""
    plain = _plain(text)
    for (kind, _), e in zip(found, entities):
        if kind in ABOUT or kind in ("near", "variant"):
            return True
        if kind == "shared" and e not in matched:
            return True
        if (len(e.keys) >= 2 or len(e.keys[0]) >= 3) and f" {' '.join(e.keys)} " in plain:
            return True
    return False


@dataclass
class FocusResult:
    entities: List[Entity] = field(default_factory=list)
    carried: bool = False  # the entities come from the previous question (a name-less follow-up)
    anchor: Optional[Tuple[date, bool]] = None
    relational: bool = False  # asks about someone by relation ("X's sister"): no drop for not naming X
    margin: float = MARGIN
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
    found = entities or carried
    return FocusResult(entities=found, carried=bool(carried), anchor=time_anchor(message),
                       relational=relational(message if entities else (list(earlier) or [""])[-1], found))


def _row_text(row: dict) -> str:
    return f"{row.get('header') or ''}\n{row.get('text') or ''}"


def _score(row: dict) -> Optional[float]:
    try:
        return float(row.get("score"))
    except (TypeError, ValueError):
        return None


def focus_candidates(message: str, rows: List[dict], limit: Optional[int] = None,
                     earlier: Sequence[str] = (), margin: float = MARGIN) -> FocusResult:
    """Check vector candidates (``rows`` in score order) against ``message``; keep at most ``limit``.

    Removal reasons (module docstring): ``after_time_anchor``, ``near_name``, ``other_person``,
    ``below_margin`` and ``candidate_limit``. Kept rows stay in score order."""
    res = analyse(message, earlier)
    res.margin = margin = parse_margin(margin)  # never nan, negative or above 1, wherever it came from
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
        # The margin arms only for a matched name of two or more words or a titled name ("Alder Brannock",
        # "Hollin Moor", "Dr Patel"), never for a lone word ("Christmas", "Paris", "Python"), and not while
        # any asked entity has no full exact hit (its entries may use a form the check cannot see).
        arming = [e for e in res.matched if len(e.keys) >= 2 or _key(e.name.split()[0]) in TITLES]
        exact = [_score(row) for rank, row in ranked
                 if any(kind == "match" and res.entities[i] in arming for i, (kind, _) in enumerate(kinds[rank]))]
        exact = [s for s in exact if s is not None]
        best = max(exact) if exact and len(res.matched) == len(res.entities) else None
        # Name-based drops need a full exact hit, and none for a relational question.
        people_asked = [] if res.relational else [e for e in res.matched if is_person(e)]
        survivors = []
        for rank, row in ranked:
            found = kinds[rank]
            if any(kind in ABOUT for kind, _ in found):
                survivors.append((rank, row))
                continue
            if any(kind == "near" and res.entities[i] in res.matched for i, (kind, _) in enumerate(found)):
                res.removed.append((rank, row, "near_name"))  # a near spelling of a name the diary has in full
                continue
            if not could_refer(found, res.entities, res.matched, _row_text(row)) and not res.relational:
                if people_asked and identified_people(_body(row)):
                    res.removed.append((rank, row, "other_person"))
                    continue
                score = _score(row)
                if best is not None and score is not None and best - score > margin + _EPSILON:
                    res.removed.append((rank, row, "below_margin"))
                    continue
            survivors.append((rank, row))
            similar = [(_RANK[kind], kind, shown, res.entities[i].name)
                       for i, (kind, shown) in enumerate(found) if kind]
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
            reasons = {reason for _, _, reason in res.removed}
            if "near_name" in reasons:
                line += (" Entries naming only a near spelling of that name were left out: the diary names the"
                         " person exactly, so the near spelling is someone else.")
            if reasons & {"other_person", "below_margin"}:
                line += (" Some entries that named none of the asked names were left out, so this list may be"
                         " incomplete: an entry about the asked person that does not name them (a pronoun or"
                         " a nickname) may be missing.")
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
