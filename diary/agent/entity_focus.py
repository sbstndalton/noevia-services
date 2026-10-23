"""Entity focus: a cheap lexical check on retrieved candidates (services#116).

Vector similarity cannot tell "Alder Brannock" from "Alder Brennock". Entries about a similarly spelled
person score as high as the entry about the person asked about, and the answer then presents one person's
detail as competing with another's. This module needs no model call.

- Asked entities: capitalised name phrases in the user's message (question words, pronouns, common verbs,
  months and weekdays are not names; a single capitalised word that only starts a sentence is ignored;
  titles such as "Dr" or "Dr." are not part of the name). A follow-up that names nobody ("and her
  email?", "Is that written in my diary?") carries the entities of the previous question in the same
  conversation (services#177: a message that refers back to the earlier exchange with "that", "this" or
  "it" is a follow-up too; a name-less message that refers back to nothing, such as "How has the week
  been?", carries nothing, and a message naming anyone never carries).
- Each candidate is classified per asked entity. Names compare case-, accent-, title- and
  punctuation-insensitively; "**Me:**"-style role labels and function words ("We", "So") are never names:
    match     names the entity in full; the surname of a name of two or more words may carry a trailing
              "s" or a possessive ("Alder Brannocks boat", "Alder Brannock's", services#148);
    extended  names it with a middle name or initial, or in reverse order ("Alder J. Brannock");
    partial   names only part of a multi-word name on its own ("Alder called", "Dr. Brannock");
    near      names a similar but different spelling (every word equal or within a small edit distance);
    variant   names a possible short form or initial ("Al", "A. Brannock", "Will" for "William", "Jan" for
              "Janet" from a small explicit table): never a different person. A short form that is also an
              ordinary word ("Will", "May", "Mark", "Bill", "Jan") counts only where its position and the next
              word read as a name: "Will you ...", "Mark the date", "Bill me", "in Jan 2030" name nobody;
    shared    names a different full name that shares a word with the entity ("Alder Pennick").
- An entry "could refer" to an asked entity when it is match, extended, partial or variant for it, shares a
  name with one that no hit names in full, or spells it in lower case. Such entries are never dropped for a
  name. A near spelling is kept (and labelled) while no hit names that entity in full; once one does, the
  near spelling is dropped as near_name. Only when some hit names an asked entity in full (a "full exact hit"):
    near_name     a near spelling of that entity: the diary names the person, so this is someone else;
    other_person  (the asked entity is a person: a full name, or a title and a name) the entry could refer
                  to no asked entity and confidently names another person in its body: a title and a
                  name, or a full name that does not look like a place, organisation, product or work. A
                  full name opening the entry or a sentence (after "Today", "Then" and similar words) counts
                  only with person evidence (services#160): a known given name first ("Tom Hardy fixed the
                  boat."), or the given name or surname (plain or with a trailing "s") of an asked person
                  ("Wren Brannocks fixed it"); a capitalised opener such as "Harbour Lights", "New Year" or
                  "Bank Holiday" has none and is kept, and so is a bold speaker label ("**Bryony
                  Castellow:** went"). The asked surname alone is evidence only for a two-word run. Known
                  limit: a capitalised verb before another person ("Met Tom Hardy.") still reads as an opener
                  and is kept. Otherwise the full name may be anywhere and must not look like a place,
                  organisation, product or work (a whole listed place phrase such as "San Francisco" or
                  "Hong Kong", or an institutional ending such as "Orbel Maps", "Harrow Bank"). A given
                  name and surname is always a person, whatever word precedes it: "Elena Bank" and "Tom Hill"
                  are people, "Northgate Bank", "Hyde Park" and "First National Bank" are not (a surname-like
                  ending that is mostly a surname, such as "bank", "park", "hill" or "green", marks an institution or place only after
                  three or more words, a first word that is no known given name, a listed whole place phrase,
                  or, for an untitled run, "along" or "at" right before it: "along Rosa Bank", "at Clara Bank";
                  mostly-place endings such as "Street", "House" or "Church", and endings that are never
                  surnames, such as "Labs" or "Maps", always do; "Rose Hill" reads as a person).
                  Single words ("Marit", "Mum", "Leeds") identify nobody;
    below_margin  the entry could refer to no asked entity and scores more than MARGIN below the best
                  full exact hit of a two-or-more-word or titled asked name, while every asked entity has a
                  full exact hit (a deliberate precision/recall trade-off, see the README).
  With no full exact hit nothing is dropped for a name (a nickname in the question must not hide the entry);
  similar names are labelled "may be a different person, check the name" instead, with one exception:
    near_name_unmatched  (services#145) the asked entity is a two-word person no hit names in full (its entry
                  was deleted, or never written), the question is not relational, and the entry names the
                  same given name with a near-spelled surname ("Caspian Juxton" for "Caspian Juxon", the same
                  near_word measure). It is someone else; the note says the diary does not record the asked
                  person and names the near spelling so the answer can ask whether it was meant. Accepted
                  trade-off: a misspelt real name now gets that abstention instead of an answer.
  A relational question ("Alder Brannock's sister-in-law", "the wife of Alder Brannock") drops nothing for
  not naming the asked entity.
- Accepted recall limits: an entry about the asked person that never names them (a pronoun only, a
  nickname unrelated to the name and not in the short-form table, such as "Ziggy") is dropped when it names another person, or
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
MAX_NEAR_SPELLINGS = 3  # #145: near spellings a note names per asked person
ABSENT_NOTE = "The diary has no entry naming"  # #145: opens the abstain sentence; kept even when nothing loads
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


# Common given-name short forms (short -> full forms). Explicit, not fuzzy; a prefix ("Al", "Alex") already counts.
SHORT_FORMS: Dict[str, Tuple[str, ...]] = {
    "will": ("william",), "bill": ("william",), "billy": ("william",), "jan": ("janet",),
    "bob": ("robert",), "rob": ("robert",), "bobby": ("robert",), "robbie": ("robert",),
    "jim": ("james",), "jimmy": ("james",), "jack": ("john",), "johnny": ("john",),
    "dick": ("richard",), "rick": ("richard",), "rich": ("richard",), "ted": ("edward",), "eddie": ("edward",),
    "tom": ("thomas",), "harry": ("henry",), "chuck": ("charles",), "charlie": ("charles",),
    "peggy": ("margaret",), "meg": ("margaret",), "maggie": ("margaret",), "liz": ("elizabeth",),
    "beth": ("elizabeth",), "betty": ("elizabeth",), "kate": ("katherine", "catherine"),
    "katie": ("katherine", "catherine"), "kathy": ("katherine",), "cathy": ("catherine",),
    "sue": ("susan",), "susie": ("susan",), "pat": ("patrick", "patricia"), "patty": ("patricia",),
    "sally": ("sarah",), "nick": ("nicholas",), "andy": ("andrew",), "tony": ("anthony",),
    "mike": ("michael",), "mick": ("michael",), "dave": ("david",), "joe": ("joseph",),
    "sam": ("samuel", "samantha"), "ben": ("benjamin",), "dan": ("daniel",), "matt": ("matthew",),
    "chris": ("christopher", "christine"), "steve": ("stephen", "steven"), "greg": ("gregory",),
    "jen": ("jennifer",), "jenny": ("jennifer",), "ron": ("ronald",), "don": ("donald",),
    "fred": ("frederick",), "freddie": ("frederick",),
}
# Words a short form must not be followed by to read as a name: pronouns, determiners and the bare verbs and
# adverbs that follow a modal or an imperative ("Will you", "May I", "Mark the date", "Bill me", "Will be").
_AFTER_VERB = {
    "i", "you", "we", "he", "she", "it", "they", "me", "him", "her", "us", "them", "this", "that", "these",
    "those", "there", "the", "a", "an", "my", "your", "our", "his", "their", "its", "not", "never", "also",
    "always", "just", "still", "be", "have", "do", "go", "get", "come", "see", "need", "try", "take", "make",
    "let", "keep", "up", "down", "out", "off", "all", "everyone", "anyone", "someone", "each",
    "both", "some", "any", "as", "so", "too", "soon", "later", "today", "tomorrow", "tonight", "again",
}
_DATE_LEAD = {"in", "on", "since", "until", "till", "by", "from", "of", "last", "next", "early", "late", "mid",
              "during", "before", "after", "through", "this"}
# Names that are also verbs or modals: read with the sentence-start context check, short form or not.
_VERB_FORMS = {"will", "may", "mark", "bill", "rob", "sue", "pat", "jack", "nick", "frank", "chuck", "drew",
               "rich", "ray", "art", "sandy", "don", "hal", "bob", "dick", "ted"}


def _ordinary(key: str) -> bool:
    """A name that is also an ordinary word: a listed short form that is a modal or a month, or a name that is
    also a verb ("Mark the date", "May I")."""
    return key in _VERB_FORMS or (key in SHORT_FORMS and key in _NOT_NAMES)


def _reads_as_name(text: str, start: int, end: int, initial: bool) -> bool:
    """A capitalised ordinary-word short form reads as a name here: not followed by a digit or by a pronoun,
    determiner or bare verb when it opens a sentence ("Will you", "Mark the date"), not a month after a date
    word or before a number ("in Jan", "Jan 2030"), and, in a question (``initial`` False), not opening the
    sentence at all."""
    opens = _sentence_start(text, start)
    if opens and not initial:
        return False
    nxt = re.match(r"[\s,]*([^\W_]+)", text[end:])
    if nxt and (nxt.group(1)[:1].isdigit() or (opens and _key(nxt.group(1)) in _AFTER_VERB)):
        return False  # mid-sentence a capitalised short form is a name ("did Bill still ..."), unless a date
    if opens and re.match(r"\s*[?!]", text[end:]):
        return False  # "Will?" on its own
    prev = re.search(r"([^\W\d_]+)[\s,]*$", text[:start])
    key = _key(_POSSESSIVE.sub("", text[start:end]))
    if (key in _MONTH_ABBR or key in MONTHS) and ((prev is not None and _key(prev.group(1)) in _DATE_LEAD)
                                                    or re.match(r"[\s,]*\d", text[end:]) is not None):
        return False
    return True


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
        if word[:1].isupper() and _ordinary(_key(word)) and not (current and joined):
            # "What did Will say?" is a name; "Will you ...?", "Mark the date", "in Jan 2030" are not. Inside a
            # run ("Alder Will") the word stays as the run decided.
            name_like = _reads_as_name(text, start, end, initial=not stop)
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
# services#177: a reference back to the previous exchange. "that", "those" and "these" count on their own
# ("Is that true?", "Where did you get that?"); "it" and "this" are too common ("Is it going to rain?", "What
# is this?") and count only beside a word asking where an answer came from (META: "Is it written in my
# diary?", "Did you infer this?"). A demonstrative that only qualifies a time word ("this week", "that
# morning") opens a new topic and refers back to nothing.
_TIME_WORDS = (r"(?:week|weekend|morning|afternoon|evening|night|day|month|year|time|summer|winter|spring|"
               r"autumn|fall|season|term|quarter|" + "|".join(_WEEKDAYS) + "|" + "|".join(MONTHS) + r")s?")
_NOT_TIME = r"\b(?!\s+" + _TIME_WORDS + r"\b)"
_REFERS_BACK = re.compile(r"\b(?:that|those|these)" + _NOT_TIME, re.I)
_WEAK_BACK = re.compile(r"\b(?:it|this)" + _NOT_TIME, re.I)
_META = re.compile(r"\b(?:written|write|wrote|recorded|record|records|noted|logged|sure|true|correct|right|"
                   r"infer|inferred|inferring|guess|guessed|guessing|assume|assumed|assuming|source|sources|"
                   r"diary|entry|entries|made\s+up|make\s+up|making\s+up|come\s+from|came\s+from|get|got|"
                   r"know|knew|say|said|mention|mentioned|confirm|confirmed|certain)\b", re.I)


def refers_back(message: str) -> bool:
    """The message is worded as a follow-up to the previous exchange: an opener such as "and" or "what
    about", a personal pronoun, or a demonstrative such as "that" or "it" that is not a time phrase."""
    text = _norm(message)
    return bool(_FOLLOW_UP.search(text) or _REFERS_BACK.search(text)
                or (_WEAK_BACK.search(text) and _META.search(text)))


def carried_entities(message: str, earlier: Sequence[str]) -> List[Entity]:
    """For a follow-up that names nobody ("and her email?"): the entities of the latest earlier user message
    of the same conversation that named any. A message that is not worded as a follow-up (refers_back)
    carries nothing, and neither does one that names anybody itself."""
    if asked_entities(message) or not refers_back(message):
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
    if want in SHORT_FORMS.get(word, ()) or word in SHORT_FORMS.get(want, ()):
        return "variant"  # a listed short form: "Will" / "William", "Jan" / "Janet", "Bob" / "Robert"
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


def _plural_of(word: str, surname: str) -> bool:
    """services#148: ``word`` is ``surname`` plus a trailing "s" ("brannocks" for "brannock"), a possessive
    written without the apostrophe or a plural. ``_words`` already strips "'s" and "’s". One direction
    only: "evan" for "evans" stays a near spelling."""
    return len(surname) >= 2 and word == surname + "s"


def _names_in_full(window: Sequence[str], entity_keys: Sequence[str]) -> bool:
    """``window`` names the entity in full: every word equal, except that the surname of a name of two or
    more words may carry a trailing "s" (services#148). A one-word name still compares exactly."""
    window, entity_keys = tuple(window), tuple(entity_keys)
    if window == entity_keys:
        return True
    return (len(entity_keys) >= 2 and len(window) == len(entity_keys) and window[:-1] == entity_keys[:-1]
            and _plural_of(window[-1], entity_keys[-1]))


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
        if any(_names_in_full(keys[i:i + n], e) for i in range(len(keys) - n + 1)):
            return "match", shown
        kinds = []
        if n >= 2 and len(keys) >= n and (_subsequence(e, keys) or set(e) <= set(keys)):
            kinds.append("extended")  # "Alder John Brannock", "Alder J. Brannock", "Brannock Alder"
        if len(keys) < n and _subsequence(keys, e):
            kinds.append("partial")  # "Alder called", "Dr. Brannock": part of a longer name on its own
        windows = [keys[i:i + n] for i in range(len(keys) - n + 1)]
        if n >= 2 and len(keys) == 1 and _plural_of(keys[0], e[-1]):
            kinds.append("partial")  # "the Brannocks": the surname on its own with a trailing "s" (#148)
        elif n >= 2 and len(keys) == 1 and len(keys[0]) > 1:  # a lone word (not a lone letter) vs each name
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
    # clearly institutional or product endings ("Orbel Maps", "Fenwick Labs", "Dunmore Airport")
    "maps", "apps", "app", "labs", "lab", "software", "systems", "technologies", "airlines", "airways", "corp",
    "corporation", "gmbh", "llp", "academy", "polytechnic", "clinic", "pharmacy", "supermarket", "brewery",
    "studios", "stadium", "magazine",
}
# Multi-word places matched as whole phrases, never by one word ("Santa Ramirez" and "Wei Kong" are people).
PLACE_PHRASES = {
    ("hong", "kong"), ("kuala", "lumpur"), ("san", "francisco"), ("san", "diego"), ("san", "jose"),
    ("san", "antonio"), ("los", "angeles"), ("los", "alamos"), ("las", "vegas"), ("buenos", "aires"),
    ("rio", "de", "janeiro"), ("sao", "paulo"), ("tel", "aviv"), ("phnom", "penh"), ("ho", "chi", "minh"),
    ("sri", "lanka"), ("costa", "rica"), ("baton", "rouge"), ("santa", "rosa"), ("santa", "fe"),
    ("santa", "barbara"), ("santa", "monica"), ("palm", "springs"), ("salt", "lake"), ("abu", "dhabi"),
    ("el", "paso"), ("la", "paz"), ("le", "havre"), ("new", "delhi"), ("cape", "town"), ("hong", "kong", "island"),
    # places ending in a surname-like word (_SURNAME_LIKE_LAST), some after a word that may be a given name
    ("hyde", "park"), ("notting", "hill"), ("golders", "green"), ("kings", "cross"), ("central", "park"),
    ("sherwood", "forest"), ("victoria", "park"), ("alexandra", "park"),
    ("albert", "park"), ("regents", "park"), ("primrose", "hill"), ("tower", "hill"), ("parliament", "hill"),
    ("greenwich", "park"), ("richmond", "park"), ("holland", "park"), ("battersea", "park"),
    ("finsbury", "park"), ("hampstead", "heath"),
}

# Endings in _NOT_PERSON_LAST that are also real surnames ("Elena Bank"). They mark an institution only when
# the run clearly names one (see _institutional): the other endings in the list ("Maps", "Labs", "Software",
# "Airways", "Academy", ...) are not surnames, so a run ending in one is always an institution or product.
# Only endings that are mostly surnames are listed. A mostly-place ending ("street", "lane", "church", "house",
# "hall", "bridge", "lake", "castle", "mill", "head") always marks a place, so "Mary Street" or "Ivy House" is
# never lost. Accepted trade-off: "hill" is a common surname, so a given name and "Hill" ("Rose Hill") reads as
# a person, and an entry naming only it is dropped beside an exact hit for someone else.
_SURNAME_LIKE_LAST = {
    "bank", "park", "hill", "hills", "wood", "woods", "green", "cross", "field", "fields", "marsh", "heath", "ward",
}
# The word right before an untitled run that marks it as a place or an institution being visited or paid, not
# a person: "along Rosa Bank", "at Clara Bank". It only decides a _SURNAME_LIKE_LAST run (see _institutional);
# a full name is a person whatever precedes it ("Met near Daniel Ostrow", "Sent it via Daniel Ostrow"). "at" is
# a documented known limit: "Met at Elena Bank" keeps the entry (a lost entry costs more than a place kept).
# "to", "from", "into" and "near" are not cues: "Waved to Elena Bank", "Heard from Ann Wood" are people.
_PLACE_CUES = frozenset({"at", "along"})
# Plausible given names, accent-folded. Together with SHORT_FORMS (keys and values) this is all the module
# knows about given names; a first word in neither is "unsure", which keeps a surname-like ending an
# institution. Names that are also places or brands ("Florence", "Austin", "Lincoln", "Chelsea") are left out.
GIVEN_NAMES = frozenset("""
    adam adrian aisha alan albert alexander alice alicia alison amanda amelia amy ana andrea angela anita anna
    anne annie antonio arthur ashley barbara beatrice benedict bernard beverly bianca brenda brian bridget
    bryony camille carl carla carmen carol caroline catherine cecilia celia charlotte cheryl chloe claire clara
    colin connor craig cynthia damian daniel daphne dawn deborah denise derek diana diane diego dmitri dolores
    dominic dorothy douglas edith edmund edwin eileen elaine eleanor elena eliza elizabeth ella ellen elsa emil
    emilia emily emma enrique eric erica erin esther eugene eva evelyn fatima felix fiona francesca francis
    frank gabriel gabriela gail gareth gavin george gerald gillian gloria grace graham gregory hannah harold
    harriet hazel heather hector helen helena henry hilda hugh hugo ian ines ingrid irene iris isaac isabel
    isabella ivan ivy jacob jade james jamie jane janet janice jasmine jason jean jeffrey jessica joan joanna
    jocelyn john jonathan jorge joseph josephine joyce juan judith julia julian juliet june karen karin karl
    katarina katherine kathleen keith kelly kenneth kevin kirsten kristina laura lauren lawrence leah leila
    lena leo leon leonard leslie lilian lillian linda lisa lola lorena louis louise lucas lucia lucy luis luke
    lydia lynn magnus malcolm marco marcus margaret margot maria marian marie marina marit marta martin martha
    mary mateo matilda maurice maxine megan melanie melissa mia michelle miguel mila mildred miriam molly mona
    monica nadia nadine naomi natalia natalie nathan neil nina noah nora norman olga olivia oscar owen paolo
    patricia paul paula pauline pedro penelope peter petra philip phoebe phyllis priya rachel rafael ramona
    raymond rebecca regina renata richard rita robin roger roland ronan rosa rosalind rose ruth ryan sabrina
    sandra sara sarah saul sean sebastian selma serena sergio sharon sheila shirley sienna simon sofia sonia
    sophia sophie stella stefan stuart sylvia tamara tania teresa thea theodore theresa tobias valentina
    valerie vanessa vera veronica vicky viktor viola violet virginia vivian walter wanda wendy wilhelm willa
    winifred xavier yasmin yolanda yvonne zara zoe
    ann lee
""".split())


def _given_name(key: str) -> bool:
    """``key`` (a comparison form) is a plausible given name by the module's own knowledge: the list above, a
    short form or the full name a short form stands for."""
    return (key in GIVEN_NAMES or key in SHORT_FORMS
            or any(key in fulls for fulls in SHORT_FORMS.values()))


def _institutional(words: Sequence[str], keys: Sequence[str], lead: str = "") -> bool:
    """The phrase ends in a word that marks a place, organisation, product or work (_NOT_PERSON_LAST).

    A surname-like ending (_SURNAME_LIKE_LAST) counts only when the run clearly names an institution: three or
    more capitalised words ("First National Bank"), or a first word that is not a known given name
    ("Northgate Bank"). A title and one or two name words ("Dr Bank"), or a given name and one other word
    ("Elena Bank"), is a person. Unsure stays an institution, which keeps the entry.

    ``lead`` is the comparison form of the word right before an untitled run. A place cue there (_PLACE_CUES:
    "along Rosa Bank", "at Clara Bank") keeps a given name and surname-like ending an institution or place. It
    decides nothing else: a full name is a person whatever precedes it."""
    if keys[-1] not in _NOT_PERSON_LAST:
        return False
    if keys[-1] not in _SURNAME_LIKE_LAST:
        return True
    name_words = [w for w in words if _key(w) not in TITLES]
    if len(name_words) >= 3:
        return True
    if len(name_words) < len(words):  # a title in front: "Dr Elena Bank", "Dr Northgate Bank"
        return False
    if lead in _PLACE_CUES:
        return True
    return not (len(keys) >= 2 and _given_name(keys[0]))


def _person_keys(words: Sequence[str], lead: str = "") -> Optional[Tuple[str, ...]]:
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
        return keys if not _institutional(words, keys) else None  # not "Mr Men"
    if any(keys[i:i + len(ph)] == ph for ph in PLACE_PHRASES for i in range(len(keys) - len(ph) + 1)):
        return None  # a whole multi-word place ("Hong Kong", "San Francisco")
    if 2 <= len(keys) <= 4 and keys[0] not in _NOT_PERSON_FIRST \
            and not _institutional(words, keys, lead):
        return keys
    return None


_HEADER_LINE = re.compile(r"^\s*(?:\[\d{4}-\d{2}-\d{2}\]\s*)?#+\s.*$", re.M)


def _body(row: dict) -> str:
    """An entry's own words: the text without its header lines or role labels (headers are often title case)."""
    return _clean(_HEADER_LINE.sub("\n", str(row.get("text") or "")))


def _person_word(keys: Sequence[str], asked: Sequence[Entity]) -> bool:
    """services#160: an untitled run opening a sentence carries evidence that it is a person name: its first
    word is a known given name (_given_name) or the given name of an asked person ("Alder Hollis"), or its
    last word is the surname of an asked two-or-more-word person, plain or with a trailing "s" ("Wren
    Brannocks"). A capitalised common opener ("Harbour Lights", "New Year", "Bank Holiday") has neither."""
    if len(keys) < 2:
        return False
    if _given_name(keys[0]):
        return True
    for e in asked:
        if len(e.keys) < 2:
            continue
        if keys[0] == e.keys[0]:
            return True
        # The surname alone is evidence only for a two-word run, so a capitalised verb in front ("Met Djordje
        # Ilic") is never read as a given name.
        if len(keys) == 2 and (keys[-1] == e.keys[-1] or _plural_of(keys[-1], e.keys[-1])):
            return True
    return False


_LABEL_AFTER = re.compile(r"[*_]*\s*:")  # "**Bryony Castellow:** went": a speaker label, not a mention


def identified_people(text: str, asked: Optional[Sequence[Entity]] = None) -> List[Tuple[str, ...]]:
    """The people ``text`` confidently names (see _person_keys), as comparison words.

    A run that opens a sentence (after list markers, quotes, dashes and brackets; leading function and time
    words such as "Then" or "Today" are skipped) counts when it starts with a title ("Then Dr Patel said").
    Without ``asked`` nothing else there counts: "Visited Leeds", "Mum Visited Leeds", "- Called Mum" and
    "Bryony Castellow came" identify nobody. With ``asked`` (the asked entities; services#160) an untitled
    full name there counts too when _person_word finds person evidence for it ("Tom Hardy fixed the boat.",
    "Today Wren Brannocks came"), so it is told apart from a capitalised opener ("Harbour Lights was busy").
    Relation words ("Mum", "Dad") are never part of a person name."""
    text = _clean(text)
    out = []
    for run in _cap_runs(text, stop=False):
        words = [w for w, _ in run]
        if _sentence_start(text, run[0][1]):
            while words and _key(words[0]) in _NOT_NAMES:
                words = words[1:]  # "Then Dr Patel said": the opening function word goes, the title stays
            if not words:
                continue
            if _key(words[0]) not in TITLES:
                last, last_start = run[-1]
                if not (asked and _person_word(_keys(words), asked)) \
                        or _LABEL_AFTER.match(text, last_start + len(last)):
                    continue
        if any(_key(w) in RELATIONS and _key(w) not in TITLES for w in words):  # "Mum", not "Aunt Peg"
            continue
        before = re.search(r"([^\W\d_]+)[ \t]+$", text[:run[0][1]])  # the word right before the run, same line
        keys = _person_keys(words, _key(before.group(1)) if before else "")
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
    absent: Dict[Entity, List[str]] = field(default_factory=dict)  # #145: asked person nobody names -> near spellings left out
    exact: Dict[int, List[str]] = field(default_factory=dict)  # #178: vector rank -> asked names it names in full

    @property
    def active(self) -> bool:
        """The message (or the follow-up it continues) names an entity, or it has a time anchor."""
        return bool(self.entities or self.anchor)

    @property
    def applied(self) -> bool:
        return bool(self.matched)


_ANCHOR_UNSET = object()


def analyse(message: str, earlier: Sequence[str] = (), *, anchor=_ANCHOR_UNSET) -> FocusResult:
    entities = asked_entities(message)
    carried = [] if entities else carried_entities(message, earlier)
    found = entities or carried
    if anchor is _ANCHOR_UNSET:
        anchor = time_anchor(message)
    return FocusResult(entities=found, carried=bool(carried), anchor=anchor,
                       relational=relational(message if entities else (list(earlier) or [""])[-1], found))


def _row_text(row: dict) -> str:
    return f"{row.get('header') or ''}\n{row.get('text') or ''}"


def _score(row: dict) -> Optional[float]:
    try:
        return float(row.get("score"))
    except (TypeError, ValueError):
        return None


def _same_given_near_surname(text: str, entity: Entity) -> Optional[str]:
    """The first name in ``text`` with ``entity``'s given name and a near-spelled surname, as written
    ("Caspian Juxton" for "Caspian Juxon"), else None. ``entity`` has exactly two comparison words."""
    given, surname = entity.keys
    for run in _cap_runs(_clean(text), stop=False):
        words = [w for w, _ in run if _key(w) not in TITLES]
        keys = _keys(words)
        if len(keys) != len(words):
            continue
        for i in range(len(keys) - 1):
            if keys[i] == given and near_word(keys[i + 1], surname) and not _plural_of(keys[i + 1], surname):
                return f"{words[i]} {words[i + 1]}"
    return None


def focus_candidates(message: str, rows: List[dict], limit: Optional[int] = None,
                     earlier: Sequence[str] = (), margin: float = MARGIN, *,
                     analysed: Optional[FocusResult] = None, temporal_partition=None) -> FocusResult:
    """Check vector candidates (``rows`` in score order) against ``message``; keep at most ``limit``.

    Removal reasons (module docstring): ``after_time_anchor``, ``near_name``, ``near_name_unmatched``,
    ``other_person``, ``below_margin`` and ``candidate_limit``. Kept rows stay in score order."""
    res = analysed if analysed is not None else analyse(message, earlier)
    res.margin = margin = parse_margin(margin)  # never nan, negative or above 1, wherever it came from
    ranked = list(enumerate(rows, 1))
    if temporal_partition is not None:
        kept_ranks, removed_ranks = temporal_partition
        ranked = [(rank, rows[rank - 1]) for rank in kept_ranks]
        res.removed.extend((rank, rows[rank - 1], "after_time_anchor") for rank in removed_ranks)
    elif res.anchor is not None:
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
        # #145: a two-word person no hit names in full; a same-given-name near spelling is someone else.
        absent = [] if res.relational else [e for e in res.entities
                                            if e not in res.matched and len(e.keys) == 2 and is_person(e)]
        survivors = []
        for rank, row in ranked:
            found = kinds[rank]
            if any(kind in ABOUT for kind, _ in found):
                survivors.append((rank, row))
                named = [res.entities[i].name for i, (kind, _) in enumerate(found) if kind == "match"]
                if named:
                    res.exact[rank] = named
                continue
            if any(kind == "near" and res.entities[i] in res.matched for i, (kind, _) in enumerate(found)):
                res.removed.append((rank, row, "near_name"))  # a near spelling of a name the diary has in full
                continue
            near = [(e, _same_given_near_surname(_row_text(row), e))
                    for i, (kind, _) in enumerate(found) if kind == "near" and (e := res.entities[i]) in absent]
            near = [(e, shown) for e, shown in near if shown]
            if near:
                res.removed.append((rank, row, "near_name_unmatched"))
                for e, shown in near:
                    spellings = res.absent.setdefault(e, [])
                    if shown not in spellings and len(spellings) < MAX_NEAR_SPELLINGS:
                        spellings.append(shown)
                continue
            if not could_refer(found, res.entities, res.matched, _row_text(row)) and not res.relational:
                if people_asked and identified_people(_body(row), res.entities):
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
        res.exact = {rank: s for rank, s in res.exact.items() if rank in kept_ranks}
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
        for e, spellings in res.absent.items():
            near = " or ".join(f'"{s}"' for s in spellings)
            what, who = (("a near spelling", "a different person") if len(spellings) == 1
                         else ("near spellings", "different people"))
            line += (f' {ABSENT_NOTE} "{e.name}". Entries naming {near}, {what} with the same first'
                     f' name, were left out as {who}: say the diary does not record this about "{e.name}", do not'
                     f' give the details of {near} as the answer, and ask whether {near} was meant.')
        unmatched = [e for e in unmatched if e not in res.absent]
        if unmatched:
            line += (f" No retrieved entry names {_names(unmatched)} exactly. An entry with a similar name may be a"
                     " different person, or the name may be spelled differently: check the name before relying"
                     " on it, and if no entry records what was asked, say so.")
        lines.append(line + ")")
    return lines


def names_exactly(row: dict, name: str) -> bool:
    """services#179: the entry (header and text) names ``name`` in full, as entity focus reads a match."""
    entities = asked_entities(name)
    return bool(entities) and classify(_row_text(row), entities[0])[0] == "match"


def entry_label(res: Optional[FocusResult], vector_rank: int) -> str:
    """Suffix for one retrieved entry's header line ('' when there is nothing to say). services#178: an entry
    naming an asked person in full says so; a near spelling is never a match, so it never gets that label."""
    if res is None:
        return ""
    if vector_rank in res.exact:
        names = res.exact[vector_rank]
        who = "the asked person" if len(names) == 1 else "the asked people"
        label = f"; names {who} {_names([Entity(n, ()) for n in names])} exactly"
        if vector_rank in res.corrected_by:
            label += f"; dated before the correction of {res.corrected_by[vector_rank]}, which wins where they disagree"
        return label
    if vector_rank in res.similar:
        kind, found, asked = res.similar[vector_rank]
        if kind == "shared":
            return f'; names "{found}", who shares a name with "{asked}": may be a different person'
        return f'; names "{found}", similar to "{asked}": may be a different person, check the name'
    if vector_rank in res.corrected_by:
        return f"; dated before the correction of {res.corrected_by[vector_rank]}, which wins where they disagree"
    return ""
