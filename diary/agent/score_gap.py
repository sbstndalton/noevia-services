"""Relative score cut for retrieved candidates (services#114).

A fixed top_k loads eight entries even when one or two are relevant: the rest are near-topic entries that only
look similar. Gate run 1 measured it on the synthetic corpus: every relevant entry scored within 0.06 of the best
candidate, while the list ran on to 0.17 below it.

The cut drops a candidate that scores more than ``retrieval.score_gap`` below the best remaining candidate. It
runs after the entity focus (entity_focus.py), so the best candidate is never an entry dated after a time anchor.
It is skipped when the message names an entity (or is a follow-up carrying one): the name check then decides what
is about the asked entity, and an entry about it, or about a name the question mistyped, is never dropped for its
score. The best candidate is always kept, order is never changed, and candidates without a
score are kept.

DIARY_SCORE_GAP_IMPL=python is the default. Explicit rust uses the fixed installed
scores-only CLI through score_gap_native; image pairing is required before
activation. Native failure retains every post-focus candidate, still computes
best, and reports unavailable through the context trace with applied=false.
No Python filtering is used as a fallback after a requested Rust failure.
"""
from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# Off by default (#114): on run 1's traced queries the cut added no precision over the entity focus, and the
# query families without traces are untested. SUGGESTED_GAP is the value to try when enabling it: in run 1 the
# largest gap between the best candidate and a relevant entry was 0.060, so 0.08 leaves headroom.
SCORE_GAP = None
SUGGESTED_GAP = 0.08
_EPSILON = 1e-9  # scores are rounded to 4 decimals: a gap of exactly the setting is kept
_OFF = ("", "off", "none", "false", "no", "0", "disabled")
_WARNED: set = set()


def parse_gap(value) -> Optional[float]:
    """retrieval.score_gap as a number in (0, 1], or None (off) for "off", "none", "false", 0, null or false.

    Anything else (other text, nan, a negative or a number above 1) logs a warning once and leaves the cut off."""
    if value is None or value is False:
        return None
    if isinstance(value, str) and value.strip().lower() in _OFF:
        return None
    try:
        gap = None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        gap = None
    if gap == 0:
        return None
    if gap is not None and math.isfinite(gap) and 0.0 < gap <= 1.0:
        return gap
    key = repr(value)
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning("retrieval.score_gap %s is not a number in (0, 1] or off; the score gap stays off", key)
    return SCORE_GAP


def _score(row: dict) -> Optional[float]:
    try:
        score = float(row.get("score"))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def cut(ranked: List[Tuple[int, dict]], gap: Optional[float], *, diagnostics=None):
    """(kept, removed, best) for ``ranked`` [(vector rank, row)] in score order.

    ``removed`` lists the (vector rank, row) pairs that score more than ``gap`` below ``best``, the best score
    in ``ranked``. With ``gap`` None nothing is cut."""
    normalized = [_score(row) for _, row in ranked]
    scores = [s for s in normalized if s is not None]
    best = max(scores) if scores else None
    # Off until a paired image supplies the reviewed CLI; never fall back to
    # Python filtering after an explicitly requested native failure.
    if os.environ.get("DIARY_SCORE_GAP_IMPL", "python").strip().lower() == "rust":
        from . import score_gap_native
        native = score_gap_native.partition(normalized, gap, best)
        if diagnostics is not None:
            diagnostics.update(implementation="rust", status="ok" if native is not None else "unavailable")
        if native is None:
            return list(ranked), [], best
        kept, removed, native_best = native
        return [ranked[i] for i in kept], [ranked[i] for i in removed], native_best
    if gap is None or best is None:
        return list(ranked), [], best
    kept, removed = [], []
    for rank, row in ranked:
        score = _score(row)
        if score is not None and best - score > gap + _EPSILON:
            removed.append((rank, row))
        else:
            kept.append((rank, row))
    return kept, removed, best
