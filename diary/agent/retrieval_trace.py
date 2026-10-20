"""Opt-in retrieval tracing for the synthetic Diary gate.

Off unless BOTH hold: the operator set DIARY_RETRIEVAL_TRACE=1 in the service environment (production
never does) and the request itself sends `X-Diary-Retrieval-Trace: 1`. A trace is built per request,
returned only in that request's own response, and never stored: there is no side channel, so a tenant
can only ever receive the trace of its own exchange.

A trace carries ids, scores, counts and budget figures only. Never entry text, entry headers,
embeddings, the query or credentials.

The diary has no reranker. The stages are (1) vector-search candidates (cosine similarity, at least
min_score; top_k * 2 requested, or top_k * 4 when the message names an entity or a time anchor), then
(1b, services#116) the entity focus (entity_focus.py): a lexical check that drops only confirmed
near-name hits, hits naming only other identified people (when an asked person is named in full) and hits
dated after a time anchor, keeping score order, at most top_k * 2, then (2) selection in score order under the token budget, capped at top_k. `candidates` lists what
stage 1b kept (`rank` is the position there, `vector_rank` the stage 1 rank) and `focus.removed` every
stage 1 candidate it dropped, with the reason. With no entity or time anchor in the message stage 1b
does not run (`focus.checked` is false) and `candidates` is stage 1. `rerank.present` is false so the
gate can tell "no reranker" from "reranker returned nothing": the focus filters and never reorders.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)

ENV_FLAG = "DIARY_RETRIEVAL_TRACE"
HEADER = "X-Diary-Retrieval-Trace"
SCHEMA_VERSION = 1
_TIME_RE = re.compile(r"^\s*(\d{1,2}:\d{2})\b")


def trace_requested(request) -> bool:
    """True only when the operator flag AND the per-request header are both exactly "1"."""
    if os.environ.get(ENV_FLAG) != "1":
        return False
    return str(request.headers.get(HEADER, "")).strip() == "1"


class RetrievalTrace:
    """Collects one exchange's retrieval stages. Created per request, never shared."""

    def __init__(self) -> None:
        self.failed = False  # set when any tracing step raised: the response then carries no trace
        self.data: dict = {
            "version": SCHEMA_VERSION,
            "status": "not_run",
            "params": {},
            "rerank": {"present": False},
            "candidates": [],
            "loaded": [],
            "budget": {},
            "counts": {"candidates": 0, "loaded": 0, "dropped_budget": 0, "dropped_top_k": 0},
            "direct_fallback": {"used": False, "found": False, "tokens": 0},
            "focus": {"checked": False},
        }

    def set_status(self, status: str) -> None:
        self.data["status"] = status

    def set_params(self, **params) -> None:
        self.data["params"] = params

    def set_budget(self, **budget) -> None:
        self.data["budget"] = budget

    @staticmethod
    def _ref(row: dict) -> dict:
        # The entry header is entry-derived text; only its leading HH:MM clock time is kept.
        m = _TIME_RE.match(str(row.get("header") or ""))
        return {"id": row.get("id"), "day": row.get("day"), "time": m.group(1) if m else None}

    def add_candidate(self, rank: int, row: dict, vector_rank: Optional[int] = None) -> dict:
        item = {"rank": rank, **self._ref(row), "score": row.get("score"), "loaded": False, "dropped": None,
                "vector_rank": rank if vector_rank is None else vector_rank}
        self.data["candidates"].append(item)
        return item

    def set_focus(self, focus, vector_candidates: int) -> None:
        """Stage 1b (services#116). Counts and entry refs only: never the asked names or entry text."""
        self.data["focus"] = {
            "checked": True,
            "entities": len(focus.entities),
            "carried": bool(focus.carried),
            "time_anchor": focus.anchor is not None,
            "applied": bool(focus.applied),
            "vector_candidates": vector_candidates,
            "removed": [{"vector_rank": rank, **self._ref(row), "score": row.get("score"), "reason": reason}
                        for rank, row, reason in focus.removed],
            "removed_by_reason": {reason: sum(1 for _, _, r in focus.removed if r == reason)
                                  for reason in sorted({r for _, _, r in focus.removed})},
            "similar_name_labelled": len(focus.similar),
            "superseded_labelled": len(focus.corrected_by),
        }

    def add_loaded(self, row: dict, tokens: int, chars: int) -> None:
        self.data["loaded"].append({**self._ref(row), "score": row.get("score"), "tokens": tokens, "chars": chars})

    def finish(self, used_chars: int) -> None:
        cands = self.data["candidates"]
        self.data["budget"]["used_chars"] = used_chars
        self.data["budget"]["used_tokens"] = sum(e["tokens"] for e in self.data["loaded"])
        self.data["counts"] = {
            "candidates": len(cands),
            "loaded": len(self.data["loaded"]),
            "dropped_budget": sum(1 for c in cands if c["dropped"] == "budget"),
            "dropped_top_k": sum(1 for c in cands if c["dropped"] == "top_k"),
        }

    def set_direct_fallback(self, found: bool, tokens: int) -> None:
        """The direct-file fallback ran (``used``); ``found`` says whether it produced any text."""
        self.data["direct_fallback"] = {"used": True, "found": bool(found), "tokens": tokens}


class SafeTrace:
    """Guards a collector so tracing can never fail an exchange.

    Every collector call goes through here: if one raises, it is logged once, the collector is marked
    failed, and this and every later call is a no-op returning None. The response then has no trace.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    @classmethod
    def wrap(cls, trace):
        if trace is None or isinstance(trace, cls):
            return trace
        return cls(trace)

    def __getattr__(self, name):
        inner = self._inner

        def call(*args, **kwargs):
            if getattr(inner, "failed", False):
                return None
            try:
                return getattr(inner, name)(*args, **kwargs)
            except Exception:  # noqa: BLE001 - tracing is optional; never take the chat down
                log.warning("retrieval trace disabled for this request: %s failed", name, exc_info=True)
                try:
                    inner.failed = True
                except Exception:  # noqa: BLE001
                    pass
                return None

        return call


def attach(result: dict, trace: Optional[RetrievalTrace]) -> dict:
    """Add the trace to a result, unless tracing is off or failed (then the result is untouched)."""
    try:
        if trace is not None and not trace.failed:
            result["retrieval_trace"] = trace.data
    except Exception:  # noqa: BLE001
        log.warning("retrieval trace dropped from the response", exc_info=True)
        result.pop("retrieval_trace", None)
    return result
