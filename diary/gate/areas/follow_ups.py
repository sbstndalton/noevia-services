#!/usr/bin/env python3
"""Workflow area: follow-ups (needs inference). Parallel-safe: own tenant, own copy of the fixture.

Two-turn sessions: the follow-up names the near-duplicate person and must be answered from the
distractor entry, not the first answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _model  # noqa: E402

PARALLEL_SAFE = True


def select(q):
    return "followup" in q


def turns(q):
    return [(q["text"], q), (q["followup"]["text"], {**q, **q["followup"], "citation_days": [], "expect_abstain": False})]


if __name__ == "__main__":
    sys.exit(_model.run("follow-ups", __doc__, select, turns=turns))
