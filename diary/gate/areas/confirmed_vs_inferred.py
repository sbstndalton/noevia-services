#!/usr/bin/env python3
"""Workflow area: confirmed-vs-inferred (needs inference). Parallel-safe: own tenant, own copy of the fixture.

After an answer, the person asks whether it is written in the diary or inferred. The judge checks
the answer separates recorded facts from inference (rubric criterion confirmed_vs_inferred).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _model  # noqa: E402

PARALLEL_SAFE = True


IDS = {"Q-needle-02", "Q-needle-09", "Q-needle-17", "Q-needle-24", "Q-needle-30", "Q-unans-1", "Q-unans-4"}
ASK = "Is that written in my diary, or are you inferring it?"


def select(q):
    return q["id"] in IDS


def turns(q):
    return [(q["text"], q), (ASK, q)]


if __name__ == "__main__":
    sys.exit(_model.run("confirmed-vs-inferred", __doc__, select, turns=turns))
