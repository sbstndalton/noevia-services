#!/usr/bin/env python3
"""Workflow area: semantic (needs inference). Parallel-safe: own tenant, own copy of the fixture.

Needles at early, middle and late positions against near-duplicate distractors, paraphrased
wording, multi-entry questions, ambiguous names and unanswerable questions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _model  # noqa: E402

PARALLEL_SAFE = True


CATEGORIES = {"needle", "paraphrase", "multi-entry", "ambiguous", "unanswerable"}


def select(q):
    return q["category"] in CATEGORIES


if __name__ == "__main__":
    sys.exit(_model.run("semantic", __doc__, select))
