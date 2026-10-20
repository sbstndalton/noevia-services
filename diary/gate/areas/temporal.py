#!/usr/bin/env python3
"""Workflow area: temporal (needs inference). Parallel-safe: own tenant, own copy of the fixture.

As-of questions answered at an earlier clock (the value before a correction) and the same facts
asked at the query clock (the corrected value).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _model  # noqa: E402

PARALLEL_SAFE = True


def select(q):
    return q["category"] in ("temporal", "correction")


if __name__ == "__main__":
    sys.exit(_model.run("temporal", __doc__, select))
