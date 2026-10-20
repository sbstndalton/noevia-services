#!/usr/bin/env python3
"""Workflow area: corrections (needs inference). Parallel-safe: own tenant, own copy of the fixture.

Scripted mutations: corrected values must win over the superseded ones, deleted details must not
reappear, paraphrased entries must still answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _model  # noqa: E402

PARALLEL_SAFE = True


def select(q):
    return bool(q["mutations"]) and q["category"] in ("correction", "deletion", "paraphrase")


if __name__ == "__main__":
    sys.exit(_model.run("corrections", __doc__, select))
