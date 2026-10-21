#!/usr/bin/env python3
"""Offline replay of retrieval precision on recorded candidates (services#114).

    replay_precision.py [--fixture FILE] [--json]

The fixture (tests/fixtures/precision_replay_114.json) is trimmed from synthetic gate run 1: per traced query,
the vector candidates the diary returned (entry id and cosine score, in score order) and the synthetic text of
every entry named. Each setting runs the real ContextAssembler over a retriever that returns those recorded
rows, then scores the trace's candidates the way gate/score.py does (precision@k and best-set recall@k, macro
means, queries with relevant entries only).

Limits: no embedding is computed, so a setting can only remove or keep recorded candidates. Run 1 recorded 16
candidates per query, while a focused query asks for 32 today; the extra candidates scored below these and so
could only add distractors further down. Not a substitute for the full gate run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

DIARY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIARY))

from agent.config import Config  # noqa: E402
from agent.context import ContextAssembler  # noqa: E402
from agent.retrieval_trace import RetrievalTrace  # noqa: E402

FIXTURE = DIARY / "tests" / "fixtures" / "precision_replay_114.json"
QUERY_DAY = date(2030, 3, 7)
K = 8

SETTINGS = {
    "run 1 (no focus, no gap)": {"entity_focus": False, "score_gap": "off"},
    "score gap 0.08 only": {"entity_focus": False, "score_gap": 0.08},
    "entity focus only (main)": {"entity_focus": True, "score_gap": "off"},
    "focus + score gap 0.08 (default)": {},
}


class _Store:
    def get_day_text(self, day, max_chars=None):
        return ""

    def get_standing_sections_text(self, max_chars=None):
        return ""


class _Recorded:
    """A retriever returning one query's recorded rows, best first, honouring top_k and min_score."""

    def __init__(self, rows):
        self.rows = rows

    def search(self, query, top_k=8, min_score=0.30, with_ids=False, diag=None):
        if diag is not None:
            diag["status"] = "ok"
        return [dict(r) for r in self.rows if r["score"] >= min_score][:top_k]


def _rows(data, query):
    out = []
    for entry_id, score in query["candidates"]:
        e = data["entries"][entry_id]
        text = f"[{e['day']}] {e['day_header']}\n### {e['header']}\n{e['body']}"
        out.append({"id": entry_id, "day": e["day"], "header": e["header"], "text": text, "score": score})
    return out


def candidates(data, query, retrieval):
    cfg = Config({"context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
                  "retrieval": {"top_k": K, "min_score": 0.30, "max_context_tokens": 5200, **retrieval},
                  "prompts": {"system_path": ""}})
    asm = ContextAssembler(_Store(), _Recorded(_rows(data, query)), cfg)
    asm._system_prompt = lambda: "SYSTEM"
    trace = RetrievalTrace()
    asm.build(QUERY_DAY, query["text"], trace=trace)
    return [c["id"] for c in trace.data["candidates"]], [c["id"] for c in trace.data["loaded"]]


def score(data, retrieval) -> dict:
    fam = defaultdict(lambda: {"precision": [], "recall": [], "loaded_recall": [], "returned": []})
    for q in data["queries"]:
        sets = q["relevant"]
        if not sets:
            continue  # gate/score.py skips queries without relevant entries (deletions)
        cands, loaded = candidates(data, q, retrieval)
        top = cands[:K]
        relevant = set().union(*map(set, sets))
        p = sum(1 for x in top if x in relevant) / len(top) if top else 0.0
        r = max(len(set(s) & set(top)) / len(s) for s in sets)
        lr = max(len(set(s) & set(loaded)) / len(s) for s in sets)
        for name in ("all", q["category"]):
            f = fam[name]
            f["precision"].append(p)
            f["recall"].append(r)
            f["loaded_recall"].append(lr)
            f["returned"].append(len(top))
    return {name: {"queries": len(v["precision"]),
                   f"precision@{K}": round(sum(v["precision"]) / len(v["precision"]), 4),
                   f"recall@{K}": round(sum(v["recall"]) / len(v["recall"]), 4),
                   "loaded_recall": round(sum(v["loaded_recall"]) / len(v["loaded_recall"]), 4),
                   "mean_returned": round(sum(v["returned"]) / len(v["returned"]), 2)}
            for name, v in sorted(fam.items())}


def replay(fixture: Path = FIXTURE) -> dict:
    data = json.loads(Path(fixture).read_text(encoding="utf-8"))
    return {label: score(data, retrieval) for label, retrieval in SETTINGS.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    result = replay(Path(args.fixture))
    if args.json:
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0
    print(f"| setting | family | queries | precision@{K} | recall@{K} | loaded recall | mean returned |")
    print("|---|---|---|---|---|---|---|")
    for label, fams in result.items():
        for name, m in fams.items():
            print(f"| {label} | {name} | {m['queries']} | {m[f'precision@{K}']:.3f} | {m[f'recall@{K}']:.3f} "
                  f"| {m['loaded_recall']:.3f} | {m['mean_returned']:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
