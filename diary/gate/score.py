#!/usr/bin/env python3
"""Score a Diary gate run: one result per stage against thresholds.json, no blended score.

    score.py --run-dir RUN --fixture FIX

Writes RUN/stages/score.json and RUN/matrix.csv (stages and workflow areas, one row each).
Stages: candidate retrieval, reranking, context load, answer. Retrieval stages need the diary's
retrieval trace; without one they are "untested: trace unavailable". Answer rows carry deterministic
signals from rubric.json and stay "needs-judge": no model judge is built in.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gatelib as g  # noqa: E402


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("’", "'").replace("‘", "'").lower()).strip()


def best_recall(sets: list, got: list) -> float:
    return max((len(set(s) & set(got)) / len(s) for s in sets if s), default=0.0)


def precision(sets: list, got: list) -> float:
    relevant = set().union(*map(set, sets)) if sets else set()
    return (sum(1 for x in got if x in relevant) / len(got)) if got else 0.0


def mrr(sets: list, got: list) -> float:
    relevant = set().union(*map(set, sets)) if sets else set()
    return next((1.0 / (i + 1) for i, x in enumerate(got) if x in relevant), 0.0)


def ndcg(sets: list, got: list, k: int) -> float:
    relevant = set().union(*map(set, sets)) if sets else set()
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(got[:k]) if x in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, min((len(s) for s in sets if s), default=0))))
    return dcg / ideal if ideal else 0.0


def mean(values: list):
    return round(sum(values) / len(values), 4) if values else None


def citation_forms(day_iso: str) -> list:
    d = date.fromisoformat(day_iso)
    month = d.strftime("%B").lower()
    return [f"{month} {d.day}", f"{d.day} {month}", day_iso]


def answer_signals(q: dict, answer: str, rubric: dict) -> dict:
    text = _norm(answer)
    facts = [f for f in q.get("expected_facts", [])]
    abstained = any(p in text for p in rubric["abstention_phrases"])
    return {"facts_present": all(_norm(f) in text for f in facts) if facts else None,
            "forbidden_present": [f for f in q.get("forbidden_facts", []) if _norm(f) in text],
            "cited": any(form in text for day in q.get("citation_days", []) for form in citation_forms(day)) if q.get("citation_days") else None,
            "abstained": abstained}


def stage_retrieval(queries: dict, traces: list, th: dict) -> tuple:
    k = th["candidate_retrieval"]["k"]
    rows = [t for t in traces if t.get("trace") and "#" not in t["id"]]
    if not rows:
        reason = "untested: trace unavailable"
        return ({"stage": "candidate_retrieval", "status": reason}, {"stage": "reranking", "status": reason},
                {"stage": "context_load", "status": reason})
    rec, prec, mrr_c, mrr_r, nd, crec, within, has_rerank, has_ctx = [], [], [], [], [], [], [], False, False
    budget = th["context_load"]["budget_tokens"]
    for t in rows:
        q = queries.get(t["id"])
        if not q or not q["relevant_entry_ids"]:
            continue
        sets, tr = q["relevant_entry_ids"], t["trace"]
        top = tr["candidates"][:k]
        rec.append(best_recall(sets, top))
        prec.append(precision(sets, top))
        if tr.get("reranked") is not None:
            has_rerank = True
            mrr_c.append(mrr(sets, tr["candidates"][:k]))
            mrr_r.append(mrr(sets, tr["reranked"][:k]))
            nd.append(ndcg(sets, tr["reranked"], k))
        if tr.get("context") is not None:
            has_ctx = True
            crec.append(best_recall(sets, tr["context"]))
            tokens = tr.get("context_tokens")
            within.append(1.0 if tokens is not None and tokens <= (tr.get("budget_tokens") or budget) else 0.0)
    c = th["candidate_retrieval"]
    cand = {"stage": "candidate_retrieval", "queries": len(rec), f"recall@{k}": mean(rec), f"precision@{k}": mean(prec)}
    cand["status"] = "pass" if rec and cand[f"recall@{k}"] >= c["recall_at_k_min"] and cand[f"precision@{k}"] >= c["precision_at_k_min"] else "fail"
    if has_rerank:
        r = th["reranking"]
        rr = {"stage": "reranking", "mrr_candidates": mean(mrr_c), "mrr_reranked": mean(mrr_r), f"ndcg@{k}": mean(nd)}
        rr["status"] = "pass" if rr["mrr_reranked"] >= rr["mrr_candidates"] and rr[f"ndcg@{k}"] >= r["ndcg_at_k_min"] else "fail"
    else:
        rr = {"stage": "reranking", "status": th["reranking"]["not_applicable"]}
    if has_ctx:
        cl = th["context_load"]
        ctx = {"stage": "context_load", "recall": mean(crec), "within_budget": mean(within), "budget_tokens": budget}
        ctx["status"] = "pass" if ctx["recall"] >= cl["recall_min"] and ctx["within_budget"] >= cl["within_budget_min"] else "fail"
    else:
        ctx = {"stage": "context_load", "status": "untested: trace has no loaded-context list"}
    return cand, rr, ctx


def stage_answer(queries: dict, answers: list, th: dict, rubric: dict) -> tuple:
    rows = []
    for a in answers:
        q = queries.get(a["query"])
        if q is None:
            continue
        spec = q if a["turn"] == 0 else {**q, **q["followup"], "citation_days": [], "expect_abstain": False}
        sig = answer_signals(spec, a["answer"], rubric) if a["status"] == 200 else None
        rows.append({"id": a["id"], "category": q["category"], "http": a["status"], "expect_abstain": spec["expect_abstain"],
                     "signals": sig, "verdict": "needs-judge" if a["status"] == 200 else "fail: http " + str(a["status"])})
    if not rows:
        return {"stage": "answer", "status": "untested: no answers recorded"}, rows
    ok = [r for r in rows if r["signals"]]
    answerable = [r for r in ok if not r["expect_abstain"]]
    facts = [r for r in answerable if r["signals"]["facts_present"] is not None]
    cites = [r for r in answerable if r["signals"]["cited"] is not None]
    unans = [r for r in ok if r["category"] == "unanswerable"]
    t = th["answer"]
    provisional = {
        "fact_accuracy": mean([1.0 if r["signals"]["facts_present"] else 0.0 for r in facts]),
        "citations": mean([1.0 if r["signals"]["cited"] else 0.0 for r in cites]),
        "stale_fact_rows": sum(1 for r in ok if r["signals"]["forbidden_present"]),
        "abstentions": f"{sum(1 for r in unans if r['signals']['abstained'])}/{len(unans)}",
        "false_abstentions": sum(1 for r in answerable if r["signals"]["abstained"]),
        "http_failures": sum(1 for r in rows if r["http"] != 200)}
    failing = provisional["http_failures"] > 0
    return {"stage": "answer", "status": "fail" if failing else "needs-judge", "provisional": provisional,
            "thresholds": {k: v for k, v in t.items() if not k.startswith("verdict")}, "rows": len(rows)}, rows


def score(run: Path, fixture: Path) -> dict:
    th = json.loads((g.GATE_DIR / "thresholds.json").read_text(encoding="utf-8"))
    rubric = json.loads((g.GATE_DIR / "rubric.json").read_text(encoding="utf-8"))
    queries = {q["id"]: q for q in g.read_jsonl(fixture / "truth" / "queries.jsonl")}
    stages_dir = run / "stages"
    traces = g.read_jsonl(stages_dir / "traces.jsonl") if (stages_dir / "traces.jsonl").exists() else []
    answers = g.read_jsonl(stages_dir / "answers.jsonl") if (stages_dir / "answers.jsonl").exists() else []
    stages = []
    for name in ("load", "mutations"):
        p = stages_dir / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            stages.append({"stage": name, "status": "pass" if data.get("ok") else "fail"})
        else:
            stages.append({"stage": name, "status": "untested: not run"})
    if traces:
        stages.extend(stage_retrieval(queries, traces, th))
    else:
        stages.extend({"stage": s, "status": "untested: no queries run"} for s in ("candidate_retrieval", "reranking", "context_load"))
    answer, rows = stage_answer(queries, answers, th, rubric)
    stages.append(answer)
    g.write_json(stages_dir / "answer-rows.json", rows)
    areas = []
    for p in sorted((run / "workflows").glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        areas.append({"area": r["area"], "status": r["status"], "parallel_safe": r["parallel_safe"], "evidence": str(p.relative_to(run))})
    result = {"stages": stages, "areas": areas, "blended_score": None,
              "verdict": "pass" if all(s["status"] in ("pass", th["reranking"]["not_applicable"]) for s in stages)
              and areas and all(a["status"] == "pass" for a in areas) else "not passed (see each row)"}
    g.write_json(stages_dir / "score.json", result)
    with open(run / "matrix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "name", "status", "metrics", "evidence"])
        for s in stages:
            metrics = {k: v for k, v in s.items() if k not in ("stage", "status")}
            w.writerow(["stage", s["stage"], s["status"], json.dumps(metrics, sort_keys=True) if metrics else "", f"stages/{s['stage']}"])
        for a in areas:
            w.writerow(["workflow", a["area"], a["status"], json.dumps({"parallel_safe": a["parallel_safe"]}), a["evidence"]])
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--fixture", required=True)
    args = ap.parse_args(argv)
    result = score(Path(args.run_dir), Path(args.fixture))
    print(json.dumps({"stages": {s["stage"]: s["status"] for s in result["stages"]},
                      "areas": {a["area"]: a["status"] for a in result["areas"]}, "verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
