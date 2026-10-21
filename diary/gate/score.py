#!/usr/bin/env python3
"""Score a Diary gate run: one result per stage against thresholds.json, no blended score.

    score.py --run-dir RUN --fixture FIX [--judgements FILE]

Writes RUN/stages/score.json and RUN/matrix.csv (stages and workflow areas, one row each).
Stages: candidate retrieval, reranking, context load, answer. Retrieval stages need the diary's
retrieval trace; without one they are "untested: trace unavailable". Answer rows carry deterministic
signals from rubric.json and stay "needs-judge" until RUN/judgements.jsonl gives a judge verdict per
row (see README); only then can the answer stage, and the run, pass. No model judge is built in.
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
    rows = [t for t in traces if isinstance(t.get("trace"), dict) and isinstance(t["trace"].get("candidates"), list)
            and "#" not in t["id"]]
    if not rows:
        reason = "untested: trace unavailable or unknown shape"
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
        rr = {"stage": "reranking", "mrr_candidates": mean(mrr_c), "mrr_reranked": mean(mrr_r), f"ndcg@{k}": mean(nd)}
        rr["status"] = "pass" if rr["mrr_reranked"] >= rr["mrr_candidates"] and rr[f"ndcg@{k}"] >= th["reranking"]["ndcg_at_k_min"] else "fail"
    else:
        rr = {"stage": "reranking", "status": th["reranking"]["not_applicable"]}
    if has_ctx:
        cl = th["context_load"]
        ctx = {"stage": "context_load", "recall": mean(crec), "within_budget": mean(within), "budget_tokens": budget}
        ctx["status"] = "pass" if ctx["recall"] >= cl["recall_min"] and ctx["within_budget"] >= cl["within_budget_min"] else "fail"
    else:
        ctx = {"stage": "context_load", "status": "untested: trace has no loaded-context list"}
    return cand, rr, ctx


def load_judgements(path: Path) -> dict:
    """judgements.jsonl: {"row": <answer row id>, "verdict": "pass"|"fail", "fact", "citation", "stale",
    "abstained" (booleans or null), "judge": <role>}. Later rows for the same id win."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    for j in g.read_jsonl(path):
        if isinstance(j, dict) and j.get("row") and j.get("verdict") in ("pass", "fail"):
            out[j["row"]] = j
    return out


def judged_metrics(rows: list, judgements: dict) -> dict:
    """Stage metrics from judge verdicts (only when every row is judged)."""
    get = lambda r, k: judgements[r["id"]].get(k)  # noqa: E731
    answerable = [r for r in rows if not r["expect_abstain"]]
    unans = [r for r in rows if r["category"] == "unanswerable"]
    facts = [r for r in answerable if get(r, "fact") is not None]
    cites = [r for r in answerable if get(r, "citation") is not None]
    return {"fact_accuracy": mean([1.0 if get(r, "fact") else 0.0 for r in facts]),
            "citations": mean([1.0 if get(r, "citation") else 0.0 for r in cites]),
            "stale_fact_rows": sum(1 for r in rows if get(r, "stale")),
            "abstentions": sum(1 for r in unans if get(r, "abstained")), "abstentions_of": len(unans),
            "false_abstentions": sum(1 for r in answerable if get(r, "abstained"))}


def stage_answer(queries: dict, answers: list, th: dict, rubric: dict, judgements: dict | None = None) -> tuple:
    judgements = judgements or {}
    rows = []
    for a in answers:
        q = queries.get(a["query"])
        if q is None:
            continue
        spec = q if a["turn"] == 0 else {**q, **q["followup"], "citation_days": [], "expect_abstain": False}
        sig = answer_signals(spec, a["answer"], rubric) if a["status"] == 200 else None
        j = judgements.get(a["id"])
        verdict = ("fail: http " + str(a["status"])) if a["status"] != 200 else (j["verdict"] if j else "needs-judge")
        rows.append({"id": a["id"], "category": q["category"], "http": a["status"], "expect_abstain": spec["expect_abstain"],
                     "signals": sig, "verdict": verdict})
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
    result = {"stage": "answer", "provisional": provisional, "rows": len(rows),
              "judged": sum(1 for r in rows if r["verdict"] in ("pass", "fail")),
              "thresholds": {k: v for k, v in t.items() if not k.startswith("verdict")}}
    if provisional["http_failures"]:
        result["status"] = "fail"
    elif any(r["verdict"] == "needs-judge" for r in rows):
        result["status"] = "needs-judge"
    else:
        m = judged_metrics(rows, judgements)
        passed = ((m["fact_accuracy"] or 0) >= t["fact_accuracy_min"] and (m["citations"] or 0) >= t["citation_min"]
                  and m["stale_fact_rows"] <= t["stale_facts_max"] and m["abstentions"] >= t["abstentions_min"]
                  and m["false_abstentions"] <= t["false_abstentions_max"])
        result.update(judged_metrics=m, status="pass" if passed else "fail")
    return result, rows


def answer_counts(rows: list) -> dict:
    """#134: how many of these answers.jsonl rows were cut off at the length cap, and how many non-2xx rows
    carry each error code ("unknown" when the row names none, as every row from before the field existed).
    Read from the rows only; it never feeds a status, a threshold or the verdict."""
    truncated, codes = 0, {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("truncated") is True or r.get("finish_reason") == "length":
            truncated += 1
        status = r.get("status", r.get("http"))
        if isinstance(status, int) and not 200 <= status < 300:
            code = r.get("error_code") if isinstance(r.get("error_code"), str) and r.get("error_code") else "unknown"
            codes[code] = codes.get(code, 0) + 1
    return {"rows": sum(1 for r in rows if isinstance(r, dict)), "truncated": truncated, "error_codes": dict(sorted(codes.items()))}


def run_counts(run: Path, stage_answers: list) -> dict:
    """answer_counts for the stage answers ("stages") and for every workflow area's answers.jsonl, plus the totals."""
    by_area = {"stages": answer_counts(stage_answers)} if stage_answers else {}
    for p in sorted((run / "workflows").glob("*/answers.jsonl")):
        by_area[p.parent.name] = answer_counts(g.read_jsonl(p))
    overall = {"rows": 0, "truncated": 0, "error_codes": {}}
    for c in by_area.values():
        overall["rows"] += c["rows"]
        overall["truncated"] += c["truncated"]
        for code, n in c["error_codes"].items():
            overall["error_codes"][code] = overall["error_codes"].get(code, 0) + n
    overall["error_codes"] = dict(sorted(overall["error_codes"].items()))
    return {"overall": overall, "by_area": by_area}


def area_status(result: dict, area_dir: Path, judgements: dict) -> str:
    """An area whose only open rows are needs-judge answers becomes pass (all judged pass) or fail."""
    status = result["status"]
    rows_path = area_dir / "answers.jsonl"
    if status != "untested" or not rows_path.exists():
        return status
    open_checks = [c for c in result["checks"] if c["status"] != "pass"]
    if any(not str(c.get("reason", "")).startswith("needs-judge") for c in open_checks):
        return status
    rows = g.read_jsonl(rows_path)
    verdicts = [judgements.get(r["id"], {}).get("verdict") for r in rows]
    if not rows or any(v is None for v in verdicts):
        return status
    return "pass" if all(v == "pass" for v in verdicts) else "fail"


def score(run: Path, fixture: Path, judgements_path: Path | None = None) -> dict:
    g.verify_fixture(fixture)
    th = json.loads((g.GATE_DIR / "thresholds.json").read_text(encoding="utf-8"))
    rubric = json.loads((g.GATE_DIR / "rubric.json").read_text(encoding="utf-8"))
    judgements = load_judgements(judgements_path or run / "judgements.jsonl")
    queries = {q["id"]: q for q in g.read_jsonl(fixture / "truth" / "queries.jsonl")}
    sd = run / "stages"
    traces = g.read_jsonl(sd / "traces.jsonl") if (sd / "traces.jsonl").exists() else []
    answers = g.read_jsonl(sd / "answers.jsonl") if (sd / "answers.jsonl").exists() else []
    plan = json.loads((sd / "queries-plan.json").read_text(encoding="utf-8")) if (sd / "queries-plan.json").exists() else None
    stages, ready, rows = [], True, []
    for name in ("load", "mutations"):
        p = sd / f"{name}.json"
        ok = p.exists() and bool(json.loads(p.read_text(encoding="utf-8")).get("ok"))
        ready = ready and ok
        stages.append({"stage": name, "status": "pass" if ok else ("fail" if p.exists() else "untested: not run")})
    later = ("candidate_retrieval", "reranking", "context_load", "answer")
    if not ready:
        stages.extend({"stage": s, "status": "untested: load or mutations not ok"} for s in later)
    elif plan is None or not answers:
        stages.extend({"stage": s, "status": "untested: no queries run"} for s in later)
    elif len(answers) < plan["planned_rows"]:
        partial = f"untested: partial {len(answers)}/{plan['planned_rows']}"
        stages.extend({"stage": s, "status": partial} for s in later[:3])
        if any(a.get("status") != 200 for a in answers):
            # A failed turn ends its query early (a follow-up is never asked): that is a failure, not a gap.
            answer, rows = stage_answer(queries, answers, th, rubric, judgements)
            stages.append({**answer, "status": "fail", "partial": partial})
        else:
            stages.append({"stage": "answer", "status": partial})
    else:
        stages.extend(stage_retrieval(queries, traces, th))
        answer, rows = stage_answer(queries, answers, th, rubric, judgements)
        stages.append(answer)
    g.write_json(sd / "answer-rows.json", rows)
    areas = []
    for p in sorted((run / "workflows").glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        areas.append({"area": r["area"], "status": area_status(r, p.parent, judgements), "parallel_safe": r["parallel_safe"],
                      "evidence": str(p.relative_to(run))})
    passing = ("pass", th["reranking"]["not_applicable"])
    result = {"stages": stages, "areas": areas, "blended_score": None, "judgements": len(judgements),
              "counts": run_counts(run, answers),
              "verdict": "pass" if all(s["status"] in passing for s in stages) and areas
              and all(a["status"] == "pass" for a in areas) else "not passed (see each row)"}
    g.write_json(sd / "score.json", result)
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
    ap.add_argument("--judgements", default=None, help="judge verdicts (default RUN/judgements.jsonl)")
    args = ap.parse_args(argv)
    result = score(Path(args.run_dir), Path(args.fixture), Path(args.judgements) if args.judgements else None)
    print(json.dumps({"stages": {s["stage"]: s["status"] for s in result["stages"]},
                      "areas": {a["area"]: a["status"] for a in result["areas"]}, "verdict": result["verdict"],
                      "counts": result["counts"]["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
