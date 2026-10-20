#!/usr/bin/env python3
"""Diary gate harness: load the synthetic fixture through the diary API, apply the scripted
mutations, run the ground-truth queries through /v1/chat/completions with signed tenant assertions,
and record answers and retrieval traces as evidence. score.py turns the evidence into stage results.

    harness.py init    --evidence-root DIR --rev REV --phase pre|post --fixture FIX
    harness.py load    --run-dir RUN --fixture FIX          (no inference: file API + read-back)
    harness.py mutate  --run-dir RUN --fixture FIX          (no inference: edits and a later entry)
    harness.py queries --run-dir RUN --fixture FIX [--only ID,...] [--limit N]
    harness.py all     --evidence-root DIR --rev REV --phase pre|post --fixture FIX

Before every inference call the MS-01 load gate runs (gatelib.load_gate): it stops when the embed or
chat engine has requests in flight or MemAvailable is under 6 GiB. Retrieval traces are recorded when
the diary returns them; otherwise every row says "trace unavailable".
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gatelib as g  # noqa: E402

CORPUS_LABEL = "gate-corpus"
TRACE_KEYS = ("trace", "retrieval_trace", "retrieval")


def _client(args, run: Path, env: dict) -> g.Client:
    return g.Client(args.base_url, env, log=lambda row: g.append_jsonl(run / "logs" / "requests.jsonl", row))


def entry_index(fixture: Path) -> dict:
    rows = g.read_jsonl(fixture / "truth" / "entries.jsonl")
    index = {}
    for e in rows:
        index[("xid", e["xid"])] = e["id"]
        index[("loc", e["path"], e["header"])] = e["id"]
        index[("dayhdr", e["day"], e["header"])] = e["id"]
    return index


def extract_trace(payload: dict):
    """The diary's retrieval trace, if this revision returns one; None otherwise."""
    if not isinstance(payload, dict):
        return None
    diary = payload.get("diary") if isinstance(payload.get("diary"), dict) else {}
    for key in TRACE_KEYS:
        if isinstance(diary.get(key), (dict, list)):
            return diary[key]
        if isinstance(payload.get(key), (dict, list)):
            return payload[key]
    return None


def _ids(items, index: dict) -> list:
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        eid = (index.get(("xid", str(it.get("xid") or "")))
               or index.get(("loc", str(it.get("document") or it.get("path") or ""), str(it.get("header") or "")))
               or index.get(("dayhdr", str(it.get("day") or ""), str(it.get("header") or "").lstrip("# ").strip())))
        out.append(eid or f"unmapped:{str(it.get('header') or it.get('xid') or '')[:40]}")
    return out


def normalize_trace(trace, index: dict) -> dict:
    """Map whatever trace shape the diary returns onto entry ids: candidates, reranked, context, tokens."""
    if isinstance(trace, list):
        trace = {"candidates": trace}
    cands = trace.get("candidates") or trace.get("retrieved") or trace.get("results") or []
    rer = trace.get("reranked")
    ctx = trace.get("context") or trace.get("loaded") or trace.get("kept")
    return {"candidates": _ids(cands, index), "reranked": _ids(rer, index) if rer is not None else None,
            "context": _ids(ctx, index) if ctx is not None else None,
            "context_tokens": trace.get("context_tokens") or trace.get("used_tokens"),
            "budget_tokens": trace.get("budget_tokens") or trace.get("max_context_tokens")}


def ask(client: g.Client, user: str, text: str, clock: str, session: str, history=None, timeout: float = 600):
    body = {"model": "diary", "messages": [*(history or []), {"role": "user", "content": text}], "session_id": session,
            "entryTime": clock.replace("Z", "+00:00"), "diary_trace": True}
    started = time.monotonic()
    r = client.request("POST", "/v1/chat/completions", user=user, json_body=body, timeout=timeout)
    payload = r.json() or {}
    answer = ""
    try:
        answer = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    return {"status": r.status, "answer": answer, "ms": round((time.monotonic() - started) * 1000),
            "decision": (payload.get("diary") or {}).get("decision") if isinstance(payload, dict) else None,
            "trace": extract_trace(payload)}


def cmd_init(args) -> Path:
    fixture = Path(args.fixture)
    fman = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    run = g.evidence_dir(args.evidence_root, args.rev, args.phase)
    shutil.copyfile(fixture / "manifest.json", run / "fixture-manifest.json")
    env = g.load_env(args.env_file)
    manifest = {"schema": "diarygate-run/1", "rev": args.rev, "phase": args.phase, "run_dir": str(run),
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base_url": args.base_url,
                "inference": "on" if g.inference_enabled(env) else "off",
                "fixture": {"tree_sha256": fman["tree_sha256"], "seed": fman["seed"], "anchor": fman["anchor"],
                            "query_clock": fman["query_clock"], "generator_commit": fman["generator_commit"]},
                "thresholds_sha256": g._sha((g.GATE_DIR / "thresholds.json").read_bytes()),
                "rubric_sha256": g._sha((g.GATE_DIR / "rubric.json").read_bytes()),
                "layout": ["manifest.json", "matrix.csv", "stages/", "workflows/<area>/", "logs/", "blockers.md"]}
    g.write_json(run / "manifest.json", manifest)
    print(str(run))
    return run


def cmd_load(args, run: Path | None = None) -> dict:
    run = Path(run or args.run_dir)
    env, fixture = g.load_env(args.env_file), Path(args.fixture)
    client, user = _client(args, run, env), g.tenant_id(args.tenant_label)
    started = time.monotonic()
    loaded = g.load_fixture(client, user, fixture)
    back = g.readback(client, user, fixture)
    ok = not loaded["failures"] and not back["mismatched"] and not back["markers_missing"] and loaded["written"] > 0
    result = {"stage": "load", "ok": ok, "tenant": user, "seconds": round(time.monotonic() - started, 1), **loaded, "readback": back}
    g.write_json(run / "stages" / "load.json", result)
    if not ok:
        g.add_blocker(run, f"fixture load/read-back failed: {len(loaded['failures'])} write failures, "
                           f"{len(back['mismatched'])} mismatched files, {len(back['markers_missing'])} missing markers")
    print(json.dumps({"stage": "load", "ok": ok, "written": loaded["written"], "files_checked": back["files_checked"],
                      "mismatched": len(back["mismatched"]), "markers_missing": len(back["markers_missing"])}))
    return result


def cmd_mutate(args, run: Path | None = None) -> dict:
    run = Path(run or args.run_dir)
    env, fixture = g.load_env(args.env_file), Path(args.fixture)
    client, user = _client(args, run, env), g.tenant_id(args.tenant_label)
    applied = g.apply_mutations(client, user, fixture)
    back = g.readback(client, user, fixture, mutated=True)
    ok = applied["applied"] == applied["total"] and not back["markers_missing"]
    result = {"stage": "mutate", "ok": ok, "tenant": user, **applied, "markers_missing": back["markers_missing"]}
    g.write_json(run / "stages" / "mutations.json", result)
    if not ok:
        g.add_blocker(run, f"scripted mutations: {applied['applied']}/{applied['total']} applied")
    print(json.dumps({"stage": "mutate", "ok": ok, "applied": applied["applied"], "total": applied["total"]}))
    return result


def cmd_queries(args, run: Path | None = None) -> int:
    run = Path(run or args.run_dir)
    env, fixture = g.load_env(args.env_file), Path(args.fixture)
    if not g.inference_enabled(env):
        g.add_blocker(run, "queries not run: inference is disabled in .env.gate")
        print(json.dumps({"stage": "queries", "status": "blocked", "reason": "inference disabled"}))
        return 3
    if not (run / "stages" / "mutations.json").exists():
        raise SystemExit("run `harness.py load` and `harness.py mutate` first")
    client, user = _client(args, run, env), g.tenant_id(args.tenant_label)
    index = entry_index(fixture)
    queries = g.read_jsonl(fixture / "truth" / "queries.jsonl")
    if args.only:
        wanted = set(args.only.split(","))
        queries = [q for q in queries if q["id"] in wanted]
    if args.limit:
        queries = queries[: args.limit]
    answers, traces = run / "stages" / "answers.jsonl", run / "stages" / "traces.jsonl"
    for path in (answers, traces):
        path.write_text("", encoding="utf-8")
    done = 0
    for q in queries:
        turns = [(q["text"], q)] + ([(q["followup"]["text"], q["followup"])] if q.get("followup") else [])
        history, session = [], f"gate-{q['id']}-{uuid.uuid4().hex[:8]}"
        for n, (text, spec) in enumerate(turns):
            gate = g.load_gate(env)
            g.append_jsonl(run / "logs" / "load-gate.jsonl", {"query": q["id"], "turn": n, **gate})
            if not gate["ok"]:
                g.add_blocker(run, f"stopped before {q['id']} turn {n}: MS-01 load gate: {'; '.join(gate['reasons'])}")
                print(json.dumps({"stage": "queries", "status": "blocked", "done": done, "reasons": gate["reasons"]}))
                return 3
            res = ask(client, user, text, q["clock"], session, history)
            trace = normalize_trace(res["trace"], index) if res["trace"] is not None else None
            row_id = q["id"] if n == 0 else q["id"] + "#followup"
            g.append_jsonl(answers, {"id": row_id, "query": q["id"], "turn": n, "category": q["category"], "status": res["status"],
                                     "answer": res["answer"], "ms": res["ms"], "decision": res["decision"],
                                     "trace": "available" if trace else "trace unavailable", "verdict": "needs-judge"})
            g.append_jsonl(traces, {"id": row_id, "trace": trace, "raw": res["trace"]} if trace else
                           {"id": row_id, "trace": None, "note": "trace unavailable"})
            history += [{"role": "user", "content": text}, {"role": "assistant", "content": res["answer"]}]
        done += 1
    print(json.dumps({"stage": "queries", "status": "done", "queries": done}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("init", "load", "mutate", "queries", "all"))
    ap.add_argument("--base-url", default=g.DEFAULT_BASE_URL)
    ap.add_argument("--env-file", default=str(g.DEFAULT_ENV))
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--evidence-root")
    ap.add_argument("--rev")
    ap.add_argument("--phase", choices=("pre", "post"))
    ap.add_argument("--tenant-label", default=CORPUS_LABEL)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.command in ("init", "all") and not (args.evidence_root and args.rev and args.phase):
        raise SystemExit("init/all need --evidence-root, --rev and --phase")
    if args.command in ("load", "mutate", "queries") and not args.run_dir:
        raise SystemExit("--run-dir is required (harness.py init prints it)")
    if args.command == "init":
        cmd_init(args)
        return 0
    if args.command == "load":
        return 0 if cmd_load(args)["ok"] else 1
    if args.command == "mutate":
        return 0 if cmd_mutate(args)["ok"] else 1
    if args.command == "queries":
        return cmd_queries(args)
    run = cmd_init(args)
    if not cmd_load(args, run)["ok"] or not cmd_mutate(args, run)["ok"]:
        return 1
    rc = cmd_queries(args, run)
    import score
    score.score(run, Path(args.fixture))
    return rc


if __name__ == "__main__":
    sys.exit(main())
