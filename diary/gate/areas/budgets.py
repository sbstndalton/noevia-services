#!/usr/bin/env python3
"""Workflow area: budgets. RUN ALONE: it sends limit-sized bodies and, with inference, measures context.

Without a model: the append limit (8000 characters), the file limit (512 KiB) and the signed-body cap
(64 MiB) hold at the boundary. With a model and a retrieval trace: every loaded context fits
retrieval.max_context_tokens; without a trace that part is "untested: trace unavailable".
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gatelib as g  # noqa: E402
import harness  # noqa: E402

PARALLEL_SAFE = False
BUDGET_TOKENS = 5200
PROBE_QUERIES = ("Q-needle-00", "Q-needle-20", "Q-multi-0", "Q-ambig-0")


def main() -> int:
    args = g.area_args(__doc__).parse_args()
    area = g.Area("budgets", args.run_dir, PARALLEL_SAFE)
    env = g.load_env(args.env_file)
    user = g.tenant_id(args.tenant_label or "area-budgets")
    c = g.Client(args.base_url, env, log=area.log_request)
    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    for name, size, want in (("append-at-limit", 8000, 200), ("append-over-limit", 8001, 400)):
        r = c.request("POST", "/api/entries/append", user=user,
                      json_body={"text": "b" * size, "title": "Budget", "requestId": str(uuid.uuid4()), "entryTime": now()})
        area.check(name, r.status == want, http=r.status)
    for name, size, want in (("file-at-limit", 512 * 1024, 200), ("file-over-limit", 512 * 1024 + 1, 400)):
        r = c.request("PUT", "/api/file", user=user,
                      json_body={"path": f"Notes/{name}.md", "content": "c" * size, "version": None}, timeout=120)
        area.check(name, r.status == want, http=r.status)
    big = b'{"messages": "' + b"d" * (64 * 1024 * 1024) + b'"}'
    r = c.request("POST", "/v1/chat/completions", user=user, raw_body=big, content_type="application/json", timeout=300)
    # The diary answers 413 as soon as the cap is crossed; the upload may be cut off mid-body (status 0).
    area.check("signed-body-cap", r.status in (413, 0), http=r.status)
    h = c.request("GET", "/api/health")
    area.check("alive-after-body-cap", h.status == 200, http=h.status)
    if not g.inference_enabled(env):
        area.mark("context-within-budget", "blocked", "inference disabled in .env.gate (smoke mode)")
        return area.finish({"tenant": user})
    if not args.fixture:
        area.mark("context-within-budget", "blocked", "--fixture not given")
        return area.finish({"tenant": user})
    fixture = Path(args.fixture)
    loaded = g.load_fixture(c, user, fixture)
    area.check("fixture-loaded", not loaded["failures"], written=loaded["written"])
    index, queries = harness.entry_index(fixture), {q["id"]: q for q in g.read_jsonl(fixture / "truth" / "queries.jsonl")}
    seen = []
    for qid in PROBE_QUERIES:
        gate = g.load_gate(env)
        if not gate["ok"]:
            area.mark("context-within-budget", "blocked", "MS-01 load gate: " + "; ".join(gate["reasons"]))
            return area.finish({"tenant": user})
        res = harness.ask(c, user, queries[qid]["text"], queries[qid]["clock"], f"budgets-{qid}")
        if res["status"] != 200:
            area.check(f"{qid}-http", False, http=res["status"])
            continue
        if res["trace"] is None:
            continue
        t = harness.normalize_trace(res["trace"], index)
        seen.append({"query": qid, "context_tokens": t["context_tokens"], "budget_tokens": t["budget_tokens"] or BUDGET_TOKENS})
    if not seen:
        area.mark("context-within-budget", "untested", "trace unavailable")
    else:
        ok = all(s["context_tokens"] is not None and s["context_tokens"] <= s["budget_tokens"] for s in seen)
        area.check("context-within-budget", ok, samples=seen)
    g.write_json(area.dir / "context-samples.json", seen)
    return area.finish({"tenant": user})


if __name__ == "__main__":
    sys.exit(main())
