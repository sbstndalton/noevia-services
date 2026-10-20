#!/usr/bin/env python3
"""Workflow area: capture (no model). Parallel-safe: own tenant.

Append-only capture through POST /api/entries/append: a note lands in today's daily file with its
xid marker exactly once, a repeated requestId is idempotent, structure in the text is refused, and
the day and month views return it.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gatelib as g  # noqa: E402

PARALLEL_SAFE = True


def main() -> int:
    args = g.area_args(__doc__).parse_args()
    area = g.Area("capture", args.run_dir, PARALLEL_SAFE)
    env = g.load_env(args.env_file)
    user = g.tenant_id(args.tenant_label or "area-capture")
    c = g.Client(args.base_url, env, log=area.log_request)

    r = c.request("GET", "/api/health")
    area.check("health", r.status == 200 and (r.json() or {}).get("ok") is True, http=r.status)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rid = str(uuid.uuid4())
    text = "Synthetic capture check: pinned the gate tide table to the studio wall."
    body = {"text": text, "title": "Gate capture", "requestId": rid, "entryTime": now}
    r = c.request("POST", "/api/entries/append", user=user, json_body=body)
    j = r.json() or {}
    area.check("append-written", r.status == 200 and j.get("queued") is False and j.get("xid") == rid, http=r.status,
               queued=j.get("queued"))
    document = j.get("document")
    r2 = c.request("POST", "/api/entries/append", user=user, json_body=body)
    area.check("append-idempotent", r2.status == 200 and (r2.json() or {}).get("xid") == rid, http=r2.status)
    rid2 = str(uuid.uuid4())
    r3 = c.request("POST", "/api/entries/append", user=user,
                   json_body={"text": "Second synthetic note, written after the first.", "title": "Gate capture two",
                              "requestId": rid2, "entryTime": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    area.check("append-second", r3.status == 200, http=r3.status)
    if document:
        f = c.request("POST", "/api/file", user=user, json_body={"path": document})
        content = (f.json() or {}).get("content") or ""
        day = datetime.now(timezone.utc).date()
        header = f"## {day.strftime('%A')}, {day.strftime('%B')} {day.day}, {day.year}"
        area.check("daily-file-layout", f.status == 200 and content.startswith(header + "\n\n### ")
                   and document == f"Entries/{day.year}/{day.strftime('%B')}/{day.strftime('%B')} {day.day}, {day.year}.md",
                   http=f.status, document=document)
        area.check("marker-once", content.count(f"<!-- xid:{rid} -->") == 1 and content.count(text) == 1)
        area.check("order-kept", 0 <= content.find(rid) < content.find(rid2))
        area.check("me-label", f"**Me:** {text}" in content)
    else:
        area.check("daily-file-layout", False, reason="append returned no document")
    d = c.request("GET", "/api/day", user=user)
    area.check("day-view", d.status == 200 and text in ((d.json() or {}).get("today_log") or ""), http=d.status)
    m = c.request("GET", "/api/months", user=user)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    months = (m.json() or {}).get("months") or []
    area.check("month-listed", m.status == 200 and any(month in str(x) for x in months), http=m.status)
    for name, bad in (("heading-refused", {"text": "line one\n## Not a heading", "title": "x"}),
                      ("marker-refused", {"text": "<!-- xid:00000000-0000-4000-8000-000000000000 -->", "title": "x"}),
                      ("title-newline-refused", {"text": "ok", "title": "two\nlines"})):
        r = c.request("POST", "/api/entries/append", user=user,
                      json_body={**bad, "requestId": str(uuid.uuid4()), "entryTime": datetime.now(timezone.utc).isoformat()})
        area.check(name, r.status == 400, http=r.status)
    f = c.request("POST", "/api/file", user=user, json_body={"path": document or ""})
    area.check("refused-writes-left-nothing", "Not a heading" not in ((f.json() or {}).get("content") or ""))
    return area.finish({"tenant": user})


if __name__ == "__main__":
    sys.exit(main())
