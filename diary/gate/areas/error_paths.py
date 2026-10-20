#!/usr/bin/env python3
"""Workflow area: error paths (no model). RUN ALONE: it deletes a tenant and probes size limits.

Every malformed request gets the documented refusal and nothing is written; none reaches inference
(all /v1 cases are refused before the exchange starts). The deleted tenant then answers 410.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gatelib as g  # noqa: E402

PARALLEL_SAFE = False


def main() -> int:
    args = g.area_args(__doc__, nomodel=True).parse_args()
    area = g.Area("error-paths", args.run_dir, PARALLEL_SAFE)
    env = g.load_env(args.env_file)
    label = args.tenant_label or "area-error-paths"
    user, doomed = g.tenant_id(label), g.tenant_id(label + "-deleted")
    c = g.Client(args.base_url, env, log=area.log_request)

    def expect(name, resp, statuses):
        area.check(name, resp.status in statuses, http=resp.status, expected=sorted(statuses))

    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    expect("month-out-of-range", c.request("GET", "/api/day", user=user, query="month=2030-13"), {400})
    expect("month-malformed", c.request("GET", "/api/day", user=user, query="month=March"), {400})
    expect("tenant-id-not-uuid", c.request("GET", "/api/months", user="not-a-uuid"), {400, 401})
    good = {"text": "fine", "title": "t", "requestId": str(uuid.uuid4()), "entryTime": now()}
    for name, patch in (("append-past-day", {"entryDay": "2030-01-06"}),
                        ("append-stale-time", {"entryTime": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()}),
                        ("append-naive-time", {"entryTime": datetime.now().replace(microsecond=0).isoformat()}),
                        ("append-empty-text", {"text": "   "}),
                        ("append-too-long", {"text": "x" * 8001}),
                        ("append-bad-request-id", {"requestId": "1234"}),
                        ("append-title-too-long", {"title": "t" * 81})):
        expect(name, c.request("POST", "/api/entries/append", user=user, json_body={**good, "requestId": str(uuid.uuid4()), **patch}), {400})
    expect("file-not-markdown", c.request("PUT", "/api/file", user=user, json_body={"path": "notes.txt", "content": "x", "version": None}), {400})
    expect("file-dot-dot", c.request("PUT", "/api/file", user=user, json_body={"path": "../escape.md", "content": "x", "version": None}), {400})
    expect("file-hidden", c.request("PUT", "/api/file", user=user, json_body={"path": ".hidden/x.md", "content": "x", "version": None}), {400})
    expect("file-no-version", c.request("PUT", "/api/file", user=user, json_body={"path": "Notes/a.md", "content": "x"}), {400})
    expect("file-too-large", c.request("PUT", "/api/file", user=user,
                                       json_body={"path": "Notes/big.md", "content": "y" * (512 * 1024 + 1), "version": None}), {400})
    first = c.request("PUT", "/api/file", user=user, json_body={"path": "Notes/once.md", "content": "one\n", "version": None})
    area.check("file-first-write", first.status == 200, http=first.status)
    expect("file-stale-version", c.request("PUT", "/api/file", user=user,
                                           json_body={"path": "Notes/once.md", "content": "two\n", "version": None}), {409})
    r = c.request("POST", "/api/file", user=user, json_body={"path": "Notes/once.md"})
    area.check("file-conflict-kept-original", (r.json() or {}).get("content") == "one\n", http=r.status)
    expect("edit-invalid-xid", c.request("POST", "/api/entries/edit", user=user, json_body={"xid": "zz", "me": "x"}), {400})
    expect("edit-unknown-xid", c.request("POST", "/api/entries/edit", user=user,
                                         json_body={"xid": str(uuid.uuid4()), "me": "x", "assistant": "y"}), {404})
    expect("edit-empty-me", c.request("POST", "/api/entries/edit", user=user,
                                      json_body={"xid": str(uuid.uuid4()), "me": "  "}), {400})
    expect("chat-invalid-json", c.request("POST", "/v1/chat/completions", user=user, raw_body=b"{not json",
                                          content_type="application/json"), {400})
    expect("chat-no-user-message", c.request("POST", "/v1/chat/completions", user=user,
                                             json_body={"messages": [{"role": "system", "content": "s"}]}), {400})
    expect("chat-blank-user-message", c.request("POST", "/v1/chat/completions", user=user,
                                                json_body={"messages": [{"role": "user", "content": "   "}]}), {400})
    expect("chat-naive-entry-time", c.request("POST", "/v1/chat/completions", user=user,
                                              json_body={"messages": [{"role": "user", "content": "hello"}],
                                                         "entryTime": "2030-03-06T12:00:00"}), {400})
    expect("chat-future-entry-day", c.request("POST", "/v1/chat/completions", user=user,
                                              json_body={"messages": [{"role": "user", "content": "hello"}],
                                                         "entryTime": "2030-03-06T12:00:00+00:00", "entryDay": "2030-03-07"}), {400})
    expect("storage-header-too-large", c.request("GET", "/api/months", user=user,
                                                 extra_headers={"X-Cowork-Storage": "A" * 5000}), {401, 413})
    expect("unknown-route", c.request("GET", "/api/does-not-exist", user=user), {404})
    expect("wrong-service-token", c.request("GET", "/api/health", extra_headers={"Authorization": "Bearer wrong"}), {401})
    m = c.request("GET", "/api/months", user=user)
    area.check("no-entries-written-by-refusals", m.status == 200 and not (m.json() or {}).get("months"), http=m.status,
               months=(m.json() or {}).get("months"))
    w = c.request("PUT", "/api/file", user=doomed, json_body={"path": "Notes/doomed.md", "content": "bye\n", "version": None})
    area.check("doomed-tenant-written", w.status == 200, http=w.status)
    expect("delete-tenant-unsigned", c.request("DELETE", "/api/internal/tenant", user=doomed, assertion="omit"), {401})
    expect("delete-tenant", c.request("DELETE", "/api/internal/tenant", user=doomed), {200})
    expect("deleted-tenant-gone", c.request("POST", "/api/file", user=doomed, json_body={"path": "Notes/doomed.md"}), {410})
    expect("other-tenant-unaffected", c.request("POST", "/api/file", user=user, json_body={"path": "Notes/once.md"}), {200})
    return area.finish({"tenants": [user, doomed]})


if __name__ == "__main__":
    sys.exit(main())
