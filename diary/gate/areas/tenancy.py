#!/usr/bin/env python3
"""Workflow area: tenancy (no model). Parallel-safe: its own two tenants.

Tenant A's writes are invisible to tenant B, and every way of acting as a tenant without a valid
signed assertion for exactly this request is refused: missing, cross-tenant, replayed, stale, wrong
key, wrong path, tampered query, tampered body, tampered storage header, and no service token.
"""
from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gatelib as g  # noqa: E402

PARALLEL_SAFE = True
PROBE = "Notes/tenancy-probe.md"


def main() -> int:
    args = g.area_args(__doc__).parse_args()
    area = g.Area("tenancy", args.run_dir, PARALLEL_SAFE)
    env = g.load_env(args.env_file)
    label = args.tenant_label or "area-tenancy"
    a, b = g.tenant_id(label + "-a"), g.tenant_id(label + "-b")
    c = g.Client(args.base_url, env, log=area.log_request)
    if not c.key:
        area.mark("tenant-key", "blocked", "DIARY_TENANT_KEY is empty in .env.gate")
        return area.finish()
    secret_text = "## Tenancy probe\n\nOnly tenant A may read this synthetic line.\n"
    r = c.request("PUT", "/api/file", user=a, json_body={"path": PROBE, "content": secret_text, "version": None})
    area.check("a-writes", r.status == 200, http=r.status)
    r = c.request("POST", "/api/file", user=a, json_body={"path": PROBE})
    area.check("a-reads-own", r.status == 200 and (r.json() or {}).get("content") == secret_text, http=r.status)
    r = c.request("POST", "/api/file", user=b, json_body={"path": PROBE})
    area.check("b-cannot-see-a", r.status in (200, 404) and (r.json() or {}).get("content") is None, http=r.status)
    r = c.request("GET", "/api/files", user=b)
    area.check("b-listing-has-no-a-file", r.status == 200 and "tenancy-probe" not in json.dumps(r.json()), http=r.status)

    def refused(name, resp):
        area.check(name, resp.status == 401, http=resp.status)

    refused("missing-assertion", c.request("GET", "/api/months", user=a, assertion="omit"))
    refused("cross-tenant-signature", c.request("GET", "/api/months", user=a, sign_as=b))
    once = g.sign(c.key, a, "GET", "/api/months")
    first = c.request("GET", "/api/months", user=a, assertion=once)
    area.check("fresh-assertion-accepted", first.status == 200, http=first.status)
    refused("replayed-assertion", c.request("GET", "/api/months", user=a, assertion=once))
    refused("stale-assertion", c.request("GET", "/api/months", user=a, assertion=g.sign(c.key, a, "GET", "/api/months", ts=int(time.time()) - 600)))
    refused("future-assertion", c.request("GET", "/api/months", user=a, assertion=g.sign(c.key, a, "GET", "/api/months", ts=int(time.time()) + 600)))
    refused("wrong-key", c.request("GET", "/api/months", user=a, assertion=g.sign(secrets.token_hex(32), a, "GET", "/api/months")))
    refused("wrong-path", c.request("GET", "/api/months", user=a, assertion=g.sign(c.key, a, "GET", "/api/day")))
    refused("wrong-method", c.request("GET", "/api/months", user=a, assertion=g.sign(c.key, a, "POST", "/api/months")))
    refused("tampered-query", c.request("GET", "/api/day", user=a, query="month=2030-02",
                                        assertion=g.sign(c.key, a, "GET", "/api/day", query=b"month=2030-01")))
    signed_body = json.dumps({"path": PROBE}).encode()
    other_body = json.dumps({"path": "Notes/other.md"}).encode()
    refused("tampered-body", c.request("POST", "/api/file", user=a, raw_body=other_body, content_type="application/json",
                                       assertion=g.sign(c.key, a, "POST", "/api/file", body=signed_body)))
    refused("tampered-storage-header", c.request("GET", "/api/months", user=a,
                                                 extra_headers={"X-Cowork-Storage": "eyJraW5kIjoibG9jYWwifQ"}))
    refused("json-signed-as-stream", c.request("POST", "/api/file", user=a, raw_body=signed_body, content_type="application/json",
                                               assertion=g.sign(c.key, a, "POST", "/api/file", body_hash="stream")))
    r = c.request("GET", "/api/months", user=a, bearer=False)
    area.check("no-service-token", r.status == 401, http=r.status)
    r = c.request("GET", "/api/months", user=a, extra_headers={"Authorization": "Bearer " + secrets.token_hex(16)})
    area.check("wrong-service-token", r.status == 401, http=r.status)
    r = c.request("GET", "/api/health")
    area.check("process-health-without-tenant", r.status == 200, http=r.status)
    r = c.request("POST", "/api/file", user=a, json_body={"path": PROBE})
    area.check("a-still-intact", r.status == 200 and (r.json() or {}).get("content") == secret_text, http=r.status)
    return area.finish({"tenants": [a, b]})


if __name__ == "__main__":
    sys.exit(main())
