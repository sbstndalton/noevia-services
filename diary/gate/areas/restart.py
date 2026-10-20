#!/usr/bin/env python3
"""Workflow area: restart (no model). RUN ALONE: it restarts the gate diary container.

Restarts ONLY the container named diarygate-diary-nomodel, after checking it belongs to compose project
diarygate. Writes survive the restart byte for byte, the journal is empty afterwards, and an
assertion signed before the restart cannot be replayed after it.
"""
from __future__ import annotations

import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gatelib as g  # noqa: E402

PARALLEL_SAFE = False
# The no-model diary: restarting it can never trigger embedding work.
CONTAINER = "diarygate-diary-nomodel"


def docker(*args) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True, timeout=120).stdout.strip()


def main() -> int:
    args = g.area_args(__doc__, nomodel=True).parse_args()
    area = g.Area("restart", args.run_dir, PARALLEL_SAFE)
    env = g.load_env(args.env_file)
    user = g.tenant_id(args.tenant_label or "area-restart")
    c = g.Client(args.base_url, env, log=area.log_request, nomodel=True)
    try:
        project = docker("inspect", CONTAINER, "--format", '{{index .Config.Labels "com.docker.compose.project"}}')
    except (subprocess.SubprocessError, OSError) as exc:
        area.mark("container", "blocked", f"cannot inspect {CONTAINER}: {type(exc).__name__}")
        return area.finish()
    if project != "diarygate":
        area.mark("container", "blocked", f"{CONTAINER} is not in project diarygate; refusing to restart it")
        return area.finish()
    content = "## Restart probe\n\nWritten before the restart; must survive it unchanged.\n"
    w = c.request("PUT", "/api/file", user=user, json_body={"path": "Notes/restart-probe.md", "content": content, "version": None})
    area.check("write-before", w.status == 200, http=w.status)
    rid = str(uuid.uuid4())
    a = c.request("POST", "/api/entries/append", user=user,
                  json_body={"text": "Synthetic note written before a restart.", "title": "Restart probe", "requestId": rid,
                             "entryTime": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    area.check("append-before", a.status == 200 and (a.json() or {}).get("queued") is False, http=a.status)
    old = g.sign(c.key, user, "GET", "/api/months") if c.key else None
    started_before = docker("inspect", CONTAINER, "--format", "{{.State.StartedAt}}")
    time.sleep(2)  # the replay guard has one-second resolution
    docker("restart", "--time", "20", CONTAINER)
    deadline, health = time.monotonic() + 180, ""
    while time.monotonic() < deadline:
        health = docker("inspect", CONTAINER, "--format", "{{.State.Health.Status}}")
        if health == "healthy":
            break
        time.sleep(2)
    started_after = docker("inspect", CONTAINER, "--format", "{{.State.StartedAt}}")
    area.check("restarted-healthy", health == "healthy" and started_after != started_before, health=health)
    r = c.request("POST", "/api/file", user=user, json_body={"path": "Notes/restart-probe.md"})
    area.check("file-survives", r.status == 200 and (r.json() or {}).get("content") == content, http=r.status)
    d = c.request("GET", "/api/day", user=user)
    area.check("append-survives", d.status == 200 and f"xid:{rid}" in ((d.json() or {}).get("today_log") or ""), http=d.status)
    h = c.request("GET", "/api/health", user=user, storage=g.LOCAL_STORAGE)
    hj = h.json() or {}
    area.check("journal-empty", h.status == 200 and hj.get("journal_pending") == 0 and hj.get("journal_quarantined") == 0,
               journal_pending=hj.get("journal_pending"), journal_quarantined=hj.get("journal_quarantined"))
    if old:
        r = c.request("GET", "/api/months", user=user, assertion=old)
        area.check("pre-restart-assertion-refused", r.status == 401, http=r.status)
    else:
        area.mark("pre-restart-assertion-refused", "untested", "no tenant key")
    return area.finish({"tenant": user})


if __name__ == "__main__":
    sys.exit(main())
