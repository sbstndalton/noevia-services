#!/usr/bin/env python3
"""Prove the diarygate stack touches only its own volumes and network (up), or that nothing is left (down).

    python3 -I isolation.py up   [--out FILE]
    python3 -I isolation.py down [--out FILE]

Reads `docker inspect` / `docker volume ls` / `docker network ls` / `docker ps -a` only; changes nothing.
Environment variable VALUES are never read out: only key names are checked.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = "diarygate"
VOLUMES = {"diarygate_data", "diarygate_corpus"}
NETWORKS = {"diarygate_net"}
FORBIDDEN_ENV_PREFIXES = ("WEBDAV_", "S3_", "CLOUD", "DIARY_LOCAL_VOLUMES", "DIARY_LEGACY_USER_ID")


def docker(*args) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True, timeout=60).stdout


def lines(text: str) -> list:
    return [x for x in text.splitlines() if x.strip()]


def check_up() -> dict:
    problems = []
    ids = lines(docker("ps", "-a", "-q", "--filter", f"label=com.docker.compose.project={PROJECT}"))
    if not ids:
        problems.append("no diarygate containers")
    containers = []
    for info in json.loads(docker("inspect", *ids)) if ids else []:
        name = info["Name"].lstrip("/")
        mounts = [{"type": m.get("Type"), "name": m.get("Name"), "destination": m.get("Destination")} for m in info.get("Mounts", [])]
        nets = sorted((info.get("NetworkSettings") or {}).get("Networks") or {})
        env_keys = sorted(e.split("=", 1)[0] for e in (info.get("Config") or {}).get("Env") or [])
        binds = (info.get("HostConfig") or {}).get("Binds") or []
        ports = (info.get("HostConfig") or {}).get("PortBindings") or {}
        for m in mounts:
            if m["type"] != "volume" or m["name"] not in VOLUMES:
                problems.append(f"{name}: mount {m} is not a diarygate volume")
        for bind in binds:  # compose lists named volumes here too ("name:/dest:rw"); a host path is a bind mount
            if bind.split(":", 1)[0] not in VOLUMES:
                problems.append(f"{name}: bind {bind.split(':', 1)[0]!r} is not a diarygate volume")
        if set(nets) != NETWORKS:
            problems.append(f"{name}: networks {nets} != {sorted(NETWORKS)}")
        for k in env_keys:
            if k.startswith(FORBIDDEN_ENV_PREFIXES):
                problems.append(f"{name}: forbidden environment key {k}")
        for spec in ports.values():
            for b in spec or []:
                if b.get("HostIp") not in ("127.0.0.1",):
                    problems.append(f"{name}: port published on {b.get('HostIp') or '0.0.0.0'}")
        containers.append({"name": name, "image": (info.get("Config") or {}).get("Image"), "mounts": mounts,
                           "networks": nets, "env_keys": env_keys, "ports": ports})
    vols = lines(docker("volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={PROJECT}"))
    nets = lines(docker("network", "ls", "--format", "{{.Name}}", "--filter", f"label=com.docker.compose.project={PROJECT}"))
    if set(vols) - VOLUMES:
        problems.append(f"unexpected project volumes {sorted(set(vols) - VOLUMES)}")
    if set(nets) - NETWORKS:
        problems.append(f"unexpected project networks {sorted(set(nets) - NETWORKS)}")
    return {"mode": "up", "ok": not problems, "problems": problems, "containers": containers,
            "project_volumes": sorted(vols), "project_networks": sorted(nets)}


def check_down() -> dict:
    vols = lines(docker("volume", "ls", "--format", "{{.Name}}", "--filter", "name=diarygate"))
    nets = lines(docker("network", "ls", "--format", "{{.Name}}", "--filter", "name=diarygate"))
    ctrs = lines(docker("ps", "-a", "--format", "{{.Names}}", "--filter", "name=diarygate"))
    labelled = lines(docker("ps", "-a", "-q", "--filter", f"label=com.docker.compose.project={PROJECT}"))
    left = {"volumes": vols, "networks": nets, "containers": sorted(set(ctrs) | set(labelled))}
    return {"mode": "down", "ok": not any(left.values()), "left": left,
            "docker_volume_ls": lines(docker("volume", "ls", "--format", "{{.Name}}")),
            "docker_network_ls": lines(docker("network", "ls", "--format", "{{.Name}}"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("up", "down"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = check_up() if args.mode == "up" else check_down()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(json.dumps({"mode": result["mode"], "ok": result["ok"], **({"problems": result["problems"]} if args.mode == "up" else {"left": result["left"]})}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
