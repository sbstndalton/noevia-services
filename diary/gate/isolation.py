#!/usr/bin/env python3
"""Prove the diarygate stack touches only its own volumes and network (up), or that nothing is left (down).

    python3 -I isolation.py up   [--out FILE]
    python3 -I isolation.py down [--out FILE]
    python3 -I isolation.py preflight          (refuse when diarygate volumes already exist)

Reads `docker inspect` / `docker volume ls` / `docker network ls` / `docker ps -a` only; changes nothing.
Secret VALUES are never read out: only key names are checked, plus the inference URL keys of the
no-model diary (which must point at the disabled name).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = "diarygate"
VOLUMES = {"diarygate_data", "diarygate_corpus", "diarygate_nomodel_data", "diarygate_nomodel_corpus"}
NETWORKS = {"diarygate_net"}
CONTAINERS = {"diarygate-diary", "diarygate-diary-nomodel"}
NOMODEL = "diarygate-diary-nomodel"
NOMODEL_URL_KEYS = ("LLM_BASE_URL", "LLM_AUX_BASE_URL", "LLM_EMBED_BASE_URL")
FORBIDDEN_ENV_PREFIXES = ("WEBDAV_", "S3_", "CLOUD", "DIARY_LOCAL_VOLUMES", "DIARY_LEGACY_USER_ID")


def docker(*args) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True, timeout=60).stdout


def lines(text: str) -> list:
    return [x for x in text.splitlines() if x.strip()]


def check_containers(inspected: list, project_volumes: list, project_networks: list) -> dict:
    """Pure check of `docker inspect` output (tests feed it fake data)."""
    problems, containers = [], []
    if {(i.get("Name") or "").lstrip("/") for i in inspected} != CONTAINERS:
        problems.append(f"containers {sorted((i.get('Name') or '').lstrip('/') for i in inspected)} != {sorted(CONTAINERS)}")
    for info in inspected:
        name = (info.get("Name") or "").lstrip("/")
        cfg, host = info.get("Config") or {}, info.get("HostConfig") or {}
        mounts = [{"type": m.get("Type"), "name": m.get("Name"), "destination": m.get("Destination")} for m in info.get("Mounts") or []]
        nets = sorted((info.get("NetworkSettings") or {}).get("Networks") or {})
        env = [e.split("=", 1) + [""] for e in cfg.get("Env") or []]
        env_keys = sorted(k for k, *_ in env)
        for m in mounts:
            if m["type"] != "volume" or m["name"] not in VOLUMES:
                problems.append(f"{name}: mount {m} is not a diarygate volume")
        for bind in host.get("Binds") or []:  # compose lists named volumes here too; a host path is a bind mount
            if bind.split(":", 1)[0] not in VOLUMES:
                problems.append(f"{name}: bind {bind.split(':', 1)[0]!r} is not a diarygate volume")
        if set(nets) != NETWORKS:
            problems.append(f"{name}: networks {nets} != {sorted(NETWORKS)}")
        for k in env_keys:
            if k.startswith(FORBIDDEN_ENV_PREFIXES):
                problems.append(f"{name}: forbidden environment key {k}")
        if name == NOMODEL:
            values = {k: v for k, v, *_ in env if k in NOMODEL_URL_KEYS}
            for k in NOMODEL_URL_KEYS:
                if "inference-disabled" not in values.get(k, ""):
                    problems.append(f"{name}: {k} is not the disabled inference name")
        for spec in (host.get("PortBindings") or {}).values():
            for b in spec or []:
                if b.get("HostIp") != "127.0.0.1":
                    problems.append(f"{name}: port published on {b.get('HostIp') or '0.0.0.0'}")
        if not str(cfg.get("Image") or "").startswith("diarygate-diary:"):
            problems.append(f"{name}: image {cfg.get('Image')!r} is not a diarygate image")
        containers.append({"name": name, "image": cfg.get("Image"), "mounts": mounts, "networks": nets, "env_keys": env_keys})
    if set(project_volumes) - VOLUMES:
        problems.append(f"unexpected project volumes {sorted(set(project_volumes) - VOLUMES)}")
    if set(project_networks) - NETWORKS:
        problems.append(f"unexpected project networks {sorted(set(project_networks) - NETWORKS)}")
    return {"mode": "up", "ok": not problems, "problems": problems, "containers": containers,
            "project_volumes": sorted(project_volumes), "project_networks": sorted(project_networks)}


def check_up() -> dict:
    ids = lines(docker("ps", "-a", "-q", "--filter", f"label=com.docker.compose.project={PROJECT}"))
    inspected = json.loads(docker("inspect", *ids)) if ids else []
    vols = lines(docker("volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={PROJECT}"))
    nets = lines(docker("network", "ls", "--format", "{{.Name}}", "--filter", f"label=com.docker.compose.project={PROJECT}"))
    return check_containers(inspected, vols, nets)


def leftovers() -> dict:
    vols = lines(docker("volume", "ls", "--format", "{{.Name}}", "--filter", "name=diarygate"))
    nets = lines(docker("network", "ls", "--format", "{{.Name}}", "--filter", "name=diarygate"))
    ctrs = lines(docker("ps", "-a", "--format", "{{.Names}}", "--filter", "name=diarygate"))
    labelled = lines(docker("ps", "-a", "-q", "--filter", f"label=com.docker.compose.project={PROJECT}"))
    return {"volumes": vols, "networks": nets, "containers": sorted(set(ctrs) | set(labelled))}


def check_down() -> dict:
    left = leftovers()
    return {"mode": "down", "ok": not any(left.values()), "left": left,
            "docker_volume_ls": lines(docker("volume", "ls", "--format", "{{.Name}}")),
            "docker_network_ls": lines(docker("network", "ls", "--format", "{{.Name}}"))}


def check_preflight() -> dict:
    left = leftovers()
    return {"mode": "preflight", "ok": not left["volumes"] and not left["containers"], "left": left}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("up", "down", "preflight"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = {"up": check_up, "down": check_down, "preflight": check_preflight}[args.mode]()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"mode": result["mode"], "ok": result["ok"],
                      **({"problems": result["problems"]} if args.mode == "up" else {"left": result["left"]})}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
