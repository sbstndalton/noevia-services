#!/usr/bin/env bash
# Copy the diarygate logs out, then `down -v --remove-orphans`, then prove nothing is left.
#
#   down.sh [--run-dir DIR]
#
# Logs go to DIR/logs/ (default: ./diarygate-logs-<utc>). The teardown proof (docker volume ls and
# docker network ls after the down) is written to DIR/logs/isolation-down.json. Removes only the
# diarygate project's own containers, volumes and network.
set -euo pipefail
GATE=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    *) echo "usage: $0 [--run-dir DIR]" >&2; exit 2 ;;
  esac
done
LOGS="${RUN_DIR:-$PWD/diarygate-logs-$(date -u +%Y%m%dT%H%M%SZ)}/logs"
mkdir -p "$LOGS"
# compose interpolation needs these set even for logs/down; the values do not matter here.
export GATE_BUILD_CONTEXT="${GATE_BUILD_CONTEXT:-/nonexistent}" GATE_REV="${GATE_REV:-down}"
compose=(docker compose -p diarygate -f "$GATE/compose.yml")
"${compose[@]}" ps -a > "$LOGS/compose-ps.txt" 2>&1 || true
"${compose[@]}" logs --no-color --timestamps > "$LOGS/diary.log" 2>&1 || true
docker inspect diarygate-diary --format '{{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}}' > "$LOGS/diary-state.txt" 2>&1 || true
"${compose[@]}" down -v --remove-orphans
python3 -I "$GATE/isolation.py" down --out "$LOGS/isolation-down.json"
