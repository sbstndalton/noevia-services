#!/usr/bin/env bash
# Copy the diarygate logs out, `down -v --remove-orphans`, remove this run's own diarygate image (by
# exact tag) and its build context, then prove nothing is left.
#
#   down.sh [--run-dir DIR] [--keep-image]
#
# Logs go to DIR/logs/ (default ./diarygate-logs-<utc>); the teardown proof (docker volume ls and
# docker network ls after the down) is DIR/logs/isolation-down.json. Removes only the diarygate
# project's containers, volumes, network, the image tag up.sh built and its git-archive context.
set -euo pipefail
GATE=$(cd "$(dirname "$0")" && pwd)
BUILD_ROOT="${GATE_BUILD_ROOT:-$HOME/noevia-work/diarygate-build}"
RUN_DIR=""; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --keep-image) KEEP=1; shift ;;
    *) echo "usage: $0 [--run-dir DIR] [--keep-image]" >&2; exit 2 ;;
  esac
done
LOGS="${RUN_DIR:-$PWD/diarygate-logs-$(date -u +%Y%m%dT%H%M%SZ)}/logs"
mkdir -p "$LOGS"
image=""; ctx=""
if [ -f "$BUILD_ROOT/last-up.env" ]; then
  image=$(sed -n 's/^GATE_IMAGE=//p' "$BUILD_ROOT/last-up.env")
  ctx=$(sed -n 's/^GATE_BUILD_CONTEXT=//p' "$BUILD_ROOT/last-up.env")
fi
export GATE_BUILD_CONTEXT="${GATE_BUILD_CONTEXT:-/nonexistent}" GATE_REV="${GATE_REV:-down}"
compose=(docker compose -p diarygate -f "$GATE/compose.yml")
"${compose[@]}" ps -a > "$LOGS/compose-ps.txt" 2>&1 || true
"${compose[@]}" logs --no-color --timestamps > "$LOGS/diary.log" 2>&1 || true
for c in diarygate-diary diarygate-diary-nomodel; do
  docker inspect "$c" --format "$c {{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}}" >> "$LOGS/diary-state.txt" 2>&1 || true
done
"${compose[@]}" down -v --remove-orphans
if [ "$KEEP" != 1 ]; then
  case "$image" in
    diarygate-diary:*) docker image rm "$image" > "$LOGS/image-rm.txt" 2>&1 || echo "image $image not removed" >> "$LOGS/image-rm.txt" ;;
  esac
  case "$ctx" in
    "$BUILD_ROOT"/*) rm -rf "$ctx" ;;
  esac
  rm -f "$BUILD_ROOT/last-up.env"
fi
python3 -I "$GATE/isolation.py" down --out "$LOGS/isolation-down.json"
