#!/usr/bin/env bash
# Build the diary image from a pinned revision and start the isolated diarygate stack.
#
#   up.sh [--rev REV] [--build-root DIR] [--run-dir DIR]
#
# REV (default HEAD) is extracted with `git archive`, so uncommitted changes never reach the image.
# After start-up, isolation.py checks with docker inspect that the stack has only diarygate volumes
# and the diarygate network; any other mount, network, published address or storage setting fails
# the run and the stack is taken down again. Touches no other compose project or container.
set -euo pipefail
GATE=$(cd "$(dirname "$0")" && pwd)
REV=HEAD
BUILD_ROOT="${GATE_BUILD_ROOT:-$HOME/noevia-work/diarygate-build}"
RUN_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rev) REV="$2"; shift 2 ;;
    --build-root) BUILD_ROOT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    *) echo "usage: $0 [--rev REV] [--build-root DIR] [--run-dir DIR]" >&2; exit 2 ;;
  esac
done
[ -f "$GATE/.env.gate" ] || { echo "missing $GATE/.env.gate; run make_env.sh first" >&2; exit 1; }
DEVRUN="$HOME/bin/devrun"; [ -x "$DEVRUN" ] || DEVRUN=""
sha=$(git -C "$GATE" rev-parse --verify "$REV^{commit}")
short=${sha:0:12}
ctx="$BUILD_ROOT/$sha"
if [ ! -f "$ctx/diary/Dockerfile" ]; then
  rm -rf "$ctx.tmp" && mkdir -p "$ctx.tmp"
  git -C "$(git -C "$GATE" rev-parse --show-toplevel)" archive "$sha" diary | tar -x -C "$ctx.tmp"
  rm -rf "$ctx" && mv "$ctx.tmp" "$ctx"
fi
export GATE_BUILD_CONTEXT="$ctx/diary" GATE_REV="$short"
compose=(docker compose -p diarygate -f "$GATE/compose.yml")
echo "diarygate: building diary at $short"
$DEVRUN "${compose[@]}" build diary
"${compose[@]}" up -d --wait --wait-timeout 180 diary
out=""; [ -n "$RUN_DIR" ] && out="$RUN_DIR/logs/isolation-up.json"
if ! python3 -I "$GATE/isolation.py" up ${out:+--out "$out"}; then
  echo "diarygate: isolation check FAILED; taking the stack down" >&2
  "${compose[@]}" down -v --remove-orphans
  exit 1
fi
[ -n "$RUN_DIR" ] && echo "$sha" > "$RUN_DIR/logs/diary-rev.txt"
echo "diarygate: up at http://127.0.0.1:${GATE_PORT:-18910} (rev $short)"
