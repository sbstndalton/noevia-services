#!/usr/bin/env bash
# Build the diary image from a pinned revision and start the isolated diarygate stack.
#
#   up.sh [--rev REV] [--build-root DIR] [--run-dir DIR]
#
# REV (default HEAD; any commit of this repo, e.g. the pre-release diary revision or one with the
# retrieval trace) is extracted with `git archive`, so uncommitted changes never reach the image.
# Refuses to start when diarygate volumes or containers already exist (a previous run not torn down).
# After start-up, isolation.py checks with docker inspect that both gate diaries have only diarygate
# volumes and the diarygate network, and that the no-model diary's inference URLs are disabled; on
# any problem the stack is taken down again. Touches no other compose project or container.
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
if ! python3 -I "$GATE/isolation.py" preflight; then
  echo "diarygate: volumes or containers from an earlier run exist; run down.sh first" >&2
  exit 1
fi
DEVRUN="$HOME/bin/devrun"; [ -x "$DEVRUN" ] || DEVRUN=""
sha=$(git -C "$GATE" rev-parse --verify "$REV^{commit}")
short=${sha:0:12}
ctx="$BUILD_ROOT/$sha"
if [ ! -f "$ctx/diary/Dockerfile" ]; then
  rm -rf "$ctx.tmp" && mkdir -p "$ctx.tmp"
  git -C "$(git -C "$GATE" rev-parse --show-toplevel)" archive "$sha" diary | tar -x -C "$ctx.tmp"
  rm -rf "$ctx" && mv "$ctx.tmp" "$ctx"
fi
mkdir -p "$BUILD_ROOT"
printf 'GATE_REV=%s\nGATE_IMAGE=diarygate-diary:%s\nGATE_BUILD_CONTEXT=%s\n' "$short" "$short" "$ctx" > "$BUILD_ROOT/last-up.env"
export GATE_BUILD_CONTEXT="$ctx/diary" GATE_REV="$short"
compose=(docker compose -p diarygate -f "$GATE/compose.yml")
echo "diarygate: building diary at $short"
$DEVRUN "${compose[@]}" build diary
"${compose[@]}" up -d --wait --wait-timeout 180 diary diary-nomodel
out=""; [ -n "$RUN_DIR" ] && out="$RUN_DIR/logs/isolation-up.json"
if ! python3 -I "$GATE/isolation.py" up ${out:+--out "$out"}; then
  echo "diarygate: isolation check FAILED; taking the stack down" >&2
  "${compose[@]}" down -v --remove-orphans
  exit 1
fi
[ -n "$RUN_DIR" ] && echo "$sha" > "$RUN_DIR/logs/diary-rev.txt"
echo "diarygate: up at http://127.0.0.1:${GATE_PORT:-18910} (no-model diary :${GATE_NOMODEL_PORT:-18911}; rev $short)"
