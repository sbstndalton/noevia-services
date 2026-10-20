#!/usr/bin/env bash
# Smoke proof of the gate tooling with NO inference:
#   1. generate the fixture twice and compare every byte and the tree hash;
#   2. bring up the diarygate stack (inference URLs point at an unresolvable name) and prove isolation;
#   3. load the fixture through the API, read every day back, apply the scripted mutations, snapshot
#      the baseline, contaminate it the way a logged exchange would and restore it byte for byte;
#   4. run the no-model areas: capture and tenancy (parallel), then error-paths and restart (alone);
#   5. score, tear down, and prove with docker volume ls / network ls that nothing is left.
# Production containers are only read (docker inspect StartedAt before and after), never touched.
#
#   smoke.sh EVIDENCE_ROOT        (a directory outside git)
set -euo pipefail
GATE=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:?usage: smoke.sh EVIDENCE_ROOT}
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)
PY="python3 -I"
rev=$(git -C "$GATE" rev-parse HEAD)
fixA="$ROOT/fixture-$rev-a"; fixB="$ROOT/fixture-$rev-b"
rm -rf "$fixA" "$fixB"
$PY "$GATE/gen_fixture.py" --seed 20261020 --anchor 2030-01-06 --days 60 --tz UTC --out "$fixA" > "$ROOT/gen-a.json"
$PY "$GATE/gen_fixture.py" --seed 20261020 --anchor 2030-01-06 --days 60 --tz UTC --out "$fixB" > "$ROOT/gen-b.json"
"$GATE/make_env.sh" --no-inference --force
run=$($PY "$GATE/harness.py" init --evidence-root "$ROOT" --rev "$rev" --phase pre --fixture "$fixA")
{
  echo "a: $(cat "$ROOT/gen-a.json")"
  echo "b: $(cat "$ROOT/gen-b.json")"
  if diff -r "$fixA" "$fixB" > /dev/null && cmp -s "$ROOT/gen-a.json" "$ROOT/gen-b.json"; then echo "IDENTICAL"; else echo "DIFFERENT"; fi
} > "$run/logs/fixture-determinism.txt"
grep -qx IDENTICAL "$run/logs/fixture-determinism.txt" || { echo "fixture is not deterministic" >&2; exit 1; }
prod() { docker ps --filter label=com.docker.compose.project=noevia-inference --format '{{.Names}}' | sort | while read -r n; do
  echo "$n $(docker inspect "$n" --format '{{.State.StartedAt}} restarts={{.RestartCount}}')"; done; }
prod > "$run/logs/production-before.txt"
cleanup() { "$GATE/down.sh" --run-dir "$run" || true; prod > "$run/logs/production-after.txt";
  if cmp -s "$run/logs/production-before.txt" "$run/logs/production-after.txt"; then echo "production untouched" >> "$run/logs/production-after.txt"; fi; }
trap cleanup EXIT
"$GATE/up.sh" --rev "$rev" --run-dir "$run"
$PY "$GATE/harness.py" load --run-dir "$run" --fixture "$fixA"
$PY "$GATE/harness.py" mutate --run-dir "$run" --fixture "$fixA"
$PY "$GATE/harness.py" restore-selftest --run-dir "$run" --fixture "$fixA"
$PY "$GATE/areas/capture.py" --run-dir "$run" & p1=$!
$PY "$GATE/areas/tenancy.py" --run-dir "$run" & p2=$!
rc=0; wait $p1 || rc=1; wait $p2 || rc=1
$PY "$GATE/areas/error_paths.py" --run-dir "$run" || rc=1
$PY "$GATE/areas/restart.py" --run-dir "$run" || rc=1
for a in semantic temporal corrections follow_ups confirmed_vs_inferred budgets; do
  $PY "$GATE/areas/$a.py" --run-dir "$run" --fixture "$fixA" > /dev/null || true
done
$PY "$GATE/harness.py" queries --run-dir "$run" --fixture "$fixA" || true
$PY "$GATE/score.py" --run-dir "$run" --fixture "$fixA"
echo "smoke evidence: $run"
exit $rc
