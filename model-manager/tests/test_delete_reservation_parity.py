"""noevia#1987: the paths a model delete reserves against new downloads (noevia#1972), and the 409
wording on either side of a delete, asserted against the cases noevia-rs asserts too.

`tests/fixtures/delete-reservation-cases.json` is shared with sbstndalton/noevia-rs
(`bins/noevia-model-manager/tests/fixtures/delete-reservation-cases.json`, the canonical copy; its
Rust test is `model_delete::tests::delete_reservation_matches_the_shared_cases`). The sections oracle
only re-hashes api.py, so this fixture is what keeps the two sides' rules from drifting apart.
Every file is a synthetic empty file in a temporary models folder; no download starts."""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from app import main, services
from app.config import settings
from app.downloader import DELETE_IN_PROGRESS, DeleteInProgress, manager

FIXTURE = Path(__file__).parent / "fixtures" / "delete-reservation-cases.json"
# sha256 of the shared fixture. Refresh BOTH copies together: change the canonical file in noevia-rs,
# copy it byte-for-byte here, and update this constant and the Rust test in the same pair of PRs.
FIXTURE_SHA256 = "4f75883a80832383ad0866942f5668e962cceb2bfd82504807114659c8469ddc"

CASES = json.loads(FIXTURE.read_text())


def test_the_shared_fixture_is_the_pinned_copy():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256, (
        "delete-reservation-cases.json changed: refresh the noevia-rs copy and this one together")
    assert CASES["format"] == "delete-reservation-cases.v1" and len(CASES["cases"]) >= 10


@pytest.fixture()
def models(monkeypatch, tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(settings, "models_dir", root)
    monkeypatch.setattr(settings, "models_ini_path", root / "models.ini")
    monkeypatch.setattr(settings, "model_download_targets", "")
    monkeypatch.setattr(manager, "_reserved", [])
    return root


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["name"])
def test_delete_reservation_matches_the_shared_cases(models, case):
    for rel in case["files"]:
        path = models / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(8)
    listed = services.snapshot_models_dir().ggufs
    for expected in case["entries"]:
        match = [g for g in listed if g.display_name == expected["base"] and g.subdir == expected["subdir"]]
        assert len(match) == 1, f"no listed model {expected['subdir']}/{expected['base']}"
        reserved = services.delete_reservation(match[0])
        assert reserved == expected["reserved"]
        # While reserved, a start into each reserved path (a folder: anything inside it) is refused
        # as DeleteInProgress, whose 409 is the shared wording; nothing is queued.
        with manager.reserve(reserved):
            for path in reserved:
                dest = path if path.endswith(".gguf") else f"{path}/blocked.gguf"
                with pytest.raises(DeleteInProgress), manager.starting([dest]):
                    pass
    assert manager._reserved == []
    assert [j for j in manager.jobs.values() if j.status in ("queued", "downloading")] == []


def test_delete_409_wording_matches_the_shared_cases(models):
    # A download blocked by a delete: the route's exception handler answers 409 with this detail.
    blocked = CASES["downloadBlockedByDelete"]
    assert DELETE_IN_PROGRESS == blocked["detail"]
    answer = asyncio.run(main._delete_in_progress(None, DeleteInProgress()))
    assert answer.status_code == blocked["status"] == 409
    assert json.loads(answer.body) == {"detail": blocked["detail"]}
    # A delete blocked by a download.
    for case in CASES["deleteBlockedByDownload"]:
        with open(models / case["base"], "wb") as f:
            f.truncate(8)
        match = next(g for g in services.snapshot_models_dir().ggufs if g.display_name == case["base"])
        assert services.downloading_message(match, case["rels"]) == case["detail"], case["name"]
