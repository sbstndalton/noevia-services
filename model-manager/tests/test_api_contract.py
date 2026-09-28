"""The /api/v1 routes the app really serves must match docs/spec-model-loader-api-v1.md (#269).

Adding or removing a route without updating the contract doc fails here. The web side of the
same contract is checked by apps/web/server/model-loader-contract.test.cjs.
"""
import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

DOC = Path(__file__).resolve().parents[3] / "docs" / "spec-model-loader-api-v1.md"


def _norm(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path)


def _documented() -> set[str]:
    block = re.search(r"<!-- route-table:start -->(.*?)<!-- route-table:end -->", DOC.read_text(), re.S)
    assert block, "route table markers missing from the contract doc"
    rows = set()
    for line in block.group(1).splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 5 and cells[1] in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
            rows.add(f"{cells[1]} {_norm(cells[2])}")
    return rows


def _served() -> set[str]:
    out = set()
    for r in app.routes:
        if isinstance(r, APIRoute) and r.path.startswith("/api/v1"):
            out.update(f"{m} {_norm(r.path)}" for m in r.methods if m not in {"HEAD", "OPTIONS"})
    return out


def test_doc_lists_every_served_route_and_nothing_else():
    served, doc = _served(), _documented()
    assert len(served) > 30
    assert not served - doc, f"served but undocumented: {sorted(served - doc)}"
    assert not doc - served, f"documented but not served: {sorted(doc - served)}"


def test_only_health_is_unauthenticated(monkeypatch):
    monkeypatch.setattr(settings, "model_loader_token", "fixture-token")
    with TestClient(app) as c:
        assert c.get("/api/v1/health").status_code == 200
        for path in ("/api/v1/models", "/api/v1/sections"):
            r = c.get(path)
            assert r.status_code == 401 and r.json() == {"error": "unauthorized"}
        assert c.put("/api/v1/models-ini", json={"baseRevision": "x", "text": "a=1"}).status_code == 401
        assert c.get("/api/v1/sections", headers={"X-Model-Loader-Token": "fixture-token"}).status_code == 200
