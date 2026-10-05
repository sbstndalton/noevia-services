"""The model manager serves JSON under /api/v1 only (#806).

Its server-rendered pages were replaced by noevia's own web screens and then removed. These
tests keep them removed and keep the helpers the API took from the old page module.
"""
import ast
import importlib
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.main import app

APP_DIR = Path(__file__).resolve().parents[1] / "app"

# Page routes that used to exist, with the method each answered.
REMOVED_PAGES = [
    ("GET", "/"), ("GET", "/models"), ("POST", "/models/delete"), ("POST", "/models/delete-bulk"),
    ("POST", "/models/check-updates"), ("GET", "/model/tiny-Q4_K_M.gguf"), ("GET", "/settings"),
    ("GET", "/search"), ("GET", "/search/results"), ("GET", "/search/repo/acme/x-GGUF"),
    ("POST", "/download"), ("POST", "/download/multi"), ("POST", "/download/url"),
    ("GET", "/downloads"), ("GET", "/downloads/rows"), ("POST", "/downloads/clear"),
    ("GET", "/config"), ("GET", "/config/section/tiny/edit"), ("POST", "/config/section/tiny"),
    ("GET", "/config/section/tiny/autoconfig"), ("POST", "/config/section/tiny/delete"),
    ("GET", "/containers"), ("POST", "/containers/sync-openwebui"), ("GET", "/containers/x/logs"),
    ("POST", "/containers/x/restart"), ("POST", "/models/openwebui-visibility"),
    ("GET", "/prompts"), ("POST", "/prompts"), ("GET", "/benchmark"), ("POST", "/benchmark/start"),
    ("GET", "/benchmark/run/1"), ("POST", "/badge"), ("POST", "/badge/clear"),
    ("GET", "/palette.json"), ("GET", "/_vendor/htmx.min.js"),
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_every_route_is_json_api_or_a_fastapi_builtin():
    paths = {r.path for r in app.routes}
    extra = {p for p in paths if not p.startswith("/api/v1") and p not in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}}
    assert not extra, f"non-API routes are back: {sorted(extra)}"
    assert any(isinstance(r, APIRoute) and r.path == "/api/v1/health" for r in app.routes)


@pytest.mark.parametrize("method,path", REMOVED_PAGES)
def test_removed_pages_are_gone(client, method, path):
    r = client.request(method, path)
    assert r.status_code in (404, 405), f"{method} {path} answered {r.status_code}"
    assert "text/html" not in r.headers.get("content-type", "")


def test_no_template_directory_or_template_engine_is_left():
    assert not (APP_DIR / "templates").exists()
    for f in APP_DIR.glob("*.py"):
        text = f.read_text()
        assert "import jinja2" not in text and "fastapi.templating" not in text, f.name
        assert "TemplateResponse" not in text and "HTMLResponse" not in text, f.name


def test_every_helper_the_api_imports_exists():
    """api.py imports these inside the handlers, so a missing one would only fail at request time."""
    helpers = importlib.import_module("app.helpers")
    wanted: set[str] = set()
    for node in ast.walk(ast.parse((APP_DIR / "api.py").read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "helpers":
            wanted.update(a.name for a in node.names)
        assert not (isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "main"), "api.py must not import from main"
    assert len(wanted) >= 10, wanted
    missing = sorted(n for n in wanted if not callable(getattr(helpers, n, None)))
    assert not missing, missing
