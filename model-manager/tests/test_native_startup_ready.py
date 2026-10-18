"""Actual child lifespan/middleware, with synthetic local storage and no live services."""
import sqlite3
import pytest
from fastapi.testclient import TestClient
from app import db, hw, model_files
from app.config import settings
from app.main import app

PATH = "/api/v1/internal/startup-ready"
TOKEN = "synthetic-startup-ready-token"
AUTH = {"x-model-loader-token": TOKEN}

@pytest.fixture()
def startup(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "migrate_cache_ram_on_start", False)
    monkeypatch.setattr(db, "_DB_PATH", None)
    monkeypatch.setattr(app.state, "model_manager_startup_ready", False)
    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED", "1")
    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_RUNNER_CONFIRMED", "1")
    def binary():
        assert not app.state.model_manager_startup_ready
    def sampler():
        assert not app.state.model_manager_startup_ready
        with sqlite3.connect(db._path()) as c:
            assert c.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='req_timing'").fetchone()[0] == 1
    monkeypatch.setattr(model_files, "require_binary", binary)
    monkeypatch.setattr(hw, "start_sampler", sampler)


def test_readiness_follows_actual_startup_and_keeps_public_health_refused(startup):
    client = TestClient(app)
    assert client.get(PATH, headers=AUTH).status_code == 503
    with TestClient(app) as client:
        assert client.get(PATH).status_code == 401
        assert client.post(PATH, headers=AUTH).status_code == 405
        answer = client.get(PATH, headers=AUTH)
        assert answer.status_code == 200 and answer.content == b'{"startup_ready":1}'
        assert client.get("/api/v1/health", headers=AUTH).status_code == 503
    assert not app.state.model_manager_startup_ready


def test_failed_initialization_never_advertises_ready(startup, monkeypatch):
    def fail():
        assert not app.state.model_manager_startup_ready
        raise RuntimeError("synthetic initialization failure")
    monkeypatch.setattr(db, "init", fail)
    with pytest.raises(RuntimeError, match="synthetic initialization failure"):
        with TestClient(app):
            pass
    assert not app.state.model_manager_startup_ready
    assert TestClient(app).get(PATH, headers=AUTH).status_code == 503
