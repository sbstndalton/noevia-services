"""sbstndalton/noevia#1783: the model manager only rotates the `.bak-<YYYYMMDD-HHMMSS>[-n]` copies it
makes itself; an operator's own `models.ini.bak-*` file beside models.ini is never removed, under
either MODEL_FILES_IMPL. Every models.ini write route, section DELETE included, leaves one rotating
pre-change copy. Synthetic files only."""
from __future__ import annotations

import os
import re

import pytest
from fastapi.testclient import TestClient

from app import config, ini
from conftest import INI, ROOT
from tests.test_model_backups import _fake, _rust

OPERATOR = ("models.ini.bak-operator", "models.ini.bak-2026-10-15", "models.ini.bak-1731-before-preset-removal",
            "models.ini.bak-20261015", "models.ini.bak-20261015-1200")
OWN = re.compile(r"models\.ini\.bak-\d{8}-\d{6}(-\d+)?")


@pytest.fixture(autouse=True)
def _python_by_default(monkeypatch):
    monkeypatch.setattr(config.settings, "model_files_impl", "python")
    monkeypatch.setattr(config.settings, "model_files_bin", "model-files")


def _seed(d):
    d.mkdir(exist_ok=True)
    (d / "models.ini").write_text("version = 1\n")
    for i in range(ini.BACKUPS_TO_KEEP + 2):  # more of the manager's own copies than it keeps
        (d / f"models.ini.bak-202609{i + 1:02}-000000").write_text(f"own {i}\n")
    for n in OPERATOR:
        (d / n).write_text(f"operator {n}\n")


def _own(d):
    return sorted(p.name for p in d.iterdir() if OWN.fullmatch(p.name))


def test_python_save_keeps_operator_bak_files_and_rotates_its_own(tmp_path, monkeypatch):
    d = tmp_path / "cfg"
    _seed(d)
    monkeypatch.setattr(config.settings, "models_ini_path", d / "models.ini")
    ini._replace_file(d / "models.ini", "version = 2\n")
    for n in OPERATOR:
        assert (d / n).read_text() == f"operator {n}\n", f"{n} was removed"
    own = _own(d)
    assert len(own) == ini.BACKUPS_TO_KEEP and own[0] == "models.ini.bak-20260904-000000"
    assert (d / own[-1]).read_text() == "version = 1\n", "this write's pre-change copy is the newest"


def test_rust_plan_naming_an_operator_file_cannot_remove_it(tmp_path, monkeypatch):
    # An older model-files binary prunes every listed `.bak-*`; ini.py drops names it did not make.
    d = tmp_path / "cfg"
    _seed(d)
    monkeypatch.setattr(config.settings, "models_ini_path", d / "models.ini")
    reply = ('{"rotating":null,"revision":null,"prune":["models.ini.bak-1731-before-preset-removal",'
             '"models.ini.bak-2026-10-15","models.ini.bak-20260901-000000"]}')
    _rust(monkeypatch, _fake(tmp_path, reply))
    ini._replace_file(d / "models.ini", "version = 2\n")
    assert (d / "models.ini").read_text() == "version = 2\n"
    for n in OPERATOR:
        assert (d / n).exists(), f"{n} was removed"
    assert not (d / "models.ini.bak-20260901-000000").exists(), "its own old copy is still pruned"


@pytest.fixture()
def client():
    with TestClient(app_module()) as c:
        yield c


def app_module():
    from app.main import app
    return app


def test_every_write_route_keeps_one_pre_change_copy_and_operator_files(client):
    d = ROOT / "models"
    path = d / "models.ini"
    path.write_text(INI)
    for i in range(ini.BACKUPS_TO_KEEP):  # a full rotation, so every write prunes
        (d / f"models.ini.bak-202609{i + 1:02}-000000").write_text(f"own {i}\n")
    for n in OPERATOR:
        (d / n).write_text(f"operator {n}\n")
    try:
        def write(call):
            before, copies = path.read_text(), set(_own(d))
            r = call(client.get("/api/v1/sections").json()["revision"])
            assert r.status_code == 200, r.text
            [new] = set(_own(d)) - copies
            assert (d / new).read_text() == before, "the copy holds the file as it was"
        write(lambda rev: client.put("/api/v1/models-ini", json={"baseRevision": rev, "text": INI + "; edited\n"}))
        values = client.get("/api/v1/sections/tiny").json()["values"]
        write(lambda rev: client.put("/api/v1/sections/tiny", json={"baseRevision": rev, "values": {**values, "ctx-size": "4096"}}))
        write(lambda rev: client.post("/api/v1/sections/tiny/rename", json={"newName": "tiny-chat", "baseRevision": rev}))
        write(lambda rev: client.delete(f"/api/v1/sections/tiny-chat?baseRevision={rev}"))
        assert "[tiny-chat]" not in path.read_text()
        for n in OPERATOR:
            assert (d / n).read_text() == f"operator {n}\n", f"{n} was removed"
    finally:
        for p in d.glob("models.ini.bak-*"):
            os.unlink(p)
