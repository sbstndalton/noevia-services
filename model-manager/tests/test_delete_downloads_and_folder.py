"""noevia#1964 and #1966: a delete removes exactly the model named (no same-named model from
another folder), and never a folder a queued or running download writes into. Every file and job
is synthetic and lives under the test's temporary models folder; no download starts."""
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient

from conftest import INI, ROOT, _gguf
from app import services
from app.downloader import DownloadJob, manager, safe_dest
from app.main import app

MODELS = ROOT / "models"
MADE = ["dup.gguf", "dupsub", "dlq", "dlpart", "models--acme--queued-GGUF"]


@pytest.fixture()
def client(monkeypatch):
    (MODELS / "models.ini").write_text(INI)
    monkeypatch.setattr(services, "_running_containers", lambda: [])
    with TestClient(app) as c:
        yield c
    for name in MADE:
        p = MODELS / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
    for job_id in [k for k, j in manager.jobs.items() if j.repo_id == "synthetic.test"]:
        del manager.jobs[job_id]


def _model(rel, pad=64):
    path = MODELS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_gguf({"general.architecture": "llama"}) + b"\0" * pad)
    return path


def _job(rel, status="queued"):
    """A job the way `_make_job` records it, without starting a transfer."""
    dest = safe_dest(rel)
    job = DownloadJob(id=uuid.uuid4().hex[:12], repo_id="synthetic.test", filename=rel,
                      url="http://127.0.0.1:9/never", dest_path=dest,
                      temp_path=dest.with_suffix(dest.suffix + ".download"), total_bytes=0)
    job.status = status
    manager.jobs[job.id] = job
    return job


def _delete(client, *keys):
    return client.post("/api/v1/models/delete", json={"models": list(keys)})


# ---------- #1964 ----------

def test_duplicate_flat_key_never_falls_back_to_a_same_named_model_in_a_folder(client):
    flat = _model("dup.gguf")
    nested = _model("dupsub/dup.gguf")
    r = _delete(client, "dup.gguf", "dup.gguf")
    assert r.status_code == 200
    first, second = r.json()["results"]
    assert first["ok"] is True and not flat.exists()
    assert second == {"key": "dup.gguf", "ok": False, "message": "not found: dup.gguf", "freed": 0, "freedH": ""}
    assert nested.exists(), "a model nobody named must survive"


def test_delete_gguf_with_an_empty_subdir_only_matches_the_flat_model():
    nested = _model("dupsub/dup.gguf")
    try:
        assert services.delete_gguf("dup.gguf", "") == (False, "not found: dup.gguf", 0)
        assert nested.exists()
    finally:
        shutil.rmtree(MODELS / "dupsub")


# ---------- #1966 ----------

def test_queued_download_beside_the_model_refuses_the_delete(client):
    a = _model("dlq/a.gguf")
    partial = MODELS / "dlq" / "b.gguf.download"
    partial.write_bytes(b"\0" * 32)
    _job("dlq/b.gguf")
    r = _delete(client, "dlq/a.gguf")
    assert r.status_code == 409
    assert r.json()["detail"] == ("a.gguf has a download queued or running in its folder: dlq/b.gguf. "
                                  "Wait for it to finish or cancel it first; nothing was deleted.")
    assert a.exists() and partial.exists()


def test_running_download_of_the_models_own_file_refuses(client):
    a = _model("dlq/a.gguf")
    _job("dlq/a.gguf", status="downloading")
    assert _delete(client, "dlq/a.gguf").status_code == 409 and a.exists()


def test_a_flat_model_is_only_blocked_by_its_own_files(client):
    flat = _model("dup.gguf")
    _job("dlq/other.gguf")
    r = _delete(client, "dup.gguf")
    assert r.status_code == 200 and r.json()["results"][0]["ok"] is True and not flat.exists()


def test_finished_or_failed_jobs_do_not_block(client):
    a = _model("dlq/a.gguf")
    _job("dlq/b.gguf", status="completed")
    _job("dlq/c.gguf", status="failed")
    assert _delete(client, "dlq/a.gguf").json()["results"][0]["ok"] is True and not a.exists()


def test_hf_cache_repository_folder_counts_as_the_models_folder(client):
    snap = "models--acme--queued-GGUF/snapshots/0123abcd"
    m = _model(f"{snap}/q-Q4.gguf")
    _job("models--acme--queued-GGUF/blobs/elsewhere.gguf")
    assert _delete(client, f"{snap}/q-Q4.gguf").status_code == 409 and m.exists()


def test_delete_gguf_backstop_refuses_too(client):
    a = _model("dlq/a.gguf")
    _job("dlq/b.gguf")
    ok, msg, freed = services.delete_gguf("a.gguf", "dlq")
    assert (ok, freed) == (False, 0) and "download queued or running" in msg and a.exists()


def test_partial_download_left_in_the_folder_keeps_the_folder(client):
    """No job (a restart lost it), but its partial file is still another model on its way: the
    delete degrades to per-file and the partial file stays."""
    a = _model("dlpart/a.gguf")
    partial = MODELS / "dlpart" / "b.gguf.download"
    partial.write_bytes(b"\0" * 32)
    r = _delete(client, "dlpart/a.gguf")
    assert r.json()["results"][0]["ok"] is True
    assert not a.exists() and partial.exists()
