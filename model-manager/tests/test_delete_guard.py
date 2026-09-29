"""#336: the delete handler refuses files a running container depends on, even if the web guard
is bypassed. Containers are synthetic `docker inspect` dictionaries."""
import pytest
from fastapi.testclient import TestClient

from conftest import INI, ROOT, _gguf
from app import services
from app.main import app

TINY = ROOT / "models" / "tiny" / "tiny-Q4_K_M.gguf"


@pytest.fixture()
def client():
    (ROOT / "models" / "models.ini").write_text(INI)
    with TestClient(app) as c:
        yield c


def _embed(cmd, env=None, mounts=None):
    return {"name": "cowork-embed-1", "Config": {"Cmd": cmd, "Env": env or []},
            "Mounts": mounts or [{"Destination": "/models", "Source": "/mnt/user/ai-models"}]}


def _use(monkeypatch, *containers):
    monkeypatch.setattr(services, "_running_containers", lambda: list(containers))


def _delete(client, key="tiny/tiny-Q4_K_M.gguf"):
    return client.post("/api/v1/models/delete", json={"models": [key]})


def _tiny():
    return next(g for g in services.snapshot_models_dir().ggufs if g.display_name == "tiny-Q4_K_M.gguf")


def test_command_line_model_path_blocks_delete(client, monkeypatch):
    _use(monkeypatch, _embed(["--embedding", "--model", "/models/tiny/tiny-Q4_K_M.gguf", "--port", "8080"]))
    r = _delete(client)
    assert r.status_code == 409
    assert "cowork-embed-1" in r.json()["detail"] and "nothing was deleted" in r.json()["detail"]
    assert TINY.exists()


def test_flag_equals_form_and_other_mount_point_block_delete(client, monkeypatch):
    _use(monkeypatch, _embed(["--model=/weights/tiny/tiny-Q4_K_M.gguf"],
                             mounts=[{"Destination": "/weights", "Source": "/mnt/user/ai-models"}]))
    assert _delete(client).status_code == 409 and TINY.exists()


def test_embedding_model_env_blocks_delete_but_default_placeholder_does_not(client, monkeypatch):
    _use(monkeypatch, {"name": "web", "Config": {"Env": ["EMBEDDING_MODEL=tiny"]}})
    assert _delete(client).status_code == 409 and TINY.exists()
    assert services.model_holders(_tiny(), [{"name": "web", "Config": {"Env": ["EMBEDDING_MODEL=default"]}}]) == []


def test_rerank_model_only_counts_while_the_reranker_feature_is_on(client):
    off = {"name": "web", "Config": {"Env": ["RERANK_MODEL=tiny"]}}
    on = {"name": "web", "Config": {"Env": ["RERANK_MODEL=tiny", "NOEVIA_FEATURE_RAG_RERANK=true"]}}
    assert services.model_holders(_tiny(), [off]) == []
    assert services.model_holders(_tiny(), [on]) == ["web (RERANK_MODEL is set to tiny)"]


def test_unrelated_container_and_same_named_file_elsewhere_do_not_block(client):
    assert services.model_holders(_tiny(), [_embed(["--model", "/models/other/tiny-Q4_K_M.gguf"])]) == []
    elsewhere = _embed(["--model", "/data/tiny/tiny-Q4_K_M.gguf"], mounts=[{"Destination": "/data", "Source": "/srv/elsewhere"}])
    services.settings.models_host_path = "/mnt/user/ai-models"
    try:
        assert services.model_holders(_tiny(), [elsewhere]) == []
    finally:
        services.settings.models_host_path = None


def test_unreadable_docker_socket_fails_closed(client, monkeypatch):
    def boom():
        raise services.ModelInUse("Could not check which containers use this model (DockerException); nothing was deleted.")
    monkeypatch.setattr(services, "_running_containers", boom)
    assert _delete(client).status_code == 409 and TINY.exists()


def test_one_held_file_in_a_bulk_request_deletes_nothing(client, monkeypatch):
    spare = ROOT / "models" / "spare-Q4.gguf"
    spare.write_bytes(_gguf({"general.architecture": "llama"}) + b"\0" * 64)
    try:
        _use(monkeypatch, _embed(["--model", "/models/tiny/tiny-Q4_K_M.gguf"]))
        r = client.post("/api/v1/models/delete", json={"models": ["spare-Q4.gguf", "tiny/tiny-Q4_K_M.gguf"]})
        assert r.status_code == 409 and spare.exists() and TINY.exists()
    finally:
        spare.unlink(missing_ok=True)


def test_delete_gguf_backstop_refuses_for_non_api_callers(client, monkeypatch):
    _use(monkeypatch, _embed(["--model", "/models/tiny/tiny-Q4_K_M.gguf"]))
    ok, msg, freed = services.delete_gguf("tiny-Q4_K_M.gguf", "tiny")
    assert (ok, freed) == (False, 0) and "in use" in msg and TINY.exists()


def test_free_model_still_deletes(client, monkeypatch):
    keep = TINY.read_bytes()
    _use(monkeypatch, _embed(["--model", "/models/somewhere-else.gguf"]))
    try:
        assert _delete(client).json()["results"][0]["ok"] is True
        assert not TINY.exists()
    finally:
        TINY.parent.mkdir(exist_ok=True)
        TINY.write_bytes(keep)
