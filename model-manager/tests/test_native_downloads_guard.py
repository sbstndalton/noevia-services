"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_DOWNLOADS_CONFIRMED=1 the private Python child
refuses the five download job routes (the native parent owns the job table and runs the
transfers) and answers the parent's private planning request: the start route runs as usual but
queues nothing, and the files it would have queued come back with its answer. Nothing is
downloaded: the Hub, HEAD requests and the job table are fakes, and every name is synthetic.
"""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import api, downloader, hf
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_DOWNLOADS_CONFIRMED"
TOKEN = "synthetic-downloads-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PLAN = {"x-noevia-download-plan": "1"}
ROUTES = [("get", "/api/v1/downloads"), ("post", "/api/v1/downloads"),
          ("post", "/api/v1/downloads/0123456789ab/cancel"), ("post", "/api/v1/downloads/clear"),
          ("post", "/api/v1/sections/synthetic-model/draft-heads/download")]
REAL_CLIENT = httpx.AsyncClient


def _file(path, size):
    return hf.HfFile(path=path, size=size, quant=None, shard_base=path, shard_index=None, shard_total=None)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "model_download_targets", "")
    monkeypatch.delenv(ENV, raising=False)
    started = []
    make = downloader.DownloadManager._make_job

    async def idle(self, job):
        return None

    def recorded(self, **kw):
        job = make(self, **kw)
        if job is not None:
            started.append(job.filename)
        return job
    monkeypatch.setattr(downloader.DownloadManager, "_run", idle)
    monkeypatch.setattr(downloader.DownloadManager, "_make_job", recorded)
    monkeypatch.setattr(downloader.manager, "jobs", {})
    heads = []

    def respond(req):
        heads.append((req.method, str(req.url)))
        return httpx.Response(200, headers={"content-length": "4096"})
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(respond), **kw))

    async def detail(repo, revision="main"):
        if repo != "synthetic-owner/synthetic-repo":
            raise RuntimeError("synthetic: no such repo")
        return hf.HfRepoDetail(id=repo, files=[
            _file("Synthetic-Q4_K_M.gguf", 1000), _file("mmproj-synthetic-f16.gguf", 200),
            _file("synthetic-draft-0.5B-Q8_0.gguf", 50)], readme_snippet=None)
    monkeypatch.setattr(hf, "repo_detail", detail)
    c = TestClient(app, raise_server_exceptions=False)
    c.started, c.heads = started, heads
    return c


def auth(**extra):
    return {"x-model-loader-token": TOKEN, **extra}


def test_confirmed_child_refuses_the_five_job_routes(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    for method, path in ROUTES:
        assert getattr(client, method)(path).status_code == 401
        kwargs = {"json": {}} if method == "post" else {}
        answer = getattr(client, method)(path, headers=auth(), **kwargs)
        assert answer.status_code == 503 and answer.content == REFUSED, path
        # A planning header on anything but a start is still refused.
        if path.endswith("/cancel") or path.endswith("/clear") or method == "get":
            answer = getattr(client, method)(path, headers=auth(**PLAN), **kwargs)
            assert answer.status_code == 503 and answer.content == REFUSED, path
    # Other download reads and the targets follow their own switches.
    assert client.get("/api/v1/download-targets", headers=auth()).status_code == 200
    assert client.get("/api/v1/downloads/clear", headers=auth()).status_code == 405
    assert client.started == []


@pytest.mark.parametrize("value", [None, "0", "true", " 1"])
def test_other_spellings_keep_python_ownership(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    # The planning header means nothing to an app that owns the job table: the route queues.
    answer = client.post("/api/v1/downloads", headers=auth(**PLAN),
                         json={"url": "https://example.test/files/synthetic.gguf"})
    assert answer.status_code == 200 and answer.json() == {"queued": ["synthetic.gguf"]}
    assert client.started == ["synthetic/synthetic.gguf"]
    assert [j["filename"] for j in client.get("/api/v1/downloads", headers=auth()).json()["jobs"]] == \
        ["synthetic/synthetic.gguf"]


def plan(client, path, body, **kw):
    kw.setdefault("headers", auth(**PLAN))
    answer = client.post(path, **({"json": body} if body is not None else {}), **kw)
    assert answer.status_code == 200, answer.content
    return answer.json()


def test_a_url_start_is_planned_not_queued(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    got = plan(client, "/api/v1/downloads", {"url": "https://example.test/files/synthetic.gguf"})
    assert got == {"status": 200, "body": '{"queued":["synthetic.gguf"]}', "jobs": [
        {"repo_id": "example.test", "filename": "synthetic/synthetic.gguf",
         "url": "https://example.test/files/synthetic.gguf", "total_bytes": 4096}]}
    assert client.heads == [("HEAD", "https://example.test/files/synthetic.gguf")]
    assert client.started == [] and downloader.manager.jobs == {}


def test_a_repo_start_plans_the_file_and_its_companions(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    got = plan(client, "/api/v1/downloads",
               {"repo": "synthetic-owner/synthetic-repo", "path": "Synthetic-Q4_K_M.gguf"})
    assert got["status"] == 200
    queued = json.loads(got["body"])["queued"]
    assert queued == [j["url"].rsplit("/", 1)[1] for j in got["jobs"]]
    assert [j["repo_id"] for j in got["jobs"]] == ["synthetic-owner/synthetic-repo"] * 3
    assert all(j["url"].startswith(f"{hf.HF_RESOLVE}/synthetic-owner/synthetic-repo/resolve/main/")
               for j in got["jobs"])
    assert [j["total_bytes"] for j in got["jobs"]] == [1000, 200, 50]
    assert client.started == [] and downloader.manager.jobs == {}


def test_refusals_come_back_byte_for_byte_with_no_jobs(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    for body, status in [({"url": "ftp://example.test/x.gguf"}, 400), ({"target": "nowhere"}, 400),
                         ({"repo": "synthetic-owner/synthetic-repo", "path": "absent.gguf"}, 404),
                         ({"url": "https://example.test/x.gguf", "filename": "../x.gguf"}, 400)]:
        got = plan(client, "/api/v1/downloads", body)
        assert got["status"] == status and got["jobs"] == [], body
        monkeypatch.delenv(ENV)
        public = client.post("/api/v1/downloads", headers=auth(), json=body)
        monkeypatch.setenv(ENV, "1")
        assert (public.status_code, public.content.decode()) == (status, got["body"]), body
    # FastAPI's own 422 for a body that is not a JSON object.
    got = plan(client, "/api/v1/downloads", None, content=b"[1]",
               headers=auth(**PLAN, **{"content-type": "application/json"}))
    assert got["status"] == 422 and got["jobs"] == []
    assert client.started == []


def test_an_exception_in_the_route_is_the_childs_500(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    answer = client.post("/api/v1/downloads", headers=auth(**PLAN),
                         json={"repo": "synthetic-owner/missing", "path": "x.gguf"})
    assert answer.status_code == 500 and answer.content == b"Internal Server Error"
    # The planning state does not leak into later requests.
    assert downloader.PLANNING.get() is None


def test_a_draft_head_start_is_planned(monkeypatch, client, tmp_path):
    monkeypatch.setenv(ENV, "1")
    folder = tmp_path / "Synthetic"
    folder.mkdir()
    monkeypatch.setattr(api, "_resolve_section_gguf",
                        lambda name: (folder / "Synthetic-Q4_K_M.gguf", "Synthetic/Synthetic-Q4_K_M.gguf",
                                      "Synthetic/Synthetic-Q4_K_M.gguf"))
    monkeypatch.setattr(api.db, "repo_for_file", lambda name: "synthetic-owner/synthetic-repo")
    got = plan(client, "/api/v1/sections/synthetic-model/draft-heads/download",
               {"path": "synthetic-draft-0.5B-Q8_0.gguf"})
    assert got == {"status": 200,
                   "body": '{"queued":["synthetic-draft-0.5B-Q8_0.gguf"],"folder":"Synthetic"}',
                   "jobs": [{"repo_id": "synthetic-owner/synthetic-repo",
                             "filename": "Synthetic/synthetic-draft-0.5B-Q8_0.gguf",
                             "url": f"{hf.HF_RESOLVE}/synthetic-owner/synthetic-repo/resolve/main/"
                                    "synthetic-draft-0.5B-Q8_0.gguf",
                             "total_bytes": 50}]}
    assert client.started == []
