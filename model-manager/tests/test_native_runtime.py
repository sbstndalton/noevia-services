"""Merged Rust binary + actual Python app, isolated from real Docker/models/hardware.

Only these tests' temporary uvicorn executable suppresses the hardware sampler and
business Docker discovery. Production startup code is unchanged. CI supplies both
feature-enabled and deliberately omitted binaries built from the verified archive.
"""
import contextlib
import http.server
import json
import os
from pathlib import Path
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import pytest

SERVICE = Path(__file__).resolve().parents[1]
TOKEN = "synthetic-runtime-token-0000000000000000000000"


def _binary(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not provided; CI builds the merged pinned binaries")
    path = Path(value).resolve()
    assert path.is_file(), name
    return str(path)


def _request(path, *, port=8090, token=TOKEN):
    headers = {} if token is None else {"X-Model-Loader-Token": token}
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers)
    # Fixed owned loopback peer; ignore ambient HTTP proxies.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with client.open(request, timeout=2) as response:
            return response.status, json.loads(response.read(2_000_000))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read(2_000_000))


def _wait(path, process, expected=200, port=8090):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        assert process.poll() is None, "owned runtime exited before readiness"
        try:
            answer = _request(path, port=port)
            if answer[0] == expected:
                return answer
        except (OSError, ValueError):
            pass
        time.sleep(0.02)
    pytest.fail("owned runtime did not become ready")


@pytest.fixture
def runtime_env(tmp_path):
    # Refuse borrowing any existing process on the fixed production ports.
    for port in (8090, 8091):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    tools = tmp_path / "bin"
    tools.mkdir()
    marker = tmp_path / "child.json"
    shim = tools / "uvicorn"
    shim.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
from app import main, services
# Test-only: no hardware sampling, real Docker discovery, engines or owner data.
main.hw.start_sampler = lambda: None
services._docker_client = lambda: None
services.container_logs = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Python must not own logs"))
Path(os.environ["S1_RUNTIME_CHILD"]).write_text(json.dumps({"confirmed": os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED"), "sections": os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED"), "argv": sys.argv[1:]}))
import uvicorn
uvicorn.run(main.app, host=sys.argv[3], port=int(sys.argv[5]), log_level="error")
''')
    shim.chmod(0o700)
    models = tmp_path / "models"
    models.mkdir()
    ini = models / "models.ini"
    ini.write_text("version = 1\n")
    data = tmp_path / "data"
    data.mkdir()
    env = os.environ.copy()
    env.update(PATH=str(tools) + os.pathsep + env["PATH"], PYTHONPATH=str(SERVICE),
               MODELS_DIR=str(models), MODELS_INI_PATH=str(ini), DATA_DIR=str(data),
               MODEL_LOADER_TOKEN=TOKEN, LLAMA_CONTAINERS="synthetic-engine",
               MIGRATE_CACHE_RAM_ON_START="false", GGUF_PARSER="python",
               MODEL_FILES_IMPL="python", MODEL_AUTOCONFIG="python",
               S1_RUNTIME_CHILD=str(marker), PYTHONDONTWRITEBYTECODE="1",
               DOCKER_HOST="unix://" + str(tmp_path / "docker.sock"))
    env.pop("MODEL_MANAGER_IMPL", None)
    with tempfile.TemporaryDirectory(prefix="s1-native-", dir="/tmp") as socket_dir:
        env["DOCKER_HOST"] = "unix://" + str(Path(socket_dir) / "docker.sock")
        yield env, marker


@contextlib.contextmanager
def _process(binary, env):
    process = subprocess.Popen([binary], cwd=SERVICE, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=7)
        except subprocess.TimeoutExpired:
            process.kill()  # This test's owned process only.
            process.communicate(timeout=3)
            pytest.fail("owned runtime exceeded bounded shutdown")


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


@contextlib.contextmanager
def _docker_peer(socket_path):
    observed = []
    state = {"malformed": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            observed.append((self.command, self.path))
            path = self.path.split("?", 1)[0]
            if path == "/containers/json":
                body = b'[{"Id":"synthetic-engine"}]'
            elif path == "/containers/synthetic-engine/json":
                body = b'{"Name":"/synthetic-engine","Image":"sha256:synthetic","Config":{"Tty":false}}'
            elif path == "/images/sha256:synthetic/json":
                body = b'{"RepoTags":["synthetic/llama.cpp:test"]}'
            elif path == "/containers/synthetic-engine/logs":
                payload = b"password\x1f=syntheticSECRET\nn_stats.password=syntheticSECRET\n"
                body = b"".join(bytes([kind, 0, 0, 0]) + struct.pack(">I", len(raw)) + raw
                                for kind, raw in [(1, payload), (2, b"warn synthetic stderr\n")])
                if state["malformed"]:
                    body = b"\x03\x00\x00\x00\x00\x00\x00\x01X"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    with _UnixServer(str(socket_path), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield observed, state
        finally:
            server.shutdown()
            worker.join(timeout=2)
            assert not worker.is_alive()


def test_default_binary_execs_actual_python_and_clears_confirmation(runtime_env):
    env, marker = runtime_env
    env["NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED"] = "1"
    with _process(_binary("MODEL_MANAGER_BIN"), env) as process:
        assert _wait("/api/v1/health", process)[1] == {"ok": True}
        assert _request("/api/v1/sections")[0] == 200
        child = json.loads(marker.read_text())
        assert child == {"confirmed": None, "sections": None, "argv": ["app.main:app", "--host", "0.0.0.0", "--port", "8090"]}


def test_native_runtime_owns_logs_and_supervises_actual_python(runtime_env):
    env, marker = runtime_env
    env["MODEL_MANAGER_IMPL"] = "rust"
    with _docker_peer(Path(env["DOCKER_HOST"].removeprefix("unix://"))) as (calls, state):
        with _process(_binary("MODEL_MANAGER_BIN"), env) as process:
            assert _wait("/api/v1/sections", process)[0] == 200
            child = json.loads(marker.read_text())
            assert child == {"confirmed": "1", "sections": "0", "argv": ["app.main:app", "--host", "127.0.0.1", "--port", "8091"]}
            assert _request("/api/v1/health", token=None) == (200, {"ok": True})
            assert calls == []
            assert _request("/api/v1/backends/synthetic-engine/logs", token=None)[0] == 401
            assert calls == []
            assert _request("/api/v1/health", port=8091)[0] == 503
            assert _request("/api/v1/backends/synthetic-engine/logs", port=8091)[0] == 503
            status, answer = _request("/api/v1/backends/synthetic-engine/logs?tail=0")
            assert status == 200 and answer["ok"] is True
            assert "syntheticSECRET" not in json.dumps(answer)
            assert "warn synthetic stderr" in answer["lines"]
            assert _request("/api/v1/backends/synthetic-engine/logs?q=syntheticSECRET")[1]["lines"] == []
            log_calls = [path for _, path in calls if path.startswith("/containers/synthetic-engine/logs?")]
            assert log_calls and all("tail=400" in path and "stdout=true" in path and "stderr=true" in path and "follow=false" in path for path in log_calls)
            assert all(method == "GET" for method, _ in calls)
            state["malformed"] = True
            status, answer = _request("/api/v1/backends/synthetic-engine/logs")
            assert status == 200 and answer["ok"] is False
            assert "syntheticSECRET" not in json.dumps(answer)
        # Rust terminates its owned child; no fixed-loopback Python survives shutdown.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", 8091))


def test_sections_switch_default_keeps_reads_with_python(runtime_env):
    # #1334: MODEL_MANAGER_SECTIONS_IMPL defaults to python; the rust parent still supervises a child that
    # is not told the sections are native, so the Python endpoint answers them.
    env, marker = runtime_env
    env["MODEL_MANAGER_IMPL"] = "rust"
    with _docker_peer(Path(env["DOCKER_HOST"].removeprefix("unix://"))):
        with _process(_binary("MODEL_MANAGER_BIN"), env) as process:
            assert _wait("/api/v1/sections", process)[0] == 200
            child = json.loads(marker.read_text())
            assert child["sections"] == "0"


@pytest.mark.skipif(sys.platform != "linux", reason="native section reads are Linux-only (noevia-rs sections/access.rs)")
def test_sections_switch_routes_reads_to_the_native_binary(runtime_env):
    # #1334: MODEL_MANAGER_SECTIONS_IMPL=rust answers GET /api/v1/sections natively and the Python child
    # refuses those reads (confirmation 1), after authentication as before.
    env, marker = runtime_env
    env.update(MODEL_MANAGER_IMPL="rust", MODEL_MANAGER_SECTIONS_IMPL="rust")
    with _docker_peer(Path(env["DOCKER_HOST"].removeprefix("unix://"))):
        with _process(_binary("MODEL_MANAGER_BIN"), env) as process:
            assert _wait("/api/v1/sections", process)[0] == 200
            child = json.loads(marker.read_text())
            assert child["sections"] == "1"
            assert _request("/api/v1/sections", port=8091)[0] == 503
            assert _request("/api/v1/sections/tiny", port=8091)[0] == 503
            assert _request("/api/v1/sections", token=None)[0] == 401


@pytest.mark.parametrize("env_update", [{"MODEL_MANAGER_SECTIONS_IMPL": "rust"},
    {"MODEL_MANAGER_SECTIONS_IMPL": "maybe", "MODEL_MANAGER_IMPL": "rust"},
    # An empty value (a compose line that expands to nothing) is not the default `python`: it is refused
    # by name, in either manager mode, rather than guessed at.
    {"MODEL_MANAGER_SECTIONS_IMPL": "", "MODEL_MANAGER_IMPL": "rust"},
    {"MODEL_MANAGER_SECTIONS_IMPL": "", "MODEL_MANAGER_IMPL": "python"}])
def test_sections_switch_refuses_before_python_spawn(runtime_env, env_update):
    # #1334: rust sections need the rust manager and a compiled native-sections capability; an
    # unknown or empty value is refused. Each refuses before the Python child is started.
    env, marker = runtime_env
    env.update(env_update)
    answer = subprocess.run([_binary("MODEL_MANAGER_BIN")], env=env, cwd=SERVICE,
                            capture_output=True, timeout=5)
    assert answer.returncode != 0
    assert not marker.exists()
    if env_update["MODEL_MANAGER_SECTIONS_IMPL"] in ("", "maybe"):
        assert b"MODEL_MANAGER_SECTIONS_IMPL must be python or rust" in answer.stderr


def test_the_python_service_alone_ignores_the_sections_selector(monkeypatch):
    # No Rust parent: only the parent's confirmation (=1) makes Python refuse section reads, so an empty or
    # absent selector leaves the Python endpoint answering them (the default).
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setenv("MODEL_MANAGER_SECTIONS_IMPL", "")
    monkeypatch.delenv("NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED", raising=False)
    with TestClient(app) as client:
        assert client.get("/api/v1/sections").status_code != 503


@pytest.mark.parametrize("setting,value", [("MODEL_MANAGER_IMPL", "invalid"),
    ("MODEL_LOADER_TOKEN", " "), ("DOCKER_HOST", "tcp://127.0.0.1:2375")])
def test_invalid_native_configuration_refuses_before_python_spawn(runtime_env, setting, value):
    env, marker = runtime_env
    env.update(MODEL_MANAGER_IMPL="rust")
    env[setting] = value
    answer = subprocess.run([_binary("MODEL_MANAGER_BIN")], env=env, cwd=SERVICE,
                            capture_output=True, timeout=5)
    assert answer.returncode != 0
    assert not marker.exists()


def test_deliberately_omitted_feature_refuses_before_python_spawn(runtime_env):
    env, marker = runtime_env
    env["MODEL_MANAGER_IMPL"] = "rust"
    answer = subprocess.run([_binary("MODEL_MANAGER_OMISSION_BIN")], env=env,
                            cwd=SERVICE, capture_output=True, timeout=5)
    assert answer.returncode != 0 and b"feature is not enabled" in answer.stderr
    assert not marker.exists()
