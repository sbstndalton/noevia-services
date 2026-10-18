"""noevia#1318 step 3d: under exact NOEVIA_NATIVE_MODEL_MANAGER_TELEMETRY_CONFIRMED=1 the native
parent ingests the backend logs and owns req_timing, server_config and the telemetry kv keys. This
child then never reads a log or writes any of them, so each table has one writer. Every log line,
name and figure is synthetic; no Docker daemon is touched.
"""
import sqlite3

import pytest

from app import db, services, telemetry

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_TELEMETRY_CONFIRMED"
LOG = (
    "2026-10-18T10:00:01.000000001Z srv    load: spawning server instance with name=syn-a on port 18001\n"
    "2026-10-18T10:00:03.123456789Z [18001] x slot print_timing: id  0 | task 1 | prompt eval time = "
    "1239.27 ms / 8297 tokens (0.15 ms per token, 6695.1 tokens per second)\n"
    "2026-10-18T10:00:03.123456789Z [18001] x slot print_timing: id  0 | task 1 |        eval time = "
    " 777.37 ms /  111 tokens (7.07 ms per token,  141.5 tokens per second)\n"
).encode()


@pytest.fixture()
def docker(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    asked = []

    class Container:
        def logs(self, **kw):
            asked.append(kw)
            return LOG

    class Containers:
        def get(self, name):
            return Container()

    class Client:
        containers = Containers()

    monkeypatch.setattr(services, "_docker_client", lambda: Client())
    db.init()
    with sqlite3.connect(db._path()) as c:
        c.execute("DELETE FROM req_timing")
        c.execute("DELETE FROM server_config")
        c.execute("DELETE FROM kv WHERE key LIKE 'telemetry%'")
    return asked


def held():
    with sqlite3.connect(db._path()) as c:
        return (c.execute("SELECT COUNT(*) FROM req_timing").fetchone()[0],
                c.execute("SELECT COUNT(*) FROM server_config").fetchone()[0],
                c.execute("SELECT COUNT(*) FROM kv WHERE key LIKE 'telemetry%'").fetchone()[0])


def test_unconfirmed_child_still_ingests(docker):
    assert telemetry.ingest(["syn-a-backend"], force=True) == 1
    assert docker and held() == (1, 1, 3)


@pytest.mark.parametrize("value", ["0", "", "true", " 1", "1 "])
def test_only_the_exact_confirmation_hands_over(docker, monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert telemetry.ingest(["syn-a-backend"], force=True) == 1
    assert held()[0] == 1


def test_confirmed_child_never_reads_a_log_nor_writes(docker, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert telemetry.ingest(["syn-a-backend"]) == 0
    assert telemetry.ingest(["syn-a-backend"], force=True) == 0
    samples, configs = telemetry.parse_log(LOG.decode(), "syn-a-backend")
    assert samples and configs
    assert db.record_timings("syn-a-backend", samples) == 0
    assert db.record_server_configs("syn-a-backend", configs) == 0
    assert docker == [] and held() == (0, 0, 0)
    # The readers keep working over whatever the native parent stored.
    assert telemetry.stats_for(model_path="/models/syn/a.gguf", alias="syn-a").n == 0
