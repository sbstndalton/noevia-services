"""A llama-server child respawned onto a port a dead child used must not lose samples or
overwrite the dead child's recorded argv (#874 item 2). Synthetic log text only."""
from __future__ import annotations

from app import db, telemetry


def _spawn(ts: str, alias: str, port: int, model: str, extra: str = "") -> list[str]:
    return [
        f"{ts} [r] 0.00.000 I srv    load: spawning server instance with name={alias} on port {port}",
        f"{ts} [r] 0.00.000 I srv    load: spawning server instance with args:",
        f"{ts} [r] 0.00.000 I srv    load:   --model",
        f"{ts} [r] 0.00.000 I srv    load:   {model}",
        f"{ts} [r] 0.00.000 I srv    load:   --ctx-size",
        f"{ts} [r] 0.00.000 I srv    load:   {extra or 4096}",
        f"{ts} [{port}] 0.00.100 I srv    load_model: loading model '{model}'",
    ]


def _request(ts: str, port: int, task: int, gen_tps: float) -> list[str]:
    return [
        f"{ts} [{port}] 0.01.000 I slot print_timing: id 0 | task {task} | prompt eval time = 100.00 ms / 50 tokens (2.00 ms per token, 500.00 tokens per second)",
        f"{ts} [{port}] 0.01.000 I slot print_timing: id 0 | task {task} |        eval time = 1000.00 ms /  100 tokens (10.00 ms per token, {gen_tps:.2f} tokens per second)",
    ]


_COVERED = 1e12   # a watermark past every synthetic timestamp: the tail overlaps the last ingest


def _log() -> str:
    lines = []
    lines += _spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf", "4096")
    lines += _request("2026-10-05T10:00:05.000000000Z", 60279, 3, 100.0)
    # child dies, router respawns onto the SAME port with different settings
    lines += _spawn("2026-10-05T11:00:00.000000000Z", "alpha", 60279, "/models/a.gguf", "8192")
    lines += _request("2026-10-05T11:00:05.000000000Z", 60279, 3, 140.0)   # same task number
    return "\n".join(lines)


def test_respawn_on_a_reused_port_gets_a_distinct_instance():
    samples, configs = telemetry.parse_log(_log())
    assert len(samples) == 2
    assert len({s.instance for s in samples}) == 2
    assert len(configs) == 2
    by_instance = {c.instance: c for c in configs}
    # each sample is attributed to the child that actually served it
    for s in samples:
        assert s.instance in by_instance
    argvs = {c.argv.get("--ctx-size") for c in configs}
    assert argvs == {"4096", "8192"}


def test_reused_port_samples_are_both_stored_and_argv_is_not_reassigned():
    db.init()
    backend = "test-reuse"
    samples, configs = telemetry.parse_log(_log())
    db.record_server_configs(backend, configs)
    assert db.record_timings(backend, samples) == 2
    # re-reading the same tail stays a no-op (idempotent ingest)
    assert db.record_timings(backend, samples) == 0
    rows = db.recent_timings(alias="alpha", min_gen_tokens=1)
    mine = [r for r in rows if r["backend"] == backend]
    assert sorted(r["gen_tps"] for r in mine) == [100.0, 140.0]
    stored = db.server_configs_for(sorted({r["instance"] for r in mine}))
    assert sorted(v.get("--ctx-size") for v in stored.values()) == ["4096", "8192"]


def test_samples_already_stored_under_the_legacy_bare_port_key_are_not_duplicated():
    db.init()
    backend = "test-legacy"
    samples, _ = telemetry.parse_log(_log())
    legacy = [telemetry.Sample(**{**s.__dict__, "instance": s.instance.split("@")[0]}) for s in samples[:1]]
    assert db.record_timings(backend, legacy) == 1
    # after upgrade the same tail yields "<port>@<ts>" keys: the legacy sample is recognised
    assert db.record_timings(backend, samples) == 1


def test_a_bare_port_fallback_does_not_re_insert_a_task_stored_under_port_at_spawn():
    """#884: once the spawn line scrolls out of the log tail the parser falls back to the bare
    port for the same task; it must not add a second row next to the "<port>@<ts>" one."""
    db.init()
    backend = "test-straddle"
    with_spawn, _ = telemetry.parse_log(_log())
    assert all("@" in s.instance for s in with_spawn)
    assert db.record_timings(backend, with_spawn) == 2
    # the tail now begins after both spawn lines, so the same tasks come back with bare keys
    tail_only = "\n".join(_request("2026-10-05T10:00:05.000000000Z", 60279, 3, 100.0)
                          + _request("2026-10-05T11:00:05.000000000Z", 60279, 3, 140.0))
    bare, _ = telemetry.parse_log(tail_only)
    assert bare and all("@" not in s.instance for s in bare)
    assert db.record_timings(backend, bare) == 0
    rows = [r for r in db.recent_timings(alias="alpha", min_gen_tokens=1) if r["backend"] == backend]
    assert len(rows) == 2 and all("@" in r["instance"] for r in rows)


def test_the_bare_guard_is_per_backend_task_and_timestamp():
    db.init()
    backend = "test-straddle-scope"
    spawned, _ = telemetry.parse_log("\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")
                                                + _request("2026-10-05T10:00:05.000000000Z", 60279, 3, 100.0)))
    assert db.record_timings(backend, spawned) == 1
    def bare(ts, port, task):
        s, _ = telemetry.parse_log("\n".join(_request(ts, port, task, 90.0)))
        return s
    # a different task, timestamp, port or backend is a genuinely different sample
    assert db.record_timings(backend, bare("2026-10-05T10:00:05.000000000Z", 60279, 4)) == 1
    assert db.record_timings(backend, bare("2026-10-05T10:09:05.000000000Z", 60279, 3)) == 1
    assert db.record_timings(backend, bare("2026-10-05T10:00:05.000000000Z", 6027, 3)) == 1   # prefix, not substring
    assert db.record_timings(backend + "-other", bare("2026-10-05T10:00:05.000000000Z", 60279, 3)) == 1


def test_instance_key_is_stable_and_degrades_to_the_port_without_a_timestamp():
    assert telemetry.instance_key("60279", 0.0) == "60279"
    assert telemetry.instance_key("60279", 1759658400.9) == telemetry.instance_key("60279", 1759658400.2)


def test_a_request_whose_spawn_line_left_the_tail_is_attributed_to_the_stored_instance():
    """#904: after 20000 lines the spawn line is gone; new tasks must still land under the
    instance's alias and model path, not under the bare port with empty fields."""
    db.init()
    backend = "test-904-tail"
    first, configs = telemetry.parse_log("\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")
                                                   + _request("2026-10-05T10:00:05.000000000Z", 60279, 3, 100.0)))
    db.record_server_configs(backend, configs)
    assert db.record_timings(backend, first) == 1
    # the tail now holds only a new task on the same port
    tail, tail_configs = telemetry.parse_log("\n".join(_request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0)), backend, _COVERED)
    assert [s.instance for s in tail] == [first[0].instance]
    assert (tail[0].alias, tail[0].model_path) == ("alpha", "/models/a.gguf")
    assert tail_configs == []                          # the stored record is not touched
    assert db.record_timings(backend, tail) == 1
    rows = [r for r in db.recent_timings(alias="alpha", min_gen_tokens=1) if r["backend"] == backend]
    assert sorted(r["gen_tps"] for r in rows) == [100.0, 120.0]
    assert all("@" in r["instance"] for r in rows)
    # re-reading the same tail stays a no-op
    again, _ = telemetry.parse_log("\n".join(_request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0)), backend, _COVERED)
    assert db.record_timings(backend, again) == 0


def test_a_reused_port_resolves_to_the_instance_that_was_newest_at_the_request_time():
    db.init()
    backend = "test-904-reuse"
    parsed, configs = telemetry.parse_log(_log())
    db.record_server_configs(backend, configs)
    db.record_timings(backend, parsed)
    old, new = sorted((c.instance for c in configs), key=lambda k: int(k.split("@")[1]))
    tail = "\n".join(_request("2026-10-05T11:30:00.000000000Z", 60279, 9, 150.0)    # after the respawn
                     + _request("2026-10-05T10:30:00.000000000Z", 60279, 8, 110.0)  # before it
                     + _request("2026-10-05T09:00:00.000000000Z", 60279, 7, 90.0))   # before any spawn we know
    samples, _ = telemetry.parse_log(tail, backend, _COVERED)
    by_task = {s.task: s for s in samples}
    assert by_task[9].instance == new and by_task[8].instance == old
    assert "@" not in by_task[7].instance and by_task[7].alias == ""


def test_other_backends_and_other_ports_are_not_resolved():
    db.init()
    backend = "test-904-scope"
    parsed, configs = telemetry.parse_log("\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")))
    db.record_server_configs(backend, configs)
    req = "\n".join(_request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0))
    assert "@" not in telemetry.parse_log(req, backend + "-other", _COVERED)[0][0].instance
    req2 = "\n".join(_request("2026-10-05T10:30:00.000000000Z", 6027, 4, 120.0))
    assert "@" not in telemetry.parse_log(req2, backend, _COVERED)[0][0].instance        # prefix, not substring


def test_ingest_keeps_attributing_requests_after_the_spawn_line_scrolls_out(monkeypatch):
    """End to end through ingest(): the second pass reads a tail with no spawn line."""
    from app import services
    db.init()
    backend = "test-904-ingest"
    logs = {"text": "\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")
                              + _request("2026-10-05T10:00:05.000000000Z", 60279, 3, 100.0))}

    class _Container:
        def logs(self, **_kw):
            return logs["text"].encode()

    class _Client:
        class containers:
            @staticmethod
            def get(_name):
                return _Container()

    monkeypatch.setattr(services, "_docker_client", lambda: _Client())
    assert telemetry.ingest([backend], force=True) == 1
    # the new tail starts at the previous pass's last line (overlap) but has lost the spawn line
    logs["text"] = "\n".join(_request("2026-10-05T10:00:05.000000000Z", 60279, 3, 100.0)
                             + _request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0))
    assert telemetry.ingest([backend], force=True) == 1
    rows = [r for r in db.recent_timings(alias="alpha", min_gen_tokens=1) if r["backend"] == backend]
    assert sorted(r["gen_tps"] for r in rows) == [100.0, 120.0]


def test_a_different_model_load_line_in_the_tail_vetoes_the_stored_instance():
    """Port reused by model B between ingests: B's requests must not land in A's stats."""
    db.init()
    backend = "test-904-veto"
    _, configs = telemetry.parse_log("\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")))
    db.record_server_configs(backend, configs)
    b_tail = ["2026-10-05T10:20:00.000000000Z [60279] 0.00.100 I srv    load_model: loading model '/models/b.gguf'"]
    b_tail += _request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0)
    samples, _ = telemetry.parse_log("\n".join(b_tail), backend, _COVERED)
    assert [s.instance for s in samples] == ["60279"] and samples[0].alias == ""
    # the same model reloading is still A
    a_tail = [b_tail[0].replace("/models/b.gguf", "/models/a.gguf")] + b_tail[1:]
    samples, _ = telemetry.parse_log("\n".join(a_tail), backend, _COVERED)
    assert samples[0].alias == "alpha" and "@" in samples[0].instance


def test_a_tail_that_does_not_overlap_the_last_ingest_stays_bare():
    db.init()
    backend = "test-904-gap"
    _, configs = telemetry.parse_log("\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")))
    db.record_server_configs(backend, configs)
    tail = "\n".join(_request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0))
    last_ingest = telemetry._parse_ts("2026-10-05T10:00:05.000000000Z ")
    for covered in (0.0, last_ingest):                       # never ingested / gap before the tail
        s, _ = telemetry.parse_log(tail, backend, covered)
        assert s[0].instance == "60279" and s[0].alias == ""
    s, _ = telemetry.parse_log(tail, backend, telemetry._parse_ts("2026-10-05T10:30:00.000000000Z "))
    assert s[0].alias == "alpha"                             # first line at or before the watermark


def test_ingest_stays_bare_across_a_gap_and_advances_the_watermark(monkeypatch):
    from app import services
    db.init()
    backend = "test-904-ingest-gap"
    logs = {"text": "\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")
                              + _request("2026-10-05T10:00:05.000000000Z", 60279, 3, 100.0))}

    class _Container:
        def logs(self, **_kw):
            return logs["text"].encode()

    class _Client:
        class containers:
            @staticmethod
            def get(_name):
                return _Container()

    monkeypatch.setattr(services, "_docker_client", lambda: _Client())
    telemetry.ingest([backend], force=True)
    assert float(db.get_setting(telemetry._WATERMARK_KEY + backend)) == telemetry._parse_ts("2026-10-05T10:00:05.000000000Z ")
    logs["text"] = "\n".join(_request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0))   # no overlap
    assert telemetry.ingest([backend], force=True) == 1
    rows = [r for r in db.recent_timings(alias="alpha", min_gen_tokens=1) if r["backend"] == backend]
    assert [r["gen_tps"] for r in rows] == [100.0]            # the orphan is not credited to alpha


def _fake_logs(monkeypatch):
    from app import services
    logs = {"text": ""}

    class _Container:
        def logs(self, **_kw):
            return logs["text"].encode()

    class _Client:
        class containers:
            @staticmethod
            def get(_name):
                return _Container()

    monkeypatch.setattr(services, "_docker_client", lambda: _Client())
    return logs


def _alpha_rows(backend):
    return sorted(r["gen_tps"] for r in db.recent_timings(alias="alpha", min_gen_tokens=1)
                  if r["backend"] == backend)


def test_a_pass_after_a_gap_does_not_vouch_for_instances_older_than_the_gap(monkeypatch):
    """The pass that follows a gap advances the watermark; the NEXT overlapping pass must not
    treat the gap as covered (the port may have been respawned unseen inside it)."""
    db.init()
    backend = "test-904-since-gap"
    logs = _fake_logs(monkeypatch)
    logs["text"] = "\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")
                             + _request("2026-10-05T10:00:05.000000000Z", 60279, 3, 100.0))
    telemetry.ingest([backend], force=True)
    logs["text"] = "\n".join(_request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0))      # gap
    telemetry.ingest([backend], force=True)
    assert float(db.get_setting(telemetry._CONTINUOUS_KEY + backend)) == telemetry._parse_ts("2026-10-05T10:30:00.000000000Z ")
    logs["text"] = "\n".join(_request("2026-10-05T10:30:00.000000000Z", 60279, 4, 120.0)         # overlaps
                             + _request("2026-10-05T10:40:00.000000000Z", 60279, 5, 130.0))
    telemetry.ingest([backend], force=True)
    assert _alpha_rows(backend) == [100.0]
    # a gap-free pass leaves the span start alone
    assert float(db.get_setting(telemetry._CONTINUOUS_KEY + backend)) == telemetry._parse_ts("2026-10-05T10:30:00.000000000Z ")


def test_first_ingest_after_deploy_does_not_trust_instances_stored_before_it(monkeypatch):
    db.init()
    backend = "test-904-since-deploy"
    logs = _fake_logs(monkeypatch)
    _, configs = telemetry.parse_log("\n".join(_spawn("2026-10-05T10:00:00.000000000Z", "alpha", 60279, "/models/a.gguf")))
    db.record_server_configs(backend, configs)                    # stored by the pre-deploy version
    logs["text"] = "\n".join(_request("2026-10-05T10:20:00.000000000Z", 60279, 4, 120.0))   # no watermark yet
    telemetry.ingest([backend], force=True)
    logs["text"] = "\n".join(_request("2026-10-05T10:20:00.000000000Z", 60279, 4, 120.0)
                             + _request("2026-10-05T10:25:00.000000000Z", 60279, 5, 130.0))
    telemetry.ingest([backend], force=True)
    assert _alpha_rows(backend) == []
