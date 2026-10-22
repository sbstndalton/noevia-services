"""Production image wiring for the qualified native service binary."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NATIVE_CAPABILITIES = (
    "native-logs", "native-host", "native-sections", "native-prompts",
    "native-diagnose", "native-settings", "native-model-updates", "native-model-detail",
    "native-download-targets", "native-badges", "native-bench-runs", "native-section-writes",
    "native-settings-write", "native-backend-control", "native-benchmark-overview",
    "native-check-updates", "native-benchmark-runner", "native-downloads", "native-model-delete",
    "native-draft-heads", "native-backends", "native-model-list", "native-search",
    "native-section-autoconfig", "native-telemetry", "native-bench-observe", "native-db-init",
    "native-token-test", "native-safe-defaults", "native-download-plan", "native-routing",
)

# Frozen supervisor selectors kept only in the explicit legacy qualification stages.
RETIRED_SELECTORS = (
    "MODEL_MANAGER_IMPL", "MODEL_FILES_IMPL", "MODEL_MANAGER_BACKENDS_IMPL",
    "MODEL_MANAGER_BACKEND_CONTROL_IMPL", "MODEL_MANAGER_BADGES_IMPL",
    "MODEL_MANAGER_BENCHMARK_OBSERVE_IMPL", "MODEL_MANAGER_BENCHMARK_OVERVIEW_IMPL",
    "MODEL_MANAGER_BENCHMARK_RUNNER_IMPL", "MODEL_MANAGER_BENCH_RUNS_IMPL",
    "MODEL_MANAGER_CHECK_UPDATES_IMPL", "MODEL_MANAGER_DB_INIT_IMPL",
    "MODEL_MANAGER_DIAGNOSE_IMPL", "MODEL_MANAGER_DOWNLOADS_IMPL",
    "MODEL_MANAGER_DOWNLOAD_PLAN_IMPL", "MODEL_MANAGER_DOWNLOAD_TARGETS_IMPL",
    "MODEL_MANAGER_DRAFT_HEADS_IMPL", "MODEL_MANAGER_HOST_IMPL", "MODEL_MANAGER_MODELS_IMPL",
    "MODEL_MANAGER_MODEL_DELETE_IMPL", "MODEL_MANAGER_MODEL_DETAIL_IMPL",
    "MODEL_MANAGER_MODEL_UPDATES_IMPL", "MODEL_MANAGER_OVERVIEW_IMPL",
    "MODEL_MANAGER_PROMPTS_IMPL", "MODEL_MANAGER_ROUTING_IMPL",
    "MODEL_MANAGER_SAFE_DEFAULTS_IMPL", "MODEL_MANAGER_SEARCH_IMPL",
    "MODEL_MANAGER_SECTIONS_IMPL", "MODEL_MANAGER_SECTION_AUTOCONFIG_IMPL",
    "MODEL_MANAGER_SECTION_WRITES_IMPL", "MODEL_MANAGER_SETTINGS_IMPL",
    "MODEL_MANAGER_SETTINGS_WRITE_IMPL", "MODEL_MANAGER_TELEMETRY_IMPL",
    "MODEL_MANAGER_TOKEN_TEST_IMPL", "GGUF_PARSER", "MODEL_AUTOCONFIG",
)


def docker_stage(source, name):
    """Return one Docker stage body, bounded by its FROM and the next FROM."""
    stages = list(re.finditer(r"^FROM .+ AS ([A-Za-z0-9_-]+)\s*$", source, re.M))
    matches = [index for index, match in enumerate(stages) if match.group(1) == name]
    assert len(matches) == 1, (name, [match.group(1) for match in stages])
    index = matches[0]
    start = stages[index].start()
    end = stages[index + 1].start() if index + 1 < len(stages) else len(source)
    return source[start:end]


def test_native_image_build_copy_capability_and_default_entrypoint():
    source = (ROOT / "model-manager" / "Dockerfile").read_text()
    builder = docker_stage(source, "gguf-meta")
    legacy = docker_stage(source, "legacy-runtime")
    qualification = docker_stage(source, "qualification")
    runtime = docker_stage(source, "runtime")
    production = docker_stage(source, "production")
    assert production.strip() == "FROM runtime AS production"
    assert len(NATIVE_CAPABILITIES) == len(set(NATIVE_CAPABILITIES)) == 31
    assert len(RETIRED_SELECTORS) == len(set(RETIRED_SELECTORS)) == 35
    native_qualification = docker_stage(source, "native-qualification")

    # Verify the complete native build and its advertised capability inventory. native-logs is
    # unconditional; each other marker is selected by an explicit Cargo feature.
    assert "native_features=$(./target/release/noevia-model-manager --features)" in builder
    build = next(line for line in builder.splitlines() if "cargo build --release --locked" in line)
    assert "-p gguf-meta -p model-files-cli -p model-autoconfig-cli -p noevia-model-manager" in build
    # The inference gate (noevia#2115) is its own cargo invocation: building it beside the model
    # manager would unify its dependencies' features into the model manager's binary.
    assert "noevia-inference-gate" not in build
    gate_builds = [line for line in builder.splitlines() if "cargo build" in line and "noevia-inference-gate" in line]
    assert gate_builds == [" && cargo build --release --locked -p noevia-inference-gate \\"]
    assert '| grep -q "GATE_UPSTREAM is required"' in builder
    for capability in NATIVE_CAPABILITIES:
        assert f"printf '%s\\n' \"$native_features\" | grep -Fx {capability} >/dev/null" in builder
        if capability != "native-logs":
            assert f"noevia-model-manager/{capability}" in builder

    # The frozen Python supervisor and helpers are a test oracle, not the shipping runtime.
    assert re.search(r"^FROM python:3\.12(?:\.\d+)?-slim AS legacy-runtime$", legacy, re.M)
    assert "ENV MODEL_MANAGER_SECTIONS_IMPL=python MODEL_MANAGER_SETTINGS_IMPL=python" in legacy
    assert "COPY requirements.txt ." in legacy
    assert "COPY app ./app" in legacy
    for helper in ("gguf-meta", "model-files", "model-autoconfig"):
        assert f"COPY --from=legacy-builder /src/target/release/{helper} /usr/local/bin/{helper}" in legacy
    assert 'HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c ' in legacy
    assert 'CMD ["noevia-model-manager"]' in legacy

    # Keep the frozen omission probes and native test-driver target explicit. Multiple CMDs and
    # helper copies can exist in test stages without entering the production image.
    assert "FROM legacy-runtime AS qualification" in qualification
    for probe in ("sections-image-probe", "prompts-image-probe", "diagnosis-image-probe"):
        assert f"COPY --from=qualification-builder /tmp/{probe} /usr/local/bin/{probe}" in qualification
    for omission in (
        "model-manager-sections-omission",
        "model-manager-prompts-omission",
        "model-manager-diagnosis-omission",
    ):
        assert f"/usr/local/bin/{omission}" in qualification
    assert "FROM runtime AS native-qualification" in native_qualification
    assert "COPY --from=native-qualification-builder /tmp/native-service-test /usr/local/bin/native-service-test" in native_qualification
    assert 'CMD ["native-service-test", "--test-threads=1"]' in native_qualification
    assert "ENV MODEL_MANAGER_PRODUCTION_BIN=/usr/local/bin/noevia-model-manager" in native_qualification

    # Scope retired selectors and non-native runtime checks to the final shipping stage.
    assert re.search(
        r"^FROM debian:bookworm-slim@sha256:[0-9a-f]{64} AS runtime$", runtime, re.M
    )
    for selector in RETIRED_SELECTORS:
        assert selector not in runtime
    assert not re.search(
        r"(?i)\bpython(?:[0-9]+(?:\.[0-9]+)*)?\b|uvicorn|/srv/app|/usr/local/bin/(?:gguf-meta|model-files|model-autoconfig)\b",
        runtime,
    )
    assert "apt-get install -y --no-install-recommends ca-certificates libgcc-s1" in runtime
    assert re.findall(r"^COPY .+$", runtime, re.M) == [
        "COPY --from=gguf-meta /src/target/release/noevia-model-manager /usr/local/bin/noevia-model-manager",
        "COPY --from=gguf-meta /src/target/release/noevia-inference-gate /usr/local/bin/noevia-inference-gate",
        "COPY LICENSE ./",
    ]
    assert 'HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["noevia-model-manager", "--healthcheck"]' in runtime
    assert 'CMD ["noevia-model-manager"]' in runtime


def test_services_share_a_verified_rust_source_pin():
    pins = []
    for service in ("model-manager", "ocr", "diary", "docling", "laya"):
        source = (ROOT / service / "Dockerfile").read_text()
        ref = re.findall(r"^ARG NOEVIA_RS_REF=([0-9a-f]{40})$", source, re.M)
        digest = re.findall(r"^ARG NOEVIA_RS_SHA256=([0-9a-f]{64})$", source, re.M)
        assert len(ref) == len(digest) == 1
        assert 'echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c -' in source
        pins.append((ref[0], digest[0]))
    assert len(set(pins)) == 1


def test_native_search_unicode_oracle_and_shipping_base_are_pinned():
    # Python 3.12 belongs to the frozen supervisor oracle, not the native Debian image. Keep
    # checking the actual test interpreter's Unicode database because its differential fixtures
    # reference the native search's pinned Unicode 15.0 behavior.
    import unicodedata

    source = (ROOT / "model-manager" / "Dockerfile").read_text()
    legacy = docker_stage(source, "legacy-runtime")
    runtime = docker_stage(source, "runtime")
    assert re.search(r"^FROM python:3\.12(?:\.\d+)?-slim AS legacy-runtime$", legacy, re.M)
    assert re.search(
        r"^FROM debian:bookworm-slim@sha256:[0-9a-f]{64} AS runtime$", runtime, re.M
    )
    assert unicodedata.unidata_version == "15.0.0"
