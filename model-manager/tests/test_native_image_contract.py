"""Production image wiring for the qualified native service binary."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_native_image_build_copy_capability_and_default_entrypoint():
    source = (ROOT / "model-manager" / "Dockerfile").read_text()
    assert "-p noevia-model-manager --features noevia-model-manager/native-host,noevia-model-manager/native-sections" in source
    assert "native_features=$(./target/release/noevia-model-manager --features)" in source
    for capability in ("native-logs", "native-host", "native-sections"):
        assert f"printf '%s\\n' \"$native_features\" | grep -Fx {capability} >/dev/null" in source
    assert "COPY --from=gguf-meta /src/target/release/noevia-model-manager /usr/local/bin/noevia-model-manager" in source
    # noevia#1071: the image always runs the native parent (pinned for older binaries).
    assert re.findall(r"^ENV MODEL_MANAGER_IMPL=(.*)$", source, re.M) == ["rust"]
    assert "MODEL_FILES_IMPL" not in source
    # #1334: the section switch is wired in the image, and its default stays python.
    assert "ENV MODEL_MANAGER_SECTIONS_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_SECTIONS_IMPL=rust", source, re.M)
    # #1318: the model-updates read is compiled in and advertised, and its switch defaults to python.
    assert "noevia-model-manager/native-model-updates" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-model-updates >/dev/null" in source
    assert "ENV MODEL_MANAGER_MODEL_UPDATES_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_MODEL_UPDATES_IMPL=rust", source, re.M)
    # #1318: the model-detail read is compiled in and advertised, and its switch defaults to python.
    assert "noevia-model-manager/native-model-detail" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-model-detail >/dev/null" in source
    assert "ENV MODEL_MANAGER_MODEL_DETAIL_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_MODEL_DETAIL_IMPL=rust", source, re.M)
    # #1318: the download-targets read is compiled in and advertised, and its switch defaults to python.
    assert "noevia-model-manager/native-download-targets" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-download-targets >/dev/null" in source
    assert "ENV MODEL_MANAGER_DOWNLOAD_TARGETS_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_DOWNLOAD_TARGETS_IMPL=rust", source, re.M)
    # #1318: the badge writes are compiled in and advertised, and their switch defaults to python.
    assert "noevia-model-manager/native-badges" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-badges >/dev/null" in source
    assert "ENV MODEL_MANAGER_BADGES_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_BADGES_IMPL=rust", source, re.M)
    # #1318: the benchmark run read is compiled in and advertised, and its switch defaults to python.
    assert "noevia-model-manager/native-bench-runs" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-bench-runs >/dev/null" in source
    assert "ENV MODEL_MANAGER_BENCH_RUNS_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_BENCH_RUNS_IMPL=rust", source, re.M)
    # #1318: the native models.ini writers are compiled in and advertised, and default to python.
    assert "noevia-model-manager/native-section-writes" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-section-writes >/dev/null" in source
    assert "ENV MODEL_MANAGER_SECTION_WRITES_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_SECTION_WRITES_IMPL=rust", source, re.M)
    # #1318: the settings writer is compiled in and advertised, and defaults to python.
    assert "noevia-model-manager/native-settings-write" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-settings-write >/dev/null" in source
    assert "ENV MODEL_MANAGER_SETTINGS_WRITE_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_SETTINGS_WRITE_IMPL=rust", source, re.M)
    # #1318: native backend control is compiled in and advertised, and defaults to python.
    assert "noevia-model-manager/native-backend-control" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-backend-control >/dev/null" in source
    assert "ENV MODEL_MANAGER_BACKEND_CONTROL_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_BACKEND_CONTROL_IMPL=rust", source, re.M)
    # #1318: the native benchmark overview is compiled in and advertised, and defaults to python.
    assert "noevia-model-manager/native-benchmark-overview" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-benchmark-overview >/dev/null" in source
    assert "ENV MODEL_MANAGER_BENCHMARK_OVERVIEW_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_BENCHMARK_OVERVIEW_IMPL=rust", source, re.M)
    # #1318: the native update check is compiled in and advertised, and defaults to python.
    assert "noevia-model-manager/native-check-updates" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-check-updates >/dev/null" in source
    assert "ENV MODEL_MANAGER_CHECK_UPDATES_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_CHECK_UPDATES_IMPL=rust", source, re.M)
    # #1318: the native benchmark runner (the whole job slot) is compiled in and advertised, and
    # defaults to python.
    assert "noevia-model-manager/native-benchmark-runner" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-benchmark-runner >/dev/null" in source
    assert "ENV MODEL_MANAGER_BENCHMARK_RUNNER_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_BENCHMARK_RUNNER_IMPL=rust", source, re.M)
    assert "noevia-model-manager/native-downloads" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-downloads >/dev/null" in source
    assert "ENV MODEL_MANAGER_DOWNLOADS_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_DOWNLOADS_IMPL=rust", source, re.M)
    # #1318: native model deletion is compiled in and advertised, and defaults to python.
    assert "noevia-model-manager/native-model-delete" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-model-delete >/dev/null" in source
    assert "ENV MODEL_MANAGER_MODEL_DELETE_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_MODEL_DELETE_IMPL=rust", source, re.M)
    # #1318: the native draft-head read is compiled in and advertised, and defaults to python.
    assert "noevia-model-manager/native-draft-heads" in source
    assert "printf '%s\\n' \"$native_features\" | grep -Fx native-draft-heads >/dev/null" in source
    assert "ENV MODEL_MANAGER_DRAFT_HEADS_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_DRAFT_HEADS_IMPL=rust", source, re.M)
    command = re.findall(r"^CMD (.+)$", source, re.M)
    assert len(command) == 1 and json.loads(command[0]) == ["noevia-model-manager"]
    for helper in ("gguf-meta", "model-files", "model-autoconfig"):
        assert f"COPY --from=gguf-meta /src/target/release/{helper} /usr/local/bin/{helper}" in source


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
