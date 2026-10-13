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
    # #1334: the section switch is wired in the image, and its default stays python.
    assert "ENV MODEL_MANAGER_SECTIONS_IMPL=python" in source
    assert not re.search(r"^ENV .*MODEL_MANAGER_SECTIONS_IMPL=rust", source, re.M)
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
