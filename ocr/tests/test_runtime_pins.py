"""Merged S4 runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_s4():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "174078a20c214a609085bc44c1d668cd18a634eb",
        "NOEVIA_RS_SHA256": "edeab963f31832122a303999fcd5acbc46132a7b15e93f50758066c04ddf2773",
    }
    for service in ("ocr", "diary", "model-manager"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_s4()
    print("three merged native runtime defaults/checksum guards verified")
