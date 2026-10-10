"""M6 merged-runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_m6():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "b150831bdb1b00a6bc6f35315e50dad3e7c15e97",
        "NOEVIA_RS_SHA256": "b707258945f72cd9e2864fc0b1beed7a2264d117a9d2bd3a3f8d1eef8157344f",
    }
    for service in ("ocr", "diary", "model-manager"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_m6()
    print("three merged native runtime defaults/checksum guards verified")
