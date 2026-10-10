"""M6 merged-runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_m6():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "f7c90b8f50c13c0b338f0acd8ce41ffe22f17ef1",
        "NOEVIA_RS_SHA256": "e4bafed3f055b6d91fb1f5a58223dc50f65ca2f1c74823cee82f4b3933ea6256",
    }
    for service in ("ocr", "diary", "model-manager"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_m6()
    print("three merged native runtime defaults/checksum guards verified")
