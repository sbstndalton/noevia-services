"""Merged S1 runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_s1():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "90357b9512db69ad81b1f4f1f1967f135f6883dc",
        "NOEVIA_RS_SHA256": "f2b8852b4b7882b2733f391b5c872828d4a2802d152a3ebcb5ad4f1d526d2217",
    }
    for service in ("ocr", "diary", "model-manager"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_s1()
    print("three merged native runtime defaults/checksum guards verified")
