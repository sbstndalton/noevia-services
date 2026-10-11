"""Merged S4 runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_s4():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "f81260f79028c1aafee20b642208196ab946b7e6",
        "NOEVIA_RS_SHA256": "d7af5543bf86b10558b8ed46bcaf289aaed32a034f0d220eda1cbca6dab9fd5b",
    }
    for service in ("ocr", "diary", "model-manager"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_s4()
    print("three merged native runtime defaults/checksum guards verified")
