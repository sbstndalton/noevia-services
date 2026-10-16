"""Merged S4 runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_s4():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "73d7753658a1982263a5bb5552d8d2d8b07917b9",
        "NOEVIA_RS_SHA256": "1200991ce8d703465143d45e29c91e32aff7991c490b71263ae46cb077e946f4",
    }
    for service in ("ocr", "diary", "model-manager", "docling", "laya"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_s4()
    print("five merged native runtime defaults/checksum guards verified")
