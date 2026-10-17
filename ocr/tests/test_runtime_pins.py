"""Merged S4 runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_s4():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "460323a3d2d8fb7aa6dce39370570a3b3e3758fe",
        "NOEVIA_RS_SHA256": "94436759c54ecf1473ccc5a6fa328c47a62e7a7f6492fccdaac3cb07629af2c8",
    }
    for service in ("ocr", "diary", "model-manager", "docling", "laya"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_s4()
    print("five merged native runtime defaults/checksum guards verified")
