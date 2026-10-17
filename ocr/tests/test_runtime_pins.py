"""Merged S4 runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_s4():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "93adc34caaf879aa76c2a25ba72c7685ff9c6890",
        "NOEVIA_RS_SHA256": "8d53df7f0ab7d9777d8432f3776ea88d287166640ddc410c24386c9979d36cf1",
    }
    for service in ("ocr", "diary", "model-manager", "docling", "laya"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_s4()
    print("five merged native runtime defaults/checksum guards verified")
