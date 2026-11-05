"""Merged S4 runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_s4():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "273135977c1ef4ef14f79d0653cc8887dbccb316",
        "NOEVIA_RS_SHA256": "fddce4c8a51d62bd4b4fd2e64b988792dd62f89f42760219b61c34a561cabada",
    }
    for service in ("ocr", "diary", "model-manager", "docling", "laya"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_s4()
    print("five merged native runtime defaults/checksum guards verified")
