"""Merged S4 runtime defaults remain consistent across native sidecar builds."""
import re
from pathlib import Path


def test_native_sidecar_runtime_defaults_match_merged_s4():
    root = Path(__file__).resolve().parents[2]
    expected = {
        "NOEVIA_RS_REF": "14899676cd6f654eff57680b6e8a2c637b0ee650",
        "NOEVIA_RS_SHA256": "08c59c1581c7a4ed841707b8bf0e3808c164003127db807d7830d532090b539b",
    }
    for service in ("ocr", "diary", "model-manager", "docling", "laya"):
        text = (root / service / "Dockerfile").read_text()
        pins = dict(re.findall(r"^ARG (NOEVIA_RS_(?:REF|SHA256))=([0-9a-f]+)$", text, re.M))
        assert pins == expected, service
        assert 'sha256sum -c -' in text, service


if __name__ == "__main__":
    test_native_sidecar_runtime_defaults_match_merged_s4()
    print("five merged native runtime defaults/checksum guards verified")
