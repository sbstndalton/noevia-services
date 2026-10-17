"""Recorded-oracle test (#964): the model-files binary (sbstndalton/noevia-rs) must reproduce every
expectation in the shared fixture table (tests/fixtures/model-files.v1.json, the same file as
noevia-rs's crates/model-files/tests/fixtures/). The expectations were recorded from the Python
reference `files_from_tree_py`; noevia#1071 removed it from the service, and it stays available at
noevia-services 9c4bd37 for anyone regenerating the table (noevia-rs tools/gen-model-files.py).

Runs only when MODEL_FILES_BIN points at a built model-files (CI builds it at the Dockerfile's
NOEVIA_RS_REF); skipped otherwise. Synthetic only: invented paths and sizes."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "model-files.v1.json"
BIN = os.environ.get("MODEL_FILES_BIN", "")
needs_bin = pytest.mark.skipif(not BIN, reason="MODEL_FILES_BIN not set: no model-files binary to compare")


def _cases() -> list[dict]:
    return json.loads(FIXTURES.read_text())["cases"]


def _rust(text: str) -> dict:
    proc = subprocess.run([BIN, "tree"], input=text.encode(), capture_output=True, timeout=30, check=False)
    if proc.returncode != 0:
        assert proc.stdout == b"", "an error must leave stdout empty"
        return {"error": proc.stderr.decode(errors="replace")}
    return json.loads(proc.stdout)


def _agree(expect: dict, rs: dict) -> bool:
    return ("error" in expect and "error" in rs) or expect == rs


def test_fixture_table_is_the_recorded_oracle():
    f = json.loads(FIXTURES.read_text())
    assert len(f["cases"]) >= 500 and f["unidata_version"]
    assert all(("files" in c["expect"]) != ("error" in c["expect"]) for c in f["cases"])


@needs_bin
def test_binary_tables_match_the_recorded_oracle():
    out = subprocess.run([BIN, "unicode-version"], capture_output=True, text=True, check=True).stdout.strip()
    assert out == json.loads(FIXTURES.read_text())["unidata_version"]


@needs_bin
def test_binary_agrees_on_every_fixture():
    cases = _cases()
    agreed = sum(_agree(c["expect"], _rust(c["input"])) for c in cases)
    assert agreed == len(cases)
    print(f"model-files fixtures: {agreed}/{len(cases)} agree")


# #964 F3: inputs past the Rust caps are refused on purpose (fail closed; the removed Python
# reference accepted them). Kept out of the shared table.
_OVER_CAP = {
    "lone_surrogate_path": [{"type": "file", "path": "m-\ud800-Q4_0.gguf", "size": 1}],
    "path_over_4096_chars": [{"type": "file", "path": "a" * 4097 + "-Q4_0.gguf", "size": 1}],
}


@needs_bin
@pytest.mark.parametrize("name", sorted(_OVER_CAP))
def test_rust_refuses_inputs_past_its_caps_and_the_listing_fails_closed(name, monkeypatch):
    from app import config, model_files
    proc = subprocess.run([BIN, "tree"], input=json.dumps(_OVER_CAP[name]).encode(),
                          capture_output=True, timeout=30, check=False)
    assert proc.returncode == 1 and proc.stdout == b""
    monkeypatch.setattr(config.settings, "model_files_bin", BIN)
    with pytest.raises(model_files.ModelFilesError, match="rejected"):
        model_files.files_from_tree(_OVER_CAP[name])
