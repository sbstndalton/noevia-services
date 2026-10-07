"""Differential test (#964): the model-files binary (sbstndalton/noevia-rs) and the Python
reference `files_from_tree_py` must agree on the shared fixture table
(tests/fixtures/model-files.v1.json, the same file as noevia-rs's
crates/model-files/tests/fixtures/) and on a seeded random corpus generated here.

The Python half (the reference still produces every committed expectation) always runs. The
binary half runs only when MODEL_FILES_BIN points at a built model-files (CI builds it at the
Dockerfile's NOEVIA_RS_REF); skipped otherwise. Synthetic only: invented paths and sizes."""
from __future__ import annotations

import json
import os
import random
import subprocess
import unicodedata
from pathlib import Path

import pytest

from app.model_files import files_from_tree_py

FIXTURES = Path(__file__).parent / "fixtures" / "model-files.v1.json"
BIN = os.environ.get("MODEL_FILES_BIN", "")
needs_bin = pytest.mark.skipif(not BIN, reason="MODEL_FILES_BIN not set: no model-files binary to compare")


def _cases() -> list[dict]:
    return json.loads(FIXTURES.read_text())["cases"]


def _python(text: str) -> dict:
    try:
        return {"files": files_from_tree_py(json.loads(text))}
    except Exception as e:  # noqa: BLE001 - any exception is the reference refusing the input
        return {"error": type(e).__name__}


def _rust(text: str) -> dict:
    proc = subprocess.run([BIN, "tree"], input=text.encode(), capture_output=True, timeout=30, check=False)
    if proc.returncode != 0:
        assert proc.stdout == b"", "an error must leave stdout empty"
        return {"error": proc.stderr.decode(errors="replace")}
    return json.loads(proc.stdout)


def _agree(py: dict, rs: dict) -> bool:
    return ("error" in py and "error" in rs) or py == rs


def test_python_reference_still_produces_every_expectation():
    cases = _cases()
    assert len(cases) >= 500
    for c in cases:
        assert _python(c["input"]) == c["expect"], c["name"]


@needs_bin
def test_binary_tables_match_this_python():
    out = subprocess.run([BIN, "unicode-version"], capture_output=True, text=True, check=True).stdout.strip()
    assert out == unicodedata.unidata_version == json.loads(FIXTURES.read_text())["unidata_version"]


@needs_bin
def test_binary_agrees_on_every_fixture():
    cases = _cases()
    agreed = sum(_agree(c["expect"], _rust(c["input"])) for c in cases)
    assert agreed == len(cases)
    print(f"model-files fixtures: {agreed}/{len(cases)} agree")


_ALPHABET = list("-._/ QqIiFfBbPpKkMmSsXxo0123456789") + [
    "İ", "ı", "ſ", "K", "٤", "٠", "４", "²", "é", "́",
    "\u0085", "　", " ", "\n", "\t", "-of-", "-00001-of-00002", "-١١١١١-of-",
    ".gguf", ".GGUF", "mmproj", "tokenizer.model", "toKenizer.model", "chat_template.jinja",
    "Q4_K_M", "IQ2_XXS", "BF16", "F16", "fp8", "FP4", "\U0001d7cf", "\U0001f600", "\\", "\""]
_SIZES = [0, 1, -7, 2**63, 10**30, 1.5, -2.5, 1e20, True, False, None, "", "12", " 3 ", "1_0", "_1",
          "٣٣", "\u00857", "x", [], [0], {}, {"a": 1}]


def _random_listing(rng: random.Random) -> list:
    entries = []
    for _ in range(rng.randint(0, 6)):
        r = rng.random()
        if r < 0.03:
            entries.append(rng.choice([1, "x", None, [], True]))
            continue
        e: dict = {}
        e["type"] = rng.choice(["file", "file", "file", "directory", 1, None])
        if rng.random() < 0.95:
            path = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 10)))
            if rng.random() < 0.6:
                path += rng.choice([".gguf", "-00002-of-00003.gguf", ".GGUF", "mmproj.gguf", ".mmproj", ""])
            e["path"] = path if rng.random() < 0.97 else rng.choice([5, None, ["a.gguf"]])
        if rng.random() < 0.8:
            e["size"] = rng.choice(_SIZES)
        if rng.random() < 0.4:
            e["lfs"] = rng.choice([{"size": rng.choice(_SIZES)}, {}, {"oid": "x"}, "x", [], 0, None])
        entries.append(e)
    return entries


@needs_bin
def test_binary_agrees_on_a_seeded_random_corpus():
    rng = random.Random(964_2026)
    texts = [json.dumps(_random_listing(rng)) for _ in range(1500)]
    # One process per batch would hide which listing diverged; one per listing is still fast.
    disagreements = [t for t in texts if not _agree(_python(t), _rust(t))]
    assert disagreements == [], disagreements[:3]
    print(f"model-files random corpus: {len(texts)}/{len(texts)} agree")
