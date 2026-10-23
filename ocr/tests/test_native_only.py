"""OCR is native only (sbstndalton/noevia#1071).

The image runs noevia-ocr directly: no entrypoint script, no interpreter, no Python OCR server and no
selector. The retired Python server stays the test oracle at services bbed9b8 (CI fetches that
checksummed archive); this tree must neither ship nor name it as a runtime path.
"""
import re
from pathlib import Path

OCR = Path(__file__).resolve().parents[1]
ROOT = OCR.parent
RETIRED_FILES = ("server.py", "docx_text.py", "pdf_reduce.py", "entrypoint.sh")
SELECTORS = ("NOEVIA_OCR_IMPL", "DOCX_TEXT_IMPL")


def stages(text):
    return re.split(r"(?m)^(?=FROM )", text)[1:]


def test_retired_files_are_gone():
    for name in RETIRED_FILES:
        assert not (OCR / name).exists(), name
    for name in ("test_server.py", "test_docx.py", "test_docx_text_switch.py", "test_pdf_reduce.py"):
        assert not (OCR / name).exists(), name


def test_runtime_stage_is_the_native_server_only():
    dockerfile = (OCR / "Dockerfile").read_text()
    runtime = stages(dockerfile)[-1]
    assert re.match(r"FROM debian:bookworm-slim@sha256:[0-9a-f]{64}\n", runtime)
    assert re.findall(r"(?m)^COPY .+$", runtime) == [
        "COPY --from=native /src/target/release/noevia-ocr /usr/local/bin/noevia-ocr",
    ]
    assert re.findall(r"(?m)^ENTRYPOINT .+$", runtime) == ['ENTRYPOINT ["/usr/local/bin/noevia-ocr"]']
    assert 'CMD ["/usr/local/bin/noevia-ocr", "--healthcheck"]' in runtime
    assert "USER 65534:65534" in runtime
    assert not re.search(r"(?im)\bpython[0-9.]*\b|\.py\b|/app\b|entrypoint\.sh", runtime)


def test_no_stage_ships_retired_files_or_fixes_a_selector():
    dockerfile = (OCR / "Dockerfile").read_text()
    for name in RETIRED_FILES:
        assert not re.search(rf"(?m)^COPY .*\b{re.escape(name)}\b", dockerfile), name
    for selector in SELECTORS:
        assert not re.search(rf"(?m)^(?:ENV|ARG) .*\b{selector}\b", dockerfile), selector
    build = stages(dockerfile)[0]
    assert "cargo build --release --locked -p noevia-ocr --features native-ocr" in build


def test_ci_tests_the_native_server_against_the_pinned_oracle_not_this_tree():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "NOEVIA_OCR_REFERENCE: ${{ runner.temp }}/ocr-oracle/ocr" in workflow
    assert "NOEVIA_OCR_REFERENCE: ${{ github.workspace }}/ws/services/ocr" not in workflow
    assert "tar.gz/bbed9b8a5af9f66ea3133fd8ee5495013d11d6c7" in workflow
    assert "d35a0351a98514411a0e87f112d041e53096b1cda6d3e3b409a4765f9601d9bb  /tmp/ocr-oracle.tgz" in workflow
    assert "test_docx_text_switch.py" not in workflow
