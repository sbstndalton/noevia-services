"""Reproducibility guard for the docling image build (#890).

A plain `docker build` that missed the layer cache once began re-resolving an unpinned torch and
re-downloading the models. These tests fail if that door is reopened: every locked requirement must
be pinned with `==` and carry a sha256 hash, torch must stay on the CPU index build, the Dockerfile
must install only through the hashed lock files, and the model download must be pinned to commits.
They are static (no network, no docling install), so CI runs them with the rest of this folder.
"""

import re
from pathlib import Path

HERE = Path(__file__).parent
LOCK_FILES = ["requirements.txt", "requirements-torch-cpu.txt"]
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _logical_lines(path: Path):
    """Requirement lines with backslash continuations joined and comments/blank lines dropped."""
    out, buf = [], ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1].strip() + " "
            continue
        out.append((buf + line).strip())
        buf = ""
    assert not buf, f"{path.name} ends in a dangling continuation"
    return out


def _requirements(path: Path):
    reqs = []
    for line in _logical_lines(path):
        if line.startswith("-"):
            continue  # pip options such as --index-url
        reqs.append(line)
    return reqs


def test_every_locked_requirement_is_pinned_with_a_hash():
    for name in LOCK_FILES:
        reqs = _requirements(HERE / name)
        assert reqs, f"{name} has no requirements"
        for line in reqs:
            spec = line.split()[0]
            assert re.fullmatch(r"[A-Za-z0-9_.\-]+(\[[^\]]+\])?==[A-Za-z0-9_.!+\-]+", spec), (
                f"{name}: {spec!r} is not an exact `name==version` pin"
            )
            assert "--hash=sha256:" in line, f"{name}: {spec} has no sha256 hash"
            for h in re.findall(r"--hash=sha256:(\w+)", line):
                assert re.fullmatch(r"[0-9a-f]{64}", h), f"{name}: {spec} has a malformed hash"


def test_no_unpinned_package_in_any_requirements_file():
    """Catches a new requirements*.txt (or an edit) that lists a package without `==`."""
    files = sorted(HERE.glob("requirements*.txt"))
    assert {f.name for f in files} >= set(LOCK_FILES)
    for f in files:
        for line in _requirements(f):
            assert "==" in line.split()[0], f"{f.name}: unpinned requirement {line.split()[0]!r}"


def test_torch_is_the_cpu_build_and_cuda_wheels_are_absent():
    torch = {
        line.split("==")[0]: line.split()[0] for line in _requirements(HERE / "requirements-torch-cpu.txt")
    }
    assert set(torch) >= {"torch", "torchvision"}
    for name, spec in torch.items():
        assert spec.endswith("+cpu"), f"{spec} is not the +cpu build; PyPI's torch pulls multi-GB CUDA wheels"
    names = {line.split("==")[0].lower() for line in _requirements(HERE / "requirements.txt")}
    assert "torch" not in names and "torchvision" not in names
    cuda = {n for n in names if n.startswith(("nvidia-", "cuda-")) or n == "triton"}
    assert not cuda, f"CUDA packages leaked into the CPU image lock: {sorted(cuda)}"


def test_docling_version_matches_between_input_lock_and_dockerfile():
    pins = {}
    for name in ("requirements.in", "requirements.txt"):
        text = (HERE / name).read_text(encoding="utf-8")
        m = re.search(r"^docling-slim(?:\[[^\]]*\])?==([\w.]+)", text, re.M)
        assert m, f"{name} does not pin docling-slim"
        pins[name] = m.group(1)
    assert len(set(pins.values())) == 1, pins


def test_dockerfile_installs_only_from_the_hashed_locks():
    text = (HERE / "Dockerfile").read_text(encoding="utf-8")
    # Join RUN continuations, drop comments, then look at each pip invocation.
    code = "\n".join(l for l in text.replace("\\\n", " ").splitlines() if not l.lstrip().startswith("#"))
    installs = re.findall(r"pip install[^\n]*", code)
    assert installs, "Dockerfile has no pip install"
    for cmd in installs:
        assert "--require-hashes" in cmd and "--no-deps" in cmd, f"unhashed pip install: {cmd}"
        assert re.search(r"-r\s+\S*requirements[\w-]*\.txt", cmd), f"pip install not from a lock file: {cmd}"
    assert "docling-tools models download" not in code, "layout resolves at floating revision 'main'"
    assert "download_models.py" in code


def test_model_downloads_are_pinned_to_commits():
    ns = {}
    exec(compile((HERE / "download_models.py").read_text(encoding="utf-8"), "download_models.py", "exec"), ns)
    pins = ns["PINS"]
    assert {"docling-project/docling-models", "docling-project/docling-layout-heron"} <= set(pins)
    for repo, (_revision, commit) in pins.items():
        assert HEX40.match(commit), f"{repo} is not pinned to a 40-hex commit: {commit!r}"


def test_overlay_docs_list_exactly_the_files_the_dockerfile_copies_to_app():
    """docs/deployment.md's app-only overlay must replace the same files the final COPY places."""
    docker = (HERE / "Dockerfile").read_text(encoding="utf-8")
    copies = re.findall(r"^COPY ((?:[\w.]+ )+)\./\s*$", docker, re.M)
    assert len(copies) == 1, "expected one `COPY <files> ./` into /app"
    app_files = set(copies[0].split())
    docs = (HERE.parent.parent / "docs" / "deployment.md").read_text(encoding="utf-8")
    copy_lines = re.findall(r"^COPY ((?:[\w.]+ )+)/app/", docs.replace("\\n", "\n"), re.M)
    assert copy_lines, "docs/deployment.md has no docling overlay `COPY ... /app/` line"
    for line in copy_lines:
        assert set(line.split()) == app_files, (set(line.split()) ^ app_files)
