"""The Diary image installs exact dependency versions (sbstndalton/noevia#2523).

requirements.txt names what the Diary needs with ranges; constraints.txt pins the version of every
distribution the image installs, so a rebuild cannot move a dependency by itself. These tests fail
when a requirement, or anything it pulls in, has no exact pin, when a pin falls outside the range
requirements.txt allows, or when the Dockerfile installs without the constraints. The static checks
need no network; the closure check reads the installed metadata (CI installs with the constraints).
"""

import os
import re
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

HERE = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.!+_-]+)$")


def _lines(path: Path):
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _constraints(root=HERE):
    pins = {}
    for line in _lines(root / "constraints.txt"):
        m = PIN.fullmatch(line)
        assert m, f"constraints.txt: {line!r} is not an exact `name==version` pin"
        name = canonicalize_name(m.group(1))
        assert name not in pins, f"constraints.txt pins {name} twice"
        pins[name] = m.group(2)
    assert pins, "constraints.txt pins nothing"
    return pins


def _requirements(root=HERE):
    reqs = []
    for line in _lines(root / "requirements.txt"):
        assert not line.startswith("-"), f"requirements.txt: pip option {line!r} is not expected here"
        reqs.append(Requirement(line))
    assert reqs, "requirements.txt is empty"
    return reqs


def _check_requirement_pins(root):
    pins = _constraints(root)
    for req in _requirements(root):
        name = canonicalize_name(req.name)
        assert name in pins, f"requirements.txt: {req.name} has no pinned constraint in constraints.txt"
        assert req.specifier.contains(Version(pins[name]), prereleases=True), (
            f"constraints.txt pins {req.name}=={pins[name]}, outside requirements.txt's {req.specifier}"
        )


def test_every_requirement_has_an_exact_pin_inside_its_range():
    _check_requirement_pins(HERE)


def test_pip_itself_is_pinned_because_the_dockerfile_upgrades_it():
    assert "pip" in _constraints(), "the Dockerfile upgrades pip, so constraints.txt must pin it"


PIP_INSTALL = re.compile(r"(^|[\s/])(pip3?(\.\d+)?|python3?(\.\d+)?\s+-m\s+pip)\s+install(\s|$)")


def _final_stage_installs(text):
    """Every pip install command of the final stage. Comment lines go first (Docker drops them before it joins
    continuations), then each RUN is split at its shell separators. A RUN form this cannot read fails."""
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    code = "\n".join(lines).replace("\\\n", " ")
    final = re.split(r"(?mi)^\s*FROM\s+python:", code)
    assert len(final) == 2, "expected one final FROM python: stage"
    stage = final[1]
    copies = re.findall(r"(?mi)^\s*COPY\s+([^\n]*)$", stage)
    assert any("constraints.txt" in c.split() for c in copies), "the final stage never copies constraints.txt"
    installs = []
    for run in re.findall(r"(?mi)^\s*RUN\s+([^\n]*)$", stage):
        assert not run.lstrip().startswith("[") and "<<" not in run, f"RUN form the pin check cannot read: {run[:60]}"
        installs += [cmd for cmd in re.split(r"&&|\|\||;|\|", run) if PIP_INSTALL.search(cmd)]
    return installs


def _check_dockerfile(text):
    installs = _final_stage_installs(text)
    assert installs, "Dockerfile has no pip install"
    for cmd in installs:
        assert re.search(r"(^|\s)-c\s+constraints\.txt(\s|$)", cmd), f"pip install without -c constraints.txt: {cmd.strip()}"


def test_dockerfile_installs_only_with_the_constraints():
    _check_dockerfile((HERE / "Dockerfile").read_text(encoding="utf-8"))


def test_the_dockerfile_check_catches_every_unconstrained_install_form():
    head = "FROM python:3.12-slim\nCOPY requirements.txt constraints.txt ./\n"
    good = "RUN pip install -c constraints.txt -r requirements.txt\n"
    _check_dockerfile(head + good)
    for bad in ("RUN pip install -c constraints.txt a; pip install b\n",
                "RUN pip3 install -r requirements.txt\n",
                "RUN python -m pip install -r requirements.txt\n",
                "RUN true || pip install b\n",
                "RUN pip install -c constraints.txt a \\\n && /usr/local/bin/pip install b\n",
                "# a comment ending in a backslash \\\nRUN pip install b\n"):
        with pytest.raises(AssertionError, match="without -c"):
            _check_dockerfile(head + good + bad)
    for unreadable in ('RUN ["pip", "install", "b"]\n', "RUN <<EOF\npip install b\nEOF\n"):
        with pytest.raises(AssertionError, match="cannot read"):
            _check_dockerfile(head + good + unreadable)


def _closure():
    """Every distribution requirements.txt pulls in (extras and markers honoured), from the installed metadata."""
    seen, todo = {}, [(r, tuple(sorted(r.extras))) for r in _requirements()]
    while todo:
        req, extras = todo.pop()
        name = canonicalize_name(req.name)
        try:
            dist = metadata.distribution(req.name)
        except metadata.PackageNotFoundError:
            # CI sets DIARY_PINS_REQUIRED=1: there a missing distribution fails instead of skipping.
            if os.environ.get("DIARY_PINS_REQUIRED") == "1":
                raise AssertionError(f"{req.name} is required but not installed")
            pytest.skip(f"{req.name} is not installed here (pip install -c constraints.txt -r requirements-dev.txt)")
        key = (name, extras)
        if key in seen:
            continue
        seen[key] = dist.version
        for spec in dist.requires or []:
            sub = Requirement(spec)
            if sub.marker is not None and not any(sub.marker.evaluate({"extra": e}) for e in (extras or ("",))):
                continue
            todo.append((sub, tuple(sorted(sub.extras))))
    return {name: version for (name, _), version in seen.items()}


def test_everything_the_requirements_pull_in_is_pinned_and_installed_at_its_pin():
    pins = _constraints()
    for name, version in sorted(_closure().items()):
        assert name in pins, f"{name} (pulled in by requirements.txt) has no pinned constraint in constraints.txt"
        assert Version(version) == Version(pins[name]), f"{name} is installed at {version}, constraints.txt pins {pins[name]}"


def test_every_pin_is_something_the_requirements_pull_in():
    """The other direction: a pin nothing needs would be a production distribution a rebuild silently drops."""
    unused = set(_constraints()) - {"pip"} - set(_closure())
    assert not unused, f"constraints.txt pins distributions requirements.txt does not pull in: {sorted(unused)}"


def test_the_pin_check_catches_a_missing_a_ranged_and_an_out_of_range_pin(tmp_path):
    """The guard itself: a requirement without a pin, a non-exact pin and a pin outside the range all fail."""
    (tmp_path / "requirements.txt").write_text("fastapi>=0.115\nnewdep>=1\n", encoding="utf-8")
    cases = [
        ("fastapi==0.146.0\n", "newdep has no pinned constraint"),
        ("fastapi>=0.146\nnewdep==1.0\n", "not an exact"),
        ("fastapi==0.100.0\nnewdep==1.0\n", "outside requirements.txt"),
    ]
    for constraints, message in cases:
        (tmp_path / "constraints.txt").write_text(constraints, encoding="utf-8")
        with pytest.raises(AssertionError, match=message):
            _check_requirement_pins(tmp_path)
    (tmp_path / "constraints.txt").write_text("fastapi==0.146.0\nnewdep==1.0\n", encoding="utf-8")
    _check_requirement_pins(tmp_path)
