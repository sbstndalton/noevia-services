"""The Diary image installs exact dependency versions (sbstndalton/noevia#2523).

requirements.txt names what the Diary needs with ranges; constraints.txt pins the version of every
distribution the image installs, so a rebuild cannot move a dependency by itself. These tests fail
when a requirement, or anything it pulls in, has no exact pin, when a pin falls outside the range
requirements.txt allows, or when the Dockerfile installs without the constraints. The static checks
need no network; the closure check reads the installed metadata (CI installs with the constraints).
"""

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


def test_dockerfile_installs_only_with_the_constraints():
    text = (HERE / "Dockerfile").read_text(encoding="utf-8")
    code = "\n".join(l for l in text.replace("\\\n", " ").splitlines() if not l.lstrip().startswith("#"))
    final = re.split(r"^FROM python:", code, flags=re.M)
    assert len(final) == 2, "expected one final FROM python: stage"
    stage = final[1]
    copies = re.findall(r"^COPY ([^\n]*)$", stage, re.M)
    assert any("constraints.txt" in c.split() for c in copies), "the final stage never copies constraints.txt"
    installs = [cmd for run in re.findall(r"^RUN ([^\n]*)$", stage, re.M) for cmd in run.split("&&") if "pip install" in cmd]
    assert installs, "Dockerfile has no pip install"
    for cmd in installs:
        assert re.search(r"(^|\s)-c\s+constraints\.txt(\s|$)", cmd), f"pip install without -c constraints.txt: {cmd.strip()}"


def _closure():
    """Every distribution requirements.txt pulls in (extras and markers honoured), from the installed metadata."""
    seen, todo = {}, [(r, tuple(sorted(r.extras))) for r in _requirements()]
    while todo:
        req, extras = todo.pop()
        name = canonicalize_name(req.name)
        try:
            dist = metadata.distribution(req.name)
        except metadata.PackageNotFoundError:
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
