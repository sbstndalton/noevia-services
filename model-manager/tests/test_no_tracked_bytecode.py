"""noevia#1911: compiled Python must never be tracked or shipped in the image build context."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_pyc_is_tracked():
    tracked = subprocess.run(["git", "ls-files", "*.pyc", "**/__pycache__/**"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    assert tracked == []


def test_dockerignore_excludes_nested_bytecode():
    lines = (ROOT / "model-manager" / ".dockerignore").read_text().split()
    assert "**/__pycache__" in lines and "**/*.pyc" in lines
