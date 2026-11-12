"""Restore the immutable Python oracle for tests only, never the production image.

The archive and every selected file are verified before creating the destination.
No tar member paths, permissions, links or executable bits are extracted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def restore(archive, destination, manifest):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Python oracle archive checksum mismatch")
    prefix = f"noevia-services-{manifest['ref']}/model-manager/app/"
    files = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.name.startswith(prefix) or member.isdir():
                continue
            name = member.name[len(prefix):]
            if (name not in manifest["files"] or name in files or not member.isfile()
                    or member.size > 2 * 1024 * 1024):
                raise ValueError("Unexpected Python oracle member")
            source = tar.extractfile(member)
            data = source.read()
            if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
                raise ValueError("Python oracle file checksum mismatch")
            files[name] = data
    if files.keys() != manifest["files"].keys():
        raise ValueError("Incomplete Python oracle archive")
    # Never merge into or overwrite another source tree. mkdir also refuses symlinks.
    destination.mkdir(parents=True, exist_ok=True)
    app = destination / "app"
    app.mkdir(mode=0o755)
    for name, data in files.items():
        with (app / name).open("xb") as output:
            output.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("python-oracle.json").read_text())
    restore(args.archive, args.destination, manifest)
