"""Qualification-only source restoration; synthetic archives, no runtime/model access."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

class Retirement(unittest.TestCase):
    def test_active_runtime_source_is_deleted(self):
        self.assertFalse((HERE.parent / "app").exists(), "retired application source remains")

    def test_archive_is_verified_before_any_write(self):
        restore = self.module()
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "bad.tar.gz"
            archive.write_bytes(b"invalid")
            dest = Path(root) / "output"
            with self.assertRaises(ValueError):
                restore.restore(archive, dest, {"ref": "synthetic", "sha256": "0" * 64, "files": {}})
            self.assertFalse(dest.exists())

    def module(self):
        spec = importlib.util.spec_from_file_location("restore", HERE / "restore_python_oracle.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, root, entries):
        archive = Path(root) / "oracle.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, value in entries:
                info = tarfile.TarInfo("noevia-services-synthetic/model-manager/app/" + name)
                if value is None:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "/outside"
                    tar.addfile(info)
                else:
                    info.size = len(value)
                    tar.addfile(info, io.BytesIO(value))
        manifest = {"ref": "synthetic", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "files": {"main.py": hashlib.sha256(b"# oracle\n").hexdigest()}}
        return archive, manifest

    def test_only_verified_regular_files_are_restored(self):
        restore = self.module()
        with tempfile.TemporaryDirectory() as root:
            archive, manifest = self.fixture(root, [("main.py", b"# oracle\n")])
            dest = Path(root) / "output"
            restore.restore(archive, dest, manifest)
            self.assertEqual((dest / "app/main.py").read_bytes(), b"# oracle\n")
            with self.assertRaises(FileExistsError):
                restore.restore(archive, dest, manifest)

    def test_incomplete_changed_link_duplicate_and_traversal_refuse_without_output(self):
        restore = self.module()
        cases = [[], [("main.py", b"changed")], [("main.py", None)],
                 [("../outside", b"bad")], [("main.py", b"# oracle\n")] * 2]
        for entries in cases:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as root:
                archive, manifest = self.fixture(root, entries)
                dest = Path(root) / "output"
                with self.assertRaises(ValueError):
                    restore.restore(archive, dest, manifest)
                self.assertFalse(dest.exists())

    def test_manifest_pins_docker_to_the_verified_oracle(self):
        manifest = json.loads((HERE / "python-oracle.json").read_text())
        self.assertEqual(len(manifest["files"]), 21)
        self.assertTrue(all(Path(name).name == name and name.endswith(".py")
                            for name in manifest["files"]))
        docker = (HERE.parent / "Dockerfile").read_text()
        self.assertIn("--checksum=sha256:" + manifest["sha256"], docker)
        self.assertIn("/tar.gz/" + manifest["ref"], docker)

    def test_production_has_no_oracle_dependency(self):
        source = (HERE.parent / "Dockerfile").read_text()
        production = source.split(" AS runtime\n", 1)[1].split("\nFROM ", 1)[0]
        self.assertNotIn("oracle", production)
        self.assertNotIn("python", production.lower())
        self.assertIn("restore_python_oracle.py", source)
        self.assertNotIn("COPY app ./app", source)

if __name__ == "__main__":
    unittest.main()
