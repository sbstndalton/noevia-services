"""#798 / #801: deleting one model never takes another model's files with it, never removes the
models root, a configured download location or a mount point, and frees Hugging Face cache blobs
for nested quant folders. All files are synthetic and live under the test's temporary models dir."""
import os
import shutil

import pytest

from conftest import ROOT, _gguf
from app import services

MODELS = ROOT / "models"
MADE = ["dl-nest", "dl-a", "dl-mnt", "dl-archive", "dl-single", "dl-pair", "models--acme--nested-GGUF",
        "models--acme--shared-GGUF", "flat-dl-Q4.gguf", "dl-outside-target.txt"]


@pytest.fixture(autouse=True)
def _clean():
    yield
    for name in MADE:
        p = MODELS / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
    (ROOT / "dl-outside-target.txt").unlink(missing_ok=True)


def _model(path, pad=256):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_gguf({"general.architecture": "llama"}) + b"\0" * pad)
    return path


def _delete(rel):
    subdir, _, name = rel.rpartition("/")
    return services.delete_gguf(name, subdir)


# ---------- #798: nested models, protected folders ----------

def test_nested_model_survives_deleting_the_parent_folders_model():
    x = _model(MODELS / "dl-a" / "x-Q4.gguf")
    proj = _model(MODELS / "dl-a" / "mmproj-x-F16.gguf")
    y = _model(MODELS / "dl-a" / "old" / "y-Q4.gguf")
    own_size = x.stat().st_size + proj.stat().st_size
    ok, msg, freed = _delete("dl-a/x-Q4.gguf")
    assert ok, msg
    assert not x.exists() and not proj.exists(), "the model and its own projector go"
    assert y.exists(), "the nested model is a different model and must survive"
    assert freed == own_size


def test_nested_projector_only_folder_is_not_swept_up():
    _model(MODELS / "dl-a" / "x-Q4.gguf")
    nested = _model(MODELS / "dl-a" / "vision" / "mmproj-other-F16.gguf")
    ok, msg, _ = _delete("dl-a/x-Q4.gguf")
    assert ok, msg
    assert nested.exists()


def test_mount_point_is_never_removed(monkeypatch):
    mnt = MODELS / "dl-mnt"
    foo = _model(mnt / "foo-Q4.gguf")
    bar = _model(mnt / "foo-Q4" / "bar-Q4.gguf")
    note = mnt / "notes.txt"
    note.write_text("synthetic")
    real = os.path.ismount
    monkeypatch.setattr(os.path, "ismount", lambda p: os.path.abspath(p) == str(mnt) or real(p))
    ok, msg, _ = _delete("dl-mnt/foo-Q4.gguf")
    assert ok, msg
    assert not foo.exists() and bar.exists() and note.exists() and mnt.is_dir()
    # Even as the last model there, the mount's own files and the folder stay.
    shutil.rmtree(mnt / "foo-Q4")
    solo = _model(mnt / "solo-Q4.gguf")
    ok, msg, _ = _delete("dl-mnt/solo-Q4.gguf")
    assert ok, msg
    assert not solo.exists() and note.exists() and mnt.is_dir()
    note.unlink()
    last = _model(mnt / "last-Q4.gguf")
    ok, msg, _ = _delete("dl-mnt/last-Q4.gguf")
    assert ok and not last.exists() and mnt.is_dir(), "an empty mount point is not rmdir'd"


def test_configured_download_location_is_never_removed(monkeypatch):
    monkeypatch.setattr(services.settings, "model_download_targets", "dl-archive")
    target = MODELS / "dl-archive"
    m = _model(target / "only-Q4.gguf")
    (target / "README.md").write_text("synthetic")
    ok, msg, _ = _delete("dl-archive/only-Q4.gguf")
    assert ok, msg
    assert not m.exists() and target.is_dir() and (target / "README.md").exists()
    (target / "README.md").unlink()
    m = _model(target / "again-Q4.gguf")
    ok, msg, _ = _delete("dl-archive/again-Q4.gguf")
    assert ok and not m.exists() and target.is_dir()


def test_single_model_folder_is_still_removed_whole():
    folder = MODELS / "dl-single"
    a = _model(folder / "solo-Q4.gguf", pad=1000)
    b = _model(folder / "mmproj-solo-F16.gguf", pad=500)
    (folder / "chat_template.jinja").write_text("{{ messages }}")
    (folder / "tok").mkdir()
    (folder / "tok" / "tokenizer.model").write_bytes(b"t" * 100)
    expected = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    ok, msg, freed = _delete("dl-single/solo-Q4.gguf")
    assert ok, msg
    assert not folder.exists() and freed == expected
    assert "removed dl-single/" in msg
    assert not a.exists() and not b.exists()


def test_shared_projector_stays_with_the_remaining_quant():
    q4 = _model(MODELS / "dl-pair" / "pair-Q4_K_M.gguf")
    q8 = _model(MODELS / "dl-pair" / "pair-Q8_0.gguf")
    proj = _model(MODELS / "dl-pair" / "mmproj-pair-F16.gguf")
    entry = next(g for g in services.snapshot_models_dir().ggufs if g.subdir == "dl-pair" and g.companion_parts)
    ok, msg, _ = _delete(f"dl-pair/{entry.display_name}")
    assert ok, msg
    survivor = q8 if entry.display_name == q4.name else q4
    assert survivor.exists() and proj.exists(), "the projector still serves the other quant"


def test_flat_file_in_models_root_deletes_only_itself():
    flat = _model(MODELS / "flat-dl-Q4.gguf")
    ok, msg, _ = _delete("flat-dl-Q4.gguf")
    assert ok, msg
    assert not flat.exists() and MODELS.is_dir() and (MODELS / "tiny" / "tiny-Q4_K_M.gguf").exists()


def test_protected_dir_rules(monkeypatch):
    monkeypatch.setattr(services.settings, "model_download_targets", " dl-archive/ ,")
    assert services._protected_dir(MODELS)
    assert services._protected_dir(MODELS / "dl-archive")
    assert services._protected_dir(ROOT)                # outside the models folder
    assert not services._protected_dir(MODELS / "dl-single")


# ---------- #801: Hugging Face cache with nested quant folders ----------

def _blob(repo, name, size):
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    b = repo / "blobs" / name
    b.write_bytes(_gguf({"general.architecture": "llama"}) + b"\0" * size)
    return b


def _link(repo, rel, blob):
    link = repo / "snapshots" / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(blob, link.parent), link)
    return link


def test_nested_quant_folder_in_hf_cache_frees_its_blobs_and_keeps_shared_ones():
    repo = MODELS / "models--acme--nested-GGUF"
    q4_blob = _blob(repo, "sha-q4", 3000)
    q8_blob = _blob(repo, "sha-q8", 5000)
    proj_blob = _blob(repo, "sha-proj", 700)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev1")
    _link(repo, "rev1/Q4_K_M/nested-Q4_K_M.gguf", q4_blob)
    _link(repo, "rev1/Q4_K_M/mmproj-nested-F16.gguf", proj_blob)
    _link(repo, "rev1/Q8_0/nested-Q8_0.gguf", q8_blob)
    _link(repo, "rev1/Q8_0/mmproj-nested-F16.gguf", proj_blob)
    q4_size, rest = q4_blob.stat().st_size, q8_blob.stat().st_size + proj_blob.stat().st_size

    ok, msg, freed = _delete("models--acme--nested-GGUF/snapshots/rev1/Q4_K_M/nested-Q4_K_M.gguf")
    assert ok, msg
    assert freed == q4_size, "only the Q4 blob is freed; the projector blob is still linked from Q8_0"
    assert not q4_blob.exists() and proj_blob.exists() and q8_blob.exists()
    assert not (repo / "snapshots" / "rev1" / "Q4_K_M").exists()
    assert (repo / "snapshots" / "rev1" / "Q8_0" / "nested-Q8_0.gguf").exists()

    ok, msg, freed = _delete("models--acme--nested-GGUF/snapshots/rev1/Q8_0/nested-Q8_0.gguf")
    assert ok, msg
    assert not repo.exists(), "the last model takes the whole repository folder"
    assert freed == rest + len("rev1"), "Q8 and projector blobs, plus the refs file"


def test_blob_linked_from_another_snapshot_is_kept():
    repo = MODELS / "models--acme--shared-GGUF"
    blob = _blob(repo, "sha-shared", 2000)
    _link(repo, "rev1/Q4_K_M/shared-Q4_K_M.gguf", blob)
    _link(repo, "rev2/Q4_K_M/shared-Q4_K_M.gguf", blob)
    ok, msg, freed = _delete("models--acme--shared-GGUF/snapshots/rev1/Q4_K_M/shared-Q4_K_M.gguf")
    assert ok, msg
    assert freed == 0 and blob.exists()
    assert not (repo / "snapshots" / "rev1").exists(), "the emptied revision folder is tidied"
    assert (repo / "snapshots" / "rev2" / "Q4_K_M" / "shared-Q4_K_M.gguf").exists()


def test_hf_link_pointing_outside_the_repo_never_deletes_its_target():
    repo = MODELS / "models--acme--shared-GGUF"
    blob = _blob(repo, "sha-m", 1000)
    _link(repo, "rev1/sub/m-Q4_K_M.gguf", blob)
    outside = ROOT / "dl-outside-target.txt"
    outside.write_text("synthetic, not part of any model")
    os.symlink(outside, repo / "snapshots" / "rev1" / "sub" / "README.md")
    ok, msg, _ = _delete("models--acme--shared-GGUF/snapshots/rev1/sub/m-Q4_K_M.gguf")
    assert ok, msg
    assert outside.exists() and not repo.exists()


def test_empty_folders_inside_a_deleted_hf_quant_folder_do_not_block_the_tidy_up():
    repo = MODELS / "models--acme--nested-GGUF"
    blob = _blob(repo, "sha-only", 800)
    _link(repo, "rev1/Q4_K_M/only-Q4_K_M.gguf", blob)
    (repo / "snapshots" / "rev1" / "Q4_K_M" / "empty" / "deeper").mkdir(parents=True)
    size = blob.stat().st_size
    ok, msg, freed = _delete("models--acme--nested-GGUF/snapshots/rev1/Q4_K_M/only-Q4_K_M.gguf")
    assert ok, msg
    assert not repo.exists() and freed == size


def test_filesystem_mounted_inside_a_model_folder_is_never_emptied(monkeypatch):
    folder = MODELS / "dl-nest"
    m = _model(folder / "nest-Q4.gguf")
    inner = folder / "shared-data"
    inner.mkdir()
    (inner / "keep.bin").write_bytes(b"k" * 64)
    real = os.path.ismount
    monkeypatch.setattr(os.path, "ismount", lambda p: os.path.abspath(p) == str(inner) or real(p))
    ok, msg, _ = _delete("dl-nest/nest-Q4.gguf")
    assert ok, msg
    assert not m.exists() and (inner / "keep.bin").exists()


def test_symlink_loop_in_a_hf_snapshot_does_not_crash_the_delete():
    repo = MODELS / "models--acme--shared-GGUF"
    blob = _blob(repo, "sha-loop", 600)
    _link(repo, "rev1/m-Q4_K_M.gguf", blob)
    snap = repo / "snapshots" / "rev1"
    os.symlink("loop-b", snap / "loop-a")
    os.symlink("loop-a", snap / "loop-b")
    ok, msg, _ = _delete("models--acme--shared-GGUF/snapshots/rev1/m-Q4_K_M.gguf")
    assert ok, msg
    assert not repo.exists()


def test_symlink_loop_runtime_error_is_handled(monkeypatch):
    # Python 3.12 raises RuntimeError from Path.resolve() on a loop (3.13 does not); simulate it.
    from pathlib import Path
    repo = MODELS / "models--acme--shared-GGUF"
    blob = _blob(repo, "sha-loop", 600)
    _link(repo, "rev1/m-Q4_K_M.gguf", blob)
    snap = repo / "snapshots" / "rev1"
    os.symlink("loop-b", snap / "loop-a")
    os.symlink("loop-a", snap / "loop-b")
    real = Path.resolve

    def resolve(self, strict=False):
        if self.name.startswith("loop-"):
            raise RuntimeError(f"Symlink loop from {self}")
        return real(self, strict)
    monkeypatch.setattr(Path, "resolve", resolve)
    ok, msg, freed = _delete("models--acme--shared-GGUF/snapshots/rev1/m-Q4_K_M.gguf")
    assert ok, msg
    assert not repo.exists() and freed >= 600
