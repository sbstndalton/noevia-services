from __future__ import annotations

import asyncio
import configparser
import os
import posixpath
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import docker
import httpx
from docker.errors import APIError, DockerException, NotFound

from . import ini
from .config import settings
from .utils import human_bytes, shard_key


# ---------- models directory ----------


@dataclass(frozen=True)
class ModelShape:
    """MoE or dense, read from the GGUF header.

    Worth surfacing because it decides what offloading costs, and the two are not close. In a
    DENSE model every weight is read for every token, so a layer moved to system RAM is paid on
    every token. In an MoE only the routed experts are read - 8 of 128, or 10 of 512 - so most
    of what sits in RAM is untouched on any given token. Measured here: Qwen3.8-Flash-Next keeps
    ~60 GB of experts in DDR4-2667 and still generates at 17 tok/s; a dense model with that much
    on the same memory would be under 1.

    It also says whether --cpu-moe / --n-cpu-moe will do anything at all. On a dense model they
    are silently inert.
    """
    arch: str = ""
    expert_count: int = 0
    expert_used: int = 0

    @property
    def known(self) -> bool:
        return bool(self.arch)

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0

    @property
    def label(self) -> str:
        if not self.known:
            return ""
        if not self.is_moe:
            return "dense"
        return f"MoE {self.expert_count}x{self.expert_used}" if self.expert_used else \
               f"MoE {self.expert_count}"


def model_shape(path: Path) -> ModelShape:
    """Read architecture + expert counts from a GGUF header.

    gguf_meta.read_raw already caches on (path, mtime, size) and only reads the head of the
    file, so calling this per row on a page render is cheap after the first hit.
    """
    from . import gguf_meta
    try:
        kv = gguf_meta.read_raw(path)
    except Exception:  # noqa: BLE001 - an unreadable header is "unknown", never an error page
        return ModelShape()
    if isinstance(kv, dict) and isinstance(kv.get("kv"), dict):
        kv = kv["kv"]
    if not isinstance(kv, dict):
        return ModelShape()
    arch = str(kv.get("general.architecture") or "")
    if not arch:
        return ModelShape()

    def _int(suffix: str) -> int:
        v = kv.get(f"{arch}.{suffix}")
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    return ModelShape(arch=arch, expert_count=_int("expert_count"),
                      expert_used=_int("expert_used_count"))

@dataclass
class GgufEntry:
    display_name: str        # shown to user (base name, without shard suffix if grouped)
    parts: list[Path]        # one entry, or many shards sorted by index
    total_bytes: int
    mtime: float             # newest part's mtime
    is_sharded: bool
    subdir: str = ""         # empty for flat files, subdir name for shard groups in a subdir
    is_companion: bool = False   # True for mmproj / other files referenced from a main model's section
    aliases: list[str] = field(default_factory=list)  # ini section names that reference this
    # If this is a main model with a companion mmproj in the same subdir, these fields
    # describe the companion. Standalone companion entries are hidden from listings.
    companion_name: str = ""
    companion_bytes: int = 0
    companion_parts: list[Path] = field(default_factory=list)

    @property
    def human_size(self) -> str:
        return human_bytes(self.total_bytes)

    @property
    def modified(self) -> str:
        return datetime.fromtimestamp(self.mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    @property
    def stem(self) -> str:
        # e.g. "gemma-4-12b-it-Q4_K_M.gguf" -> "gemma-4-12b-it-Q4_K_M"
        return self.display_name[:-5] if self.display_name.lower().endswith(".gguf") else self.display_name

    @property
    def model_id(self) -> str:
        """The id llama-server serves this file under.

        That is its ini section name when it has one — which is NOT necessarily the filename
        stem, since a section can be renamed to give the model a short API id — and otherwise
        the stem, which is what llama-server falls back to.
        """
        if self.stem in self.aliases:
            return self.stem      # a section named after the file: the natural id
        return self.aliases[0] if self.aliases else self.stem

    @property
    def route_key(self) -> str:
        """URL-safe key for /model/<key> routes; for subdir'd entries this includes the subdir."""
        return f"{self.subdir}/{self.display_name}" if self.subdir else self.display_name

    @property
    def first_shard_rel(self) -> str:
        """Relative path (from models_dir) to the first shard/file. Used in ini `model = ...`."""
        first = self.parts[0].name
        return f"{self.subdir}/{first}" if self.subdir else first


@dataclass
class DiskInfo:
    total: int
    free: int
    used_pct: float

    @property
    def total_h(self) -> str: return human_bytes(self.total)
    @property
    def free_h(self) -> str: return human_bytes(self.free)


@dataclass
class ModelsDirSnapshot:
    path: Path
    exists: bool
    error: str | None = None
    disk: DiskInfo | None = None
    ggufs: list[GgufEntry] = field(default_factory=list)
    ini_aliases: list[str] = field(default_factory=list)
    ini_present: bool = False


def read_ini_aliases(ini_path: Path) -> list[str]:
    if not ini_path.exists():
        return []
    try:
        from . import ini
        return ini.parse_ini_text(ini_path.read_text(encoding="utf-8"), str(ini_path)).sections()
    except (OSError, configparser.Error):
        return []


def snapshot_models_dir() -> ModelsDirSnapshot:
    path = settings.models_dir
    snap = ModelsDirSnapshot(path=path, exists=path.exists())
    if not path.exists():
        snap.error = "directory not found (is the volume mounted?)"
        return snap

    try:
        u = shutil.disk_usage(path)
        snap.disk = DiskInfo(total=u.total, free=u.free, used_pct=round(100 * (u.total - u.free) / u.total, 1))
    except OSError as e:
        snap.error = f"disk usage failed: {e}"

    aliases = read_ini_aliases(settings.models_ini_path)
    snap.ini_aliases = aliases
    # file -> section, so a section renamed to a short API id still owns its GGUF
    by_file = ini.sections_by_file()
    snap.ini_present = settings.models_ini_path.exists()

    # collect gguf files, grouping shards. Also descend one level into subdirs
    # (multi-shard downloads land in /models/<base>/ to keep the top level tidy).
    # Key = (subdir, shard_base). subdir="" for flat.
    groups: dict[tuple[str, str], list[Path]] = {}
    try:
        from .utils import iter_gguf
        for reldir, p in iter_gguf(path):
            base, _, _ = shard_key(p.name)
            groups.setdefault((reldir, base), []).append(p)
    except OSError as e:
        snap.error = f"listing failed: {e}"
        return snap

    entries: list[GgufEntry] = []
    for (subdir, base), parts in groups.items():
        parts.sort(key=lambda pp: pp.name)
        total = sum(pp.stat().st_size for pp in parts)
        mtime = max(pp.stat().st_mtime for pp in parts)
        stem_no_ext = base[:-5] if base.lower().endswith(".gguf") else base
        rel = f"{subdir}/{parts[0].name}" if subdir else parts[0].name
        matched = by_file.get(rel) or [a for a in aliases if a == stem_no_ext]
        # Use the same predicate the ini layer uses to decide what may become its own section,
        # rather than a second, narrower rule. This one only tested for mmproj, so speculative
        # draft heads - Qwen MTP files, generic -draft- - were listed as standalone models even
        # though nothing can be done with them: they cannot be run or configured alone, and they
        # are already referenced from their main model's section via `model-draft`.
        is_comp = ini._is_companion(base)
        entries.append(GgufEntry(
            display_name=base,
            parts=parts,
            total_bytes=total,
            mtime=mtime,
            is_sharded=len(parts) > 1,
            subdir=subdir,
            is_companion=is_comp,
            aliases=matched,
        ))

    # Fold companion entries into their same-subdir main model.
    # A companion is a file that only makes sense paired with its main model: a multimodal
    # projector (mmproj) or a speculative draft head (MTP). You can't run one alone, can't
    # configure it, can't do anything with it. So we hide it from the list and expose it as a
    # badge on the main model instead. Un-paired companions (one sitting in a subdir with no
    # main model) stay visible so they can be cleaned up.
    main_by_subdir: dict[str, GgufEntry] = {
        e.subdir: e for e in entries if not e.is_companion and e.subdir
    }
    surviving: list[GgufEntry] = []
    for e in entries:
        if e.is_companion and e.subdir and e.subdir in main_by_subdir:
            main = main_by_subdir[e.subdir]
            # ACCUMULATE. A subdir can hold several projectors (mmproj-BF16 + mmproj-F32 is
            # a common upload pattern). Assigning here instead of appending meant the second
            # one overwrote the first, so deleting the model stranded a projector on disk and
            # left the subdir non-empty -- which then silently defeated the rmdir below.
            main.companion_name = (f"{main.companion_name}, {e.display_name}"
                                   if main.companion_name else e.display_name)
            main.companion_bytes += e.total_bytes
            main.companion_parts.extend(e.parts)
            continue  # drop the standalone companion row
        surviving.append(e)
    # sort: (remaining) companions after main models, alpha within group
    surviving.sort(key=lambda e: (e.subdir.lower(), e.is_companion, e.display_name.lower()))
    snap.ggufs = surviving
    return snap


# ---------- delete guard: files a running container depends on (#336) ----------

class ModelInUse(Exception):
    """A model file is referenced by a running container; deleting it would break that container."""


_MODEL_ENV_KEYS = ("EMBEDDING_MODEL", "EMBED_MODEL", "RERANK_MODEL")


def _running_containers() -> list[dict]:
    """Inspect data for every running container except this one.

    Raises ModelInUse (fail closed) when the Docker socket is there but cannot be read: a delete
    is irreversible, so "could not check" must not read as "nobody uses it". With no socket
    configured at all (a plain models-folder install) there is nothing to consult, so [].
    """
    client = _docker_client()
    if client is None:
        return []
    try:
        listed = client.containers.list()
    except (DockerException, OSError) as e:
        raise ModelInUse(f"Could not check which containers use this model ({type(e).__name__}); nothing was deleted.") from e
    own = os.environ.get("HOSTNAME") or ""
    out = []
    for c in listed:
        if c.name == "model-loader" or (own and c.id.startswith(own)):
            continue
        out.append({"name": c.name, **(c.attrs or {})})
    return out


def _container_args(attrs: dict) -> list[str]:
    cfg = attrs.get("Config") or {}
    args: list[str] = []
    for part in (cfg.get("Entrypoint"), cfg.get("Cmd"), attrs.get("Args")):
        if isinstance(part, str):
            args.append(part)
        elif isinstance(part, list):
            args.extend(str(a) for a in part)
    return args


def _env_map(attrs: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in (attrs.get("Config") or {}).get("Env") or []:
        k, _, v = str(item).partition("=")
        out[k] = v
    return out


def model_holders(entry: "GgufEntry", containers: list[dict] | None = None) -> list[str]:
    """Running containers that use this model, each with the reason.

    Two signals, both read from `docker inspect` (the loader has the Docker socket; it does not
    share a PID namespace with the engines, so /proc/<pid>/fd of their processes is not visible
    and cannot be used):
      1. a command-line argument naming one of the model's files (the embed sidecar runs
         `--model /models/<file>`), matched by models-dir-relative path so a different mount
         point in the other container still counts;
      2. EMBEDDING_MODEL / EMBED_MODEL / RERANK_MODEL in the container's environment naming this
         model ("default" is a placeholder; RERANK_MODEL only counts while the reranker feature
         is on, matching the web-side guard).
    """
    root = settings.models_dir
    rels: set[str] = set()
    for p in list(entry.parts) + list(entry.companion_parts):
        try:
            rels.add(p.relative_to(root).as_posix())
        except ValueError:
            rels.add(p.name)
    names = {entry.model_id, entry.stem, entry.display_name, *entry.aliases}
    names.discard("")
    host_root = (settings.models_host_path or "").rstrip("/")
    found: list[str] = []
    for c in containers if containers is not None else _running_containers():
        why = ""
        mounts = [(str(m.get("Destination") or "").rstrip("/"), str(m.get("Source") or "").rstrip("/")) for m in c.get("Mounts") or []]
        for raw in _container_args(c):
            arg = raw.split("=", 1)[1] if raw.startswith("--") and "=" in raw else raw
            norm = posixpath.normpath(arg) if arg.startswith("/") else arg
            for rel in rels:
                if norm != rel and not norm.endswith("/" + rel):
                    continue
                # When the container's mount source and the host models path are both known, the
                # file must really come from the models folder, not a same-named file elsewhere.
                mount = next(((d, s) for d, s in mounts if d and norm.startswith(d + "/")), None)
                if host_root and mount and posixpath.normpath(mount[1] + norm[len(mount[0]):]) != f"{host_root}/{rel}":
                    continue
                why = f"its command runs {rel}"
                break
            if why:
                break
        if not why:
            env = _env_map(c)
            rerank_on = env.get("NOEVIA_FEATURE_RAG_RERANK", "").lower() in ("1", "true", "on")
            for key in _MODEL_ENV_KEYS:
                v = env.get(key, "").strip()
                if not v or v.lower() == "default" or (key == "RERANK_MODEL" and not rerank_on):
                    continue
                if v in names:
                    why = f"{key} is set to {v}"
                    break
        if why:
            found.append(f"{c['name']} ({why})")
    return found


def in_use_message(entry: "GgufEntry", holders: list[str]) -> str:
    return f"{entry.display_name} is in use by a running container: {'; '.join(holders)}. Stop or reconfigure it first; nothing was deleted."


def delete_gguf(display_name: str, subdir: str = "") -> tuple[bool, str, int]:
    """Delete a GGUF (and all shards). Returns (ok, message, bytes_freed).
    If subdir is empty, matches only flat files with that display_name; otherwise the entry in that subdir."""
    snap = snapshot_models_dir()
    if snap.error and not snap.ggufs:
        return False, snap.error, 0
    # match by display_name AND subdir (allows same base filename to exist both flat and in a subdir)
    match = next((g for g in snap.ggufs if g.display_name == display_name and g.subdir == subdir), None)
    if match is None and subdir == "":
        # fall back: match any subdir if only display_name given (bulk-delete callers pass just display_name)
        match = next((g for g in snap.ggufs if g.display_name == display_name), None)
    if match is None:
        return False, f"not found: {display_name}", 0
    # Backstop for callers other than the JSON API (which checks up front and answers 409).
    try:
        holders = model_holders(match)
    except ModelInUse as e:
        return False, str(e), 0
    if holders:
        return False, in_use_message(match, holders), 0
    freed = 0
    removed: list[str] = []

    # Whole-directory delete when this model is the only model in its folder tree.
    #
    # Downloads land one model per directory, and everything beside the weights there is
    # support material for THAT model: projectors (often more than one), chat_template.jinja,
    # tokenizer.model. Deleting file-by-file means anything not explicitly enumerated is
    # stranded -- and a single leftover keeps the directory non-empty, so the tidy-up rmdir
    # below silently does nothing and the orphans persist invisibly. Measured on a real
    # deletion: 3.3 GB of projectors left behind.
    #
    # #798: guarded on a RECURSIVE search finding no other model anywhere below the folder (the
    # scan reads four levels, so `A/x.gguf` beside `A/old/y.gguf` are two models), and on the
    # folder not being the models root, a configured download location or a mount point. In any
    # of those cases it degrades to per-file deletion rather than taking a neighbour with it.
    hf = _HF_CACHE_RE.match(match.subdir or "")
    if hf:
        return _delete_hf_snapshot(match, hf.group(1))
    own = {p for p in list(match.parts) + list(match.companion_parts)}
    if match.subdir:
        subpath = settings.models_dir / match.subdir
        if subpath.is_dir() and not _protected_dir(subpath) and not _other_models_below(subpath, own):
            try:
                for q in sorted(subpath.rglob("*")):
                    if q.is_file() and not q.is_symlink():
                        freed += q.stat().st_size
                    if q.is_file() or q.is_symlink():
                        removed.append(q.name)
                shutil.rmtree(subpath)
                return True, f"deleted {len(removed)} file(s), removed {match.subdir}/", freed
            except OSError as e:
                return False, f"failed to remove {match.subdir}/: {e}", freed

    # Per-file: a shared or protected directory, or a flat file with no directory of its own.
    # The model's own shards always go. Its companions (projectors, draft heads) only go when it
    # is the sole model in its folder: Q4 and Q8 of one model routinely share one mmproj, and
    # deleting Q4 must not strand Q8 without its projector.
    paths_to_remove: list[Path] = list(match.parts)
    if match.companion_parts and not _other_models_beside(match):
        paths_to_remove += list(match.companion_parts)
    for p in paths_to_remove:
        try:
            size = p.stat().st_size if not p.is_symlink() else 0
            p.unlink()
            freed += size
            removed.append(p.name)
        except OSError as e:
            return False, f"failed to delete {p.name}: {e}. removed so far: {removed}", freed
    # if this entry lived in a subdir, remove the dir when empty (never a protected one)
    if match.subdir:
        subpath = settings.models_dir / match.subdir
        try:
            if subpath.is_dir() and not _protected_dir(subpath) and not any(subpath.iterdir()):
                subpath.rmdir()
        except OSError:
            pass
    return True, f"deleted {len(removed)} file(s)", freed


# models--owner--repo/snapshots/<revision>[/<nested folder>...], optionally inside a download
# location (e.g. a Hugging Face cache mounted at /models/hf). Group 1 is the repository folder.
_HF_CACHE_RE = re.compile(r"^((?:[^/]+/)*?models--[^/]+)/snapshots/[^/]+(?:/.+)?$")


def _download_target_dirs() -> set[Path]:
    """Folders the user configured as download locations (MODEL_DOWNLOAD_TARGETS)."""
    root = settings.models_dir
    out: set[Path] = set()
    for name in (settings.model_download_targets or "").split(","):
        name = name.strip().strip("/")
        if name:
            out.add(Path(os.path.abspath(root / name)))
    return out


MOUNTINFO_PATH = "/proc/self/mountinfo"


def _unescape_mountinfo(field: str) -> str:
    """Decode the \\NNN octal escapes the kernel uses for space, tab, newline and backslash."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def _mount_points() -> set[str] | None:
    """Mount points listed in /proc/self/mountinfo (field 5 of each line), or None where that file
    is unavailable (not Linux, no /proc), in which case os.path.ismount is all there is."""
    try:
        with open(MOUNTINFO_PATH, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    points: set[str] = set()
    for line in lines:
        fields = line.split(" ")
        if len(fields) > 4:
            points.add(os.path.normpath(_unescape_mountinfo(fields[4])))
    return points


def _is_mount_point(path, points: set[str] | None = None) -> bool:
    """True when `path` is a mount point. os.path.ismount compares st_dev with the parent's, so it
    misses a bind mount of a folder from the same filesystem (equal st_dev); mountinfo does not.
    Pass `points` (from _mount_points) to avoid re-reading the file for every folder of a walk."""
    if os.path.ismount(path):
        return True
    if points is None:
        points = _mount_points()
    if not points:
        return False
    here = os.path.abspath(path)
    return here in points or os.path.realpath(path) in points


def _protected_dir(path: Path) -> bool:
    """True for a folder model deletion must never remove: the models root, a configured download
    location, a mount point, a symlinked folder, or anything outside the models folder."""
    root = Path(os.path.abspath(settings.models_dir))
    here = Path(os.path.abspath(path))
    if here == root or root not in here.parents:
        return True
    if here in _download_target_dirs():
        return True
    try:
        return path.is_symlink() or _is_mount_point(path)
    except OSError:
        return True


def _is_gguf_name(name: str) -> bool:
    return name.lower().endswith(".gguf")


def _other_models_below(folder: Path, own: set[Path]) -> bool:
    """True if any GGUF under `folder` (any depth) is not this model's own file.

    In the folder itself only main models count -- a companion there is folded into this model
    (or is an un-paired leftover the whole-folder delete may take). Below it, ANY GGUF counts:
    a nested folder is a different listing row, even when all it holds is a projector.
    """
    def _raise(err: OSError) -> None:
        raise err
    mounts = _mount_points()
    try:
        for dirpath, _dirnames, filenames in os.walk(folder, followlinks=False, onerror=_raise):
            here = Path(dirpath)
            nested = here != folder
            if nested and _is_mount_point(dirpath, mounts):
                return True   # a filesystem mounted inside: rmtree would empty it
            for name in filenames:
                if not _is_gguf_name(name):
                    continue
                q = here / name
                if q in own:
                    continue
                if nested or not ini._is_companion(name):
                    return True
    except OSError:
        return True   # cannot tell: fail closed to per-file deletion
    return False


def _other_models_beside(match: "GgufEntry") -> bool:
    """True if another main model shares this model's folder (directly, not below it)."""
    folder = match.parts[0].parent if match.parts else settings.models_dir / match.subdir
    own = set(match.parts)
    try:
        return any(q.is_file() and _is_gguf_name(q.name) and q not in own and not ini._is_companion(q.name)
                   for q in folder.iterdir())
    except OSError:
        return True


def _delete_hf_snapshot(match: "GgufEntry", repo_dir: str) -> tuple[bool, str, int]:
    """Delete a model stored in Hugging Face's cache layout.

    Snapshot files are links into blobs/, so removing the link alone frees nothing. Remove
    this model's files and the blobs they point to (unless another link anywhere in the repo's
    snapshots still uses a blob), and the whole repository folder once no snapshot is left.
    The model may sit directly in snapshots/<rev>/ or in a nested quant folder below it (#801).
    """
    root = settings.models_dir
    folder = root / match.subdir
    repo = root / repo_dir
    snaps = repo / "snapshots"
    own = set(match.parts) | set(match.companion_parts)
    if _other_models_below(folder, own):
        # Other models share this folder or sit below it: this model's files only. Companions
        # only when no other main model sits beside it (a shared projector stays).
        victims = list(match.parts)
        if not _other_models_beside(match):
            victims += list(match.companion_parts)
    else:
        victims = sorted(q for q in folder.rglob("*") if q.is_file() or q.is_symlink())
    victim_set = set(victims)
    # Blobs still referenced by any link that is not being deleted: another snapshot, a sibling
    # quant folder in the same snapshot, or a neighbour file in this one.
    in_use: set[Path] = set()
    if snaps.is_dir():
        for q in snaps.rglob("*"):
            if q.is_symlink() and q not in victim_set:
                try:
                    in_use.add(q.resolve())
                except (OSError, RuntimeError):   # RuntimeError: a symlink loop before 3.13
                    pass
    freed, removed = 0, []
    try:
        repo_real = repo.resolve()
        for q in victims:
            try:
                target = q.resolve()
            except RuntimeError:   # a symlink loop before 3.13: nothing behind it to free
                target = None
            if q.is_symlink():
                if target is not None and target.is_file() and target not in in_use and repo_real in target.parents:
                    freed += target.stat().st_size
                    target.unlink()
            elif q.is_file():
                freed += q.stat().st_size
            q.unlink()
            removed.append(q.name)
        # Tidy empty folders inside the model's folder, then from it up to (not including)
        # snapshots/.
        if folder.is_dir():
            for d in sorted((d for d in folder.rglob("*") if d.is_dir() and not d.is_symlink()),
                            key=lambda d: len(d.parts), reverse=True):
                if not any(d.iterdir()) and not _protected_dir(d):
                    d.rmdir()
        here = folder
        while here != snaps and snaps in here.parents and not _protected_dir(here):
            if here.is_dir() and not any(here.iterdir()):
                here.rmdir()
                here = here.parent
            else:
                break
        if (not snaps.is_dir() or not any(snaps.iterdir())) and not _protected_dir(repo):
            for q in repo.rglob("*"):
                if q.is_file() and not q.is_symlink():
                    freed += q.stat().st_size
            shutil.rmtree(repo)
            return True, f"deleted {len(removed)} file(s), removed {repo_dir}/", freed
    except (OSError, RuntimeError) as e:
        return False, f"failed while deleting {match.display_name}: {e}", freed
    return True, f"deleted {len(removed)} file(s)", freed


# ---------- docker / containers ----------

def _docker_client() -> docker.DockerClient | None:
    try:
        return docker.from_env()
    except DockerException:
        return None


@dataclass
class ContainerInfo:
    name: str
    found: bool
    status: str | None = None
    image: str | None = None
    id: str | None = None


# ---------- llama backends: richer view + restart + probe ----------

# error-of-last-restart per container, shown in the card until it clears
_restart_errors: dict[str, str] = {}
_restart_errors_lock = threading.Lock()


@dataclass
class LlamaBackend:
    name: str
    found: bool
    status: str = ""           # running | exited | restarting | not_found | error
    image: str = ""
    short_id: str = ""
    started_at: str = ""
    # Time since StartedAt, only while the container is running: an exited or stopped one has no
    # uptime, and StartedAt is when it *last* started (#603), so both stay None.
    uptime: str | None = None
    # The same span in seconds, so a client can localise the units itself; `uptime` stays for
    # older clients (#597).
    uptime_s: int | None = None
    host_ports: list[str] = field(default_factory=list)
    internal_port: int | None = None
    loaded_model: str | None = None
    probe_error: str | None = None
    # True when the probe could not connect but Docker's own healthcheck reports the container
    # healthy: the engine is up and merely not reachable from this service's network (#549).
    unreachable_but_healthy: bool = False
    last_restart_error: str | None = None


def _parse_started_at(iso: str) -> tuple[str, str, int | None]:
    # docker returns e.g. "2025-08-11T00:51:00.123456789Z"
    if not iso or iso.startswith("0001"):
        return "", "", None
    try:
        # trim nanoseconds to microseconds
        core, _, frac = iso.partition(".")
        if frac:
            frac = frac.rstrip("Z")[:6]
            iso_norm = f"{core}.{frac}+00:00"
        else:
            iso_norm = core.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_norm)
    except ValueError:
        return iso, "", None
    local = dt.astimezone().strftime("%Y-%m-%d %H:%M")
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        up = f"{secs}s"
    elif secs < 3600:
        up = f"{secs // 60}m"
    elif secs < 86400:
        up = f"{secs // 3600}h {(secs % 3600) // 60}m"
    else:
        up = f"{secs // 86400}d {(secs % 86400) // 3600}h"
    return local, up, max(0, secs)


def _extract_ports(attrs: dict) -> tuple[list[str], int | None]:
    """Return (['8082->8080/tcp'], 8080)."""
    ports_map = (attrs.get("NetworkSettings") or {}).get("Ports") or {}
    result: list[str] = []
    internal: int | None = None
    for cont_port, bindings in ports_map.items():
        # cont_port like "8080/tcp"
        try:
            internal = int(cont_port.split("/")[0])
        except ValueError:
            pass
        if not bindings:
            continue
        for b in bindings:
            hp = b.get("HostPort")
            if hp:
                result.append(f"{hp}->{cont_port}")
    return result, internal


_PORT_FLAG_RE = re.compile(r"--port[=\s]+(\d+)")


def _port_from_command_or_env(attrs: dict) -> int | None:
    """Read the llama.cpp server's listen port from its own launch config, for containers
    that publish no port at all (issue #341): checks `--port N` in Cmd/Entrypoint, then the
    LLAMA_ARG_PORT env var llama.cpp's server also honors."""
    config = attrs.get("Config") or {}
    parts: list[str] = []
    for key in ("Cmd", "Entrypoint"):
        val = config.get(key)
        if isinstance(val, list):
            parts.extend(str(p) for p in val)
    m = _PORT_FLAG_RE.search(" ".join(parts))
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    for env_entry in (config.get("Env") or []):
        if not isinstance(env_entry, str) or "=" not in env_entry:
            continue
        name, _, value = env_entry.partition("=")
        if name == "LLAMA_ARG_PORT" and value.strip():
            try:
                return int(value.strip())
            except ValueError:
                continue
    return None


def _resolve_internal_port(attrs: dict) -> tuple[list[str], int | None]:
    """Like `_extract_ports`, but falls back to the container's own `--port`/env when Docker
    publishes no port, then to the configured default (llama.cpp server default 8080). An
    internal-network-only container legitimately has no port mapping, but the probe still
    needs *a* port to reach it on the docker network -- returning None here previously made
    the probe silently skip, which issue #341 showed as a false "no model loaded"."""
    host_ports, internal = _extract_ports(attrs)
    if internal is not None:
        return host_ports, internal
    internal = _port_from_command_or_env(attrs)
    if internal is not None:
        return host_ports, internal
    default = settings.llama_default_port
    if default:
        return host_ports, default
    return host_ports, None


# Messages `_probe_loaded_model` returns when the probe itself succeeded (reached the server,
# got a 200 with parseable JSON) but no model happens to be loaded. Distinct from a skipped or
# failed probe: only the latter should surface as `probe_error` (issue #341's "Expected" -
# "no model loaded" only after a successful probe; probe_error only when skipped or failed).
_SOFT_PROBE_NOTE_RE = re.compile(r"^(no models configured|\d+ configured, none loaded)$")


_UNREACHABLE_ERRORS = {"connection refused", "timeout"}
_UNREACHABLE_HEALTHY_NOTE = "not reachable from the model loader (Docker reports it healthy)"


def _is_soft_probe_note(message: str | None) -> bool:
    return bool(message) and bool(_SOFT_PROBE_NOTE_RE.match(message))


async def _probe_loaded_model(container_name: str, internal_port: int | None) -> tuple[str | None, str | None]:
    if internal_port is None:
        # Sanitized: never leak internals, but never silently pretend the probe ran either.
        return None, "port unknown"
    url = f"http://{container_name}:{internal_port}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, read=3.0)) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            data = r.json()
            items = data.get("data") or []
            if not items:
                return None, "no models configured"
            # A router lists every configured model with a per-model `status`; a fixed
            # single-model llama-server (the embed sidecar, #580) lists just the model it
            # serves, with no `status` at all -- and it only answers once that model is loaded.
            loaded_ids = [
                str(it.get("id") or "")
                for it in items
                if not isinstance(it.get("status"), dict) or it["status"].get("value") == "loaded"
            ]
            if loaded_ids:
                return ", ".join(i for i in loaded_ids if i), None
            return None, f"{len(items)} configured, none loaded"
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.ConnectError:
        return None, "connection refused"
    except httpx.HTTPError:
        # Sanitized: httpx exception text can embed the full request URL; the container
        # name/port aren't secret, but there's no reason to echo raw exception internals.
        return None, "request failed"
    except ValueError:
        return None, "invalid response"


def discover_llama_containers() -> list[dict]:
    """Return metadata for every container whose image looks like llama.cpp:server-*.
    Includes stopped containers so the user can see and start them.
    """
    client = _docker_client()
    if client is None:
        return []
    out: list[dict] = []
    try:
        containers = client.containers.list(all=True)
    except DockerException:
        return []
    for c in containers:
        try:
            tags = c.image.tags or [c.image.short_id]
            img = (tags[0] if tags else "").lower()
        except DockerException:
            img = ""
        if "ghcr.io/ggml-org/llama.cpp" not in img and "llama.cpp" not in img:
            continue
        if c.name == "model-loader":
            continue
        vendor = "rocm" if "rocm" in img else "cuda" if "cuda" in img else ("cpu" if "server" in img else "unknown")
        out.append({"name": c.name, "image": img, "vendor": vendor})
    return out


def _effective_container_names() -> list[str]:
    """Union of LLAMA_CONTAINERS env (retained even if not currently running) and any
    live-discovered llama.cpp containers on the docker socket. Order: env names first, then
    newly-discovered ones. Duplicates removed while preserving order."""
    whitelist = settings.llama_container_names
    discovered = [c["name"] for c in discover_llama_containers()]
    seen: set[str] = set()
    result: list[str] = []
    for n in list(whitelist) + discovered:
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result


async def snapshot_llama_backends() -> list[LlamaBackend]:
    client = _docker_client()
    effective = _effective_container_names()
    if client is None:
        return [LlamaBackend(name=n, found=False, status="docker unreachable") for n in effective]

    out: list[LlamaBackend] = []
    probe_targets: list[tuple[int, str, int | None]] = []  # (idx, name, internal_port)
    health_states: dict[int, str] = {}  # idx -> Docker healthcheck status ("" when none defined)

    for i, name in enumerate(effective):
        b = LlamaBackend(name=name, found=False, status="not_found")
        with _restart_errors_lock:
            b.last_restart_error = _restart_errors.get(name)
        try:
            c = client.containers.get(name)
            b.found = True
            b.status = c.status
            b.image = (c.image.tags or [c.image.short_id])[0]
            b.short_id = c.short_id
            attrs = c.attrs or {}
            state = (attrs.get("State") or {})
            started, up, up_s = _parse_started_at(state.get("StartedAt", ""))
            b.started_at = started
            if b.status == "running":
                b.uptime = up or None
                b.uptime_s = up_s
            b.host_ports, b.internal_port = _resolve_internal_port(attrs)
            health_states[i] = str((state.get("Health") or {}).get("Status") or "")
        except NotFound:
            pass
        except DockerException as e:
            b.status = f"error: {e}"
        out.append(b)
        if b.found and b.status == "running":
            probe_targets.append((i, name, b.internal_port))

    if probe_targets:
        results = await asyncio.gather(
            *[_probe_loaded_model(n, p) for _, n, p in probe_targets],
            return_exceptions=False,
        )
        for (i, _n, _p), (loaded, err) in zip(probe_targets, results):
            out[i].loaded_model = loaded
            # A successful probe (loaded, or confirmed nothing loaded) is not an error --
            # only a skipped/failed probe should tell Hardware the state is unavailable.
            out[i].probe_error = None if (loaded or _is_soft_probe_note(err)) else err
            if out[i].probe_error in _UNREACHABLE_ERRORS and health_states.get(i) == "healthy":
                # Docker's healthcheck runs inside the container's own network, so "healthy" plus
                # a refused/timed-out connection from here means this service is not on the
                # engine's network -- not that the engine is down (issue #549).
                out[i].probe_error = _UNREACHABLE_HEALTHY_NOTE
                out[i].unreachable_but_healthy = True
    return out


def restart_llama_backend(name: str) -> tuple[bool, str]:
    """Fire-and-forget restart. Errors captured in _restart_errors."""
    if name not in _effective_container_names():
        return False, "container not in configured list"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        c = client.containers.get(name)
    except NotFound:
        return False, "container not found"
    except DockerException as e:
        return False, f"docker error: {e}"

    with _restart_errors_lock:
        _restart_errors.pop(name, None)

    def _do() -> None:
        try:
            c.restart(timeout=15)
        except (DockerException, APIError) as e:
            with _restart_errors_lock:
                _restart_errors[name] = f"{type(e).__name__}: {e}"

    threading.Thread(target=_do, daemon=True, name=f"restart-{name}").start()
    return True, "restart queued"


def _fit_backends() -> dict[str, float]:
    """{backend_name: total VRAM GiB} for everything we can actually plan against.

    Prefers the live probe over settings.gpu_vram_map. The static map is an optional
    override and is empty by default — relying on it alone silently disabled the fit
    chips entirely once the hardcoded example values were removed.
    """
    from . import hw
    out: dict[str, float] = {}
    for name in _effective_container_names():
        vram = float(settings.gpu_vram_map.get(name, 0) or 0)
        if vram <= 0:
            st = hw.stats_for(name)
            if st.ok and st.gpu and st.gpu.vram_total_gb > 0:
                vram = float(st.gpu.vram_total_gb)
        if vram > 0:
            out[name] = vram
    return out


# Above this, a CPU-resident DENSE model is technically loadable and practically unusable.
# Calibrated on measured tok/s on a dual-channel DDR5 box: a 6.6 GB dense model managed
# 4.4 tok/s, implying roughly 29 GB/s of effective read bandwidth, so 30 GB of dense weights
# lands near 1 tok/s.
#
# KNOWN LIMITATION: this is size-based, and size is the wrong axis for MoE. A mixture-of-
# experts model reads only its active experts per token, so a 30 GB MoE can be several times
# faster than a 30 GB dense one — measured here, a 15.8 GB MoE beat a 6.6 GB dense by 2x.
# Telling them apart needs expert_count/expert_used_count from the GGUF header, which the
# chips do not have (they receive a file size and nothing else). The tooltip says so rather
# than pretending the number applies to both.
_CPU_CRAWL_GB = 30.0


def _cpu_backends() -> list[str]:
    """Names of discovered llama containers with no GPU — they run on the CPU.

    _fit_backends() keys on VRAM and so cannot see these at all. They are still real places
    to run a model: the constraint is system RAM rather than VRAM, and ctx-size is the only
    fit lever, since ngl, n-cpu-moe and tensor-split all presuppose a GPU.
    """
    names = set(_effective_container_names())
    out: list[str] = []
    for d in discover_llama_containers():
        n = d.get("name") or ""
        if n in names and (d.get("vendor") or "").lower() in ("cpu", "", "unknown"):
            from . import hw
            st = hw.stats_for(n)
            if not (st.ok and st.gpu and st.gpu.vram_total_gb > 0):
                out.append(n)
    return out


def vram_fit_chips(size_bytes: int) -> list[dict]:
    """Per-backend fit verdicts: {'name', 'vram_gb', 'verdict', 'ratio_pct'}.

    verdict is one of {fits, tight, oom, impossible}.

    `oom` and `impossible` are genuinely different answers and must not look alike. `oom`
    means "not on the GPU alone" — offload moves layers into system RAM and it runs, slower.
    `impossible` means the model exceeds VRAM **plus** RAM, so there is nowhere for those
    layers to go and no setting rescues it. Rendering both as a red cross against a backend
    name tells the user the GPU is the constraint, when for `impossible` the machine is.

    An impossible model returns a SINGLE machine-level chip rather than one per backend,
    because naming backends implies picking a different one would help.
    """
    from . import hw
    gb = size_bytes / (1024 ** 3)

    # usable, not total: the OS reserve is already spoken for (see hw.usable_ram_gb).
    ram = hw.usable_ram_gb()
    pooled = max((v for v in _fit_backends().values() if v > 0), default=0.0)
    if ram > 0 and pooled > 0 and gb > (pooled + ram):
        return [{
            "name": "this machine",
            "vram_gb": round(pooled, 1),
            "host_ram_gb": round(ram, 1),
            "verdict": "impossible",
            "ratio_pct": round(gb / (pooled + ram) * 100),
            "needs_gb": round(gb),
            "ceiling_gb": round(pooled + ram),
        }]

    out: list[dict] = []
    for name, vram in _fit_backends().items():
        if vram <= 0:
            continue
        ratio = gb / vram
        if ratio < 0.65:
            verdict = "fits"
        elif ratio < 0.9:
            verdict = "tight"
        else:
            verdict = "oom"
        out.append({
            "name": name,
            "vram_gb": vram,
            "verdict": verdict,
            "ratio_pct": round(ratio * 100),
        })

    # CPU backends are sized against usable RAM. "Fits" and "usable" diverge sharply here:
    # generation is RAM-bandwidth-bound, so a large dense model can occupy memory perfectly
    # well and still produce well under a token per second. A plain green tick would be
    # true and misleading, so anything past a threshold gets its own 'slow' verdict.
    for name in _cpu_backends():
        if ram <= 0:
            continue
        ratio = gb / ram
        if ratio >= 1.0:
            verdict = "oom"
        elif gb >= _CPU_CRAWL_GB:
            verdict = "slow"
        elif ratio < 0.75:
            verdict = "fits"
        else:
            verdict = "tight"
        out.append({
            "name": name,
            "vram_gb": ram,
            "verdict": verdict,
            "ratio_pct": round(ratio * 100),
            "is_cpu": True,
        })
    return out


async def test_prompt(container_name: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send a short chat completion to a container's llama-server. Returns dict with reply, tokens, elapsed_s, err."""
    client = _docker_client()
    internal_port: int | None = None
    if client is not None:
        try:
            c = client.containers.get(container_name)
            _, internal_port = _resolve_internal_port(c.attrs or {})
        except (NotFound, DockerException):
            pass
    if internal_port is None:
        return {"ok": False, "err": "container not reachable"}

    # discover a loaded model id first
    loaded, err = await _probe_loaded_model(container_name, internal_port)
    if not loaded:
        return {"ok": False, "err": err or "no model loaded on this backend"}
    model_id = loaded.split(",")[0].strip()

    url = f"http://{container_name}:{internal_port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    import time as _time
    t0 = _time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as ac:
            r = await ac.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "model": model_id}
    elapsed = _time.time() - t0

    reply = ""
    choices = data.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        reply = msg.get("content") or ""
    usage = data.get("usage") or {}
    completion_toks = int(usage.get("completion_tokens") or 0)
    tps = round(completion_toks / elapsed, 1) if elapsed > 0 and completion_toks else None
    return {
        "ok": True,
        "model": model_id,
        "reply": reply,
        "completion_tokens": completion_toks,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "elapsed_s": round(elapsed, 2),
        "tokens_per_s": tps,
    }


def container_logs(name: str, tail: int = 200) -> tuple[bool, str]:
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        c = client.containers.get(name)
    except NotFound:
        return False, "container not found"
    except DockerException as e:
        return False, f"docker error: {e}"
    try:
        raw = c.logs(tail=tail, stdout=True, stderr=True, timestamps=False)
    except DockerException as e:
        return False, f"log fetch failed: {e}"
    return True, raw.decode("utf-8", errors="replace")
