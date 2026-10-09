"""Helpers shared by the JSON API (api.py).

These lived beside the server-rendered pages in main.py until those pages were removed
(#806). They are unchanged: the API still builds its responses from them.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import autoconfig, db, gguf_meta, hw, ini, services
from .config import settings
from .utils import shard_key


def _update_status_map(snap) -> dict[str, dict]:
    """filename -> {'status': 'up-to-date'|'stale'|'unknown', 'remote': iso, 'checked_at': ts, 'delta_days': int|None}"""
    from datetime import datetime, timezone
    checks = db.all_update_checks()
    out: dict[str, dict] = {}
    for g in snap.ggufs:
        row = checks.get(g.display_name)
        if not row:
            continue
        remote = row.get("hf_last_modified") or ""
        if not remote:
            out[g.display_name] = {"status": "unknown", "remote": "", "checked_at": row["checked_at"], "delta_days": None}
            continue
        try:
            core, _, frac = remote.partition(".")
            if frac:
                frac = frac.rstrip("Z")[:6]
                remote_dt = datetime.fromisoformat(f"{core}.{frac}+00:00")
            else:
                remote_dt = datetime.fromisoformat(core.replace("Z", "+00:00"))
        except ValueError:
            out[g.display_name] = {"status": "unknown", "remote": remote, "checked_at": row["checked_at"], "delta_days": None}
            continue
        local_dt = datetime.fromtimestamp(g.mtime, tz=timezone.utc)
        delta_days = (remote_dt - local_dt).days
        status = "stale" if remote_dt > local_dt else "up-to-date"
        out[g.display_name] = {"status": status, "remote": remote_dt.strftime("%Y-%m-%d"), "checked_at": row["checked_at"], "delta_days": delta_days}
    return out


def _badges_for_files(snap) -> dict:
    """{display_name: [badge rows]} for the models list.

    Badges are keyed by models.ini alias while the list is keyed by file, and one file can
    carry several aliases, so the join happens here rather than in the template. Duplicates
    are collapsed per category keeping the newest - a model served under two names would
    otherwise render "Coding 4/5" twice with no way to tell which one you meant.
    """
    by_alias = db.badges_by_alias()
    out: dict = {}
    for g in snap.ggufs:
        best: dict = {}
        for a in (g.aliases or []):
            for b in by_alias.get(a, []):
                cur = best.get(b["category"])
                if cur is None or (b["created_at"] or 0) > (cur["created_at"] or 0):
                    best[b["category"]] = b
        if best:
            out[g.display_name] = [best[k] for k, _ in db.BADGE_CATEGORIES if k in best]
    return out


def _files_present_on_disk() -> set[str]:
    """Every GGUF currently in the models directory, keyed the way download_history stores it.

    History rows use two shapes: bare "model.gguf" for downloads that predate the per-model
    subdirectory layout, and "stem/model.gguf" for everything since. Both are indexed so a
    caller can match either.
    """
    names: set[str] = set()
    try:
        snap = services.snapshot_models_dir()
    except OSError:
        return names
    for g in snap.ggufs:
        for part in list(g.parts) + list(g.companion_parts):
            names.add(part.name)
            if g.subdir:
                names.add(f"{g.subdir}/{part.name}")
    return names


def _downloaded_and_still_present() -> dict[str, list[str]]:
    """{repo_id: [filename, ...]} for repos whose files are ACTUALLY on disk right now.

    download_history is a log, not an inventory: a row stays `done` forever, so a model that
    was downloaded and later deleted keeps being reported as owned. Search then tells you that
    you already have something you do not, which is exactly backwards from the point of the
    flag — it exists to stop you re-downloading, and a false positive stops you downloading
    at all. Intersect the log with the filesystem.
    """
    present = _files_present_on_disk()
    out: dict[str, list[str]] = {}
    for repo, files in db.downloaded_files_by_repo().items():
        kept = [f for f in files if f in present or f.rsplit("/", 1)[-1] in present]
        if kept:
            out[repo] = kept
    return out


_log = logging.getLogger(__name__)
_estimate_reasons_logged: set[str] = set()


def _why_no_estimate(reason: str, **detail: object) -> None:
    """Say once per distinct reason why a repo's context estimates came back empty (noevia#1159).

    The search page used to show "No context estimate" for every file with nothing in the logs to
    say whether the machine had no GPU backend with a measured VRAM, autoconfig raised, or its
    check refused the plan. Once per reason keeps it from repeating for every file of every repo.
    """
    key = reason
    if key in _estimate_reasons_logged:
        return
    _estimate_reasons_logged.add(key)
    _log.warning("context estimate is empty for a repo file: %s%s", reason,
                 "".join(f" {k}={v}" for k, v in detail.items()))


def _preset_estimates(summary: dict, size_bytes: int, mmproj_gb: float = 0.0) -> list[dict]:
    """Fast / Balanced / Long-ctx context estimates for a model we have NOT downloaded.

    Runs the very same autoconfig fit math used on local models, so a search-page estimate
    and the eventual Config recommendation agree instead of being two different guesses.
    Returns [] when there is nothing meaningful to show (no GPU backend, or the GGUF header
    lacks the fields needed to size a KV cache).
    """
    backends = []
    for name, vram in services._fit_backends().items():
        backends.append({
            "name": name, "vendor": "cuda", "vram_gb": vram,
            "gpu_count": hw.gpu_count_for(name), "card_vram_gb": hw.card_vram_gb_for(name),
            "host_ram_gb": hw.host_ram_gb(),
            "baseline": {},
        })
    if not summary:
        return []
    if not backends:
        _why_no_estimate("no llama backend reports GPU VRAM (CPU-only backends are not sized here)",
                         cpu_backends=len(services._cpu_backends()))
        return []
    out: list[dict] = []
    for key, label in (("fast", "Fast"), ("balanced", "Balanced"), ("long-ctx", "Long ctx")):
        try:
            rec = autoconfig.analyze(
                summary=summary, file_size=size_bytes, backends=backends,
                preset=key, models_dir=None, section_name="",
                mmproj_gb_override=(mmproj_gb or None),
            )
        except Exception as e:  # noqa: BLE001 — an estimate must never break the search page
            _why_no_estimate("autoconfig raised", preset=key, error=type(e).__name__)
            return []
        if rec.error or not rec.recommended_ctx:
            _why_no_estimate("autoconfig gave no context" if not rec.error else "autoconfig refused the plan",
                             preset=key, error=(rec.error or "")[:160])
            continue
        chosen = next((p for p in rec.presets if p.key == rec.active_preset), None)
        out.append({
            "key": key,
            "label": label,
            "ctx": rec.recommended_ctx,
            "ctx_h": autoconfig.format_ctx(rec.recommended_ctx),
            "gpu_layers": chosen.gpu_layers if chosen else 0,
            "total_layers": chosen.total_layers if chosen else 0,
            "speed_pct": round((chosen.speed_score if chosen else 1.0) * 100),
            "offload": bool(chosen and chosen.offload_kind),
        })
        # Collapse duplicates: a model that fits fully at native has one real answer, and
        # three identical chips would imply choices that don't exist.
        if len(out) > 1 and out[-1]["ctx"] == out[0]["ctx"] and not out[-1]["offload"]:
            out.pop()
    return out


def _model_stem(filename: str) -> str:
    """Basename minus .gguf extension. Used as the subdir name for one-dir-per-model layout."""
    b = Path(filename).name
    return b[:-5] if b.lower().endswith(".gguf") else b


def _dest_for_main(main_filename: str) -> tuple[str, str]:
    """(subdir, dest_filename) for a MAIN model file. Subdir = filename stem."""
    stem = _model_stem(main_filename)
    return stem, f"{stem}/{Path(main_filename).name}"


def _dest_for_companion(main_stem: str, companion_filename: str) -> str:
    """Place a companion file (mmproj, tokenizer.model, chat_template) inside the main model's subdir."""
    return f"{main_stem}/{Path(companion_filename).name}"


def _resolve_section_gguf(name: str) -> tuple[Path | None, str, str | None]:
    """(gguf_path, model_rel, rel) for a section. rel is models-dir-relative, model_rel is
    the container-absolute /models/... path llama-server wants, or "" for flat layouts.

    Resolution order matters. An explicit `model =` wins over deriving the file from the
    section name, because (a) a section can be renamed to give the model a short API id, and
    (b) a subdir can hold several quants, where re-deriving would silently pick whichever
    sorts first rather than the one this section is actually configured for.
    """
    rel = ini.section_file_rel(name)
    explicit = ((ini.get_section(name) or {}).get("model") or "").strip()
    if explicit and rel and (settings.models_dir / rel).is_file():
        return settings.models_dir / rel, (explicit if "/" in rel else ""), rel

    if rel is not None and "/" in rel:
        subdir_name = rel.rsplit("/", 1)[0]
        subdir_path = settings.models_dir / subdir_name
        if subdir_path.is_dir():
            # Prefer non-mmproj as the main GGUF (mmproj is the companion multimodal projector)
            gguf_files = sorted([q for q in subdir_path.iterdir() if q.is_file() and q.suffix.lower() == ".gguf"])
            main_files = [q for q in gguf_files if "mmproj" not in q.name.lower()] or gguf_files
            if main_files:
                # first shard has the metadata
                return main_files[0], f"/models/{subdir_name}/{main_files[0].name}", rel
        return None, "", rel
    if rel is not None:
        return settings.models_dir / rel, "", rel
    return settings.models_dir / f"{name}.gguf", "", rel


def _gguf_hints_for(name: str) -> tuple[dict[str, str], list[str]]:
    gguf_path, model_rel, rel = _resolve_section_gguf(name)

    if gguf_path is None or not gguf_path.is_file():
        return {}, [f"No GGUF found for `{name}` — cannot suggest defaults from metadata."]
    try:
        summary = gguf_meta.summarize_path(gguf_path)
    except (gguf_meta.GgufMetaError, OSError) as e:
        return {}, [f"Could not read GGUF: {e}"]
    values, hints = ini.suggest_defaults(summary)
    if model_rel:
        values["model"] = model_rel
        # Auto-detect a companion mmproj (multimodal projector — vision, audio, etc.)
        # in the same subdir so multimodal models load out of the box.
        subdir_path = Path(model_rel).parent  # e.g. /models/<stem>
        try:
            for p in Path(str(subdir_path).replace("/models", str(settings.models_dir), 1)).iterdir():
                if p.is_file() and p.suffix.lower() == ".gguf" and "mmproj" in p.name.lower():
                    values["mmproj"] = f"{Path(model_rel).parent.as_posix()}/{p.name}"
                    hints.insert(0, f"Companion mmproj found → `mmproj = {values['mmproj']}` pre-filled.")
                    break
        except OSError:
            pass
        # Distinguish real sharding (multi-part files) from "single file in a subdir"
        _, part_idx, part_total = shard_key(Path(model_rel).name)
        if part_idx is not None and part_total and part_total > 1:
            hints.insert(0, f"Sharded model ({part_total} parts) → `model = {model_rel}` pre-filled to the first shard; llama-server auto-loads the rest.")
        else:
            hints.insert(0, f"Model lives in a subdir → `model = {model_rel}` pre-filled with the absolute path.")
    return values, hints


def _container_baseline(name: str) -> list[str]:
    """Return the Config.Cmd list for a running container, or [] on failure."""
    try:
        client = services._docker_client()
        if client is None:
            return []
        c = client.containers.get(name)
        return list((c.attrs or {}).get("Config", {}).get("Cmd") or [])
    except Exception:  # noqa: BLE001
        return []


def _backend_list() -> list[dict]:
    """Backend descriptors for autoconfig, from the env whitelist or auto-discovery.

    Extracted so the benchmark results page can ask for the same fit estimate the autoconfig
    panel shows, rather than growing a second, drifting copy of this.
    """
    out = []
    for bn in services._effective_container_names():
        vram = hw.vram_gb_for(bn)
        if vram <= 0:
            continue  # CPU backends have no VRAM budget; autoconfig can't do KV math on them yet
        # detect vendor via hw sampler cache (or docker inspect image)
        vendor = "unknown"
        try:
            client = services._docker_client()
            if client is not None:
                img = ((client.containers.get(bn).image.tags or [""]) or [""])[0].lower()
                if "rocm" in img:
                    vendor = "rocm"
                elif "cuda" in img:
                    vendor = "cuda"
        except Exception:  # noqa: BLE001
            pass
        cmd = _container_baseline(bn)
        base = autoconfig.parse_baseline(cmd) if cmd else {}
        out.append({"name": bn, "vendor": vendor, "vram_gb": float(vram),
                    "gpu_count": hw.gpu_count_for(bn), "card_vram_gb": hw.card_vram_gb_for(bn),
                    "host_ram_gb": hw.host_ram_gb(),
                    # The command the baseline was parsed from, so MODEL_AUTOCONFIG=rust can
                    # confirm parse_baseline beside the rest of the recommendation.
                    "baseline": base, "baseline_args": list(cmd)})
    return out


def _predicted_vram_gb(section: str) -> float | None:
    """Autoconfig's budget for this section: weights + KV only, in GB.

    NOT the total the cards will hold. Compute buffers and the projector are budgeted elsewhere
    and are not small - llama.cpp's own estimator puts gemma-4-E4B's compute buffers at 4 GB
    against 2.4 GB of weights - so a measured peak reads several GB above this by construction.
    The chart labels both bars accordingly; presenting them as rivals would make a correct
    estimate look 5 GB wrong.
    """
    try:
        gguf_path, model_rel, rel = _resolve_section_gguf(section)
        if gguf_path is None or not gguf_path.is_file():
            return None
        summary = gguf_meta.summarize_path(gguf_path)
        backends = _backend_list()
        if not backends:
            return None
        subdir = rel.rsplit("/", 1)[0] if rel and "/" in rel else ""
        vals = ini.get_section(section) or {}
        rec = autoconfig.analyze(
            summary=summary, file_size=gguf_path.stat().st_size, backends=backends,
            model_rel=model_rel, current_section=vals, models_dir=settings.models_dir,
            section_name=section, model_subdir=subdir)
        if not rec.frontier:
            return None
        # Compare at the context the section is actually configured for, not at whatever
        # autoconfig would recommend - the measurement was taken under the former.
        try:
            want = int(str(vals.get("ctx-size", "")).strip() or 0)
        except ValueError:
            want = 0
        want = want or rec.recommended_total_ctx or rec.recommended_ctx
        row = min(rec.frontier, key=lambda r: abs(r.ctx - want)) if want else rec.frontier[0]
        return round(row.gpu_gb + row.kv_gb, 2)
    except Exception:  # noqa: BLE001
        return None


def _median(xs: list[float]) -> float:
    """Median, not mean. Same reasoning as everywhere else here: one request that queued behind
    a model load reads as a fraction of the true rate and drags an average with it."""
    xs = sorted(x for x in xs if x)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _bench_charts(run_id: int, results: list, sweeps: list) -> dict:
    """Series for the charts, shaped so the template just hands them to Chart.js.

    Cold and contended rows are excluded from every aggregate: one includes a model load, the
    other ran while something else had the GPU, and neither describes steady-state speed. They
    stay in the table, where the badges explain them.
    """
    import json as _json
    clean = [r for r in results if not r["cold"] and not r["contended"] and not r["err"]]

    aliases = sorted({r["alias"] for r in clean})
    prompts = sorted({r["prompt_name"] for r in clean})

    def _pick(alias, prompt, field, digits):
        """Median for one cell, or None when there is nothing clean to report.

        None rather than 0.0: Chart.js skips a null point but draws a zero, and a zero bar on
        a speed chart reads as "this model produced nothing" when the truth is "every row for
        it was cold, contended or failed". A model that would not load looked like a model
        that ran infinitely slowly.
        """
        vals = [r[field] for r in clean
                if r["alias"] == alias and r["prompt_name"] == prompt and r[field]]
        return round(_median(vals), digits) if vals else None

    gen_series = [{"label": p, "data": [_pick(a, p, "gen_tps", 1) for a in aliases]}
                  for p in prompts]
    ttft_series = [{"label": p, "data": [_pick(a, p, "ttft_ms", 0) for a in aliases]}
                   for p in prompts]

    # Depth decay, straight from llama bench. One line per model per test kind, x = depth.
    depths = sorted({w["n_depth"] or 0 for w in sweeps})
    decay: list[dict] = []
    for a in sorted({w["alias"] for w in sweeps}):
        for kind, want_gen in (("tg", True), ("pp", False)):
            pts = []
            for d in depths:
                m = [w["avg_ts"] for w in sweeps
                     if w["alias"] == a and (w["n_depth"] or 0) == d
                     and bool(w["n_gen"]) == want_gen and w["avg_ts"]]
                pts.append(round(m[0], 1) if m else None)
            if any(p is not None for p in pts):
                decay.append({"label": f"{a} {kind}", "data": pts, "kind": kind})

    # Measured VRAM against what autoconfig predicted. The one chart that can falsify the fit
    # maths: everything else here measures speed, which the estimator never claimed to know.
    vram_labels, vram_measured, vram_predicted = [], [], []
    capacity = 0.0
    try:
        for b in _backend_list():
            capacity = max(capacity, float(b.get("vram_gb") or 0))
    except Exception:  # noqa: BLE001
        capacity = 0.0
    for a in aliases:
        peaks = []
        for r in clean:
            if r["alias"] != a:
                continue
            try:
                peaks.append(sum(_json.loads(r["peak_vram_json"] or "[]")))
            except ValueError:
                pass
        if not peaks:
            continue
        vram_labels.append(a)
        vram_measured.append(round(max(peaks), 2))
        vram_predicted.append(_predicted_vram_gb(a))

    return {
        "capacity_gb": round(capacity, 2),
        "aliases": aliases,
        "gen": gen_series,
        "ttft": ttft_series,
        "depths": depths,
        "decay": decay,
        "vram_labels": vram_labels,
        "vram_measured": vram_measured,
        "vram_predicted": vram_predicted,
    }
