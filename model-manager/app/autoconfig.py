"""System-aware ini recommendation engine.

Given a GGUF summary + on-disk file size + backend inventory (VRAM, vendor,
baseline flags parsed from the container's CLI), produce a Recommendation
that says: which backend, at what ctx, with which values — and why.
"""
from __future__ import annotations
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import autoconfig_core, gguf_meta, ini
from .autoconfig_core import (  # noqa: F401 - the sizing core, re-exported under its old names
    _CACHE_BYTES_PER_ELEM, _CACHE_DEFAULT, _CACHE_RAM_CONVOS, _CACHE_RAM_DEFAULT_MIB,
    _CACHE_RAM_HEADROOM_GB, _CPU_LAYER_PENALTY, _CTX_CANDIDATES, _MODEL_OVERHEAD_SINGLE,
    _MODEL_OVERHEAD_SPLIT, _RESERVE_PER_GPU, _SSM_STATE_BYTES, UNMEASURED_CTX_CAP, FitRow,
    PresetOption, _cache_dtype_bytes, _dense_frontier, _find_fit, _frontier_options,
    _layer_costs, _moe_ratio, _pareto_frontier, _pattern_sample, _period_of,
    _presets_from_dense_frontier, _presets_from_frontier, _split_feasible, cap_context,
    cap_reason_text, kv_cache_bytes, kv_shape, kv_shape_bytes,
    # input prep (slice 2), re-exported under their old names
    _MMPROJ_COMPUTE_GB, _MMPROJ_VRAM_MULT, MAX_BLOCK_COUNT, _card_gb, _kv_first_int,
)
from .config import settings


_PRESET_KEYS = ("fast", "balanced", "long-ctx")

# The rules below moved to autoconfig_core (pure, stdlib-only) so MODEL_AUTOCONFIG=rust can confirm
# them; they are re-exported here under their old names. AUTOCONFIG_DOMAIN: every key autoconfig
# has an opinion about (see there for why Fill depends on it); _NEVER_CLEAR: never displaced.
from .autoconfig_core import (  # noqa: E402,F401 - slices 3-6, re-exported under their old names
    _NEVER_CLEAR, _SHORT_TO_KEY, _SUB_Q4, AUTOCONFIG_DOMAIN, HEAD_MAX_BYTES, MIN_USEFUL_CTX,
    MODE_SPEC_PROFILE, SPEC_PROFILE_BY_KEY, SPEC_PROFILE_KEYS, SPEC_PROFILES, SPEC_SECTION_KEYS,
    SUB_Q4_PARAM_LIMIT, SpecProfile, _domain_gaps, _fmt_ctx, _is_mmproj, _looks_like_draft,
    match_spec_profile, quality_warnings,
)


# ---- baseline parsing (compose command → ini keys) ----


def parse_baseline(cmd_args: list[str]) -> dict[str, str]:
    """Parse the container's llama-server command args into a {ini_key: value} dict (the rule
    is autoconfig_core.parse_baseline; the long options it knows are ini.ALL_KNOWN_KEYS)."""
    return autoconfig_core.parse_baseline(cmd_args, ini.ALL_KNOWN_KEYS)


def baseline_request(backends: list[dict]) -> tuple[list[dict], list] | None:
    """The `check` baseline part for the backends that carry the command they were parsed from
    (helpers._backend_list sets `baseline_args`), with the baselines analyze() used; None when
    no backend carries one."""
    known = sorted(ini.ALL_KNOWN_KEYS)
    reqs, used = [], []
    for b in backends:
        args = b.get("baseline_args") if isinstance(b, dict) else None
        if isinstance(args, list):
            reqs.append({"args": args, "known": known})
            base = b.get("baseline") or {}
            used.append([[k, v] for k, v in base.items()] if isinstance(base, dict) else base)
    return (reqs, used) if reqs else None


# ---- recommendation ----


@dataclass
class BackendPlan:
    name: str
    vendor: str
    vram_gb: float
    rows: list[FitRow]                       # per-ctx breakdown
    max_ctx: int                             # largest ctx that fits with reserve
    fits_at_all: bool




@dataclass
class Recommendation:
    plans: list[BackendPlan]
    recommended_backend: str
    recommended_ctx: int                     # per-session ctx (each user's window)
    recommended_total_ctx: int = 0           # ctx * n_sessions (llama-server --ctx-size)
    n_sessions: int = 1                      # concurrent parallel slots
    values: dict[str, str] = field(default_factory=dict)              # full-form values
    values_minimal: dict[str, str] = field(default_factory=dict)      # only non-baseline-redundant values
    baseline_redundant: dict[str, str] = field(default_factory=dict)  # values already covered by compose baseline
    quirks: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)              # which knobs don't apply to this model
    current_diff: list[str] = field(default_factory=list)             # human-readable diff vs existing section (empty if new)
    displaced: list[str] = field(default_factory=list)                # keys Fill must CLEAR, not set — see note at the assignment
    presets: list[PresetOption] = field(default_factory=list)  # Fast/Balanced/Long-ctx samples
    # Every point on the offload frontier, ascending by ctx. The presets above are three
    # samples from this; the UI exposes the whole curve via a slider so any achievable
    # speed/context tradeoff can be picked directly.
    frontier: list[PresetOption] = field(default_factory=list)
    # True when the model fits entirely on the GPU *at its native context*. In that case
    # there is nothing to trade — offloading layers cannot buy context beyond native — so
    # the UI shows a single option instead of three chips that would all be identical.
    fits_full_gpu: bool = False
    native_ctx: int = 0
    # What memory alone would allow, before the caps in cap_context(). Memory is only an upper
    # bound: a context this machine cannot read in reasonable time is not a usable context.
    estimated_ctx: int = 0
    ctx_cap_reason: str = ""
    # Policy warnings (small context, sub-Q4 quantisation) — advice, not refusals.
    warnings: list[str] = field(default_factory=list)
    # Which preset (if any) the CURRENTLY SAVED ini section corresponds to. Selecting a chip
    # only previews a recommendation — nothing is written until Fill form + Save — so the UI
    # needs to distinguish "previewing" from "actually running" or the two look identical.
    current_preset: str = ""
    has_unsaved: bool = False
    active_preset: str = ""                                     # which preset the current values reflect
    # Speculative decoding, which is orthogonal to the offload presets above: the preset picks
    # where the weights live, this picks how aggressively the draft head guesses ahead.
    spec_profiles: list[SpecProfile] = field(default_factory=list)   # empty = not offerable
    active_spec_profile: str = ""                               # previewed profile ("custom" if hand-tuned)
    current_spec_profile: str = ""                              # what the SAVED section has
    spec_head_rel: str = ""                                     # matched draft head, "" if none
    error: str = ""
    # noevia: projector found for this model (the vision switch applies) and the switch state.
    vision_available: str = ""
    vision: bool = True





def _partition_min_max(costs: list[float], k: int, pinned_gb: float,
                       caps: list[float] | None = None) -> list[int]:
    """Split `costs` into k CONTIGUOUS runs, minimizing the heaviest run RELATIVE to its card.

    Device 0 additionally carries `pinned_gb`: the projector, compute buffer and cuBLAS
    workspace are not layer-split, they all land on the main GPU.

    `caps` are per-card capacities. They matter because the goal is not equal bytes, it is
    equal PRESSURE: on a 24 GB card beside a 12 GB card, an even split wastes half the big
    card and overfills the small one. Minimizing load/capacity balances both cases with one
    rule, and reduces to plain byte-balancing when the cards are identical. Falls back to
    equal weighting when capacities are unknown.

    Returns the layer count per device. Binary-searches the ratio and greedily fills, and
    guarantees every device gets at least one layer whenever there are enough layers to go
    round — a device left with zero would be idle hardware.
    """
    if k <= 1 or not costs:
        return [len(costs)]
    n = len(costs)
    caps = list(caps) if caps and len(caps) == k and all(c > 0 for c in caps) else [1.0] * k

    def feasible(ratio: float) -> list[int] | None:
        counts: list[int] = []
        run = 0.0
        used = 0
        for i, c in enumerate(costs):
            idx = len(counts)                       # device the current run belongs to
            budget = caps[idx] * ratio - (pinned_gb if idx == 0 else 0.0)
            layers_left = n - i
            groups_left = k - idx                   # current device plus the ones after it
            # Close the run when the next layer would overflow this card, or when holding on
            # would leave a later card with no layers at all.
            if used > 0 and groups_left > 1 and (run + c > budget or layers_left <= groups_left - 1):
                counts.append(used)
                run, used = 0.0, 0
                idx = len(counts)
                budget = caps[idx] * ratio - (pinned_gb if idx == 0 else 0.0)
            # Checked AFTER any close, and unconditionally: without this the FINAL device
            # accumulates without limit, and the search "succeeds" by piling every layer onto
            # the last card — which is the very imbalance this function exists to prevent.
            if run + c > budget:
                return None
            run += c
            used += 1
        counts.append(used)
        while len(counts) < k:                      # fewer layers than cards
            counts.append(0)
        return counts if len(counts) == k else None

    lo, hi = 0.0, (sum(costs) + pinned_gb) / min(caps)
    best: list[int] | None = None
    for _ in range(64):
        mid = (lo + hi) / 2
        got = feasible(mid)
        if got is not None:
            best, hi = got, mid
        else:
            lo = mid
    if best is None:
        base, rem = divmod(n, k)
        best = [base + (1 if i < rem else 0) for i in range(k)]
    return best



def _head_sized(path: "Path") -> bool:
    try:
        return path.stat().st_size <= HEAD_MAX_BYTES
    except OSError:
        return False


def _listing(folder: "Path") -> list | None:
    """A directory as the companion-file rules read it (autoconfig_core.pick_file): sorted
    [[name, kind, size]...] - "file" with its size (None where stat failed), "other" for anything
    is_file() rejects, "error" where is_file() itself raised - or None if it cannot be listed."""
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return None
    out: list = []
    for p in entries:
        try:
            is_file = p.is_file()
        except OSError:
            out.append([p.name, "error", None])
            continue
        if not is_file:
            out.append([p.name, "other", None])
            continue
        try:
            size: int | None = p.stat().st_size
        except OSError:
            size = None
        out.append([p.name, "file", size])
    return out


def _pick(calls: list | None, req: dict[str, Any]) -> Any:
    """Apply one companion-file rule; record it when analyze() is collecting them for the check."""
    got = autoconfig_core.pick_file(req)
    if calls is not None:
        calls.append((req, got))
    return got


def _find_mtp(models_dir: "Path | None", section_name: str, subdir: str = "",
              _calls: list | None = None) -> str:
    """Look for THIS model's speculative-decoding draft head. Absolute /models path, or "".

    Same co-location rule as _find_mmproj: a draft head only counts when it sits beside the
    model it drafts for. A head is matched to the WRONG model at best wastes VRAM and at worst
    produces garbage, since speculative decoding requires a shared tokenizer.

    Where several heads are present — repos commonly ship BF16, F16, Q8_0 and Q4_0 of the same
    head — the SMALLEST is chosen. These are tiny to begin with (60-170 MB), they sit in VRAM
    for the whole session, and draft quality below Q8 barely moves the acceptance rate while
    the VRAM saving is real. (The name rules are autoconfig_core.pick_file's.)
    """
    if models_dir is None:
        return ""
    if subdir:
        base = models_dir / subdir
        # Repos often ship the heads in their own MTP/ folder; the downloader may preserve it.
        for sub in (base / "MTP", base / "mtp", base):
            if sub.is_dir():
                rel = sub.relative_to(models_dir).as_posix()
                found = _pick(_calls, {"rule": "mtp_folder", "listing": _listing(sub),
                                       "prefix": f"/models/{rel}/", "section": section_name})
                if found:
                    return found
        return ""
    return _pick(_calls, {"rule": "mtp_flat", "listing": _listing(models_dir), "section": section_name})


def _find_mmproj(models_dir: "Path | None", section_name: str, subdir: str = "",
                 _calls: list | None = None) -> str:
    """Look for THIS model's mmproj companion. Returns an absolute /models path, or "".

    Deliberately strict. A projector only counts as this model's companion when it is
    co-located with the model:

      * model in a subdir  -> only that subdir is searched (the smallest projector there)
      * model at top level -> only top-level projectors whose filename contains the
                              model's full stem

    Earlier revisions also tried fuzzy prefix matching on subdir/file names and an
    "if there's only one mmproj anywhere, use it" fallback. Those were flat-layout
    legacy and actively wrong under one-directory-per-model: `Qwen3.8-27B-Q4_K_M`
    picked up the projector belonging to `Qwen3.8-27B-OBLITERATED-Q4_K_M` because
    both normalize to a shared `qwen3827` prefix, which then tripped the multimodal
    ctx cap on a text-only model. Same-directory co-location is the only signal that
    does not cross-contaminate model families. (The name rules are autoconfig_core.pick_file's.)
    """
    if models_dir is None:
        return ""
    if subdir:
        return _pick(_calls, {"rule": "mmproj_subdir", "listing": _listing(models_dir / subdir),
                              "subdir": subdir})
    return _pick(_calls, {"rule": "mmproj_flat", "listing": _listing(models_dir), "section": section_name})


def _prep_refusal(prep: dict[str, Any]) -> Recommendation:
    """The Recommendation for one of autoconfig_core.prepare()'s early refusals."""
    kind = prep["refuse"]
    if kind == "block_count":
        error = (f"This model's metadata declares an implausible block_count ({prep['layers']}; "
                 f"real models have well under {MAX_BLOCK_COUNT}). The file is corrupt or "
                 "hostile, so no context size is recommended.")
    elif kind == "no_backends":
        # Callers drop backends whose VRAM probes as 0 (CPU-only containers, or a GPU probe that
        # failed), so an empty list is ambiguous with "model too big" downstream - the panel would
        # otherwise tell the user to try a smaller quant when the real problem is that no GPU
        # backend was discovered.
        error = ("No GPU backend available to size against. Either no llama.cpp container was "
                 "discovered, or its VRAM probe returned 0 (CPU-only build, or nvidia-smi / "
                 "rocm-smi not usable inside the container). Check the Containers page: a backend "
                 "must appear there with a non-zero VRAM total before Autoconfig can plan. "
                 "This is not a statement about whether the model would fit.")
    elif kind == "ram":
        # CPU-resident layers live in system memory, so there is nowhere left to put them.
        # Offering a context estimate at 2% speed is worse than saying nothing, because it reads
        # as "slow but possible" when the honest answer is "not on this machine".
        _vram, _ram = prep["vram_gb"], prep["ram_gb"]
        error = (f"This model needs about {prep['model_gb']:.0f} GB, more than this machine's "
                 f"{_vram:.0f} GB VRAM plus {_ram:.0f} GB RAM ({_vram + _ram:.0f} GB total). "
                 "CPU offload moves layers into system memory, so there is no offload setting "
                 "that makes it fit. A smaller quantisation of the same model is the option.")
    elif kind == "unsized_vram":
        # Sizing against zero produces a negative budget, so every context fails and the panel
        # says "doesn't fit at any context" - a verdict on the model when it is really a missing
        # probe. Vulkan images are the common case: they carry neither nvidia-smi nor rocm-smi.
        _names = ", ".join(prep["names"])
        error = (f"Backend(s) found ({_names}) but none report their VRAM, so there is nothing "
                 "to size against. This is usually a Vulkan build: the image ships no vendor "
                 "SMI tool, so VRAM cannot be probed. Declare it instead — set "
                 "GPU_VRAM=<container>:<GB> on the model-loader service (e.g. "
                 "GPU_VRAM=llama-vulkan:16) and restart it. Nothing here says the model "
                 "does not fit; the size of the card is simply unknown.")
    elif kind == "kv_layers":
        # A hybrid model (#1159): attention.head_count_kv marks its non-attention layers with 0,
        # and only those holding a KV cache may be counted. The GGUF summary keeps a prefix of a
        # long array, so which later layers attend is unknown; any count would be a guess.
        known, count, layers = prep["known"], prep["count"], prep["layers"]
        if prep.get("shared"):
            where = (f"it also declares {prep['shared']} layers sharing KV, which cannot be "
                     "matched to its attention layers")
        elif count == layers:
            where = f"only the first {known} of its {count} entries are readable"
        else:
            where = f"it has {count} entries for {layers} layers"
        error = ("Cannot size the KV cache from this model's metadata: it is a hybrid model whose "
                 "attention_head_count_kv marks non-attention layers with 0, and "
                 f"{where}, so its KV cache cannot be sized safely. Autoconfig will not "
                 "guess a context size. Set ctx-size manually in the form and verify it loads.")
    else:
        # kv_cache_bytes() is 0 when block_count / attention_head_count_kv / head_dim are missing
        # or zero (architectures whose keys we don't parse yet). Zero KV silently means "the cache
        # is free", so every candidate fits and the picker returns the largest context in the
        # table - a confident recommendation that OOMs the instant it loads.
        missing = prep["missing"]
        error = ("Cannot size the KV cache from this model's metadata"
                 + (" — missing/zero: " + ", ".join(missing) if missing else "")
                 + ". Autoconfig will not guess a context size, because an unsized KV cache "
                   "would look free and produce a recommendation that OOMs on load. "
                   "Set ctx-size manually in the form and verify it loads.")
    return Recommendation(plans=[], recommended_backend="", recommended_ctx=0, error=error)


def analyze(*,
            summary: dict,
            file_size: int,
            backends: list[dict],           # [{name, vendor, vram_gb, baseline: {ini_key: val}}]
            model_rel: str = "",
            current_section: dict[str, str] | None = None,
            preset: str = "",
            n_sessions: int = 1,
            models_dir: "Path | None" = None,
            section_name: str = "",
            model_subdir: str = "",
            spec_profile: str = "",
            mmproj_gb_override: float | None = None,
            vision: bool = True,
            prompt_tps: float = 0.0,
            prompt_budget_s: float = 120.0,
            verified_ctx: int = 0,
            mode: str = "") -> Recommendation:
    # Clamp n_sessions to a sensible range for a homelab (prepare() clamps again, idempotently).
    n_sessions = max(1, min(int(n_sessions or 1), 8))
    # Input prep and the early refusals (autoconfig_core.prepare): the GGUF summary's fields read
    # with the conversions they always had, the backends filtered, and a refusal when there is
    # nothing honest to recommend. MODEL_AUTOCONFIG=rust confirms it with the Rust port below.
    _prep_in: dict[str, Any] = {"n_sessions": n_sessions, "arch": summary.get("arch"),
                                "model": summary.get("model"), "file_size": file_size,
                                "backends": backends}
    _prep = autoconfig_core.prepare(_prep_in)
    if _prep["refuse"] is not None:
        refusal = _prep_refusal(_prep)
        autoconfig_core.confirm_refusal(
            prep_in=dict(_prep_in, projector={"has_mmproj": False, "mmproj_gb": 0.0, "mtp_gb": 0.0}),
            prep=_prep, model=model_rel or section_name)
        return refusal
    m = summary.get("model") or {}
    layers = _prep["layers"]
    native_ctx = _prep["native_ctx"]
    experts = m.get("expert_count")
    model_gb_raw = _prep["model_gb_raw"]
    _all_backends = backends
    backends = [backends[i] for i in _prep["sized"]]

    # NOTE: Autoconfig DOES wire speculative decoding now, but only from a head that shipped
    # beside these weights - see SpecProfile. The rule this replaced came from pairing an
    # unrelated community draft with a main model, which segfaulted llama.cpp on every quant
    # tried; a head trained against this exact base is a different proposition. `spec-type`
    # is only ever proposed for a section that does not exist yet, so clearing it and saving
    # stays cleared, and the ngram-* profiles need no draft model at all.

    # Detect a companion mmproj (multimodal projector — vision, audio, or other) and
    # reserve its real VRAM cost, rather than capping ctx at an arbitrary ceiling.
    #
    # A projector is not layer-split; it loads entirely onto the main GPU, so pooled-VRAM
    # budgeting overshoots by its full size, and the modality encoder allocates scratch on
    # top. An earlier revision handled that by clamping every multimodal model to 32K ctx,
    # which was crude and wrong: it slashed a 262K-native MoE coder to 32K purely because a
    # projector sat in its directory, and (because so many offload points then tied at the
    # same clamped ctx) collapsed the MoE preset frontier to a single option.
    # Reserving the measured cost instead lets the normal fit math decide.
    # The companion-file rules (autoconfig_core.pick_file) are recorded here, so that
    # MODEL_AUTOCONFIG=rust can confirm each one; the listings and stats stay in this module.
    _files: list = []
    _files_on = models_dir is not None and bool(section_name)
    _mm_saved = (current_section or {}).get("mmproj", "") if _files_on else ""
    _mm_found = ""
    _mm_gb = 0.0
    if _files_on:
        # Size against the projector that will ACTUALLY be loaded. If the preset already
        # names one, that file wins — sizing against a different (possibly smaller) file
        # in the same directory silently under-reserves and the model OOMs on device 0.
        _mm_path = _mm_saved.strip() or (_mm_found := _find_mmproj(models_dir, section_name, model_subdir,
                                                                   _files))
        if _mm_path:
            try:
                _mp = Path(str(_mm_path).replace("/models", str(models_dir), 1))
                _mm_gb = _mp.stat().st_size / (1024 ** 3)
            except OSError:
                _mm_gb = 0.0
    # Callers that can see a projector but have no local file (the HF search estimator, which
    # knows sizes from the repo tree but hasn't downloaded anything) pass its size directly.
    # Without this the estimate silently ignores the projector and reads optimistically high.
    # noevia: vision is a switch. Off, the projector is neither budgeted nor kept in the
    # section, so the fit table shows the context that frees up.
    _proj = _pick(_files, {"rule": "projector", "files": _files_on, "current": _mm_saved, "found": _mm_found,
                           "stat_gb": _mm_gb, "vision": vision, "override": mmproj_gb_override})
    mmproj_rel, mmproj_gb, available_mmproj = _proj["mmproj_rel"], _proj["mmproj_gb"], _proj["available"]

    # A draft head is small but it is real VRAM, resident for the whole session, and it is
    # pinned to the main GPU exactly like the projector. Budget it the same way, or the fit
    # maths approves a context that leaves no room for the head it is about to recommend.
    mtp_rel = ""
    mtp_gb = 0.0
    _mtp_found = ""
    if _files_on:
        mtp_rel = (current_section or {}).get("spec-draft-model", "").strip() \
            or (_mtp_found := _find_mtp(models_dir, section_name, model_subdir, _files))
        if mtp_rel:
            try:
                _mt = Path(str(mtp_rel).replace("/models", str(models_dir), 1))
                mtp_gb = _mt.stat().st_size / (1024 ** 3)
            except OSError:
                mtp_gb = 0.0
    # Built-in MTP layers count as a head for profile purposes; there is no file to place.
    builtin_mtp = int(m.get("nextn_predict_layers") or 0) > 0
    has_mmproj = bool(mmproj_rel)
    # VRAM pinned to the MAIN GPU: projector weights + encoder scratch, the draft head, and the
    # vision ubatch's larger compute buffer (autoconfig_core.main_gpu_reserve_gb).
    mmproj_vram_gb = autoconfig_core.main_gpu_reserve_gb(has_mmproj, mmproj_gb, mtp_gb, layers,
                                                         _prep["hidden"])

    # Symmetric q8_0 K/V, always.
    #
    # An earlier revision escalated V to q4_0 when the model could not reach its context
    # ceiling, on the theory that V tolerates quantization better than K (K feeds QK^T
    # through softmax, which amplifies error; V is only weighted-summed afterwards). The
    # VRAM math was real — 2.1 GiB freed on Qwen3.8-27B at 131K — and short-prompt
    # benchmarks looked fine, which is exactly why it shipped.
    #
    # It was a bad trade. There is no CUDA flash-attention kernel for a q4_0 V cache at
    # this model's 256-wide head dim, so attention silently falls back to CPU. Measured
    # on the same 4178-token prompt: q8_0 = 1737 tok/s prompt eval (2.4 s), q4_0 = 115
    # tok/s (36.2 s). A 15x regression, invisible on short prompts because they do almost
    # no attention work. Whether a given (quant, head_dim, arch) combination has a CUDA
    # kernel is not something we can determine from GGUF metadata, so do not guess:
    # trading a predictable slice of context for an unpredictable 15x cliff is not worth
    # it. Users who want asymmetric KV can set `cache-type-v` by hand and benchmark a
    # LONG prompt — short ones will not reveal the fallback. (autoconfig_core writes both.)

    # The size core (autoconfig_core.size_plan): the per-backend fit sweep, the recommended
    # backend and context, the usable-context cap, the offload presets and the prompt cache.
    # Everything it needs that touches files or the GGUF's raw per-layer arrays is resolved
    # above first.
    _req = {
        "shape": _prep["shape"],
        "layers": layers,
        "native_ctx": native_ctx,
        "model_gb_raw": model_gb_raw,
        "moe_ratio": _prep["moe_ratio"],
        "is_moe": _prep["is_moe"],
        "mmproj_vram_gb": mmproj_vram_gb,
        "n_sessions": n_sessions,
        "backends": autoconfig_core.size_backends(backends),
        # Anything that is not a preset key matches none, exactly as "" does: the dense branch
        # then takes presets[0] (always "fast") and the MoE branch the middle one either way.
        "preset": preset if preset in _PRESET_KEYS else "",
        "prompt_tps": prompt_tps,
        "prompt_budget_s": prompt_budget_s,
        "verified_ctx": verified_ctx,
        "cache_ram_cap_mib": settings.cache_ram_limits[0],
    }
    _plan = autoconfig_core.size_plan(_req)
    plans = [BackendPlan(name=b["name"], vendor=b.get("vendor", ""), vram_gb=float(b["vram_gb"]),
                         rows=[FitRow(**r) for r in p["rows"]], max_ctx=p["max_ctx"],
                         fits_at_all=(p["max_ctx"] > 0))
             for b, p in zip(backends, _plan["plans"])]
    recommended: BackendPlan | None = (plans[_plan["recommended"]]
                                       if _plan["recommended"] is not None else None)
    estimated_ctx = _plan["estimated_ctx"]
    ctx_cap_reason = cap_reason_text(_plan["cap"], prompt_tps=prompt_tps,
                                     prompt_budget_s=prompt_budget_s, verified_ctx=verified_ctx)

    # Resolved once, here, because three later blocks need it and each used to derive it for
    # itself inside its own conditional. That worked only while at least one of those branches
    # was guaranteed to run first; when a cheaper KV estimate meant a model no longer needed
    # expert offload, the branch that happened to bind it was skipped and the one that read it
    # was not, raising UnboundLocalError on a model that had rendered fine the day before.
    rec_backend = next((b for b in backends if b["name"] == recommended.name), None) if recommended else None

    # Speculative decoding, driven by the selected workload profile (see SpecProfile).
    #
    # A draft head is only ever the one that shipped beside these weights. Heads are trained
    # against a specific base, so this is never a generic "make it faster" switch — that is
    # what the earlier attempt got wrong, pairing an unrelated small model as a draft and
    # segfaulting llama-server. draft-mtp rather than draft-simple because these heads are a
    # distinct architecture (gemma4-assistant here: four layers against the base model's
    # forty-two) that llama.cpp drives down a different path. The ngram-* profiles need no
    # head at all — they replay literal repeats out of the prompt — which is why speculation
    # is offered even for models that never shipped one.
    #
    # PROPOSED, not applied. Like everything else here this only pre-fills the form; nothing
    # reaches models.ini until the user saves. That distinction matters more than usual for
    # this setting: speculative decoding is a throughput trade, not a free win. With a poor
    # acceptance rate it is slower than not using it at all, and it interacts with continuous
    # batching unpredictably. Benchmark it.
    #
    # Which profile applies is resolved in three steps, because the panel is a preview: an
    # explicit pick from the UI wins, otherwise the saved section is read back, otherwise a
    # brand-new section gets a conservative default.
    # (autoconfig_core.resolve_spec; MODEL_AUTOCONFIG=rust confirms it.)
    _spec_in: dict[str, Any] = {
        "section": section_name,
        "current": ({k: current_section[k] for k in SPEC_SECTION_KEYS if k in current_section}
                    if current_section is not None else None),
        "spec_profile": spec_profile, "mode": mode, "files": _files_on, "found_mtp": _mtp_found,
        "nextn": m.get("nextn_predict_layers"),
    }
    _spec = autoconfig_core.resolve_spec(_spec_in)
    _saved_spec, _spec_key, _resolved_head = _spec["saved"], _spec["key"], _spec["head"]
    # The speculative keys, in the order analyze() always wrote them; assemble_values places them.
    spec_values: dict[str, str] = dict(_spec["values"])

    # The offload presets for the recommended backend, and what they change (computed by the
    # size core above; the reasoning for each rule is kept beside it in autoconfig_core).
    active_preset = _plan["active_preset"]
    presets: list[PresetOption] = [PresetOption(backend=recommended.name, **p)
                                   for p in _plan["presets"]] if recommended else []
    frontier_opts: list[PresetOption] = [PresetOption(backend=recommended.name, **p)
                                         for p in _plan["frontier"]] if recommended else []
    # True when the model fits entirely on the GPU at its native context: nothing to trade.
    _fits_full_gpu = _plan["fits_full_gpu"]
    rec_ctx = _plan["ctx"] if _plan["sized"] else _plan["initial_ctx"]

    # The values (autoconfig_core.assemble_values; the reasoning for each setting is kept beside
    # it there): context, slots, offload, cache, prompt cache, templating and reasoning flags,
    # the projector and its batch/ubatch/image-token bounds, rope, split-mode.
    _cs = current_section or {}
    _values_in: dict[str, Any] = {
        "model_rel": model_rel, "n_sessions": n_sessions,
        # Only its truthiness is read; templates run past the Rust reader's 4096-char strings.
        "chat_template": bool(summary.get("chat_template")),
        "features": summary.get("chat_template_features"),
        "section": section_name, "vision": vision,
        "current": {k: _cs[k] for k in ("mmproj", "ubatch-size", "batch-size") if k in _cs},
        "mmproj_rel": mmproj_rel, "has_mmproj": has_mmproj,
        "spec": [[k, v] for k, v in spec_values.items()],
        "plan": {k: _plan[k] for k in ("initial_ctx", "sized", "ctx", "ngl", "fit", "cache_ram")},
        "rope": m, "native_ctx": native_ctx,
        "rec_gpu_count": (_req["backends"][_req["backends"][_plan["recommended"]]["same_as"]]["gpu_count"]
                          if recommended else None),
    }
    values = autoconfig_core.assemble_values(_values_in)

    # The report beside the values (autoconfig_core.present; the reasoning for each item is kept
    # beside it there): baseline redundancy and conflicts, the quirks, the knobs that do not
    # apply, the preset the SAVED section matches, the diff against it (#1152: changed keys in
    # `values` order, then the superseded ones sorted), the keys Fill must clear, the warnings.
    _present_in: dict[str, Any] = {
        "values": [[k, v] for k, v in values.items()],
        "current": current_section,
        "recommended": recommended is not None,
        "rec_name": recommended.name if recommended else None,
        "rec_backend": ({k: rec_backend[k] for k in ("baseline", "gpu_count") if k in rec_backend}
                        if rec_backend is not None else None),
        "arch": summary.get("arch"), "chat_template": bool(summary.get("chat_template")),
        "features": summary.get("chat_template_features"),
        "n_sessions": n_sessions, "rec_ctx": rec_ctx,
        "has_mmproj": has_mmproj, "mmproj_vram_gb": mmproj_vram_gb, "mmproj_gb": mmproj_gb,
        "rope": m, "native_ctx": native_ctx, "layers": layers, "experts": experts,
        "offload": list(_plan["offload"]),
        "presets": [{"key": p.key, "ctx": p.ctx, "offload_kind": p.offload_kind, "ngl": p.ngl,
                     "n_cpu_moe": p.n_cpu_moe} for p in presets],
        "model_rel": model_rel, "general": summary.get("general"),
    }
    _present = autoconfig_core.present(_present_in)
    _baselines = baseline_request(_all_backends)
    try:
        autoconfig_core.confirm(
            prep_in=dict(_prep_in, projector={"has_mmproj": has_mmproj, "mmproj_gb": mmproj_gb,
                                              "mtp_gb": mtp_gb}),
            prep=dict(_prep, mmproj_vram_gb=mmproj_vram_gb, backends=_req["backends"]),
            size_req=_req, plan=_plan, values_in=_values_in, values=values,
            extra_in={"spec": _spec_in, "files": [c for c, _ in _files] or None,
                      "present": _present_in, "baseline": _baselines[0] if _baselines else None},
            extra={"spec": _spec, "files": [r for _, r in _files], "present": _present,
                   "baseline": _baselines[1] if _baselines else None},
            model=model_rel or section_name)
    except autoconfig_core.AutoconfigCoreError as e:
        return Recommendation(
            plans=[], recommended_backend="", recommended_ctx=0,
            error=(f"The Rust autoconfig check (MODEL_AUTOCONFIG=rust) could not confirm this "
                   f"recommendation ({e}), so none is offered rather than one that might be "
                   "larger than this machine can hold. Set MODEL_AUTOCONFIG=python to use the "
                   "Python autoconfig alone, and report the model so the two can be reconciled."),
        )

    current_diff = _present["current_diff"]
    return Recommendation(
        plans=plans,
        displaced=_present["displaced"],
        recommended_backend=(recommended.name if recommended else ""),
        recommended_ctx=rec_ctx,
        warnings=_present["warnings"],
        estimated_ctx=estimated_ctx,
        ctx_cap_reason=ctx_cap_reason,
        recommended_total_ctx=rec_ctx * n_sessions if rec_ctx > 0 else 0,
        n_sessions=n_sessions,
        values=values,
        values_minimal=dict(_present["minimal"]),
        baseline_redundant=dict(_present["redundant"]),
        quirks=_present["quirks"],
        unavailable=_present["unavailable"],
        current_diff=current_diff,
        presets=presets,
        frontier=frontier_opts,
        fits_full_gpu=_fits_full_gpu,
        native_ctx=native_ctx,
        current_preset=_present["current_preset"],
        has_unsaved=bool(current_diff),
        active_preset=active_preset,
        # Offered for any named section, not just ones with a head: the n-gram strategies
        # need no draft model at all, which is the only speculative option most of these
        # models have.
        spec_profiles=list(SPEC_PROFILES) if section_name else [],
        active_spec_profile=_spec_key if section_name else "",
        current_spec_profile=_saved_spec,
        spec_head_rel=_resolved_head,
        vision_available=available_mmproj,
        vision=vision,
    )


def format_ctx(n: int) -> str:
    return _fmt_ctx(n)
