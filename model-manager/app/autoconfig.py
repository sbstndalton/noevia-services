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


# Every key autoconfig has an opinion about. For each one it either SETS a value or wants the
# key GONE — nothing here may survive a Fill untouched. The list is what makes "Fill form" honest:
# Fill writes the keys present in `values` and clears the rest of this set, so a recommendation
# cannot leave a stale placement pin behind and report success.
#
# This replaces a hand-maintained set that held only {cpu-moe, n-cpu-moe}. Because ngl and
# tensor-split were missing from it, the panel reported "n-cpu-moe -> unset" while Fill silently
# left ngl=999 and tensor-split=17,13 in place — Save then wrote them straight back and the user
# saw no change at all. _domain_gaps() below guards against that returning: any key assigned but
# not declared here is surfaced as a quirk rather than silently escaping.
AUTOCONFIG_DOMAIN: frozenset[str] = frozenset({
    # placement / fit — the ones that decide whether the model loads
    "ngl", "tensor-split", "split-mode", "cpu-moe", "n-cpu-moe", "fit",
    # context and cache
    "ctx-size", "parallel", "batch-size", "ubatch-size", "keep", "cache-reuse", "cache-ram",
    "cache-type-k", "cache-type-v", "flash-attn", "cont-batching", "context-shift",
    # rope
    "rope-scaling", "rope-scale",
    # speculative decoding
    "spec-type", "spec-draft-model", "spec-draft-ngl",
    # multimodal
    "mmproj", "mmproj-offload", "image-max-tokens",
    # templating / reasoning
    "jinja", "chat-template-kwargs", "reasoning", "reasoning-format", "reasoning-preserve",
})

# Clearing this would break the section outright — a section with no model file is not a model.
# It stays out of the displaced list even when a recommendation happens not to set it.
_NEVER_CLEAR: frozenset[str] = frozenset({"model"})


def _domain_gaps(values: dict[str, str]) -> list[str]:
    """Keys a recommendation set that AUTOCONFIG_DOMAIN does not declare.

    Non-fatal on purpose: a missing declaration should be visible, not a 500 on a page the user
    is trying to read. Surfaced as a quirk so it gets noticed and fixed.
    """
    return sorted(set(values) - AUTOCONFIG_DOMAIN - _NEVER_CLEAR)


# ---- baseline parsing (compose command → ini keys) ----

_SHORT_TO_KEY = {
    "-ngl": "ngl", "-fa": "flash-attn", "-ctk": "cache-type-k",
    "-ctv": "cache-type-v", "-np": "parallel", "-c": "ctx-size",
    "-b": "batch-size", "-ub": "ubatch-size", "-t": "threads",
    "-tb": "threads-batch", "-sm": "split-mode", "-mg": "main-gpu",
    "-fit": "fit", "-fitt": "fit-target", "-fitc": "fit-ctx",
    "-cmoe": "cpu-moe", "-ncmoe": "n-cpu-moe",
    "-kvo": "kv-offload",
}


def parse_baseline(cmd_args: list[str]) -> dict[str, str]:
    """Parse the container's llama-server command args into a {ini_key: value} dict."""
    out: dict[str, str] = {}
    n = len(cmd_args)
    for i, a in enumerate(cmd_args):
        key: str | None = None
        if a in _SHORT_TO_KEY:
            key = _SHORT_TO_KEY[a]
        elif a.startswith("--"):
            k = a[2:]
            if k in ini.ALL_KNOWN_KEYS:
                key = k
        if not key:
            continue
        nxt = cmd_args[i + 1] if i + 1 < n else None
        if nxt is not None and not nxt.startswith("-"):
            out[key] = nxt
        else:
            out[key] = "true"
    return out


# ---- recommendation ----


@dataclass
class BackendPlan:
    name: str
    vendor: str
    vram_gb: float
    rows: list[FitRow]                       # per-ctx breakdown
    max_ctx: int                             # largest ctx that fits with reserve
    fits_at_all: bool




@dataclass(frozen=True)
class SpecProfile:
    """A workload-shaped preset for speculative decoding.

    Speculative decoding is a bet, not a free win: a cheap drafter proposes N tokens, the main
    model verifies all N in a single forward pass, and everything from the first mismatch
    onward is discarded. Whether the bet pays depends on how PREDICTABLE the text is — which
    is a property of the workload, not of the model. The same weights drafting Python and
    drafting prose accept at wildly different rates (measured here: 31% to 100% on one model),
    so the knobs are grouped by what you intend to do with it rather than left as eleven
    independent numbers nobody can reason about in isolation.

    Two levers do the real work:
      n-max  how deep to guess. Multiplies the win on predictable text and the waste on
             unpredictable text. llama.cpp defaults to 3.
      p-min  how confident the drafter must be before it bothers. Raising it declines the
             marginal bets, which is what you want when acceptance is poor.

    These are reasoned starting points, not measured optima — llama-server prints the real
    acceptance rate per request and that is the number to tune against.
    """
    key: str
    label: str
    icon: str
    blurb: str                  # one line under the chip
    spec_type: str              # "" = speculation off
    needs_head: bool            # requires a draft/MTP model to sit beside the weights
    knobs: dict[str, str] = field(default_factory=dict)


# The knobs each profile sets. Anything not listed is deliberately left unset so llama-server
# applies its own default; writing out a value identical to the default only creates noise in
# the diff and a second place to keep in sync when upstream changes it.
SPEC_PROFILES: tuple[SpecProfile, ...] = (
    SpecProfile(
        key="off", label="Off", icon="x",
        blurb="No speculation. Always correct, never slower than the model itself.",
        spec_type="", needs_head=False),
    SpecProfile(
        key="balanced", label="Balanced", icon="gauge",
        blurb="Mixed use. Barely above llama.cpp's own defaults, with a mild confidence gate.",
        spec_type="draft-mtp", needs_head=True,
        knobs={"spec-draft-n-max": "4", "spec-draft-p-min": "0.25"}),
    SpecProfile(
        key="coding", label="Coding", icon="terminal",
        blurb="Deep drafts, loose gate. Code repeats itself, so guesses land and guessing further pays.",
        spec_type="draft-mtp,ngram-simple", needs_head=True,
        knobs={"spec-draft-n-max": "8", "spec-draft-n-min": "1", "spec-draft-p-min": "0.05"}),
    SpecProfile(
        key="writing", label="Creative writing", icon="pencil",
        blurb="Shallow drafts, tight gate. Prose is unpredictable, so bad bets cost more than good ones pay.",
        spec_type="draft-mtp", needs_head=True,
        knobs={"spec-draft-n-max": "2", "spec-draft-p-min": "0.60"}),
    SpecProfile(
        key="ngram", label="N-gram only", icon="hash",
        blurb="No draft head required. Replays literal repeats from the prompt: strong on edits and refactors, inert on new prose.",
        spec_type="ngram-simple", needs_head=False),
)

# n-gram strategies read their draft length from --spec-ngram-*-size-m, not --spec-draft-n-max
# (upstream's removal note for --draft-max spells the split out). Setting the draft-model knobs
# for a head-free profile would write keys llama-server ignores.
SPEC_PROFILE_BY_KEY: dict[str, SpecProfile] = {p.key: p for p in SPEC_PROFILES}

# Default speculative profile per workload when a head (file or built-in) is available. Agent turns
# are mostly tool JSON and short answers, which draft like code; chat is mixed.
MODE_SPEC_PROFILE: dict[str, str] = {"chat": "balanced", "code": "coding", "agent": "coding", "writing": "writing"}

# Keys a profile owns. Switching profiles must clear whatever the previous one set, or a move
# from Coding to Writing would silently keep n-min = 1 from the profile that was abandoned.
SPEC_PROFILE_KEYS: tuple[str, ...] = (
    "spec-draft-n-max", "spec-draft-n-min", "spec-draft-p-min",
)

# Fold the profile-owned keys into the domain by REFERENCE rather than restating them, so the
# two can't drift. These are written from a profile's `knobs` dict rather than by a literal
# `values[...] =`, which is exactly why enumerating assignments by eye missed them — the
# _domain_gaps() guard caught all three on its first run.
AUTOCONFIG_DOMAIN = AUTOCONFIG_DOMAIN | frozenset(SPEC_PROFILE_KEYS)


def match_spec_profile(section: dict[str, str] | None) -> str:
    """Which profile an existing section corresponds to: a key, "custom", or "".

    Returns "" for a section that does not exist yet (no opinion — the caller picks a default).
    A saved section with no spec-type is "off": that is a deliberate opt-out, and re-proposing
    speculation would make turning it off impossible to make stick.
    """
    if section is None:
        return ""
    stype = (section.get("spec-type") or "").strip()
    if not stype or stype == "none":
        return "off"
    for prof in SPEC_PROFILES:
        if prof.spec_type != stype:
            continue
        if all((section.get(k) or "").strip() == v for k, v in prof.knobs.items()):
            # Any knob the profile does not set must also be absent, or a hand-tuned section
            # that merely happens to share a spec-type would be mislabelled as this profile
            # and get its edits overwritten on the next Fill.
            extra = [k for k in SPEC_PROFILE_KEYS
                     if k not in prof.knobs and (section.get(k) or "").strip()]
            if not extra:
                return prof.key
    return "custom"


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


def _fmt_ctx(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        k = n // 1024
        return f"{k}K"
    return f"{n:,}"



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



# Real draft/MTP heads are tens to a few hundred MB; anything larger with "mtp" in its name is a
# full model build that includes MTP layers (e.g. Unsloth's "-MTP-GGUF" repos), not a head.
HEAD_MAX_BYTES = 2 * 1024 ** 3


def _head_sized(path: "Path") -> bool:
    try:
        return path.stat().st_size <= HEAD_MAX_BYTES
    except OSError:
        return False


def _looks_like_draft(filename: str) -> bool:
    """Heuristic: is this GGUF a speculative-decoding draft head?

    Match on isolated tokens (`-draft-`, `-mtp-`, or the filename starting/ending with those)
    so we don't false-match main models that have "noMTP" or "nodraft" in their name — those
    are variants explicitly WITHOUT MTP support (e.g. `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf`).
    """
    low = filename.lower()
    # Explicit "no MTP" or "no draft" variants — these are main models, not drafts
    for negation in ("nomtp", "no-mtp", "no_mtp", "nodraft", "no-draft", "no_draft"):
        if negation in low:
            return False
    # Positive matches — token boundaries only
    for token in ("-draft-", "-draft.", "_draft_", ".draft.", "-mtp-", "_mtp_", ".mtp."):
        if token in low:
            return True
    # Also match "draft" or "mtp" as leading/trailing tokens
    stem = low[:-5] if low.endswith(".gguf") else low
    parts = stem.replace("_", "-").split("-")
    return parts and (parts[0] in ("draft", "mtp") or parts[-1] in ("draft", "mtp"))


def _find_mtp(models_dir: "Path | None", section_name: str, subdir: str = "") -> str:
    """Look for THIS model's speculative-decoding draft head. Absolute /models path, or "".

    Same co-location rule as _find_mmproj: a draft head only counts when it sits beside the
    model it drafts for. A head is matched to the WRONG model at best wastes VRAM and at worst
    produces garbage, since speculative decoding requires a shared tokenizer.

    Where several heads are present — repos commonly ship BF16, F16, Q8_0 and Q4_0 of the same
    head — the SMALLEST is chosen. These are tiny to begin with (60-170 MB), they sit in VRAM
    for the whole session, and draft quality below Q8 barely moves the acceptance rate while
    the VRAM saving is real.
    """
    if models_dir is None:
        return ""

    def _pick(folder: "Path", prefix: str) -> str:
        cands = []
        try:
            for p in sorted(folder.iterdir()):
                if not (p.is_file() and p.suffix.lower() == ".gguf"):
                    continue
                if "mmproj" in p.name.lower() or not _looks_like_draft(p.name):
                    continue
                # The model itself is never its own head: "-MTP-" builds carry the name too.
                if p.stem == section_name or not _head_sized(p):
                    continue
                cands.append(p)
        except OSError:
            return ""
        if not cands:
            return ""
        best = min(cands, key=lambda q: q.stat().st_size)
        return f"{prefix}{best.name}"

    if subdir:
        base = models_dir / subdir
        # Repos often ship the heads in their own MTP/ folder; the downloader may preserve it.
        for sub in (base / "MTP", base / "mtp", base):
            if sub.is_dir():
                rel = sub.relative_to(models_dir).as_posix()
                found = _pick(sub, f"/models/{rel}/")
                if found:
                    return found
        return ""

    stem_key = section_name.lower().replace("-", "").replace("_", "").replace(".", "")
    if not stem_key:
        return ""
    try:
        cands = [p for p in sorted(models_dir.iterdir())
                 if p.is_file() and p.suffix.lower() == ".gguf"
                 and _looks_like_draft(p.name) and "mmproj" not in p.name.lower()
                 and stem_key in p.name.lower().replace("-", "").replace("_", "").replace(".", "")]
    except OSError:
        return ""
    if not cands:
        return ""
    return f"/models/{min(cands, key=lambda q: q.stat().st_size).name}"


def _find_mmproj(models_dir: "Path | None", section_name: str, subdir: str = "") -> str:
    """Look for THIS model's mmproj companion. Returns an absolute /models path, or "".

    Deliberately strict. A projector only counts as this model's companion when it is
    co-located with the model:

      * model in a subdir  -> only that subdir is searched
      * model at top level -> only top-level projectors whose filename contains the
                              model's full stem

    Earlier revisions also tried fuzzy prefix matching on subdir/file names and an
    "if there's only one mmproj anywhere, use it" fallback. Those were flat-layout
    legacy and actively wrong under one-directory-per-model: `Qwen3.8-27B-Q4_K_M`
    picked up the projector belonging to `Qwen3.8-27B-OBLITERATED-Q4_K_M` because
    both normalize to a shared `qwen3827` prefix, which then tripped the multimodal
    ctx cap on a text-only model. Same-directory co-location is the only signal that
    does not cross-contaminate model families.
    """
    if models_dir is None:
        return ""

    def _is_mmproj(name: str) -> bool:
        return name.lower().endswith(".gguf") and "mmproj" in name.lower()

    # Model lives in its own directory: its companion is in there or it has none.
    # A repo often ships several precisions of the same projector (BF16 / F16 / F32).
    # Prefer the SMALLEST: the projector is pinned to the main GPU and competes directly
    # with the KV cache for that card's VRAM, and an F32 projector is typically 2x the
    # size of the BF16 one for no meaningful quality gain.
    if subdir:
        subdir_path = models_dir / subdir
        best: tuple[int, str] | None = None
        try:
            for p in sorted(subdir_path.iterdir()):
                if p.is_file() and _is_mmproj(p.name):
                    sz = p.stat().st_size
                    if best is None or sz < best[0]:
                        best = (sz, p.name)
        except OSError:
            pass
        return f"/models/{subdir}/{best[1]}" if best else ""

    # Flat layout: require the projector filename to carry the model's full stem,
    # so `foo-Q4_K_M.gguf` matches `foo-Q4_K_M-mmproj.gguf` but never a sibling model.
    stem_key = section_name.lower().replace("-", "").replace("_", "").replace(".", "")
    if not stem_key:
        return ""
    try:
        for p in sorted(models_dir.iterdir()):
            if not (p.is_file() and _is_mmproj(p.name)):
                continue
            name_key = p.name.lower().replace("-", "").replace("_", "").replace(".", "")
            if stem_key in name_key:
                return f"/models/{p.name}"
    except OSError:
        pass
    return ""


# Operator policy (2026-09-17): a context this small leaves no room for tool definitions, results and
# a conversation, and a quantisation below Q4 costs more quality than it saves memory on a model
# this size. Both are warnings on the recommendation, never silent refusals.
MIN_USEFUL_CTX = 16384
SUB_Q4_PARAM_LIMIT = 100_000_000_000
_SUB_Q4 = re.compile(r'(?:^|[-_.])(?:UD-)?(IQ[123]\w*|Q[123](?:_[\w]+)*)(?:[-_.]|$)', re.I)


def quality_warnings(*, model_rel: str, params: float | int | None, recommended_ctx: int, native_ctx: int = 0) -> list[str]:
    """Settings that will disappoint before they are measured: too little context, too few bits."""
    out = []
    name = (model_rel or "").rsplit("/", 1)[-1]
    quant = (_SUB_Q4.search(name) or [None, None])[1] if name else None
    if quant and (not params or float(params) < SUB_Q4_PARAM_LIMIT):
        out.append(f"{quant.upper()} is below Q4; on a model this size that usually costs more quality than the memory it saves.")
    usable = recommended_ctx or native_ctx
    if usable and usable < MIN_USEFUL_CTX:
        out.append(f"{usable:,} tokens of context is little use once tool definitions and results are in the prompt; {MIN_USEFUL_CTX:,} is a sensible floor.")
    return out



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
    arch = (summary.get("arch") or "").lower()
    m = summary.get("model") or {}
    layers = _prep["layers"]
    native_ctx = _prep["native_ctx"]
    experts = m.get("expert_count")
    model_gb_raw = _prep["model_gb_raw"]
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
    mmproj_rel = ""
    mmproj_gb = 0.0
    if models_dir is not None and section_name:
        # Size against the projector that will ACTUALLY be loaded. If the preset already
        # names one, that file wins — sizing against a different (possibly smaller) file
        # in the same directory silently under-reserves and the model OOMs on device 0.
        mmproj_rel = (current_section or {}).get("mmproj", "").strip() \
            or _find_mmproj(models_dir, section_name, model_subdir)
        if mmproj_rel:
            try:
                _mp = Path(str(mmproj_rel).replace("/models", str(models_dir), 1))
                mmproj_gb = _mp.stat().st_size / (1024 ** 3)
            except OSError:
                mmproj_gb = 0.0
    # Callers that can see a projector but have no local file (the HF search estimator, which
    # knows sizes from the repo tree but hasn't downloaded anything) pass its size directly.
    # Without this the estimate silently ignores the projector and reads optimistically high.
    # noevia: vision is a switch. Off, the projector is neither budgeted nor kept in the
    # section, so the fit table shows the context that frees up.
    available_mmproj = mmproj_rel
    if not vision:
        mmproj_rel, mmproj_gb = "", 0.0
    if vision and mmproj_gb_override is not None and mmproj_gb_override > 0:
        mmproj_gb = mmproj_gb_override
        mmproj_rel = mmproj_rel or "(remote projector)"

    # A draft head is small but it is real VRAM, resident for the whole session, and it is
    # pinned to the main GPU exactly like the projector. Budget it the same way, or the fit
    # maths approves a context that leaves no room for the head it is about to recommend.
    mtp_rel = ""
    mtp_gb = 0.0
    if models_dir is not None and section_name:
        mtp_rel = (current_section or {}).get("spec-draft-model", "").strip()             or _find_mtp(models_dir, section_name, model_subdir)
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
    _saved_spec = match_spec_profile(current_section)
    _spec_key = (spec_profile or "").strip()
    if _spec_key not in SPEC_PROFILE_BY_KEY and _spec_key != "custom":
        # No explicit pick. Fall back to what is saved; a new section gets Balanced when a
        # head was found beside the weights and Off when there is nothing to draft with.
        _spec_key = _saved_spec or (MODE_SPEC_PROFILE.get(mode, "balanced") if (mtp_rel or builtin_mtp) else "off")
    # A profile that needs a head but has none cannot run — llama-server would start and then
    # fail to load the draft. Fall back rather than offering a configuration that cannot work.
    _resolved_head = (current_section or {}).get("spec-draft-model", "").strip() or mtp_rel
    _prof = SPEC_PROFILE_BY_KEY.get(_spec_key)
    if _prof and _prof.needs_head and not _resolved_head and not builtin_mtp:
        _prof = SPEC_PROFILE_BY_KEY["off"]
        _spec_key = "off"

    # The speculative keys, in the order analyze() always wrote them; assemble_values places them.
    spec_values: dict[str, str] = {}
    if section_name:
        if _spec_key == "custom":
            # Hand-tuned. Echo every spec key verbatim: Fill writes exactly `values`, so a
            # setting that is not repeated here is silently erased on the next save.
            for _k in ("spec-type", "spec-draft-model", "spec-draft-ngl", *SPEC_PROFILE_KEYS):
                _v = (current_section or {}).get(_k, "").strip()
                if _v:
                    spec_values[_k] = _v
        elif _prof is not None:
            # Every owned key is written even when empty, so switching profiles CLEARS what
            # the previous one set. Without this, Coding -> Writing would leave n-min = 1
            # behind and the result would match neither profile.
            spec_values["spec-type"] = _prof.spec_type
            for _k in SPEC_PROFILE_KEYS:
                spec_values[_k] = _prof.knobs.get(_k, "")
            if _prof.spec_type and _prof.needs_head and not _resolved_head:
                # Built-in nextn layers: the engine drafts from the model itself.
                spec_values["spec-draft-model"] = ""
                spec_values["spec-draft-ngl"] = ""
            elif _prof.spec_type and _prof.needs_head:
                spec_values["spec-draft-model"] = _resolved_head
                # Without this the head lands on the CPU, and a draft evaluated on the CPU is
                # slower than the main model it is meant to be racing ahead of.
                spec_values["spec-draft-ngl"] = (current_section or {}).get("spec-draft-ngl", "").strip() or "999"
            else:
                # n-gram strategies have no model to place, and Off has nothing at all.
                spec_values["spec-draft-model"] = ""
                spec_values["spec-draft-ngl"] = ""

    rope_type = m.get("rope_scaling_type")

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
    features = summary.get("chat_template_features") or {}
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
    rope_owned = gguf_meta.rope_owned_by_gguf(m)
    try:
        autoconfig_core.confirm(
            prep_in=dict(_prep_in, projector={"has_mmproj": has_mmproj, "mmproj_gb": mmproj_gb,
                                              "mtp_gb": mtp_gb}),
            prep=dict(_prep, mmproj_vram_gb=mmproj_vram_gb, backends=_req["backends"]),
            size_req=_req, plan=_plan, values_in=_values_in, values=values,
            model=model_rel or section_name)
    except autoconfig_core.AutoconfigCoreError as e:
        return Recommendation(
            plans=[], recommended_backend="", recommended_ctx=0,
            error=(f"The Rust autoconfig check (MODEL_AUTOCONFIG=rust) could not confirm this "
                   f"recommendation ({e}), so none is offered rather than one that might be "
                   "larger than this machine can hold. Set MODEL_AUTOCONFIG=python to use the "
                   "Python autoconfig alone, and report the model so the two can be reconciled."),
        )

    # baseline-redundant: any key that matches the recommended backend's baseline
    baseline_redundant: dict[str, str] = {}
    minimal = dict(values)
    if recommended:
        base = (rec_backend or {}).get("baseline") or {}
        for k, v in list(values.items()):
            bv = base.get(k)
            if bv is not None and str(bv).lower() == str(v).lower():
                baseline_redundant[k] = v
                minimal.pop(k, None)

    # ensure minimal keeps the essential differentiators even if redundant on paper
    for essential in ("model", "ctx-size", "jinja", "cpu-moe", "n-cpu-moe",
                      "parallel", "cont-batching", "context-shift", "keep",
                      "batch-size", "ubatch-size", "cache-reuse", "reasoning",
                      "reasoning-preserve", "mmproj", "mmproj-offload", "image-max-tokens",
                      # tensor-split is a correctness setting under expert offload, not a
                      # tuning nicety: dropping it restores the even layer split that OOMs.
                      "tensor-split", "split-mode"):
        if essential in values and essential not in minimal:
            minimal[essential] = values[essential]

    # quirks
    quirks: list[str] = []

    # Baseline CONFLICTS — the container's own CLI args win over anything in models.ini,
    # so a preset value that disagrees with the compose command is silently discarded.
    # This bit us hard: `-np 1` in the compose command overrode `parallel = 2` in the ini,
    # so multi-session serving never actually ran, and `-ctv q8_0` overrode `cache-type-v`.
    # Surface it loudly instead of letting the preset look like it took effect.
    if recommended:
        _rb2 = next((b for b in backends if b["name"] == recommended.name), None)
        _base2 = (_rb2 or {}).get("baseline") or {}
        conflicts = [
            f"`{k}`: preset wants {v}, container forces {_base2[k]}"
            for k, v in values.items()
            if k in _base2 and str(_base2[k]).lower() != str(v).lower()
        ]
        if conflicts:
            quirks.append(
                "CONFLICT — the container's CLI args override models.ini, so these preset values "
                "will NOT take effect: " + "; ".join(conflicts) + ". "
                f"Fix by removing those flags from the `{recommended.name}` command in your compose file "
                "so per-model presets can control them. Keep only router-level args there "
                "(--models-dir, --models-preset, --host, --port, --models-max)."
            )

    if arch.startswith("gemma"):
        quirks.append("Gemma sliding-window attention: the ctx→KV math above assumes swa-full=false (default). "
                      "Enabling swa-full multiplies full-attn KV ~5× and will OOM.")
    # (MoE hint is emitted later, tailored to whichever offload the recommendation actually applies)
    if not summary.get("chat_template"):
        quirks.append("No embedded chat template — you'll need to set `chat-template` or `chat-template-file` "
                      "manually to get correct multi-turn formatting.")

    # Chat-template reasoning-capability hints (from template scanning)
    detected: list[str] = []
    if features.get("accepts_enable_thinking"):
        detected.append("`enable_thinking` kwarg (set true/false)")
    if features.get("accepts_reasoning_effort"):
        detected.append("`reasoning_effort` kwarg (low/medium/high)")
    if features.get("accepts_preserve_thinking"):
        detected.append("`preserve_thinking` kwarg (keep reasoning across turns)")
    if detected:
        quirks.append(
            "Chat template supports " + ", ".join(detected) + ". "
            "Thinking is enabled via the dedicated `reasoning = on` flag (setting enable_thinking through "
            "chat-template-kwargs is deprecated in current llama.cpp); anything without a dedicated flag, "
            "such as reasoning_effort, is pre-filled into `chat-template-kwargs`. Override either in the form."
        )
    if features.get("uses_think_tags") or features.get("uses_channel_thought"):
        quirks.append(
            "Model emits <think> or channel-based thought tags. Set `reasoning-format = deepseek` so OpenAI-compatible "
            "clients (OpenWebUI etc.) render thoughts as a collapsible instead of inline in the answer."
        )

    # Multi-session disclosure — makes the ctx-size vs per-session split visible
    if n_sessions > 1 and rec_ctx > 0:
        quirks.append(
            f"Sizing for {n_sessions} concurrent sessions: each user gets {_fmt_ctx(rec_ctx)} of context, "
            f"llama-server allocates ctx-size = {_fmt_ctx(rec_ctx * n_sessions)} total across the -np {n_sessions} slots. "
            f"KV cache is sized against the total; per-session throughput drops roughly linearly with load."
        )
        quirks.append(
            f"When a session hits its {_fmt_ctx(rec_ctx)} cap it will SLIDE: oldest tokens drop, "
            f"generation continues. Set `context-shift = on` and `keep = 256` (tune to your system-prompt "
            f"length in tokens so instructions survive the shift). To fail hard instead of forgetting old "
            f"turns, set `context-shift = off` in the form."
        )
        quirks.append(
            "TTFT tuning for multi-slot: set `batch-size = 4096` (safe, negligible VRAM). "
            "To also improve prompt-eval when a second session arrives mid-generation, bump `ubatch-size` "
            "in the form — this is the main lever, but it's costly. On layer-split multi-GPU, compute-buffer "
            "VRAM grows as ~8 × ubatch × layers × hidden. A dense 27B (64L, 5120H) at ubatch=2048 costs ~4.7 GB "
            "of compute buffers across cards. Don't bump ubatch above 512 unless you have 2+ GB of measured "
            "free VRAM after boot. MoE with CPU offload has much more room to work with."
        )

    # Multi-GPU overhead disclosure — user should know the math accounted for split-mode costs
    if recommended:
        rec_b = next((b for b in backends if b["name"] == recommended.name), None)
        rec_gpu_count = int((rec_b or {}).get("gpu_count", 1))
        if rec_gpu_count > 1:
            quirks.append(
                f"Multi-GPU backend ({rec_gpu_count} cards, layer-split): reserved "
                f"{_RESERVE_PER_GPU * rec_gpu_count:.1f} GB total for CUDA runtime "
                f"(0.5 GB × {rec_gpu_count}) and applied a {int((_MODEL_OVERHEAD_SPLIT - 1) * 100)}% model-VRAM "
                f"multiplier for cross-card handoffs. Real cap will be a bit lower than pure sum-of-VRAMs."
            )

    # Multimodal VRAM accounting
    if has_mmproj:
        quirks.append(
            f"Multimodal model (mmproj companion present — vision, audio, or other modality). "
            f"Reserved {mmproj_vram_gb:.2f} GB for the projector ({mmproj_gb:.2f} GB weights + "
            f"{_MMPROJ_COMPUTE_GB:g} GB encoder scratch). If several projector precisions ship in the "
            "directory the smallest is chosen, since it competes directly with the KV cache.\n"
            "TREAT THIS CTX AS OPTIMISTIC AND VERIFY IT LOADS. The projector and its encoder buffer are "
            "NOT layer-split — both land entirely on the main GPU — so the real limit is that one card, "
            "not the pooled total this estimate is based on. Worse, measured encoder scratch varies ~4x "
            "between models (0.55 GB on a 27B with a 0.87 GB projector vs 2.3 GB on Qwen3-VL-4B with a "
            "0.78 GB one) and is not derivable from GGUF metadata, so no single constant fits all. "
            "If it OOMs on device 0 while the other card still shows free VRAM, that is exactly this "
            "limitation — step ctx-size down until it loads."
        )

    # RoPE extension quirk (only when we set linear scaling ourselves)
    if values.get("rope-scaling") == "linear" and not rope_owned and native_ctx > 0:
        scale = values.get("rope-scale", "?")
        quirks.append(f"Extended ctx from native {_fmt_ctx(native_ctx)} to {_fmt_ctx(rec_ctx)} "
                      f"via `rope-scaling=linear, rope-scale={scale}`. Linear scaling degrades quality gracefully up "
                      f"to ~2× native; beyond that outputs get progressively worse. Drop ctx-size in the form to back off.")

    # unavailable knobs
    unavailable: list[str] = []
    is_moe = isinstance(experts, int) and experts > 1
    if not is_moe:
        unavailable.append("cpu-moe / n-cpu-moe (not MoE)")
    if rope_owned:
        unavailable.append("rope-scaling (the GGUF sets it per layer; a preset value would override it)")
    elif not rope_type or str(rope_type).lower() == "none":
        unavailable.append("rope-scaling (model doesn't declare one)")

    # MoE-specific quirk: reflect what offload is being applied
    if is_moe:
        if recommended:
            off_kind, n_cm = _plan["offload"]
            if off_kind in ("cpu-moe", "n-cpu-moe"):
                est = (f"all {layers} layers'" if off_kind == "cpu-moe"
                       else f"roughly the first {n_cm} of {layers} layers'")
                quirks.append(
                    f"MoE model ({experts} experts): needs expert weights on the CPU — estimated {est} worth. "
                    "Placement is left to llama.cpp: `fit = on` with ngl, tensor-split and n-cpu-moe all unset, "
                    "so llama-server sizes it at load time against real free VRAM. That estimate is advisory; "
                    "the loader decides. Expect slower generation than a fully-GPU model."
                )
                quirks.append(
                    "Pinning ngl or tensor-split here would DISABLE that fitting (`--fit` only adjusts unset "
                    "arguments — the log says \"n_gpu_layers already set by user to 999, abort\"), and our own "
                    "placement maths has no term for compute buffers, which reached 3.6 GiB on a single card "
                    "on a 177B model at 256K ctx. Leave them unset unless you are tuning by measurement."
                )
            else:
                quirks.append(f"MoE model ({experts} experts): fits fully on GPU at this ctx — no CPU offload needed.")
        else:
            quirks.append(f"MoE model ({experts} experts): does not fit even with all experts offloaded to CPU. "
                          f"You need a bigger GPU, a smaller quant, or a shorter context.")

    # Work out which preset the SAVED section currently matches, by comparing the knobs
    # that presets actually set (total ctx plus the offload level). Used to badge the chip
    # that is really running, so a previewed chip can't be mistaken for the live config.
    _current_preset = ""
    if current_section and presets:
        _cur = {k: str(v) for k, v in current_section.items()}
        try:
            _cur_ctx = int(_cur.get("ctx-size") or 0)
        except ValueError:
            _cur_ctx = 0
        _cur_ngl = (_cur.get("ngl") or "").strip()
        _cur_ncm = (_cur.get("n-cpu-moe") or "").strip()
        for _p in presets:
            if _cur_ctx != _p.ctx * n_sessions:
                continue
            if _p.offload_kind == "ngl":
                ok = _cur_ngl == str(_p.ngl)
            elif _p.offload_kind == "n-cpu-moe":
                ok = _cur_ncm == str(_p.n_cpu_moe)
            elif _p.offload_kind == "cpu-moe":
                ok = (_cur.get("cpu-moe") or "").lower() in ("true", "on", "1")
            else:
                ok = _cur_ngl in ("", "999") and not _cur_ncm
            if ok:
                _current_preset = _p.key
                break

    # diff vs current section — only report on keys autoconfig actually opinions on.
    # Anything the user set that we don't touch (mmproj, chat-template-file, lora, override-*, etc.)
    # is left alone: not reported as a diff, and Fill/Fill minimal doesn't overwrite it.
    current_diff: list[str] = []
    # Everything autoconfig opinions on, minus what this recommendation actually set: that is
    # exactly the set it wants gone. Declared once in AUTOCONFIG_DOMAIN rather than remembered
    # per-branch, so a key can no longer be quietly left behind.
    _displaces = set(AUTOCONFIG_DOMAIN) - _NEVER_CLEAR
    if current_section:
        cur = {k: str(v) for k, v in current_section.items()}
        for k, v in values.items():
            if cur.get(k, "") != v:
                if k in cur:
                    current_diff.append(f"{k}: {cur[k]!r} → {v!r}")
                else:
                    current_diff.append(f"{k}: unset → {v!r}")
        # Only report removals for keys we actively displace
        for k in _displaces:
            if k in cur and cur[k] and k not in values:
                current_diff.append(f"{k}: {cur[k]!r} → unset (superseded)")

    # Keys the recommendation wants GONE. "Fill form" walks `values` and writes each key it
    # finds, so a key we deliberately omit is simply left holding whatever the form already had
    # — the current section — and Save writes it straight back. The diff would promise
    # "n-cpu-moe: '6' -> unset" while nothing changed. Fill has to be told what to clear.
    displaced = sorted(k for k in _displaces if k not in values)

    # A key we set but never declared would escape both the diff and Fill's clearing, which is
    # how the "Save does nothing" bug worked. Make it visible instead of silent.
    _gaps = _domain_gaps(values)
    if _gaps:
        quirks.append(
            "Autoconfig set %s, which AUTOCONFIG_DOMAIN does not declare. Fill will not clear "
            "%s on a later run, so a stale value could survive. Add them to the domain."
            % (", ".join("`%s`" % g for g in _gaps), "them" if len(_gaps) > 1 else "it")
        )

    return Recommendation(
        plans=plans,
        displaced=displaced,
        recommended_backend=(recommended.name if recommended else ""),
        recommended_ctx=rec_ctx,
        warnings=quality_warnings(model_rel=model_rel, params=(summary.get("general") or {}).get("params_raw"), recommended_ctx=rec_ctx, native_ctx=native_ctx),
        estimated_ctx=estimated_ctx,
        ctx_cap_reason=ctx_cap_reason,
        recommended_total_ctx=rec_ctx * n_sessions if rec_ctx > 0 else 0,
        n_sessions=n_sessions,
        values=values,
        values_minimal=minimal,
        baseline_redundant=baseline_redundant,
        quirks=quirks,
        unavailable=unavailable,
        current_diff=current_diff,
        presets=presets,
        frontier=frontier_opts,
        fits_full_gpu=_fits_full_gpu,
        native_ctx=native_ctx,
        current_preset=_current_preset,
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
