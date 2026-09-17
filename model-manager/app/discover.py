"""Model search that answers "can this server run it, and is the quant any good?".

Hugging Face ranks by popularity alone. Here a result is judged first on whether it fits this
machine (a Q4-or-better file inside the GPU budget), then on publisher trust, then on popularity
weighted towards recent activity — an old model with a million downloads should not outrank a good
one released last month. Nothing is hidden silently: an unsuitable result carries the reason, and
the caller can ask for them.

Rules come from the operator policy (master-prompt D19/D20): no quantisation below Q4 under 100B
parameters, no context at or below 16K (checked after download, where the real value is known), and
mixture-of-experts preferred at equal size.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

# Labs that publish their own weights, and quantisation specialists with a track record. The
# operator can extend this; an unknown publisher is not blocked, only ranked below and hidden
# behind the "everyone" toggle.
TRUSTED_QUANTISERS = {
    "unsloth", "bartowski", "ggml-org", "lmstudio-community", "mradermacher", "nvidia",
}
TRUSTED_LABS = {
    "qwen", "google", "meta-llama", "mistralai", "ibm-granite", "openai", "microsoft", "deepseek-ai",
    "zai-org", "moonshotai", "allenai", "openbmb", "liquidai", "cohicanlabs", "cohereforai", "ai21labs",
    "nousresearch", "internlm", "baai", "tiiuae", "stabilityai", "ornith-ai",
}
SUB_Q4 = re.compile(r"(?:^|[-_.])(?:UD-)?(IQ[123]\w*|Q[123](?:_[\w]+)*)(?:[-_.]|$)", re.I)
QUANT = re.compile(r"(?:^|[-_.])((?:UD-)?(?:IQ\d\w*|Q\d(?:_[\w]+)*|MXFP4|F16|BF16|F32))(?:[-_.]|$)", re.I)
# "35B-A3B" = 35B total, 3B active (mixture of experts); "27B" = dense.
PARAMS = re.compile(r"(?:^|[-_.])(\d+(?:\.\d+)?)\s*B(?:-A(\d+(?:\.\d+)?)B)?(?:[-_.]|$)", re.I)
COMPANION = re.compile(r"mmproj|projector|\bmtp\b|draft|eagle\d?|medusa", re.I)
SUB_Q4_PARAM_LIMIT = 100.0  # billions
MIN_MODEL_GB = 0.3          # below this it is not a servable model file


@dataclass
class Candidate:
    id: str
    owner: str
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    trending: float = 0.0
    tags: list[str] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)   # [{path, size}]
    license: str | None = None


def params_from(name: str, tags: list[str] | None = None) -> tuple[float | None, float | None]:
    """(total B, active B) from a repo or file name; active is set only for MoE names."""
    best = None
    for m in PARAMS.finditer(name):
        total, active = float(m.group(1)), (float(m.group(2)) if m.group(2) else None)
        if best is None or total > best[0]:
            best = (total, active)
    return best or (None, None)


def is_moe(candidate: Candidate, active: float | None) -> bool:
    if active is not None:
        return True
    blob = " ".join([candidate.id, *(candidate.tags or [])]).lower()
    return any(k in blob for k in ("moe", "mixture-of-experts", "-a3b", "-a4b", "a22b"))


def quant_of(path: str) -> str | None:
    m = QUANT.search(path.rsplit("/", 1)[-1])
    return m.group(1).upper() if m else None


def _age_days(last_modified: str) -> float:
    try:
        from datetime import datetime, timezone
        return max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(last_modified.replace("Z", "+00:00"))).total_seconds() / 86400)
    except Exception:  # noqa: BLE001 - unknown date sorts as old
        return 3650.0


def popularity(candidate: Candidate) -> float:
    """Downloads and likes, discounted by age: recent activity counts for more (user's choice).

    A half-life of 90 days means last month's release competes with a year-old favourite.
    """
    raw = candidate.downloads + candidate.likes * 50
    return (candidate.trending * 1000.0) + raw * math.pow(0.5, _age_days(candidate.last_modified) / 90.0)


def build_options(candidate: Candidate, budget_gb: float) -> list[dict]:
    """One row per downloadable model file (companions excluded), with size, quant and fit."""
    total_b, active_b = params_from(candidate.id, candidate.tags)
    out: list[dict] = []
    grouped: dict[str, dict] = {}
    for f in candidate.files:
        path = f.get("path") or ""
        if not path.lower().endswith(".gguf") or COMPANION.search(path.rsplit("/", 1)[-1]):
            continue
        base = re.sub(r"-\d{5}-of-\d{5}", "", path)
        row = grouped.setdefault(base, {"path": base, "bytes": 0, "shards": 0})
        row["bytes"] += int(f.get("size") or 0)
        row["shards"] += 1
    for row in grouped.values():
        gb = row["bytes"] / 1e9
        quant = quant_of(row["path"])
        # Repos carry stray GGUFs (index shards, tiny extras). A model has a quantisation in its
        # name and real size; without both, offering it as a download only misleads.
        if quant is None or gb < MIN_MODEL_GB:
            continue
        params = total_b or params_from(row["path"])[0]
        sub_q4 = bool(SUB_Q4.search(row["path"].rsplit("/", 1)[-1])) and (params is None or params < SUB_Q4_PARAM_LIMIT)
        reasons = []
        if gb > budget_gb:
            reasons.append(f"{gb:.1f} GB does not fit the {budget_gb:.1f} GB the GPU can hold")
        if sub_q4:
            reasons.append(f"{quant or 'this quantisation'} is below Q4")
        out.append({**row, "gb": round(gb, 2), "quant": quant, "params": params, "activeParams": active_b,
                    "fits": not reasons, "reasons": reasons})
    out.sort(key=lambda r: (-r["fits"], -r["gb"]))
    return out


def judge(candidate: Candidate, *, budget_gb: float, trusted: set[str]) -> dict:
    """A search result with everything the UI ranks and filters on."""
    options = build_options(candidate, budget_gb)
    usable = [o for o in options if o["fits"]]
    total_b, active_b = params_from(candidate.id, candidate.tags)
    owner = candidate.owner.lower()
    is_trusted = owner in trusted or owner in TRUSTED_LABS
    best = max(usable, key=lambda o: o["gb"], default=None)
    reasons = []
    if not options:
        reasons.append("no GGUF model files in this repository")
    elif not usable:
        reasons.append(min((o["reasons"][0] for o in options if o["reasons"]), key=len))
    return {
        "id": candidate.id, "owner": candidate.owner, "downloads": candidate.downloads, "likes": candidate.likes,
        "lastModified": candidate.last_modified, "ageDays": round(_age_days(candidate.last_modified)),
        "license": candidate.license, "tags": candidate.tags[:12],
        "params": total_b, "activeParams": active_b, "moe": is_moe(candidate, active_b),
        "vision": any("mmproj" in (f.get("path") or "").lower() for f in candidate.files),
        "trusted": is_trusted, "options": options[:12],
        "best": best, "suitable": bool(usable), "reasons": reasons,
        "score": (2.0 if usable else 0.0) + (0.5 if is_trusted else 0.0),
        "popularity": round(popularity(candidate), 2),
    }


def rank(results: list[dict]) -> list[dict]:
    """Fit first (the user's choice), then trusted publishers, then age-weighted popularity."""
    return sorted(results, key=lambda r: (-r["score"], -r["popularity"], r["id"]))


def apply_filters(results: list[dict], *, trusted_only: bool = True, show_unsuitable: bool = False,
                  min_gb: float | None = None, max_gb: float | None = None, quants: list[str] | None = None,
                  min_params: float | None = None, max_params: float | None = None, moe: str = "any",
                  vision: bool = False, license_contains: str = "", owner: str = "") -> list[dict]:
    out = []
    for r in results:
        if trusted_only and not r["trusted"]:
            continue
        if not show_unsuitable and not r["suitable"]:
            continue
        if moe == "moe" and not r["moe"]:
            continue
        if moe == "dense" and r["moe"]:
            continue
        if vision and not r["vision"]:
            continue
        if owner and owner.lower() not in r["owner"].lower():
            continue
        if license_contains and license_contains.lower() not in (r["license"] or "").lower():
            continue
        if min_params is not None and (r["params"] is None or r["params"] < min_params):
            continue
        if max_params is not None and (r["params"] is None or r["params"] > max_params):
            continue
        options = r["options"]
        if quants:
            wanted = {q.upper() for q in quants}
            options = [o for o in options if (o["quant"] or "").upper() in wanted]
        if min_gb is not None:
            options = [o for o in options if o["gb"] >= min_gb]
        if max_gb is not None:
            options = [o for o in options if o["gb"] <= max_gb]
        if (quants or min_gb is not None or max_gb is not None) and not options:
            continue
        out.append({**r, "options": options} if options is not r["options"] else r)
    return out


def hub_url(query: str, *, quants: list[str] | None = None, owner: str = "") -> str:
    """The same search on Hugging Face, so the full hub is always one click away."""
    from urllib.parse import urlencode
    params = {"filter": "gguf"}
    if query:
        params["search"] = query
    if owner:
        params["author"] = owner
    return "https://huggingface.co/models?" + urlencode(params) + "&sort=trending"
