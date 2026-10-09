"""The pure sizing core of autoconfig: which context, offload, preset and prompt cache fit.

Moved out of autoconfig.py unchanged in behaviour so that it can be ported in slices to Rust
(MODEL_AUTOCONFIG=rust; see the bottom of this module). Everything here is a pure function of
its arguments: no files, processes, network, clock or settings. It imports only the stdlib at
load time, so noevia-rs's fixture generator can import it without the service's dependencies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable

# candidate contexts to try, smallest → largest
_CTX_CANDIDATES = (
    4096, 8192, 12288, 16384, 24576, 32768, 40960, 49152, 57344, 65536,
    73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072, 139264,
    147456, 151552, 155648, 159744, 163840, 172032, 180224, 188416,
    196608, 204800, 212992, 221184, 229376, 237568, 245760, 253952, 262144,
    294912, 327680, 360448, 393216, 425984, 458752, 491520, 524288,
    589824, 655360, 720896, 786432, 851968, 917504, 983040, 1048576,
)
# Every value above is a multiple of 4096 (most are multiples of 8192), which keeps them on
# llama.cpp's internal block-alignment boundaries — the reason arbitrary values like 160000
# behave badly. The upper range used to step by 16384, which was too coarse: a model that
# already fits near its ceiling had only one or two reachable steps left, so the offload
# frontier collapsed to two points and "Balanced" came out identical to "Fast".

_CACHE_BYTES_PER_ELEM = {
    "": 2.0, "f16": 2.0, "bf16": 2.0, "f32": 4.0,
    "q8_0": 1.0625, "q5_0": 0.75, "q5_1": 0.8125, "q4_0": 0.6250, "q4_1": 0.6875,
    "iq4_nl": 0.6250,
}

_RESERVE_PER_GPU = 1.0   # CUDA runtime + driver context + scratch/cuBLAS workspace.
                         # Bumped from 0.5 -> 1.0 for llama.cpp 0.3.0-dev (commit d222767+) which
                         # recognizes MTP nextn tensors and allocates larger cuBLAS workspaces on
                         # inference. Empirical: Qwen3.8-27B at ctx=159744 loaded fine but OOM'd
                         # on first inference (cuBLAS workspace on device 1). At ctx=131072 fits
                         # cleanly. If you pin an older image and want more ctx, drop this back to 0.5.

# --- prompt cache (--cache-ram) sizing. See the block that consumes these for the measurements.
_CACHE_RAM_DEFAULT_MIB = 8192   # llama-server's own default; never suggest worse without cause
_CACHE_RAM_CONVOS = 4           # conversations to keep warm at the recommended context
_CACHE_RAM_HEADROOM_GB = 8.0    # left for the OS, the other containers and page cache churn

_MODEL_OVERHEAD_SINGLE = 1.00  # Q_K_M loads at ~file size when everything's on one card
_MODEL_OVERHEAD_SPLIT = 1.08   # +8% for cross-GPU handoffs, duplicated activation buffers, layer-imbalance.
_CACHE_DEFAULT = "q8_0"  # symmetric K/V; K stays q8, V could drop to q4 for +20% ctx (future preset)
_SSM_STATE_BYTES = 4 * 1024 * 1024  # ~4 MB per SSM layer, derived from typical state_size×inner_size
_CPU_LAYER_PENALTY = 20.0  # how much slower one CPU-resident DENSE layer is than a GPU one.
                           # Used only to rank presets, not to decide fit. Dense offload is
                           # brutal compared to MoE expert offload: every token traverses every
                           # CPU layer, whereas MoE only touches a few active experts.
                           # VERIFIED on Qwen3.8-27B-OBLITERATED (65 layers, 2x RTX 5070):
                           #   ngl=999 (all GPU)  -> 34.2 tok/s   (predicted 100%)
                           #   ngl=56  (9 on CPU) ->  9.7 tok/s   (predicted 28%, measured 28.4%)
                           # Still hardware-dependent (RAM bandwidth, PCIe width), so treat the
                           # speed % as a well-calibrated estimate rather than a guarantee.

def _cache_dtype_bytes(k: str) -> float:
    return _CACHE_BYTES_PER_ELEM.get((k or "").lower(), 2.0)


def _pattern_sample(val: Any, layers: int) -> list:
    """Per-layer GGUF arrays arrive as {_array, count, sample} holding only a PREFIX of the
    array. Returns the prefix when it describes this model's layer stack, else []."""
    if isinstance(val, dict) and val.get("_array"):
        if int(val.get("count") or 0) == layers:
            return list(val.get("sample") or [])
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    return []


def _period_of(seq: list) -> int:
    """Shortest p with seq[i] == seq[i % p] for all i. These per-layer patterns repeat, and the
    sample is only a prefix, so the period is what lets one cycle stand for the whole stack."""
    for p in range(1, len(seq) + 1):
        if all(seq[i] == seq[i % p] for i in range(len(seq))):
            return p
    return len(seq)


def kv_shape(arch: str, layers: int, kv_heads: int, head_dim: int,
             key_length: int | None = None, value_length: int | None = None,
             full_attention_interval: int | None = None,
             ssm_state_size: int | None = None,
             sliding_window: int | None = None,
             sliding_window_pattern: Any = None,
             key_length_swa: int | None = None,
             value_length_swa: int | None = None,
             shared_kv_layers: int | None = None,
             kv_heads_pattern: Any = None) -> dict | None:
    """Everything kv_cache_bytes needs that does not depend on the context, resolved once.

    None when the cache cannot be sized at any context (kv_cache_bytes is then 0). The per-layer
    GGUF arrays (which may hold anything a hostile header puts there) are reduced here, in
    Python, to one repeating period of (is_local, kv_heads) pairs, with the same conversions and
    the same exceptions as before; what remains is plain integer and float arithmetic, which is
    what MODEL_AUTOCONFIG=rust ports. A JSON-able dict, so it can be handed to the Rust side.
    """
    if not (layers > 0 and kv_heads > 0):
        return None
    # Use explicit key/value lengths when the model declares them; else fall back to head_dim
    k_dim = int(key_length) if key_length else head_dim
    v_dim = int(value_length) if value_length else head_dim
    if k_dim <= 0 or v_dim <= 0:
        return None
    shape: dict[str, Any] = {
        "gemma": (arch or "").lower().startswith("gemma"),
        "layers": layers, "kv_heads": kv_heads, "k_dim": k_dim, "v_dim": v_dim,
        "hybrid_interval": None, "window": None, "k_swa": None, "v_swa": None,
        "shared": None, "period": None,
    }
    # Hybrid attention + SSM (Mamba-2 style) — only every Nth layer holds a real KV cache
    is_hybrid = bool(ssm_state_size) or (full_attention_interval and full_attention_interval > 1)
    if is_hybrid and full_attention_interval and full_attention_interval > 1:
        shape["hybrid_interval"] = int(full_attention_interval)
        return shape
    if sliding_window and sliding_window > 0:
        # Read every per-layer quantity off the SAME repeating period (see _kv_bytes).
        pattern = _pattern_sample(sliding_window_pattern, layers)
        heads_seq = _pattern_sample(kv_heads_pattern, layers)
        shape["shared"] = max(0, min(int(shared_kv_layers or 0), layers))
        shape["k_swa"] = int(key_length_swa) if key_length_swa else k_dim
        shape["v_swa"] = int(value_length_swa) if value_length_swa else v_dim
        shape["window"] = int(sliding_window)
        if pattern:
            period = []
            for i in range(_period_of(pattern)):
                # Fall back to the scalar head count for any position the sample does not
                # reach; models that declare a scalar (gemma-4-E4B) take this path throughout.
                h = kv_heads
                if i < len(heads_seq):
                    try:
                        h = max(1, int(heads_seq[i]))
                    except (TypeError, ValueError):
                        h = kv_heads
                # float(h) is the conversion the product below always made.
                period.append([bool(pattern[i]), float(h)])
            shape["period"] = period
        return shape
    if not (shape["gemma"] and layers >= 6):
        # Only the last branch of kv_shape_bytes reads it, as before.
        shape["shared"] = int(shared_kv_layers or 0)
    return shape


def kv_shape_bytes(shape: dict | None, ctx: int, bytes_per_elem: float,
                   v_bytes_per_elem: float | None = None) -> int:
    """KV cache bytes at `ctx` for a kv_shape(). The arithmetic of kv_cache_bytes, unchanged."""
    if shape is None or not ctx > 0:
        return 0
    layers, kv_heads = shape["layers"], shape["kv_heads"]
    k_dim, v_dim = shape["k_dim"], shape["v_dim"]
    v_bytes = bytes_per_elem if v_bytes_per_elem is None else v_bytes_per_elem
    per_layer_per_token = kv_heads * (k_dim * bytes_per_elem + v_dim * v_bytes)

    interval = shape["hybrid_interval"]
    if interval is not None:
        full_layers = max(1, (layers + interval - 1) // interval)
        ssm_layers = layers - full_layers
        kv_full = full_layers * per_layer_per_token * ctx
        return int(kv_full + ssm_layers * _SSM_STATE_BYTES)

    # ---- Sliding-window attention, from what the model DECLARES ----
    #
    # Earlier this was guessed: 1-in-6 layers global, a 4096-token window, and the same head
    # dim for every layer. Gemma-4 declares all of it and none of the guesses match — window
    # 512 not 4096, a 42-entry per-layer pattern rather than a ratio, SWA head dims of 256
    # against 512 for global layers, and 18 layers that share another layer's KV and so
    # allocate none of their own. Guessing over-estimated this model's cache by roughly an
    # order of magnitude, which shows up as far less offered context than it can really do.
    if shape["window"] is not None:
        # Gemma declares both the local/global pattern and the KV head count as per-layer
        # arrays, and they are aligned: on gemma-4-12b the global layer (pattern False) carries
        # 1 KV head against 8 on the local ones, and on 26B-A4B it is 2 against 8. Collapsing
        # that array to one representative value charged every global layer 8x too much, and
        # since the global term is the only one that scales with ctx, it dominated the entire
        # estimate - 14.5 GB predicted against llama.cpp's own 2.0 GB at 208K.
        #
        # Layers that reuse another layer's KV allocate none of their own. They are spread
        # through the stack rather than clustered at one end, so the saving lands on local and
        # global layers alike; charging it entirely to the local layers (the cheap ones) made
        # it nearly worthless and left gemma-4-E4B 70% high.
        shared = shape["shared"]
        alloc_frac = (layers - shared) / layers if layers else 1.0
        per_local_elem = shape["k_swa"] * bytes_per_elem + shape["v_swa"] * v_bytes
        per_global_elem = k_dim * bytes_per_elem + v_dim * v_bytes
        window = min(shape["window"], ctx)
        period = shape["period"]
        if period:
            reps = (layers / len(period)) * alloc_frac
            total = 0.0
            for is_local, h in period:
                if is_local:
                    total += reps * h * per_local_elem * window
                else:
                    total += reps * h * per_global_elem * ctx
            return int(total)

        # No usable pattern: fall back to gemma's 5-local-to-1-global ratio at one head count.
        local_layers = max(0, min(layers, layers - max(1, layers // 6)))
        global_layers = layers - local_layers
        return int(alloc_frac * (global_layers * kv_heads * per_global_elem * ctx
                                 + local_layers * kv_heads * per_local_elem * window))

    if shape["gemma"] and layers >= 6:
        # Older gemma with no declared window: fall back to the 1-in-6 / 4096 approximation.
        full_layers = max(1, layers // 6)
        swa_layers = layers - full_layers
        kv_full = full_layers * per_layer_per_token * ctx
        kv_swa = swa_layers * per_layer_per_token * 4096
        return int(kv_full + kv_swa)

    # No sliding window declared, but layers may still share KV.
    effective_layers = max(1, layers - max(0, min(shape["shared"], layers - 1)))
    return int(effective_layers * per_layer_per_token * ctx)


def kv_cache_bytes(arch: str, ctx: int, layers: int, kv_heads: int,
                   head_dim: int, bytes_per_elem: float,
                   key_length: int | None = None, value_length: int | None = None,
                   full_attention_interval: int | None = None,
                   ssm_state_size: int | None = None,
                   v_bytes_per_elem: float | None = None,
                   sliding_window: int | None = None,
                   sliding_window_pattern: Any = None,
                   key_length_swa: int | None = None,
                   value_length_swa: int | None = None,
                   shared_kv_layers: int | None = None,
                   kv_heads_pattern: Any = None) -> int:
    """Compute KV cache bytes. Handles:
       - explicit key/value_length overrides (Qwen3.x, Yi, etc.)
       - hybrid attention+SSM (Qwen3.5, Zamba) via full_attention_interval + ssm_state_size
       - Gemma sliding-window carve-out
       - asymmetric K/V quantization (bytes_per_elem is K; v_bytes_per_elem defaults to it)
    """
    if not ctx > 0:
        return 0
    shape = kv_shape(arch, layers, kv_heads, head_dim, key_length=key_length,
                     value_length=value_length, full_attention_interval=full_attention_interval,
                     ssm_state_size=ssm_state_size, sliding_window=sliding_window,
                     sliding_window_pattern=sliding_window_pattern,
                     key_length_swa=key_length_swa, value_length_swa=value_length_swa,
                     shared_kv_layers=shared_kv_layers, kv_heads_pattern=kv_heads_pattern)
    return kv_shape_bytes(shape, ctx, bytes_per_elem, v_bytes_per_elem)


@dataclass
class FitRow:
    ctx: int                 # per-session ctx (what each user sees)
    total_ctx: int           # ctx * n_sessions — what llama-server allocates as --ctx-size
    model_gb: float          # GPU-resident weight VRAM (accounts for MoE offload)
    kv_gb: float
    total_gb: float          # model_gb + kv_gb
    fits: bool
    free_gb: float
    offload_kind: str = ""   # "" | "cpu-moe" | "n-cpu-moe"
    n_cpu_moe: int = 0       # populated when offload_kind == "n-cpu-moe"
    gpu_pct: int = 100       # share of the model's WEIGHTS resident on the GPU, 0-100.
                             # Measured in bytes rather than layers because for a MoE the
                             # layer count says little: attention stays resident while only
                             # experts move, so "layers on GPU" overstates what is really there.


@dataclass
class PresetOption:
    key: str                # "fast" | "balanced" | "long-ctx"
    label: str
    icon: str               # lucide name
    backend: str
    ctx: int
    n_cpu_moe: int          # 0 = no offload; layers = all offloaded (cpu-moe=true)
    offload_kind: str       # "" | "cpu-moe" | "n-cpu-moe" | "ngl"
    gpu_layers: int         # layers kept on GPU (for MoE: layers whose experts stay on GPU)
    total_layers: int
    gpu_gb: float           # weights VRAM
    kv_gb: float
    speed_score: float      # 0..1 relative (1.0 = no offload)
    ngl: int = -1           # dense offload only: value to write as `ngl`. -1 = leave at 999 (all)


def _moe_ratio(expert_count: Any) -> float:
    """Approximate share of a GGUF's weights that live in MoE experts.
    Heuristic: more experts → more of the weights are experts.
      - 8 experts → ~0.75 in experts
      - 16 experts → ~0.85
      - 32-128 experts → ~0.9-0.92
    Clamped to [0.6, 0.92]. Used for sizing cpu-moe / n-cpu-moe recommendations."""
    if not isinstance(expert_count, int) or expert_count < 2:
        return 0.0
    if expert_count >= 64:
        return 0.92
    if expert_count >= 32:
        return 0.90
    if expert_count >= 16:
        return 0.85
    if expert_count >= 8:
        return 0.78
    return 0.65


def _layer_costs(layers: int, n_cpu_moe: int, attention_gb: float,
                 expert_per_layer_gb: float, kv_gb: float) -> list[float]:
    """Per-layer GPU cost: attention + KV share, plus experts above the n-cpu-moe threshold."""
    if layers <= 0:
        return []
    att = attention_gb / layers
    kv = kv_gb / layers
    return [att + kv + (expert_per_layer_gb if i >= n_cpu_moe else 0.0) for i in range(layers)]


def _split_feasible(layers: int, n_cpu_moe: int, attention_gb: float,
                    expert_per_layer_gb: float, kv_gb: float, gpu_count: int,
                    pinned_gb: float, caps: list[float]) -> bool:
    """Does ANY contiguous per-card partition fit? One greedy pass.

    The fit search asks only whether a configuration is placeable, never how to place it
    best. Answering that with _partition_min_max ran a 64-step binary search per call, and
    the search makes thousands of calls per page load — measured at 2239 for one model,
    around nine million inner steps, which turned a snappy preset click into a visible wait.

    Filling each card to capacity in order is optimal for CONTIGUOUS feasibility: taking
    less on an earlier card can only leave more for a later one, never less.
    """
    if gpu_count <= 1 or not caps:
        return True
    costs = _layer_costs(layers, n_cpu_moe, attention_gb, expert_per_layer_gb, kv_gb)
    if not costs:
        return True
    idx, n = 0, len(costs)
    for card in range(gpu_count):
        budget = caps[card] - (pinned_gb if card == 0 else 0.0)
        took = 0
        while idx < n and costs[idx] <= budget:
            budget -= costs[idx]
            idx += 1
            took += 1
        if idx >= n:
            return True
        if took == 0:
            return False          # this card cannot hold even one more layer
    return idx >= n


def _find_fit(model_gb_full: float, kv_gb: float, budget_gb: float,
              layers: int, moe_ratio: float,
              card_ok: "Callable[[float, str, int], bool] | None" = None) -> tuple[bool, float, str, int]:
    """Return (fits, gpu_model_gb, offload_kind, n_layers) for a given (model, kv, budget).

    Strategy: try no offload first; then offload.
      * MoE  -> smallest n-cpu-moe (fewest expert layers moved) that fits; n=layers means cpu-moe
      * dense -> smallest number of whole layers moved to CPU via `ngl`

    For offload_kind == "ngl" the trailing int is the count of CPU-resident LAYERS, not experts.

    The dense branch used to be missing entirely: anything whose weights+KV exceeded the budget
    was reported as "doesn't fit on any backend", even though a dense model runs perfectly well
    with some layers on the CPU — that is exactly what the Fast/Balanced/Long presets do. The
    result was a red X and no options for, say, a 27 GB Q8 on 24 GB of VRAM, when the honest
    answer is "yes, with N layers offloaded, and here is the speed cost".

    `card_ok(gpu_gb, kind, n)` is an optional second test, used on multi-GPU backends to check
    that the per-card split of that configuration actually fits each card. It has to be part
    of the SEARCH, not a filter applied afterwards: the pooled budget and the per-card limit
    are satisfied at different offload levels, so the first configuration that clears the pool
    may still overflow one card while offloading one more layer clears both. Rejecting that
    first candidate instead of continuing produced a band of contexts reported as impossible
    while both smaller AND larger ones fitted — non-monotonic, and wrong.
    """
    def _ok(gpu: float, kind: str, n: int) -> bool:
        if gpu + kv_gb > budget_gb:
            return False
        return card_ok is None or card_ok(gpu, kind, n)

    if _ok(model_gb_full, "", 0):
        return True, model_gb_full, "", 0
    if layers <= 0:
        return False, model_gb_full, "", 0

    if moe_ratio <= 0:
        # Dense: move whole layers to the CPU. Keep at least one on the GPU — a fully
        # offloaded model is just CPU inference and the GPU backend has nothing to do.
        per_layer_gb = model_gb_full / layers
        for cpu_layers in range(1, layers):
            gpu = per_layer_gb * (layers - cpu_layers)
            if _ok(gpu, "ngl", cpu_layers):
                return True, gpu, "ngl", cpu_layers
        return False, model_gb_full, "", 0
    attention_gb = model_gb_full * (1 - moe_ratio)
    expert_per_layer_gb = model_gb_full * moe_ratio / layers
    # smallest n where fits — n counts CPU-offloaded layers
    for n in range(1, layers + 1):
        gpu = attention_gb + max(0, layers - n) * expert_per_layer_gb
        kind = "cpu-moe" if n >= layers else "n-cpu-moe"
        if _ok(gpu, kind, n):
            if n >= layers:
                return True, attention_gb, "cpu-moe", 0
            return True, gpu, "n-cpu-moe", n
    return False, attention_gb, "cpu-moe", 0  # even full offload can't fit (attention too big for GPU)


def _pareto_frontier(layers: int, kv_gb_at: Callable[[int], float],
                     model_gb: float, moe_ratio: float,
                     budget_gb: float, ctx_candidates: list[int],
                     n_sessions: int = 1,
                     card_ok: "Callable[[float, float, int], bool] | None" = None,
                     ) -> list[tuple[int, int, float, float]]:
    """For a MoE model on a given backend, sweep n-cpu-moe from 0..layers.

    `card_ok(weight_gb, kv_gb, n_cpu_moe)` is the same per-card feasibility test the fit table
    applies. Without it the frontier can propose a point that clears the pooled budget but
    overflows one card, so a preset card recommends a configuration the table beside it marks
    as not fitting.
    Return list of (n_cpu_moe, max_per_session_ctx_fitting, gpu_weight_gb, kv_gb_at_max_ctx) — pareto frontier.
    Higher n_cpu_moe → higher max_ctx (more offloaded = less VRAM for weights = more room for KV).

    Candidates are interpreted as PER-SESSION ctx. KV is sized against total_ctx = ctx * n_sessions.
    """

    if moe_ratio <= 0 or layers <= 0:
        return []
    attention_gb = model_gb * (1 - moe_ratio)
    per_layer_gb = model_gb * moe_ratio / layers
    frontier: list[tuple[int, int, float, float]] = []
    n = max(1, int(n_sessions))
    for ncm in range(0, layers + 1):
        gpu_layers = layers - ncm
        weight_gb = attention_gb + gpu_layers * per_layer_gb
        if weight_gb >= budget_gb:
            continue  # can't fit even at ctx=0
        # find max per-session ctx that fits with this weight footprint
        best_ctx = 0
        best_kv = 0.0
        for ctx in ctx_candidates:
            kv_gb = kv_gb_at(ctx * n)
            if card_ok is not None and not card_ok(weight_gb, kv_gb, ncm):
                continue
            if weight_gb + kv_gb <= budget_gb:
                if ctx > best_ctx:
                    best_ctx = ctx
                    best_kv = kv_gb
        if best_ctx == 0:
            continue
        # only keep this point if it strictly improves ctx over prior (higher ncm) — pareto step
        if frontier and best_ctx <= frontier[-1][1]:
            continue
        frontier.append((ncm, best_ctx, weight_gb, best_kv))
    return frontier


def _dense_frontier(layers: int, kv_gb_at: Callable[[int], float],
                    model_gb: float, budget_gb: float,
                    ctx_candidates: list[int], n_sessions: int = 1,
                    card_ok: "Callable[[float, float], bool] | None" = None,
                    ) -> list[tuple[int, int, float, float]]:
    """Context-vs-speed frontier for a DENSE model, by moving whole layers off the GPU.

    MoE models offload expert weights (`--cpu-moe` / `--n-cpu-moe`), which is cheap because
    only a few experts are active per token. A dense model has no experts, so the only lever
    is `--n-gpu-layers` below the total: layers past that stay in host RAM (mmap'd from the
    GGUF) and every single token has to traverse them on the CPU. Freeing VRAM this way buys
    context, but it is far more expensive per layer than MoE offload — see _CPU_LAYER_PENALTY.

    Returns (cpu_layers, max_per_session_ctx, gpu_weight_gb, kv_gb_at_that_ctx), same tuple
    shape as _pareto_frontier so the preset builder can consume either.

    `card_ok(weight_gb, kv_gb)` applies the same per-card feasibility test the fit table uses.
    Without it the frontier proposes points that clear the pooled budget but overflow one
    card, so a preset card could recommend an ngl the table beside it marks as not fitting —
    and, worse, that llama.cpp would OOM.
    """
    if layers <= 0 or model_gb <= 0:
        return []
    per_layer_gb = model_gb / layers
    n = max(1, int(n_sessions))
    frontier: list[tuple[int, int, float, float]] = []
    for cpu_layers in range(0, layers):  # keep at least 1 layer on GPU
        gpu_layers = layers - cpu_layers
        weight_gb = per_layer_gb * gpu_layers
        if weight_gb >= budget_gb:
            continue
        best_ctx, best_kv = 0, 0.0
        for ctx in ctx_candidates:
            kv_gb = kv_gb_at(ctx * n)
            if weight_gb + kv_gb <= budget_gb and ctx > best_ctx:
                if card_ok is not None and not card_ok(weight_gb, kv_gb):
                    continue
                best_ctx, best_kv = ctx, kv_gb
        if best_ctx == 0:
            continue
        # pareto step: only keep points that strictly improve max ctx
        if frontier and best_ctx <= frontier[-1][1]:
            continue
        frontier.append((cpu_layers, best_ctx, weight_gb, best_kv))
    return frontier


def _frontier_options(frontier: list[tuple[int, int, float, float]], backend: str,
                      layers: int, dense: bool) -> list[PresetOption]:
    """Turn EVERY frontier point into a PresetOption, ascending by ctx.

    Feeds the custom slider, which lets any achievable point on the speed/context curve be
    selected rather than only the three named samples.
    """
    out: list[PresetOption] = []
    for off, ctx, gpu_gb, kv_gb in sorted(frontier, key=lambda f: f[1]):
        if dense:
            gpu_layers = layers - off
            speed = round(layers / (gpu_layers + off * _CPU_LAYER_PENALTY), 3) if layers else 0.0
            out.append(PresetOption(
                key=f"pt{off}", label=f"{gpu_layers}/{layers} layers", icon="sliders",
                backend=backend, ctx=ctx, n_cpu_moe=0,
                offload_kind=("ngl" if off > 0 else ""),
                gpu_layers=gpu_layers, total_layers=layers,
                gpu_gb=round(gpu_gb, 2), kv_gb=round(kv_gb, 2),
                speed_score=speed, ngl=(gpu_layers if off > 0 else 999),
            ))
        else:
            speed = round((layers - off) / layers, 3) if layers else 0.0
            out.append(PresetOption(
                key=f"pt{off}", label=f"ncm {off}", icon="sliders",
                backend=backend, ctx=ctx, n_cpu_moe=off,
                offload_kind=("cpu-moe" if off >= layers else ("n-cpu-moe" if off > 0 else "")),
                gpu_layers=layers - off, total_layers=layers,
                gpu_gb=round(gpu_gb, 2), kv_gb=round(kv_gb, 2),
                speed_score=speed,
            ))
    return out


def _presets_from_dense_frontier(frontier: list[tuple[int, int, float, float]], backend: str,
                                 layers: int) -> list[PresetOption]:
    """Fast / Balanced / Long-context picks from a dense layer-offload frontier."""
    if not frontier:
        return []

    def _speed(cpu_layers: int) -> float:
        # Relative throughput estimate. GPU layer = 1 unit of time, CPU layer = _CPU_LAYER_PENALTY.
        # Dense offload hurts far more than the MoE equivalent: with a 20x penalty, moving just
        # 10% of layers off already costs roughly two thirds of your tokens/sec.
        if layers <= 0:
            return 0.0
        gpu = layers - cpu_layers
        return round(layers / (gpu + cpu_layers * _CPU_LAYER_PENALTY), 3)

    def _make(key: str, label: str, icon: str, e: tuple[int, int, float, float]) -> PresetOption:
        cpu_layers, ctx, gpu_gb, kv_gb = e
        gpu_layers = layers - cpu_layers
        return PresetOption(
            key=key, label=label, icon=icon, backend=backend, ctx=ctx,
            n_cpu_moe=0,
            offload_kind=("ngl" if cpu_layers > 0 else ""),
            gpu_layers=gpu_layers, total_layers=layers,
            gpu_gb=round(gpu_gb, 2), kv_gb=round(kv_gb, 2),
            speed_score=_speed(cpu_layers),
            # 999 keeps llama-server's "everything on GPU" behaviour when nothing is offloaded.
            ngl=(gpu_layers if cpu_layers > 0 else 999),
        )

    fast_entry = min(frontier, key=lambda f: f[0])                 # fewest layers offloaded
    long_entry = max(frontier, key=lambda f: (f[1], -f[0]))        # most context
    fi, li = frontier.index(fast_entry), frontier.index(long_entry)
    if fi > li:
        fi, li = li, fi
    mi = (fi + li) // 2
    # Never let Balanced collapse onto Fast/Long: when they are adjacent the midpoint
    # rounds onto one of them and the chip disappears. Step inward instead.
    if mi in (fi, li) and li - fi >= 2:
        mi = fi + 1
    balanced_entry = frontier[mi]

    out: list[PresetOption] = []
    seen: set[tuple[int, int]] = set()
    for key, label, icon, entry in (
        ("fast", "Fast", "gauge", fast_entry),
        ("balanced", "Balanced", "cpu", balanced_entry),
        ("long-ctx", "Long context", "layers-3", long_entry),
    ):
        sig = (entry[0], entry[1])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(_make(key, label, icon, entry))
    return out


def _presets_from_frontier(frontier: list[tuple[int, int, float, float]], backend: str,
                           layers: int, native_ctx: int) -> list[PresetOption]:
    """Pick Fast / Balanced / Long-ctx from the frontier."""
    if not frontier:
        return []
    fast_min_ctx = 8192

    def _speed_score(ncm: int) -> float:
        return (layers - ncm) / layers if layers > 0 else 0.0

    def _make(key: str, label: str, icon: str, entry: tuple[int, int, float, float]) -> PresetOption:
        ncm, ctx, gpu_gb, kv_gb = entry
        return PresetOption(
            key=key, label=label, icon=icon, backend=backend,
            ctx=ctx, n_cpu_moe=ncm,
            offload_kind=("cpu-moe" if ncm >= layers else ("n-cpu-moe" if ncm > 0 else "")),
            gpu_layers=layers - ncm, total_layers=layers,
            gpu_gb=round(gpu_gb, 2), kv_gb=round(kv_gb, 2),
            speed_score=round(_speed_score(ncm), 3),
        )

    # Fast: highest speed (lowest ncm) where ctx meets a chat minimum
    fast_candidates = [f for f in frontier if f[1] >= fast_min_ctx] or frontier
    fast_entry = min(fast_candidates, key=lambda f: f[0])

    # Long-ctx: max ctx (any ncm)
    long_entry = max(frontier, key=lambda f: (f[1], -f[0]))

    # Balanced: a point STRICTLY between fast and long, so it is never a duplicate of
    # either. Plain midpoint rounding collapses onto fast when the two are adjacent
    # (e.g. a 2-point frontier gives (0+1)//2 == 0), which is why Balanced used to vanish.
    fi = frontier.index(fast_entry)
    li = frontier.index(long_entry)
    if fi > li: fi, li = li, fi
    mi = (fi + li) // 2
    if mi in (fi, li) and li - fi >= 2:
        mi = fi + 1
    balanced_entry = frontier[mi]

    out: list[PresetOption] = []
    seen: set[tuple[int, int]] = set()
    for key, label, icon, entry in (
        ("fast", "Fast", "gauge", fast_entry),
        ("balanced", "Balanced", "cpu", balanced_entry),
        ("long-ctx", "Long context", "layers-3", long_entry),
    ):
        sig = (entry[0], entry[1])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(_make(key, label, icon, entry))
    return out


# Without a prompt-speed measurement, never propose more than this: the memory estimate alone
# happily recommends a model's full 262K window on hardware that reads ~500 tokens/s (9 min).
UNMEASURED_CTX_CAP = 32768


# Why cap_context bound the context, as a code: the size core returns the code and the text is
# rendered from it (cap_reason_text), so the Rust port never has to reproduce Python's number
# formatting.
CAP_VERIFIED, CAP_TIME, CAP_UNMEASURED = "verified", "time", "unmeasured"


def cap_context_kind(memory_ctx: int, candidates: list[int], *, prompt_tps: float = 0.0,
                     prompt_budget_s: float = 120.0, verified_ctx: int = 0) -> tuple[int, str]:
    """cap_context with the reason as a code ("" when memory was the binding limit)."""
    if memory_ctx <= 0:
        return memory_ctx, ""
    limit, reason = memory_ctx, ""
    if verified_ctx and verified_ctx > 0:
        if verified_ctx < limit:
            limit, reason = verified_ctx, CAP_VERIFIED
    elif prompt_tps and prompt_tps > 0:
        by_time = int(prompt_tps * max(prompt_budget_s, 1))
        if by_time < limit:
            limit, reason = by_time, CAP_TIME
    elif UNMEASURED_CTX_CAP < limit:
        limit, reason = UNMEASURED_CTX_CAP, CAP_UNMEASURED
    fitting = sorted(c for c in candidates if 0 < c <= limit)
    if not reason:
        return memory_ctx, ""
    return (fitting[-1] if fitting else min(limit, memory_ctx)), reason


def cap_reason_text(kind: str, *, prompt_tps: float = 0.0, prompt_budget_s: float = 120.0,
                    verified_ctx: int = 0) -> str:
    if kind == CAP_VERIFIED:
        return f"verified on this machine at {verified_ctx:,} tokens"
    if kind == CAP_TIME:
        return f"a full prompt must finish in {int(prompt_budget_s)} s at the measured {prompt_tps:.0f} tokens/s"
    if kind == CAP_UNMEASURED:
        return "prompt speed not measured yet; measure context to go higher"
    return ""


def cap_context(memory_ctx: int, candidates: list[int], *, prompt_tps: float = 0.0,
                prompt_budget_s: float = 120.0, verified_ctx: int = 0) -> tuple[int, str]:
    """Largest candidate context that memory allows AND this machine can use.

    Order of evidence: a calibration-verified context (measured load + full prompt) wins outright
    as an upper bound; otherwise the measured prompt rate × the time budget; otherwise a
    conservative default until something is measured. Returns (ctx, reason) with reason "" when
    memory was the binding limit.
    """
    ctx, kind = cap_context_kind(memory_ctx, candidates, prompt_tps=prompt_tps,
                                 prompt_budget_s=prompt_budget_s, verified_ctx=verified_ctx)
    return ctx, cap_reason_text(kind, prompt_tps=prompt_tps, prompt_budget_s=prompt_budget_s,
                                verified_ctx=verified_ctx)


# ---- the size core: analyze()'s fit sweep, pick, context cap, presets and prompt cache ----
#
# One request in, one plan out, both plain JSON. analyze() builds the request from the GGUF
# summary and the backend inventory (resolving everything that touches files or hostile
# per-layer arrays first, in Python), and applies the plan to the values it writes. The request:
#
#   shape            kv_shape() of the model (never None here: analyze refuses that first)
#   layers           block_count, 0..MAX_BLOCK_COUNT
#   native_ctx       context_length (any integer)
#   model_gb_raw     file size in GiB
#   moe_ratio        _moe_ratio(expert_count); is_moe: expert_count is an int above 1
#   mmproj_vram_gb   VRAM pinned to the main GPU (projector, encoder scratch, draft head)
#   n_sessions       1..8
#   backends         [{vram_gb, gpu_count (>= 1), cards: [GB...], host_ram_gb, same_as}];
#                    same_as is the index of the first backend whose name equals this one's,
#                    which is how analyze() finds "the" backend of a recommended name
#   preset           the requested preset key ("" = default)
#   prompt_tps, prompt_budget_s (floats), verified_ctx (int), cache_ram_cap_mib (int)
#
# The plan (every float is what analyze() stores; presets carry no backend name):
#
#   plans            per backend, in order: {rows: [FitRow...], max_ctx}
#   recommended      index of the recommended backend, or None
#   offload          [kind, n] the fit sweep needed at the recommended backend's max context
#   estimated_ctx, initial_ctx (after the cap), capped_ctx, cap (a CAP_* code or "")
#   sized            whether the preset/prompt-cache block ran (a recommendation at ctx > 0)
#   ctx              the final per-session context
#   presets, frontier, active_preset, fits_full_gpu
#   ngl              dense preset's ngl to write, or None to leave it
#   fit              True when placement is handed to llama.cpp's fitter (fit = on)
#   cache_ram        --cache-ram in MiB, or None to leave it unset

def _preset_dict(p: PresetOption) -> dict[str, Any]:
    d = asdict(p)
    del d["backend"]
    return d


def size_plan(req: dict[str, Any]) -> dict[str, Any]:
    """The reference implementation (and the answer under MODEL_AUTOCONFIG=python)."""
    shape = req["shape"]
    layers = req["layers"]
    native_ctx = req["native_ctx"]
    model_gb_raw = req["model_gb_raw"]
    moe_ratio = req["moe_ratio"]
    is_moe = req["is_moe"]
    mmproj_vram_gb = req["mmproj_vram_gb"]
    n_sessions = req["n_sessions"]
    backends = req["backends"]
    preset = req["preset"]
    bytes_per = _cache_dtype_bytes(_CACHE_DEFAULT)

    def kv_gb_at(total_ctx: int) -> float:
        return kv_shape_bytes(shape, total_ctx, bytes_per, bytes_per) / (1024 ** 3)

    # Candidate ctx values: default cap is the model's native ctx. Linear RoPE extension
    # to 2× is possible but (a) degrades quality noticeably, (b) inflates compute buffers
    # unpredictably. Not worth the OOM risk as an autoconfig default — users who want
    # extension can dial ctx-size up in the form manually.
    cands = sorted(set(list(_CTX_CANDIDATES) + ([native_ctx] if native_ctx else [])))
    if native_ctx:
        cands = [c for c in cands if c <= native_ctx]

    def fit_backend(b: dict) -> tuple[list[FitRow], int, tuple[str, int]]:
        rows: list[FitRow] = []
        gpu_count = b["gpu_count"]
        vram = b["vram_gb"]
        # Reserve scales per GPU (each CUDA context takes ~500 MB just to be initialized).
        # NOTE: compute buffer VRAM (prompt-eval scratch, ~1-2 GB at stock ub) is NOT
        # subtracted from budget. Real-world calibration on this rig shows the 1.08
        # split-overhead + 0.5 GB/GPU reserve already over-estimates enough to absorb
        # the compute buffer for models we've tested.
        # The projector is pinned to the main GPU, not layer-split. Under an even layer
        # split every GB pinned to one card costs gpu_count GB of usable POOLED capacity.
        budget = vram - _RESERVE_PER_GPU * gpu_count - mmproj_vram_gb
        # Per-card capacities for the single-device feasibility check below. Falls back to an
        # even division of the pool when the sampler has not reported individual cards.
        card_caps = [c - _RESERVE_PER_GPU for c in b["cards"]]
        if gpu_count > 1 and not card_caps:
            card_caps = [(vram / gpu_count) - _RESERVE_PER_GPU] * gpu_count
        # The projector, compute buffer and cuBLAS workspace are not layer-split: they all
        # land on the main GPU, so device 0 starts with less room than its siblings.
        pinned_gb = mmproj_vram_gb
        # Model overhead: single-GPU is basically file size; layer-split adds ~5% for cross-card handoffs
        overhead_mul = _MODEL_OVERHEAD_SPLIT if gpu_count > 1 else _MODEL_OVERHEAD_SINGLE
        model_gb = model_gb_raw * overhead_mul
        eff_moe = moe_ratio
        max_fit = 0
        max_fit_offload: tuple[str, int] = ("", 0)
        for per_session_ctx in cands:
            # ctx-size llama-server sees = per_session * n_sessions. KV cache is sized against
            # the TOTAL because each slot is allocated a contiguous chunk in the shared cache.
            total_ctx = per_session_ctx * n_sessions
            kv_gb = kv_gb_at(total_ctx)

            # The per-card test is handed to the search rather than applied to its answer, so
            # it can keep offloading until BOTH the pool and every individual card are happy.
            def _card_ok(gpu_gb: float, kind: str, n: int,
                         _kv=kv_gb, _caps=card_caps, _pin=pinned_gb) -> bool:
                if gpu_count <= 1 or not _caps:
                    return True
                if eff_moe > 0:
                    att = model_gb * (1 - eff_moe)
                    exp = (model_gb * eff_moe / layers) if layers else 0.0
                    ncm = n if kind == "n-cpu-moe" else (layers if kind == "cpu-moe" else 0)
                else:
                    att, exp, ncm = gpu_gb, 0.0, 0
                return _split_feasible(layers, ncm, att, exp, _kv, gpu_count, _pin, _caps)

            fits, gpu_model_gb, offload_kind, n_cm = _find_fit(
                model_gb, kv_gb, budget, layers, eff_moe, card_ok=_card_ok)
            total = gpu_model_gb + kv_gb
            rows.append(FitRow(
                ctx=per_session_ctx, total_ctx=total_ctx,
                model_gb=round(gpu_model_gb, 2), kv_gb=round(kv_gb, 2),
                total_gb=round(total, 2), fits=fits,
                free_gb=round(vram - total, 2),
                offload_kind=offload_kind, n_cpu_moe=n_cm,
                gpu_pct=(round(100.0 * gpu_model_gb / model_gb) if model_gb > 0 else 100),
            ))
            if fits:
                max_fit = per_session_ctx
                max_fit_offload = (offload_kind, n_cm)
        return rows, max_fit, max_fit_offload

    # Symmetric q8_0 K/V, always (see analyze()).
    plans: list[tuple[list[FitRow], int]] = []
    offload_by_name: dict[int, tuple[str, int]] = {}
    for b in backends:
        rows, max_fit, max_fit_offload = fit_backend(b)
        plans.append((rows, max_fit))
        if max_fit:
            offload_by_name[b["same_as"]] = max_fit_offload

    # pick recommendation:
    #   Rule of thumb: pick the largest ctx we can, on the smallest GPU that hosts it,
    #   BUT if MoE requires offload at that ctx, prefer the bigger GPU (less offload = faster).
    rec: int | None = None
    rec_ctx = 0
    estimated_ctx = 0
    capped_ctx = 0
    cap = ""
    fitting = [i for i, (_, max_fit) in enumerate(plans) if max_fit > 0]
    if fitting:
        global_max_ctx = max(plans[i][1] for i in fitting)
        top = [i for i in fitting if plans[i][1] >= global_max_ctx]

        def _needs_offload_at(i: int, ctx: int) -> bool:
            r = next((row for row in plans[i][0] if row.ctx == ctx), None)
            return bool(r and r.offload_kind)
        if is_moe and any(_needs_offload_at(i, global_max_ctx) for i in top):
            rec = sorted(top, key=lambda i: backends[i]["vram_gb"], reverse=True)[0]
        else:
            rec = sorted(top, key=lambda i: backends[i]["vram_gb"])[0]
        rec_ctx = plans[rec][1]
        estimated_ctx = rec_ctx
        rec_ctx, cap = cap_context_kind(
            rec_ctx, [r.ctx for r in plans[rec][0] if r.ctx <= plans[rec][1]],
            prompt_tps=req["prompt_tps"], prompt_budget_s=req["prompt_budget_s"],
            verified_ctx=req["verified_ctx"])
        capped_ctx = rec_ctx
    initial_ctx = rec_ctx

    active_preset = ""
    presets: list[PresetOption] = []
    frontier_opts: list[PresetOption] = []
    fits_full_gpu = False
    sized = False
    ngl: int | None = None
    fit = False
    cache_ram: int | None = None
    if rec is not None and rec_ctx > 0 and native_ctx > 0:
        # If the model fits entirely on the GPU AND reaches its native context that way,
        # there is nothing to trade and one option is the whole truth.
        no_offload_ctx = max((r.ctx for r in plans[rec][0] if r.fits and not r.offload_kind),
                             default=0)
        fits_full_gpu = no_offload_ctx >= native_ctx
    if rec is not None and rec_ctx > 0:
        sized = True
        rec_vram = backends[rec]["vram_gb"]
        rb = backends[backends[rec]["same_as"]]
        off_kind, n_cm = offload_by_name.get(backends[rec]["same_as"], ("", 0))
        gpu_count = rb["gpu_count"]
        overhead_mul = _MODEL_OVERHEAD_SPLIT if gpu_count > 1 else _MODEL_OVERHEAD_SINGLE
        model_gb_rec = model_gb_raw * overhead_mul
        if is_moe and layers > 0:
            budget = rec_vram - _RESERVE_PER_GPU * gpu_count - mmproj_vram_gb
            # Same per-card test the fit table applies. For a MoE the split is attention
            # (always resident) plus the experts of every layer at or above n-cpu-moe.
            mcaps = [c - _RESERVE_PER_GPU for c in rb["cards"]]
            if gpu_count > 1 and len(mcaps) != gpu_count:
                mcaps = [(rec_vram / gpu_count) - _RESERVE_PER_GPU] * gpu_count

            def _moe_card_ok(weight_gb: float, kv_gb: float, ncm: int) -> bool:
                if gpu_count <= 1 or not mcaps:
                    return True
                att = model_gb_rec * (1 - moe_ratio)
                exp = (model_gb_rec * moe_ratio / layers) if layers else 0.0
                return _split_feasible(layers, ncm, att, exp, kv_gb,
                                       gpu_count, mmproj_vram_gb, mcaps)

            frontier = _pareto_frontier(layers, kv_gb_at, model_gb_rec, moe_ratio, budget, cands,
                                        n_sessions=n_sessions, card_ok=_moe_card_ok)
            presets = _presets_from_frontier(frontier, "", layers, native_ctx)
            frontier_opts = _frontier_options(frontier, "", layers, dense=False)
            if presets:
                chosen = next((p for p in presets if p.key == preset), presets[len(presets) // 2])
                active_preset = chosen.key
                rec_ctx = chosen.ctx
                off_kind = chosen.offload_kind
                n_cm = chosen.n_cpu_moe
        elif not is_moe and layers > 0:
            # DENSE model: no experts to offload, so trade whole layers for context via `ngl`.
            budget = rec_vram - _RESERVE_PER_GPU * gpu_count - mmproj_vram_gb
            fcaps = [c - _RESERVE_PER_GPU for c in rb["cards"]]
            if gpu_count > 1 and len(fcaps) != gpu_count:
                fcaps = [(rec_vram / gpu_count) - _RESERVE_PER_GPU] * gpu_count

            def _front_card_ok(weight_gb: float, kv_gb: float) -> bool:
                if gpu_count <= 1 or not fcaps:
                    return True
                return _split_feasible(layers, 0, weight_gb, 0.0, kv_gb,
                                       gpu_count, mmproj_vram_gb, fcaps)

            dfront = _dense_frontier(layers, kv_gb_at, model_gb_rec, budget, cands,
                                     n_sessions=n_sessions, card_ok=_front_card_ok)
            presets = _presets_from_dense_frontier(dfront, "", layers)
            frontier_opts = _frontier_options(dfront, "", layers, dense=True)
            if presets:
                # Dense offload is OPT-IN: without an explicit pick, stay on "fast".
                want = preset or "fast"
                chosen = next((p for p in presets if p.key == want), presets[0])
                active_preset = chosen.key
                rec_ctx = chosen.ctx
                if chosen.ngl > 0:
                    ngl = chosen.ngl
        # The presets above pick from the memory frontier; the usable-context cap still binds.
        if cap and rec_ctx > capped_ctx:
            rec_ctx = capped_ctx
        # Expert offload: placement is handed to llama.cpp's own fitter (see analyze()).
        fit = off_kind in ("cpu-moe", "n-cpu-moe")

        # ---- cache-ram: host-RAM budget for the server-side prompt cache (see analyze()).
        kv_convo_gb = kv_gb_at(rec_ctx * n_sessions)
        if kv_convo_gb > 0:
            host_ram_gb = rb["host_ram_gb"]
            # Weights that will live in host RAM must stay in page cache; the prompt cache
            # must not crowd them out.
            cpu_weight_gb = max(0.0, model_gb_raw - rec_vram) if fit else 0.0
            upper_gb = host_ram_gb - cpu_weight_gb - _CACHE_RAM_HEADROOM_GB
            want_gb = _CACHE_RAM_CONVOS * kv_convo_gb
            cache_gb = min(want_gb, upper_gb) if upper_gb > 0 else 0.0
            if cache_gb > 0:
                # Never suggest worse than llama.cpp's own default unless headroom forbids it.
                mib = int(round(cache_gb * 1024))
                if upper_gb * 1024 >= _CACHE_RAM_DEFAULT_MIB:
                    mib = max(mib, _CACHE_RAM_DEFAULT_MIB)
                # #697: never above the configured cap. On a shared-memory GPU the prompt
                # cache competes with the model.
                cache_ram = min(mib, req["cache_ram_cap_mib"])

    return {
        "plans": [{"rows": [asdict(r) for r in rows], "max_ctx": max_fit} for rows, max_fit in plans],
        "recommended": rec,
        "offload": list(offload_by_name.get(backends[rec]["same_as"], ("", 0))) if rec is not None else ["", 0],
        "estimated_ctx": estimated_ctx,
        "initial_ctx": initial_ctx,
        "capped_ctx": capped_ctx,
        "cap": cap,
        "sized": sized,
        "ctx": rec_ctx,
        "presets": [_preset_dict(p) for p in presets],
        "frontier": [_preset_dict(p) for p in frontier_opts],
        "active_preset": active_preset,
        "fits_full_gpu": fits_full_gpu,
        "ngl": ngl,
        "fit": fit,
        "cache_ram": cache_ram,
    }


# ---- input prep (slice 2): reading the GGUF summary and the backends, and the early refusals ----
#
# analyze()'s first steps, unchanged in behaviour: the hostile int() conversions of the GGUF
# summary's model fields, _kv_first_int, kv_shape, _moe_ratio, the backend filter and the four
# early refusals, then (once the projector files are resolved) the main-GPU reservation and the
# backends as the size core reads them. `prepare_all` is the whole of it as one JSON request:
#
#   n_sessions       analyze()'s n_sessions (already clamped there; clamped again, idempotently)
#   arch, model      summary["arch"] and summary["model"], raw
#   file_size        the model's bytes
#   backends         the backend dicts analyze() was given, raw
#   projector        {has_mmproj, mmproj_gb, mtp_gb}: resolved from files by analyze()
#
# and answers either a refusal, {"refuse": kind, ...the numbers its message needs}, or
# {"refuse": None, n_sessions, layers, native_ctx, hidden, is_moe, moe_ratio, model_gb_raw,
#  shape, sized (indices of the backends that report VRAM), mmproj_vram_gb, backends (the size
#  core's form)}. Python raises where it always raised (a corrupt summary still raises).

# Deepest public LLMs have ~60-130 transformer blocks (Llama 3.1 405B: 126). Fit search loops
# over layers x context candidates, and the block count can come from an untrusted remote GGUF
# header, so an absurd value (0xFFFFFFFF) would spin for minutes on the event loop. 4096 is
# ~30x the deepest real model: no legitimate file is refused, and the worst accepted input
# costs a few hundred thousand iterations.
MAX_BLOCK_COUNT = 4096

_MMPROJ_VRAM_MULT = 1.0  # projector weights land in VRAM at ~their file size.
                         # Verified on Qwen3-VL-4B: mmproj-F32.gguf is 1.55 GB on disk and
                         # llama-server allocates 1584.43 MiB for it. (An earlier 2.0 here was
                         # a mis-calibration: that same 1584 MiB was compared against the 0.78 GB
                         # BF16 projector in the directory rather than the F32 one actually
                         # referenced by the preset.)
_MMPROJ_COMPUTE_GB = 0.5  # modality-encoder scratch beyond the projector weights.
                          # This is deliberately small. An earlier 1.8 here was calibrated off a
                          # FAILED allocation line in a log (an attempt at a much larger ctx, on a
                          # different model) and then multiplied by gpu_count — which reserved
                          # 5.34 GB for a 0.87 GB projector and made a working model report
                          # "doesn't fit at any context". Ground truth from a loaded
                          # Qwen3.8-27B-OBLITERATED at ctx=130768: 22.63 GiB used total, of which
                          # model+KV+projector accounts for ~20.96 GiB — so ALL remaining overhead,
                          # both cards' CUDA contexts included, is ~1.67 GiB. _RESERVE_PER_GPU
                          # already covers most of that.


def _kv_first_int(kv_heads: Any, default: int = 8) -> int:
    """kv_heads may be an int, or a per-layer array dict from GGUF; extract a representative int."""
    if isinstance(kv_heads, int):
        return kv_heads
    if isinstance(kv_heads, dict) and kv_heads.get("_array"):
        sample = kv_heads.get("sample") or []
        # pick the most common value in the sample; for gemma it's usually 8 with a few 1s
        if sample:
            counts: dict[int, int] = {}
            for v in sample:
                if v is None:  # a non-finite float in the GGUF, nulled by summarize (#901)
                    continue
                counts[int(v)] = counts.get(int(v), 0) + 1
            if counts:
                return max(counts, key=lambda k: counts[k])
    if isinstance(kv_heads, list) and kv_heads and kv_heads[0] is not None:
        return int(kv_heads[0])
    return default


def _card_gb(c: Any) -> float:
    """One card's VRAM as the fit maths uses it (`c - reserve`): a number, else TypeError as
    that subtraction raised before the size core existed."""
    if isinstance(c, (int, float)):
        return float(c)
    raise TypeError(f"card_vram_gb entry {type(c).__name__!r} is not a number")


def prepare(inp: dict[str, Any]) -> dict[str, Any]:
    """analyze()'s reading of the summary and backends, up to and including the early refusals."""
    # Clamp n_sessions to a sensible range for a homelab. Above 8 the per-slot ctx
    # shrinks below usability for real chat, and llama-server continuous batching
    # overhead starts dominating.
    n_sessions = max(1, min(int(inp["n_sessions"] or 1), 8))
    arch = (inp["arch"] or "").lower()
    m = inp["model"] or {}
    layers = int(m.get("block_count") or 0)
    if layers > MAX_BLOCK_COUNT or layers < 0:
        return {"refuse": "block_count", "layers": layers}
    heads = int(m.get("attention_head_count") or 1)
    embed = int(m.get("embedding_length") or 0)
    head_dim = embed // heads if heads > 0 else 0
    kv_heads = _kv_first_int(m.get("attention_head_count_kv"), default=heads)
    native_ctx = int(m.get("context_length") or 0)
    experts = m.get("expert_count")
    # explicit K/V lengths + hybrid markers (Qwen 3.5, Zamba, etc.)
    key_length = int(m.get("key_length")) if isinstance(m.get("key_length"), int) else None
    value_length = int(m.get("value_length")) if isinstance(m.get("value_length"), int) else None
    full_attention_interval = int(m.get("full_attention_interval")) if isinstance(m.get("full_attention_interval"), int) else None
    ssm_state_size = int(m.get("ssm_state_size")) if isinstance(m.get("ssm_state_size"), int) else None
    # Sliding-window attention parameters travel together and are threaded through every
    # kv_cache_bytes call as one bundle. Absent for models that declare none, in which case
    # kv_cache_bytes falls back to its previous behaviour.
    _swa = {
        "sliding_window": m.get("sliding_window") if isinstance(m.get("sliding_window"), int) else None,
        "sliding_window_pattern": m.get("sliding_window_pattern"),
        "key_length_swa": m.get("key_length_swa") if isinstance(m.get("key_length_swa"), int) else None,
        "value_length_swa": m.get("value_length_swa") if isinstance(m.get("value_length_swa"), int) else None,
        "shared_kv_layers": m.get("shared_kv_layers") if isinstance(m.get("shared_kv_layers"), int) else None,
        # Raw, not collapsed: which layers are global decides which head count applies.
        "kv_heads_pattern": m.get("attention_head_count_kv"),
    }
    # Model VRAM depends on whether we'll be layer-splitting across multiple GPUs; the size core
    # applies the split multiplier per backend.
    model_gb_raw = inp["file_size"] / (1024 ** 3)
    moe_ratio = _moe_ratio(experts)
    # Plan around q8_0 KV cache: near-identical quality to f16 in practice,
    # and lets us fit ~2× the context. Users can override in the form if they want f16.
    bytes_per = _cache_dtype_bytes(_CACHE_DEFAULT)

    # HARD STOP if we have nothing to fit against (see analyze() for the message and why).
    backends = inp["backends"]
    if not backends:
        return {"refuse": "no_backends"}
    # A model larger than VRAM + system RAM cannot run at ANY offload setting.
    _ram = max((float(b.get("host_ram_gb") or 0) for b in backends), default=0.0)
    _vram = max((float(b.get("vram_gb") or 0) for b in backends), default=0.0)
    if _ram > 0 and model_gb_raw > (_vram + _ram):
        return {"refuse": "ram", "model_gb": model_gb_raw, "vram_gb": _vram, "ram_gb": _ram}
    # A backend that is discovered but reports 0 GB is a different failure from one that is
    # genuinely too small, and must not be reported as the latter.
    sized = [i for i, b in enumerate(backends) if float(b.get("vram_gb") or 0) > 0]
    if not sized:
        return {"refuse": "unsized_vram", "names": [str(b.get("name") or "?") for b in backends]}

    # HARD STOP if the KV cache cannot be sized from this GGUF's metadata: zero KV silently means
    # "the cache is free" and every candidate would fit.
    shape = kv_shape(arch, layers, kv_heads, head_dim,
                     key_length=key_length, value_length=value_length,
                     full_attention_interval=full_attention_interval,
                     ssm_state_size=ssm_state_size, **_swa)
    if kv_shape_bytes(shape, 4096, bytes_per) <= 0:
        missing = [k for k, v in (("block_count", layers),
                                  ("attention_head_count_kv", kv_heads),
                                  ("head_dim (embedding_length / attention_head_count)", head_dim))
                   if not v]
        return {"refuse": "kv", "missing": missing}
    return {"refuse": None, "n_sessions": n_sessions, "layers": layers, "native_ctx": native_ctx,
            "hidden": embed, "is_moe": isinstance(experts, int) and experts > 1,
            "moe_ratio": moe_ratio, "model_gb_raw": model_gb_raw, "shape": shape, "sized": sized}


def main_gpu_reserve_gb(has_mmproj: bool, mmproj_gb: float, mtp_gb: float, layers: int,
                        hidden: int) -> float:
    """VRAM pinned to the MAIN GPU: projector weights + encoder compute buffer, the draft head,
    and the larger vision ubatch's compute buffer (see analyze())."""
    mmproj_vram_gb = (mmproj_gb * _MMPROJ_VRAM_MULT + _MMPROJ_COMPUTE_GB) if has_mmproj else 0.0
    # The draft head rides along in the same non-layer-split reservation. Its own KV is small
    # (a handful of layers over the drafted window) and folded into this rather than modelled.
    mmproj_vram_gb += mtp_gb * 1.15
    # Raising ubatch for the vision encoder (see image-max-tokens) grows the compute buffer,
    # which is not layer-split and lands on the main GPU with everything else here.
    # Calibrated from a measured point: a 27B (64 layers, 5120 hidden) at ub=2048 allocated
    # ~4.7 GB, i.e. ~7 bytes per (ub x layer x hidden). Scaled against the default ub of 512,
    # only the INCREASE is charged.
    if has_mmproj and layers > 0:
        if hidden > 0:
            _extra_ub = max(0, 1024 - 512)
            mmproj_vram_gb += 7.0 * _extra_ub * layers * hidden / 1e9
    return mmproj_vram_gb


def size_backends(backends: list[dict]) -> list[dict[str, Any]]:
    """The sized backends as the size core reads them. same_as is the first backend whose name
    equals this one's (Python ==), which is how analyze() finds "the" backend of a name."""
    return [{
        "vram_gb": float(b["vram_gb"]),
        "gpu_count": max(1, int(b.get("gpu_count", 1))),
        "cards": [_card_gb(c) for c in (b.get("card_vram_gb") or [])],
        "host_ram_gb": float(b.get("host_ram_gb") or 0.0),
        "same_as": next(j for j, o in enumerate(backends) if o["name"] == b["name"]),
    } for b in backends]


def prepare_all(inp: dict[str, Any]) -> dict[str, Any]:
    """The reference for the Rust port's `prep` part: prepare(), then (unless it refused) the
    main-GPU reservation and the size core's backends, in analyze()'s order."""
    p = prepare(inp)
    if p["refuse"] is not None:
        return p
    pj = inp["projector"]
    reserve = main_gpu_reserve_gb(pj["has_mmproj"], pj["mmproj_gb"], pj["mtp_gb"], p["layers"], p["hidden"])
    return dict(p, mmproj_vram_gb=reserve,
                backends=size_backends([inp["backends"][i] for i in p["sized"]]))


# ---- values assembly (slice 2): the settings analyze() writes, before the diff and quirks ----
#
# One JSON request (`assemble_values`); the answer is the ordered values dict (key order is part
# of it). The request:
#
#   model_rel        the model path ("" = none)
#   n_sessions       1..8
#   chat_template    whether summary["chat_template"] is set (a bool: only its truthiness counts)
#   features         summary["chat_template_features"], raw
#   section          the section name ("" for an estimate); vision: the vision switch
#   current          the saved section's mmproj / ubatch-size / batch-size, where set (strings)
#   mmproj_rel       the projector analyze() resolved ("" = none); has_mmproj: one is attached
#   spec             [[key, value]...]: the speculative-decoding keys, in order (slice 3 ports them)
#   plan             the size plan's {initial_ctx, sized, ctx, ngl, fit, cache_ram}
#   rope             summary["model"] (arch, rope_scaling_type, rope_scaling_factor), raw
#   native_ctx       context_length as prepare() read it
#   rec_gpu_count    GPUs of the recommended backend (its first same-named one), or None

# The GGUF already decides rope scaling for these architectures (llama.cpp derives per-layer
# scaling itself); a preset value would override it (#568).
_ROPE_FROM_GGUF_ARCHS = ("gemma3",)


def rope_owned_by_gguf(model: dict[str, Any]) -> bool:
    """True when the GGUF already decides rope scaling and a preset must not override it:
    the metadata declares a scaling type or factor, or the architecture is one whose per-layer
    scaling llama.cpp derives itself. `model` is the `summary["model"]` dict."""
    arch = str(model.get("arch") or "").lower()
    if any(arch.startswith(a) for a in _ROPE_FROM_GGUF_ARCHS):
        return True
    rtype = str(model.get("rope_scaling_type") or "").strip().lower()
    if rtype not in ("", "none"):
        return True
    factor = model.get("rope_scaling_factor")
    return isinstance(factor, (int, float)) and factor > 0


def assemble_values(inp: dict[str, Any]) -> dict[str, str]:
    """analyze()'s values, from "build values" through split-mode (see analyze() for the why of
    each setting)."""
    n_sessions = inp["n_sessions"]
    current = inp["current"]
    plan = inp["plan"]
    rec_ctx = plan["initial_ctx"]
    values: dict[str, str] = {}
    if inp["model_rel"]:
        values["model"] = inp["model_rel"]
    if rec_ctx > 0:
        # ctx-size llama-server allocates = per-session ctx × n_sessions.
        values["ctx-size"] = str(rec_ctx * n_sessions)
    # ALWAYS emit parallel: llama-server's own default is 4 slots, which quarters each slot.
    values["parallel"] = str(n_sessions)
    if n_sessions > 1:
        values["cont-batching"] = "on"
        values["context-shift"] = "on"
        values["keep"] = "256"
        values["batch-size"] = "4096"
    values["ngl"] = "999"
    values["flash-attn"] = "on"
    values["cache-type-k"] = _CACHE_DEFAULT
    values["cache-type-v"] = _CACHE_DEFAULT
    values["cache-reuse"] = "1"
    if inp["chat_template"]:
        values["jinja"] = "true"

    # Multimodal: the user's projector wins; otherwise the one analyze() resolved.
    if inp["section"] and inp["vision"]:
        current_mmproj = current.get("mmproj", "").strip()
        if current_mmproj:
            values["mmproj"] = current_mmproj
        else:
            if inp["mmproj_rel"]:
                values["mmproj"] = inp["mmproj_rel"]

    # Speculative decoding (resolved by analyze(); see SpecProfile there).
    for k, v in inp["spec"]:
        values[k] = v

    if plan["sized"]:
        rec_ctx = plan["ctx"]
        values["ctx-size"] = str(rec_ctx * n_sessions)
        if plan["ngl"] is not None:
            values["ngl"] = str(plan["ngl"])
        if plan["fit"]:
            # Expert offload: placement handed to llama.cpp's fitter, which only adjusts UNSET args.
            values["fit"] = "on"
            values.pop("ngl", None)
            values.pop("cpu-moe", None)
            values.pop("n-cpu-moe", None)
            values.pop("tensor-split", None)
        if plan["cache_ram"] is not None:
            values["cache-ram"] = str(plan["cache_ram"])

    # Reasoning / thinking — inferred from chat-template scanning.
    features = inp["features"] or {}
    tpl_kwargs: dict[str, Any] = {}
    if features.get("accepts_enable_thinking"):
        values["reasoning"] = "on"
    if features.get("accepts_reasoning_effort"):
        tpl_kwargs["reasoning_effort"] = "medium"
    if tpl_kwargs:
        values["chat-template-kwargs"] = json.dumps(tpl_kwargs)
    if features.get("uses_think_tags") or features.get("uses_channel_thought"):
        values["reasoning-format"] = "deepseek"
    if features.get("accepts_preserve_thinking"):
        values["reasoning-preserve"] = "on"

    # Multimodal extras — only meaningful when a projector is actually attached.
    if inp["has_mmproj"]:
        values["mmproj-offload"] = "on"
        # Bound how many tokens a single image may consume (an unbounded decode buffer OOMs at
        # inference time, long after the fit maths approved the load).
        values["image-max-tokens"] = "1024"
        # ubatch-size MUST be at least image-max-tokens, or the model aborts on the first image
        # (non-causal attention: the whole image has to be in one micro-batch).
        try:
            _imt = int(values.get("image-max-tokens") or 0)
        except ValueError:
            _imt = 0
        if _imt > 0:
            try:
                _cur_ub = int(current.get("ubatch-size") or 0)
            except ValueError:
                _cur_ub = 0
            values["ubatch-size"] = str(max(_imt, _cur_ub))
            # batch-size must be >= ubatch-size, but only RAISE it: an absent batch-size is
            # llama-server's 2048, comfortably above a 1024 ubatch.
            try:
                _cur_b = int(values.get("batch-size") or current.get("batch-size") or 0)
            except ValueError:
                _cur_b = 0
            if _cur_b and _cur_b < _imt:
                values["batch-size"] = str(_imt)
            elif not _cur_b and _imt > 2048:      # above llama-server's own default
                values["batch-size"] = str(_imt)

    # RoPE: only when the chosen ctx exceeds n_ctx_train on a model that declares no scaling.
    native_ctx = inp["native_ctx"]
    if not rope_owned_by_gguf(inp["rope"]) and rec_ctx > native_ctx and native_ctx > 0:
        values["rope-scaling"] = "linear"
        values["rope-scale"] = f"{round(rec_ctx / native_ctx, 1)}"

    # Multi-GPU: be explicit about the split strategy.
    if inp["rec_gpu_count"] is not None and inp["rec_gpu_count"] > 1:
        values["split-mode"] = "layer"
    return values


# ---- speculative-decoding profiles (slice 3) ----


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

# The saved-section keys resolve_spec() reads (what analyze() sends of the section).
SPEC_SECTION_KEYS: tuple[str, ...] = ("spec-type", "spec-draft-model", "spec-draft-ngl", *SPEC_PROFILE_KEYS)


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


# One JSON request (`resolve_spec`); the answer is {mtp_rel, saved, key, head, values}:
#
#   section       the section name ("" for an estimate)
#   current       the saved section's SPEC_SECTION_KEYS where set (strings), or None for no section
#   spec_profile  the profile picked in the UI (raw string); mode: the workload ("" = none)
#   files         whether analyze() looked for files (a models_dir and a section name)
#   found_mtp     the draft head _find_mtp found ("" when it found none or was not asked)
#   nextn         summary["model"]["nextn_predict_layers"], raw (built-in MTP layers)
#
#   mtp_rel       the draft head analyze() budgets (the saved one wins over the found one)
#   saved         match_spec_profile(current); key: the profile in effect ("custom" = hand-tuned)
#   head          the draft head the profile uses; values: [[key, value]...], the spec keys in order


def resolve_spec(inp: dict[str, Any]) -> dict[str, Any]:
    """analyze()'s speculative-decoding resolution (see SpecProfile)."""
    current = inp["current"]
    cs = current or {}
    mtp_rel = ""
    if inp["files"]:
        mtp_rel = cs.get("spec-draft-model", "").strip() or inp["found_mtp"]
    builtin_mtp = int(inp["nextn"] or 0) > 0
    saved = match_spec_profile(current)
    key = (inp["spec_profile"] or "").strip()
    if key not in SPEC_PROFILE_BY_KEY and key != "custom":
        # No explicit pick. Fall back to what is saved; a new section gets Balanced when a
        # head was found beside the weights and Off when there is nothing to draft with.
        key = saved or (MODE_SPEC_PROFILE.get(inp["mode"], "balanced") if (mtp_rel or builtin_mtp) else "off")
    # A profile that needs a head but has none cannot run — llama-server would start and then
    # fail to load the draft. Fall back rather than offering a configuration that cannot work.
    head = cs.get("spec-draft-model", "").strip() or mtp_rel
    prof = SPEC_PROFILE_BY_KEY.get(key)
    if prof and prof.needs_head and not head and not builtin_mtp:
        prof = SPEC_PROFILE_BY_KEY["off"]
        key = "off"

    values: dict[str, str] = {}
    if inp["section"]:
        if key == "custom":
            # Hand-tuned. Echo every spec key verbatim: Fill writes exactly `values`, so a
            # setting that is not repeated here is silently erased on the next save.
            for k in SPEC_SECTION_KEYS:
                v = cs.get(k, "").strip()
                if v:
                    values[k] = v
        elif prof is not None:
            # Every owned key is written even when empty, so switching profiles CLEARS what
            # the previous one set. Without this, Coding -> Writing would leave n-min = 1
            # behind and the result would match neither profile.
            values["spec-type"] = prof.spec_type
            for k in SPEC_PROFILE_KEYS:
                values[k] = prof.knobs.get(k, "")
            if prof.spec_type and prof.needs_head and not head:
                # Built-in nextn layers: the engine drafts from the model itself.
                values["spec-draft-model"] = ""
                values["spec-draft-ngl"] = ""
            elif prof.spec_type and prof.needs_head:
                values["spec-draft-model"] = head
                # Without this the head lands on the CPU, and a draft evaluated on the CPU is
                # slower than the main model it is meant to be racing ahead of.
                values["spec-draft-ngl"] = cs.get("spec-draft-ngl", "").strip() or "999"
            else:
                # n-gram strategies have no model to place, and Off has nothing at all.
                values["spec-draft-model"] = ""
                values["spec-draft-ngl"] = ""
    return {"mtp_rel": mtp_rel, "saved": saved, "key": key, "head": head,
            "values": [[k, v] for k, v in values.items()]}


# ---- companion files: the mmproj and draft-head name rules (slice 6) ----
#
# The directory listing and every stat stay in autoconfig.py; these rules only read a listing:
# [[name, kind, size]...] in sorted order (kind "file" with its size, or None where stat failed;
# "other" for anything that is not a regular file; "error" where is_file() itself raised), or
# None when the directory could not be listed. One request per rule call (`pick_file`):
#
#   {"rule": "mmproj_subdir", "listing", "subdir"}    the smallest projector in the model's folder
#   {"rule": "mmproj_flat", "listing", "section"}     the first top-level projector naming the model
#   {"rule": "mtp_folder", "listing", "prefix", "section"}  the smallest head in a folder
#   {"rule": "mtp_flat", "listing", "section"}        the smallest top-level head naming the model
#   {"rule": "projector", "files", "current", "found", "stat_gb", "vision", "override"}
#       which projector analyze() budgets: {mmproj_rel, mmproj_gb, available}

# Real draft/MTP heads are tens to a few hundred MB; anything larger with "mtp" in its name is a
# full model build that includes MTP layers (e.g. Unsloth's "-MTP-GGUF" repos), not a head.
HEAD_MAX_BYTES = 2 * 1024 ** 3


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


def _is_mmproj(name: str) -> bool:
    return name.lower().endswith(".gguf") and "mmproj" in name.lower()


def _name_key(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(".", "")


def _is_head(name: str, kind: str, size: int | None, section: str) -> bool:
    """_find_mtp's per-folder filter: a .gguf head, not a projector, not the model itself,
    head-sized."""
    p = PurePosixPath(name)
    if not (kind == "file" and p.suffix.lower() == ".gguf"):
        return False
    if "mmproj" in name.lower() or not _looks_like_draft(name):
        return False
    # The model itself is never its own head: "-MTP-" builds carry the name too.
    return p.stem != section and size is not None and size <= HEAD_MAX_BYTES


def pick_file(inp: dict[str, Any]) -> Any:
    """One companion-file rule (see above)."""
    rule = inp["rule"]
    if rule == "projector":
        return _resolve_projector(inp)
    listing = inp["listing"]
    if rule == "mmproj_subdir":
        # A repo often ships several precisions of the same projector (BF16 / F16 / F32).
        # Prefer the SMALLEST: it is pinned to the main GPU and competes with the KV cache.
        if listing is None:
            return ""
        best: tuple[int, str] | None = None
        for name, kind, size in listing:
            if kind == "error":
                break       # is_file() raised: the scan stops, keeping what it found
            if kind == "file" and _is_mmproj(name):
                if size is None:
                    break   # stat raised: likewise
                if best is None or size < best[0]:
                    best = (size, name)
        return f"/models/{inp['subdir']}/{best[1]}" if best else ""
    if rule == "mmproj_flat":
        # Flat layout: the projector filename must carry the model's full stem, so
        # `foo-Q4_K_M.gguf` matches `foo-Q4_K_M-mmproj.gguf` but never a sibling model.
        stem_key = _name_key(inp["section"])
        if not stem_key or listing is None:
            return ""
        for name, kind, _size in listing:
            if kind == "error":
                return ""
            if kind == "file" and _is_mmproj(name) and stem_key in _name_key(name):
                return f"/models/{name}"
        return ""
    if rule == "mtp_folder":
        if listing is None:
            return ""
        cands = []
        for name, kind, size in listing:
            if kind == "error":
                return ""
            if _is_head(name, kind, size, inp["section"]):
                cands.append((name, size))
        if not cands:
            return ""
        return f"{inp['prefix']}{min(cands, key=lambda c: c[1])[0]}"
    if rule == "mtp_flat":
        stem_key = _name_key(inp["section"])
        if not stem_key or listing is None:
            return ""
        cands = []
        for name, kind, size in listing:
            if kind == "error":
                return ""
            if (kind == "file" and PurePosixPath(name).suffix.lower() == ".gguf" and _looks_like_draft(name)
                    and "mmproj" not in name.lower() and stem_key in _name_key(name)):
                cands.append((name, size))
        if not cands:
            return ""
        if any(size is None for _, size in cands):
            raise OSError("a draft head could not be sized")
        return f"/models/{min(cands, key=lambda c: c[1])[0]}"
    raise ValueError(f"unknown file rule {rule!r}")


def _resolve_projector(inp: dict[str, Any]) -> dict[str, Any]:
    """Size against the projector that will ACTUALLY be loaded: the saved one wins over the one
    found beside the model. `stat_gb` is its size as analyze() read it (0.0 when unreadable).
    Vision off drops it; a caller that knows a remote projector's size (the HF estimator)
    passes it as `override`."""
    mmproj_rel, mmproj_gb = "", 0.0
    if inp["files"]:
        mmproj_rel = inp["current"].strip() or inp["found"]
        if mmproj_rel:
            mmproj_gb = inp["stat_gb"]
    available = mmproj_rel
    if not inp["vision"]:
        mmproj_rel, mmproj_gb = "", 0.0
    override = inp["override"]
    if inp["vision"] and override is not None and override > 0:
        mmproj_gb = override
        mmproj_rel = mmproj_rel or "(remote projector)"
    return {"mmproj_rel": mmproj_rel, "mmproj_gb": mmproj_gb, "available": available}


# ---- baseline parsing (slice 5): the container's llama-server command -> ini keys ----

_SHORT_TO_KEY = {
    "-ngl": "ngl", "-fa": "flash-attn", "-ctk": "cache-type-k",
    "-ctv": "cache-type-v", "-np": "parallel", "-c": "ctx-size",
    "-b": "batch-size", "-ub": "ubatch-size", "-t": "threads",
    "-tb": "threads-batch", "-sm": "split-mode", "-mg": "main-gpu",
    "-fit": "fit", "-fitt": "fit-target", "-fitc": "fit-ctx",
    "-cmoe": "cpu-moe", "-ncmoe": "n-cpu-moe",
    "-kvo": "kv-offload",
}


def parse_baseline(cmd_args: list[str], known: Any) -> dict[str, str]:
    """Parse the container's llama-server command args into a {ini_key: value} dict. `known`
    holds the long option names that are ini keys (ini.ALL_KNOWN_KEYS)."""
    out: dict[str, str] = {}
    n = len(cmd_args)
    for i, a in enumerate(cmd_args):
        key: str | None = None
        if a in _SHORT_TO_KEY:
            key = _SHORT_TO_KEY[a]
        elif a.startswith("--"):
            k = a[2:]
            if k in known:
                key = k
        if not key:
            continue
        nxt = cmd_args[i + 1] if i + 1 < n else None
        if nxt is not None and not nxt.startswith("-"):
            out[key] = nxt
        else:
            out[key] = "true"
    return out


# ---- diff and presentation (slice 4): what analyze() reports beside the values ----

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
#
# The profile-owned keys are folded in by REFERENCE rather than restated, so the two can't drift.
# They are written from a profile's `knobs` dict rather than by a literal `values[...] =`, which
# is exactly why enumerating assignments by eye missed them — the _domain_gaps() guard caught all
# three on its first run.
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
}) | frozenset(SPEC_PROFILE_KEYS)

# Clearing this would break the section outright — a section with no model file is not a model.
# It stays out of the displaced list even when a recommendation happens not to set it.
_NEVER_CLEAR: frozenset[str] = frozenset({"model"})


def _domain_gaps(values: dict[str, str]) -> list[str]:
    """Keys a recommendation set that AUTOCONFIG_DOMAIN does not declare.

    Non-fatal on purpose: a missing declaration should be visible, not a 500 on a page the user
    is trying to read. Surfaced as a quirk so it gets noticed and fixed.
    """
    return sorted(set(values) - AUTOCONFIG_DOMAIN - _NEVER_CLEAR)


def _fmt_ctx(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        k = n // 1024
        return f"{k}K"
    return f"{n:,}"


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


# Kept even when the baseline already covers them (see present()).
_ESSENTIAL = ("model", "ctx-size", "jinja", "cpu-moe", "n-cpu-moe",
              "parallel", "cont-batching", "context-shift", "keep",
              "batch-size", "ubatch-size", "cache-reuse", "reasoning",
              "reasoning-preserve", "mmproj", "mmproj-offload", "image-max-tokens",
              # tensor-split is a correctness setting under expert offload, not a
              # tuning nicety: dropping it restores the even layer split that OOMs.
              "tensor-split", "split-mode")

# One JSON request (`present`); the answer is {minimal, redundant, quirks, unavailable,
# current_preset, current_diff, displaced, warnings} (minimal and redundant as [[key, value]...]):
#
#   values        Python's values, [[key, value]...] in order
#   current       the saved section (strings), or None
#   recommended   whether a backend was recommended; rec_name: its name, raw
#   rec_backend   the first backend of that name: its "baseline" and "gpu_count" where present
#   arch          summary["arch"], raw; chat_template: whether summary["chat_template"] is set
#   features      summary["chat_template_features"], raw
#   n_sessions    1..8; rec_ctx: the per-session context recommended
#   has_mmproj, mmproj_vram_gb, mmproj_gb   the projector and its reservation
#   rope          summary["model"], raw (rope ownership, the declared scaling type)
#   native_ctx, layers   as prepare() read them; experts: summary["model"]["expert_count"], raw
#   offload       the size plan's [kind, n_cpu_moe]; presets: [{key, ctx, offload_kind, ngl, n_cpu_moe}...]
#   model_rel     the model path; general: summary["general"], raw (its params_raw)


def _lower_eq(a: Any, b: Any) -> bool:
    return str(a).lower() == str(b).lower()


def present(inp: dict[str, Any]) -> dict[str, Any]:
    """analyze()'s report beside the values, in analyze()'s order: baseline redundancy, the
    quirks, the unavailable knobs, the saved preset, the diff, the displaced keys, the warnings."""
    values = {k: v for k, v in inp["values"]}
    current_section = inp["current"]
    recommended = inp["recommended"]
    rec_backend = inp["rec_backend"]
    n_sessions = inp["n_sessions"]
    rec_ctx = inp["rec_ctx"]
    m = inp["rope"]
    native_ctx = inp["native_ctx"]
    layers = inp["layers"]
    experts = inp["experts"]
    features = inp["features"] or {}

    # baseline-redundant: any key that matches the recommended backend's baseline
    baseline_redundant: dict[str, str] = {}
    minimal = dict(values)
    if recommended:
        base = (rec_backend or {}).get("baseline") or {}
        for k, v in list(values.items()):
            bv = base.get(k)
            if bv is not None and _lower_eq(bv, v):
                baseline_redundant[k] = v
                minimal.pop(k, None)

    # ensure minimal keeps the essential differentiators even if redundant on paper
    for essential in _ESSENTIAL:
        if essential in values and essential not in minimal:
            minimal[essential] = values[essential]

    quirks: list[str] = []

    # Baseline CONFLICTS — the container's own CLI args win over anything in models.ini,
    # so a preset value that disagrees with the compose command is silently discarded.
    # This bit us hard: `-np 1` in the compose command overrode `parallel = 2` in the ini,
    # so multi-session serving never actually ran, and `-ctv q8_0` overrode `cache-type-v`.
    # Surface it loudly instead of letting the preset look like it took effect.
    if recommended:
        _base2 = (rec_backend or {}).get("baseline") or {}
        conflicts = [
            f"`{k}`: preset wants {v}, container forces {_base2[k]}"
            for k, v in values.items()
            if k in _base2 and not _lower_eq(_base2[k], v)
        ]
        if conflicts:
            quirks.append(
                "CONFLICT — the container's CLI args override models.ini, so these preset values "
                "will NOT take effect: " + "; ".join(conflicts) + ". "
                f"Fix by removing those flags from the `{inp['rec_name']}` command in your compose file "
                "so per-model presets can control them. Keep only router-level args there "
                "(--models-dir, --models-preset, --host, --port, --models-max)."
            )

    if (inp["arch"] or "").lower().startswith("gemma"):
        quirks.append("Gemma sliding-window attention: the ctx→KV math above assumes swa-full=false (default). "
                      "Enabling swa-full multiplies full-attn KV ~5× and will OOM.")
    # (MoE hint is emitted later, tailored to whichever offload the recommendation actually applies)
    if not inp["chat_template"]:
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
        rec_gpu_count = int((rec_backend or {}).get("gpu_count", 1))
        if rec_gpu_count > 1:
            quirks.append(
                f"Multi-GPU backend ({rec_gpu_count} cards, layer-split): reserved "
                f"{_RESERVE_PER_GPU * rec_gpu_count:.1f} GB total for CUDA runtime "
                f"(0.5 GB × {rec_gpu_count}) and applied a {int((_MODEL_OVERHEAD_SPLIT - 1) * 100)}% model-VRAM "
                f"multiplier for cross-card handoffs. Real cap will be a bit lower than pure sum-of-VRAMs."
            )

    # Multimodal VRAM accounting
    if inp["has_mmproj"]:
        quirks.append(
            f"Multimodal model (mmproj companion present — vision, audio, or other modality). "
            f"Reserved {inp['mmproj_vram_gb']:.2f} GB for the projector ({inp['mmproj_gb']:.2f} GB weights + "
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
    rope_owned = rope_owned_by_gguf(m)
    if values.get("rope-scaling") == "linear" and not rope_owned and native_ctx > 0:
        scale = values.get("rope-scale", "?")
        quirks.append(f"Extended ctx from native {_fmt_ctx(native_ctx)} to {_fmt_ctx(rec_ctx)} "
                      f"via `rope-scaling=linear, rope-scale={scale}`. Linear scaling degrades quality gracefully up "
                      f"to ~2× native; beyond that outputs get progressively worse. Drop ctx-size in the form to back off.")

    # unavailable knobs
    rope_type = m.get("rope_scaling_type")
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
            off_kind, n_cm = inp["offload"]
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
    presets = inp["presets"]
    current_preset = ""
    if current_section and presets:
        _cur = {k: str(v) for k, v in current_section.items()}
        try:
            _cur_ctx = int(_cur.get("ctx-size") or 0)
        except ValueError:
            _cur_ctx = 0
        _cur_ngl = (_cur.get("ngl") or "").strip()
        _cur_ncm = (_cur.get("n-cpu-moe") or "").strip()
        for _p in presets:
            if _cur_ctx != _p["ctx"] * n_sessions:
                continue
            if _p["offload_kind"] == "ngl":
                ok = _cur_ngl == str(_p["ngl"])
            elif _p["offload_kind"] == "n-cpu-moe":
                ok = _cur_ncm == str(_p["n_cpu_moe"])
            elif _p["offload_kind"] == "cpu-moe":
                ok = (_cur.get("cpu-moe") or "").lower() in ("true", "on", "1")
            else:
                ok = _cur_ngl in ("", "999") and not _cur_ncm
            if ok:
                current_preset = _p["key"]
                break

    # diff vs current section — only report on keys autoconfig actually opinions on.
    # Anything the user set that we don't touch (mmproj, chat-template-file, lora, override-*, etc.)
    # is left alone: not reported as a diff, and Fill/Fill minimal doesn't overwrite it.
    #
    # Order (#1152): first every key the recommendation sets whose value differs, in the order of
    # `values`; then every key it supersedes, sorted by key (code point order, as `displaced`).
    # The removals used to follow set iteration order, which changes with PYTHONHASHSEED, so the
    # same request listed them differently from one process to the next.
    current_diff: list[str] = []
    # Everything autoconfig opinions on, minus what this recommendation actually set: that is
    # exactly the set it wants gone. Declared once in AUTOCONFIG_DOMAIN rather than remembered
    # per-branch, so a key can no longer be quietly left behind.
    _displaces = sorted(AUTOCONFIG_DOMAIN - _NEVER_CLEAR)
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
    displaced = [k for k in _displaces if k not in values]

    # A key we set but never declared would escape both the diff and Fill's clearing, which is
    # how the "Save does nothing" bug worked. Make it visible instead of silent.
    _gaps = _domain_gaps(values)
    if _gaps:
        quirks.append(
            "Autoconfig set %s, which AUTOCONFIG_DOMAIN does not declare. Fill will not clear "
            "%s on a later run, so a stale value could survive. Add them to the domain."
            % (", ".join("`%s`" % g for g in _gaps), "them" if len(_gaps) > 1 else "it")
        )

    warnings = quality_warnings(model_rel=inp["model_rel"], params=(inp["general"] or {}).get("params_raw"),
                                recommended_ctx=rec_ctx, native_ctx=native_ctx)
    return {"minimal": [[k, v] for k, v in minimal.items()],
            "redundant": [[k, v] for k, v in baseline_redundant.items()],
            "quirks": quirks, "unavailable": unavailable, "current_preset": current_preset,
            "current_diff": current_diff, "displaced": displaced, "warnings": warnings}


def check_reference(parts: dict[str, Any]) -> dict[str, Any]:
    """What `model-autoconfig check` answers for `parts` ({prep?, size?, values?, spec?, files?,
    present?, baseline?}), from the Python reference: prep as prepare_all, size as size_plan,
    values as ordered [key, value], spec as resolve_spec, files as pick_file of each rule call,
    present as present, baseline as parse_baseline of each {args, known} (ordered pairs)."""
    out: dict[str, Any] = {}
    if parts.get("prep") is not None:
        out["prep"] = prepare_all(parts["prep"])
    if parts.get("size") is not None:
        out["size"] = size_plan(parts["size"])
    if parts.get("values") is not None:
        out["values"] = [[k, v] for k, v in assemble_values(parts["values"]).items()]
    if parts.get("spec") is not None:
        out["spec"] = resolve_spec(parts["spec"])
    if parts.get("files") is not None:
        out["files"] = [pick_file(c) for c in parts["files"]]
    if parts.get("present") is not None:
        out["present"] = present(parts["present"])
    if parts.get("baseline") is not None:
        out["baseline"] = [[[k, v] for k, v in parse_baseline(c["args"], set(c["known"])).items()]
                           for c in parts["baseline"]]
    return out


# ---- MODEL_AUTOCONFIG: python (default) | rust ----
#
# rust runs the same steps - input prep, the size core, values assembly, and (slices 3-6) the
# speculative-decoding resolution, the companion-file rules, the report beside the values and
# the baseline parse - in the bounded Rust leaf from sbstndalton/noevia-rs (`model-autoconfig
# check`, one process per analyze(), baked into the image at the Dockerfile's NOEVIA_RS_REF)
# BESIDE the Python, which stays authoritative: what analyze() returns is always Python's, and
# only when the Rust answer confirms it. Input prep must agree exactly. The size plan and the
# values must agree exactly, or Python's must be the conservative one: the same backend,
# placement mode and every other value, and no larger context, GPU layer count, prompt cache,
# batch, ubatch or image-token bound. The spec part likewise, with the draft limits
# (SPEC_LIMITS: draft depth, draft minimum, the head's GPU layers) as its only bounds; the
# files, present and baseline parts must agree exactly (every message, key and order). Anything
# else, and every Rust fault (missing binary, timeout, refusal, malformed output), refuses the
# recommendation with AutoconfigCoreError. When Python's own input prep
# refuses (an implausible block count, no backend, ...), that refusal is returned either way and
# a Rust disagreement is only logged. DaServer's iGPU memory is system RAM with no swap (#697),
# so a disagreement never buys a bigger setting.

IMPLS = ("python", "rust")
# Where the model-manager image installs the binary; MODEL_AUTOCONFIG_BIN overrides it (tests).
DEFAULT_BINARY = "/usr/local/bin/model-autoconfig"
CORE_TIMEOUT_S = 10.0
CORE_STDIN_CAP = 4 * 1024 * 1024
CORE_STDOUT_CAP = 16 * 1024 * 1024

_log = logging.getLogger(__name__)
_LOGGED: set[str] = set()
_LOGGED_LOCK = threading.Lock()


class AutoconfigCoreError(RuntimeError):
    """MODEL_AUTOCONFIG=rust could not confirm the Python plan (fails closed)."""


_LOGGED_MAX = 1024


def _log_once(reason: str, message: str) -> None:
    with _LOGGED_LOCK:
        if reason in _LOGGED:
            return
        if len(_LOGGED) >= _LOGGED_MAX:   # bounded: a long-lived service sees many request shapes
            _LOGGED.clear()
        _LOGGED.add(reason)
    _log.warning(message)


def _setting(name: str, env: str, default: str) -> str:
    try:
        from .config import settings
        return str(getattr(settings, name, default) or "")
    except ImportError:
        return os.environ.get(env, default)


def impl_choice() -> str:
    """The configured implementation: "python" (default) or "rust". Anything else is python."""
    value = _setting("model_autoconfig", "MODEL_AUTOCONFIG", "python").strip().lower() or "python"
    if value not in IMPLS:
        _log_once("invalid_setting",
                  f"MODEL_AUTOCONFIG={value!r} is not one of {', '.join(IMPLS)}; using python")
        return "python"
    return value


def _rust_binary() -> str | None:
    configured = (_setting("model_autoconfig_bin", "MODEL_AUTOCONFIG_BIN", DEFAULT_BINARY).strip()
                  or DEFAULT_BINARY)
    if os.sep in configured:
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None
    return shutil.which(configured)


def _shape_key(req: dict[str, Any]) -> str:
    """A short hash of the request: one log line per request shape, not per request."""
    try:
        return hashlib.sha256(canonical(req).encode()).hexdigest()[:12]
    except (TypeError, ValueError):
        return "unhashable"


def _fail(reason: str, detail: str, *, shape: str = "", model: str = "") -> AutoconfigCoreError:
    which = f" [model {model[:200]!r}, request {shape}]" if shape else ""
    _log_once(f"rust:{reason}:{shape}" if shape else f"rust:{reason}",
              f"model-autoconfig failed ({reason}: {detail}){which}; refusing the recommendation")
    return AutoconfigCoreError(f"model-autoconfig {reason}")


def _run_bounded(proc: "subprocess.Popen[bytes]", payload: bytes) -> tuple[str, bytes, bytes]:
    """Feed `payload`, read stdout up to CORE_STDOUT_CAP + 1 bytes and stderr's first 4 KiB, all
    within CORE_TIMEOUT_S; the child is killed (and reaped) the moment the cap is passed or the
    time is up, so a runaway binary can never fill memory or outlive the request.
    Returns ("ok" | "too_large" | "timeout", stdout, stderr)."""
    out, err = bytearray(), bytearray()
    over = threading.Event()

    def feed() -> None:
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (OSError, ValueError):
            pass   # the child went away early; its exit status says why

    def read_out() -> None:
        fd = proc.stdout.fileno()
        while len(out) <= CORE_STDOUT_CAP:
            try:
                chunk = os.read(fd, min(65536, CORE_STDOUT_CAP + 1 - len(out)))
            except OSError:
                return
            if not chunk:
                return
            out.extend(chunk)
        over.set()

    def read_err() -> None:
        fd = proc.stderr.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            if len(err) < 4096:
                err.extend(chunk[:4096 - len(err)])

    threads = [threading.Thread(target=f, daemon=True) for f in (feed, read_out, read_err)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + CORE_TIMEOUT_S
    status = "ok"
    try:
        while True:
            if over.is_set():
                status = "too_large"
                break
            if not any(t.is_alive() for t in threads[1:]) and proc.poll() is not None:
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                break
            over.wait(0.01)
    finally:
        if status != "ok" or proc.poll() is None:
            proc.kill()
        proc.wait()
        for t in threads:
            t.join(1.0)
        for f in (proc.stdin, proc.stdout, proc.stderr):
            try:
                f.close()
            except (OSError, ValueError):
                pass
    return status, bytes(out), bytes(err)


def check_rust(parts: dict[str, Any]) -> dict[str, Any]:
    """`model-autoconfig check` on `parts` ({prep?, size?, values?}): the Rust answer, unchecked."""
    binary = _rust_binary()
    if binary is None:
        raise _fail("missing_binary", "model-autoconfig not found or not executable")
    try:
        payload = json.dumps(parts, ensure_ascii=True).encode()
    except (TypeError, ValueError) as e:
        raise _fail("unencodable_input", type(e).__name__) from None
    if len(payload) > CORE_STDIN_CAP:
        raise _fail("input_too_large", f"{len(payload)} bytes")
    try:
        proc = subprocess.Popen([binary, "check"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                # A minimal environment: the child needs nothing of ours.
                                env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")})
    except OSError as e:
        raise _fail("spawn", type(e).__name__) from None
    status, stdout, stderr = _run_bounded(proc, payload)
    if status == "timeout":
        raise _fail("timeout", f"no result within {CORE_TIMEOUT_S:g} s") from None
    if status == "too_large":
        raise _fail("output_too_large", f"more than {CORE_STDOUT_CAP} bytes")
    if proc.returncode != 0:
        detail = stderr[:300].decode("utf-8", "replace").strip()
        raise _fail("rejected", f"exit {proc.returncode}: {detail}")
    try:
        out = json.loads(stdout)
    except ValueError:
        raise _fail("malformed_output", "not JSON") from None
    if not isinstance(out, dict) or set(out) != {k for k, v in parts.items() if v is not None}:
        raise _fail("malformed_output", "not an object with the parts asked for")
    return out


def canonical(plan: Any) -> str:
    """The comparison form: key order and int/float types both count, NaN equals NaN."""
    return json.dumps(plan, sort_keys=True, ensure_ascii=True, allow_nan=True)


def _int_or(v: Any, default: int | None) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else default


def _written(plan: dict[str, Any], n_sessions: int) -> tuple | None:
    """What a plan makes analyze() write that costs memory, or None if it is malformed:
    (backend, fit, ctx-size, gpu layers, cache-ram). No ngl is 999 (every layer); no cache-ram
    is llama-server's own default, counted as larger than any value autoconfig writes."""
    rec = plan.get("recommended")
    if rec is not None and _int_or(rec, None) is None:
        return None
    ctx = _int_or(plan.get("ctx") if plan.get("sized") else plan.get("initial_ctx"), None)
    ngl, cache, fit = plan.get("ngl"), plan.get("cache_ram"), plan.get("fit")
    if (ctx is None or not isinstance(fit, bool) or (ngl is not None and _int_or(ngl, None) is None)
            or (cache is not None and _int_or(cache, None) is None)):
        return None
    return (rec, fit, max(ctx, 0) * n_sessions, 999 if ngl is None else ngl,
            math.inf if cache is None else cache)


def python_is_conservative(py: dict[str, Any], rs: dict[str, Any], n_sessions: int) -> bool:
    """True when Python's plan writes nothing larger than Rust's: the same backend and placement
    mode, and no larger context, GPU layer count or prompt cache."""
    p, r = _written(py, n_sessions), _written(rs, n_sessions)
    if p is None or r is None or p[:2] != r[:2]:
        return False
    return all(pv <= rv for pv, rv in zip(p[2:], r[2:]))


# The values a recommendation writes that cost memory, and what their absence means: llama-server's
# own default (batch 2048, ubatch 512) or, for the rest, no bound at all - counted as larger than
# any value autoconfig writes.
VALUE_LIMITS: dict[str, float] = {
    "ctx-size": math.inf, "ngl": math.inf, "cache-ram": math.inf, "image-max-tokens": math.inf,
    "batch-size": 2048, "ubatch-size": 512,
}


def _as_values(v: Any) -> dict[str, str] | None:
    """The Rust values ([[key, value]...], keys unique, all strings) as a dict, else None."""
    if not isinstance(v, list):
        return None
    out: dict[str, str] = {}
    for pair in v:
        if (not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair)
                or pair[0] in out):
            return None
        out[pair[0]] = pair[1]
    return out


def _value_int(v: str | None, key: str) -> float | None:
    """A written number as an int (its absence as VALUE_LIMITS says), or None if malformed."""
    if v is None:
        return VALUE_LIMITS[key]
    if len(v) > 40 or not (v.isascii() and (v.isdigit() or (v[:1] == "-" and v[1:].isdigit()))):
        return None
    return int(v)


def values_are_conservative(py: dict[str, str], rs: dict[str, str]) -> bool:
    """True when Python's values write nothing larger than Rust's: every key outside VALUE_LIMITS
    present in both or neither, with the same value and the shared keys in the same order, and
    no VALUE_LIMITS key larger in Python's."""
    for k in set(py) | set(rs):
        if k in VALUE_LIMITS:
            p, r = _value_int(py.get(k), k), _value_int(rs.get(k), k)
            if p is None or r is None or p > r:
                return False
        elif py.get(k) != rs.get(k):
            return False
    return [k for k in py if k in rs] == [k for k in rs if k in py]


def size_plan_accepted(py: dict[str, Any], rs: Any, n_sessions: int) -> str:
    """"same", "conservative" or "" (refuse) for a Rust size plan against Python's."""
    if not isinstance(rs, dict):
        return ""
    if canonical(rs) == canonical(py):
        return "same"
    return "conservative" if python_is_conservative(py, rs, n_sessions) else ""


# The speculative-decoding values that bound work or memory: how deep the drafter guesses and how
# many of the head's layers go on the GPU. Python's may be smaller than Rust's; anything else in
# the spec part must agree exactly.
SPEC_LIMITS: tuple[str, ...] = ("spec-draft-n-max", "spec-draft-n-min", "spec-draft-ngl")


def _digits(v: str) -> int | None:
    return int(v) if len(v) <= 40 and v.isascii() and v.isdigit() else None


def spec_is_conservative(py: dict[str, Any], rs: Any) -> bool:
    """True when Python's spec answer writes nothing larger than Rust's: every field but the
    values the same, the same value keys in the same order, every value the same except a
    SPEC_LIMITS one that is a smaller (or equal) plain number in Python's."""
    if not isinstance(rs, dict) or set(rs) != set(py):
        return False
    if any(canonical(rs[k]) != canonical(py[k]) for k in py if k != "values"):
        return False
    pv, rv = _as_values(py["values"]), _as_values(rs["values"])
    if pv is None or rv is None or list(pv) != list(rv):
        return False
    for k, p in pv.items():
        r = rv[k]
        if p == r:
            continue
        a, b = _digits(p), _digits(r)
        if k not in SPEC_LIMITS or a is None or b is None or a > b:
            return False
    return True


# The parts added in slices 3-6, in the order they are checked.
EXTRA_PARTS = ("spec", "files", "present", "baseline")


def confirm(*, prep_in: dict[str, Any], prep: dict[str, Any], size_req: dict[str, Any],
            plan: dict[str, Any], values_in: dict[str, Any], values: dict[str, str],
            extra_in: dict[str, Any] | None = None, extra: dict[str, Any] | None = None,
            model: str = "") -> None:
    """Under MODEL_AUTOCONFIG=rust, confirm Python's prep, size plan and values - and the parts
    in `extra_in` (spec, files, present, baseline: requests, None to skip one), whose Python
    answers are in `extra` - with the Rust port (one process); raise AutoconfigCoreError unless
    every part is confirmed (see above). `model` only labels the log lines; it is never part of
    the request."""
    if impl_choice() != "rust":
        return
    asked = {k: v for k, v in (extra_in or {}).items() if k in EXTRA_PARTS and v is not None}
    out = check_rust({"prep": prep_in, "size": size_req, "values": values_in, **asked})
    shape = _shape_key(size_req)
    if canonical(out["prep"]) != canonical(prep):
        raise _fail("mismatch", "the Rust input prep disagrees", shape=shape, model=model)
    verdict = size_plan_accepted(plan, out["size"], size_req["n_sessions"])
    if not verdict:
        raise _fail("mismatch", "the Rust plan disagrees and the Python plan is not the smaller one",
                    shape=shape, model=model)
    rs_values = _as_values(out["values"])
    if rs_values is None:
        raise _fail("malformed_output", "values are not [[key, value]...]", shape=shape, model=model)
    if list(rs_values.items()) != list(values.items()):
        if not values_are_conservative(values, rs_values):
            raise _fail("mismatch", "the Rust values disagree and Python's are not the smaller ones",
                        shape=shape, model=model)
        verdict = "conservative"
    for part in EXTRA_PARTS:
        if part not in asked:
            continue
        py, rs = (extra or {}).get(part), out[part]
        if canonical(rs) == canonical(py):
            continue
        if part == "spec" and spec_is_conservative(py, rs):
            verdict = "conservative"
            continue
        raise _fail("mismatch", f"the Rust {part} part disagrees", shape=shape, model=model)
    if verdict == "conservative":
        _log_once(f"mismatch:conservative:{shape}",
                  f"model-autoconfig disagreed with the Python autoconfig for model {model[:200]!r} "
                  f"(request {shape}); using the Python recommendation, which is the more conservative one")


def confirm_refusal(*, prep_in: dict[str, Any], prep: dict[str, Any], model: str = "") -> None:
    """Under MODEL_AUTOCONFIG=rust, compare a Python input-prep refusal with the Rust one. The
    Python refusal stands either way (no recommendation is the most conservative answer), so a
    disagreement or a Rust fault is only logged; this never raises."""
    if impl_choice() != "rust":
        return
    shape = _shape_key(prep_in)
    try:
        out = check_rust({"prep": prep_in})
    except AutoconfigCoreError:
        return    # _fail logged it
    if canonical(out["prep"]) != canonical(prep):
        _log_once(f"mismatch:refusal:{shape}",
                  f"model-autoconfig disagreed with the Python input prep for model {model[:200]!r} "
                  f"(request {shape}), which refused ({prep.get('refuse')}); the refusal stands")
