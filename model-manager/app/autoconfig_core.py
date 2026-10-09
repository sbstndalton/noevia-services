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
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
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


# ---- MODEL_AUTOCONFIG: python (default) | rust ----
#
# rust runs the same size core in the bounded Rust leaf from sbstndalton/noevia-rs
# (`model-autoconfig size`, baked into the image at the Dockerfile's NOEVIA_RS_REF) BESIDE the
# Python, which stays authoritative: the plan used is always Python's, and only when the Rust
# answer agrees with it exactly. On a mismatch the Python plan is still used if it is the
# conservative one (never a larger context, prompt cache or GPU layer count, and the same
# backend and placement mode); otherwise, and on any Rust fault (missing binary, timeout,
# refusal, malformed output), the recommendation is refused with AutoconfigCoreError. DaServer's
# iGPU memory is system RAM with no swap (#697), so a disagreement never buys a bigger setting.

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


def size_plan_rust(req: dict[str, Any]) -> dict[str, Any]:
    binary = _rust_binary()
    if binary is None:
        raise _fail("missing_binary", "model-autoconfig not found or not executable")
    try:
        payload = json.dumps(req, ensure_ascii=True).encode()
    except (TypeError, ValueError) as e:
        raise _fail("unencodable_input", type(e).__name__) from None
    if len(payload) > CORE_STDIN_CAP:
        raise _fail("input_too_large", f"{len(payload)} bytes")
    try:
        proc = subprocess.Popen([binary, "size"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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
    if not isinstance(out, dict):
        raise _fail("malformed_output", "not an object")
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


def plan_sizes(req: dict[str, Any], model: str = "") -> dict[str, Any]:
    """The size plan by the configured implementation; always Python's answer (see above).
    `model` only labels the log lines; it is never part of the request."""
    py = size_plan(req)
    if impl_choice() != "rust":
        return py
    rs = size_plan_rust(req)
    if canonical(rs) == canonical(py):
        return py
    shape = _shape_key(req)
    if python_is_conservative(py, rs, req["n_sessions"]):
        _log_once(f"mismatch:conservative:{shape}",
                  f"model-autoconfig disagreed with the Python size core for model {model[:200]!r} "
                  f"(request {shape}); using the Python plan, which is the more conservative one")
        return py
    raise _fail("mismatch", "the Rust plan disagrees and the Python plan is not the smaller one",
                shape=shape, model=model)
