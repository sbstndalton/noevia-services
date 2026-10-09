"""#1152: analyze()'s current_diff is deterministic. It used to list the superseded keys in set
iteration order, which follows PYTHONHASHSEED, so the same request gave a differently ordered
diff in each process. The documented order: changed keys in the order of `values`, then the
superseded ones sorted by key. Synthetic sections only; nothing loads a model."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app import autoconfig

MM = Path(__file__).resolve().parents[1]

# Many superseded keys at once, so a hash-ordered walk is all but certain to differ across seeds.
SECTION = {"ngl": "30", "tensor-split": "1,1", "n-cpu-moe": "6", "cpu-moe": "on", "keep": "64",
           "rope-scaling": "yarn", "rope-scale": "4", "image-max-tokens": "512", "mmproj-offload": "off",
           "reasoning": "on", "reasoning-format": "none", "cont-batching": "on", "context-shift": "off",
           "chat-template-kwargs": "{}", "split-mode": "row", "fit": "off", "ctx-size": "999"}


def _analyze() -> list[str]:
    summary = {"arch": "llama", "chat_template": "x",
               "model": {"block_count": 32, "attention_head_count": 32, "embedding_length": 4096,
                         "attention_head_count_kv": 8, "context_length": 131072}}
    backends = [{"name": "llama-cuda", "vendor": "cuda", "vram_gb": 24.0, "gpu_count": 1,
                 "host_ram_gb": 64.0, "baseline": {}}]
    rec = autoconfig.analyze(summary=summary, file_size=int(4.7 * 2**30), backends=backends,
                             current_section=dict(SECTION), section_name="", n_sessions=1)
    return rec.current_diff


def test_superseded_keys_follow_values_then_sorted_order():
    diff = _analyze()
    superseded = [line.split(":", 1)[0] for line in diff if line.endswith("→ unset (superseded)")]
    changed = [line.split(":", 1)[0] for line in diff if not line.endswith("→ unset (superseded)")]
    assert len(superseded) >= 10
    assert superseded == sorted(superseded)
    # every changed line precedes every superseded one
    assert diff[:len(changed)] == [line for line in diff if not line.endswith("(superseded)")]


def test_current_diff_is_the_same_under_every_hash_seed():
    code = ("import json, sys; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "from test_autoconfig_diff_order import _analyze; print(json.dumps(_analyze()))"
            % (str(MM), str(MM / "tests")))
    outs = set()
    for seed in ("0", "1", "2", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, timeout=120, check=True)
        outs.add(proc.stdout)
    assert len(outs) == 1, "current_diff order depends on PYTHONHASHSEED"
    assert json.loads(outs.pop()) == _analyze()
