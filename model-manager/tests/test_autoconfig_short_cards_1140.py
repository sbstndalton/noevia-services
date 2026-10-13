"""#1140: a backend that reports FEWER per-card VRAM entries than gpu_count must not 500 analyze().

_split_feasible used to index caps[card] for every card of gpu_count, so a short `cards` list raised
IndexError (HTTP 500). A short list now takes the same even-division fallback an empty list already
takes, and a card with no reported capacity holds nothing. Synthetic inputs only: nothing loads a
model or starts llama.cpp."""
from __future__ import annotations

import dataclasses

import pytest

from app import autoconfig, autoconfig_core, config


def _summary(layers=32, native=131072):
    m = {"block_count": layers, "attention_head_count": 32, "embedding_length": 4096,
         "attention_head_count_kv": 8, "context_length": native}
    return {"arch": "llama", "model": m, "chat_template": "x"}


def _backend(gpu_count, cards, vram=36.0):
    b = {"name": "llama-cuda", "vendor": "cuda", "vram_gb": vram, "gpu_count": gpu_count,
         "host_ram_gb": 64.0, "baseline": {}}
    if cards is not None:
        b["card_vram_gb"] = cards
    return b


def _analyze(backends, file_size=30 * 2**30):
    return autoconfig.analyze(summary=_summary(), file_size=file_size, backends=backends, n_sessions=1)


@pytest.fixture(autouse=True)
def _python_mode(monkeypatch):
    autoconfig_core._LOGGED.clear()
    monkeypatch.setattr(config.settings, "model_autoconfig", "python")
    yield
    autoconfig_core._LOGGED.clear()


@pytest.mark.parametrize("cards", [[12.0], [12.0, 12.0]], ids=["one-of-three", "two-of-three"])
def test_short_card_list_does_not_raise(cards):
    rec = _analyze([_backend(3, cards)])
    assert isinstance(rec.recommended_ctx, int) and rec.recommended_ctx >= 0


@pytest.mark.parametrize("cards", [[12.0], [12.0, 12.0]], ids=["one-of-three", "two-of-three"])
def test_short_card_list_takes_the_even_division_fallback(cards):
    """A short list is treated exactly like an absent one: the pool divided evenly across the GPUs."""
    short = dataclasses.asdict(_analyze([_backend(3, cards)]))
    absent = dataclasses.asdict(_analyze([_backend(3, None)]))
    assert short == absent


def test_full_card_list_is_still_used_per_card():
    """Matching lengths keep the per-card capacities: an uneven split is not replaced by the even one."""
    uneven = dataclasses.asdict(_analyze([_backend(3, [4.0, 14.0, 14.0], vram=32.0)]))
    even = dataclasses.asdict(_analyze([_backend(3, None, vram=32.0)]))
    assert uneven != even


def test_split_feasible_missing_card_holds_nothing():
    """A card with no reported capacity cannot take a layer, so the partition is refused, not raised."""
    assert autoconfig_core._split_feasible(
        layers=8, n_cpu_moe=0, attention_gb=2.0, expert_per_layer_gb=0.0, kv_gb=0.0,
        gpu_count=3, pinned_gb=0.0, caps=[1.0]) is False
    assert autoconfig_core._split_feasible(
        layers=8, n_cpu_moe=0, attention_gb=2.0, expert_per_layer_gb=0.0, kv_gb=0.0,
        gpu_count=3, pinned_gb=0.0, caps=[1.0, 1.0, 1.0]) is True
