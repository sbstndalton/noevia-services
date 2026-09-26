"""Search judging: fit first, trusted publishers, age-weighted popularity (user's rules, 2026-09-17)."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT
from app import discover
from app.main import app

RECENT = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat().replace("+00:00", "Z")
OLD = (datetime.now(timezone.utc) - timedelta(days=700)).isoformat().replace("+00:00", "Z")


def cand(id, owner, files, **kw):
    return discover.Candidate(id=id, owner=owner, files=[{"path": p, "size": int(g * 1e9)} for p, g in files], **kw)


def test_params_and_moe_from_names():
    assert discover.params_from("unsloth/Qwen3.6-35B-A3B-GGUF") == (35.0, 3.0)
    assert discover.params_from("ibm-granite/granite-4.2-30b-GGUF") == (30.0, None)
    assert discover.params_from("some/model-GGUF") == (None, None)
    assert discover.is_moe(discover.Candidate(id="x/y-30B-A3B", owner="x"), 3.0) is True
    assert discover.is_moe(discover.Candidate(id="x/dense-27B", owner="x", tags=["text-generation"]), None) is False
    assert discover.is_moe(discover.Candidate(id="x/y", owner="x", tags=["moe"]), None) is True


def test_a_repo_is_suitable_only_when_a_q4_file_fits():
    budget = 13.5
    ok = discover.judge(cand("unsloth/M-20B-GGUF", "unsloth", [("M-20B-Q4_K_M.gguf", 11.6), ("M-20B-Q8_0.gguf", 21.0)]), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS)
    assert ok["suitable"] and ok["best"]["quant"] == "Q4_K_M" and ok["trusted"]
    too_big = discover.judge(cand("unsloth/M-70B-GGUF", "unsloth", [("M-70B-Q4_K_M.gguf", 40.0)]), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS)
    assert not too_big["suitable"] and "does not fit" in too_big["reasons"][0]
    sub_q4 = discover.judge(cand("unsloth/M-35B-A3B-GGUF", "unsloth", [("M-35B-A3B-UD-IQ3_XXS.gguf", 13.2)]), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS)
    assert not sub_q4["suitable"] and "below Q4" in sub_q4["reasons"][0]
    # Over 100B, a sub-Q4 quant is expected and only size decides.
    huge = discover.judge(cand("unsloth/M-400B-GGUF", "unsloth", [("M-400B-UD-IQ2_M.gguf", 12.0)]), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS)
    assert huge["suitable"]


def test_shards_are_one_option_and_companions_are_not_offered():
    judged = discover.judge(cand("unsloth/M-30B-GGUF", "unsloth", [
        ("Q4/M-30B-Q4_K_M-00001-of-00002.gguf", 6.0), ("Q4/M-30B-Q4_K_M-00002-of-00002.gguf", 6.0),
        ("mmproj-BF16.gguf", 0.9), ("MTP/mtp-M-30B-Q4_0.gguf", 1.3),
        ("eagle3-M-30B-BF16.gguf", 1.6)]), budget_gb=13.5, trusted=discover.TRUSTED_QUANTISERS)
    assert [o["path"] for o in judged["options"]] == ["Q4/M-30B-Q4_K_M.gguf"]
    assert judged["options"][0]["gb"] == 12.0 and judged["options"][0]["shards"] == 2
    assert judged["vision"] is True


def test_ranking_puts_fit_first_then_trust_then_recent_popularity():
    budget = 13.5
    rows = [
        discover.judge(cand("nobody/huge-70B-GGUF", "nobody", [("huge-70B-Q4_K_M.gguf", 40.0)], downloads=5_000_000, last_modified=RECENT), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
        discover.judge(cand("random/fits-8B-GGUF", "random", [("fits-8B-Q4_K_M.gguf", 5.0)], downloads=400_000, last_modified=RECENT), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
        discover.judge(cand("unsloth/old-8B-GGUF", "unsloth", [("old-8B-Q4_K_M.gguf", 5.0)], downloads=9_000_000, last_modified=OLD), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
        discover.judge(cand("unsloth/new-9B-GGUF", "unsloth", [("new-9B-Q4_K_M.gguf", 6.0)], downloads=300_000, last_modified=RECENT), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
    ]
    order = [r["id"] for r in discover.rank(rows)]
    assert order[0] == "unsloth/new-9B-GGUF", "recent trusted and fitting wins over an older favourite"
    assert order[1] == "unsloth/old-8B-GGUF"
    assert order[2] == "random/fits-8B-GGUF", "fits but untrusted still beats a model that cannot run"
    assert order[-1] == "nobody/huge-70B-GGUF"


def test_filters_hide_untrusted_and_unsuitable_by_default_and_can_be_widened():
    budget = 13.5
    rows = [
        discover.judge(cand("unsloth/a-8B-GGUF", "unsloth", [("a-8B-Q4_K_M.gguf", 5.0)], last_modified=RECENT), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
        discover.judge(cand("someone/b-8B-GGUF", "someone", [("b-8B-Q4_K_M.gguf", 5.0)], last_modified=RECENT), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
        discover.judge(cand("unsloth/c-70B-GGUF", "unsloth", [("c-70B-Q4_K_M.gguf", 40.0)], last_modified=RECENT), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
    ]
    assert [r["id"] for r in discover.apply_filters(rows)] == ["unsloth/a-8B-GGUF"]
    assert len(discover.apply_filters(rows, trusted_only=False)) == 2
    assert len(discover.apply_filters(rows, show_unsuitable=True)) == 2
    assert len(discover.apply_filters(rows, trusted_only=False, show_unsuitable=True)) == 3


def test_size_quant_param_and_shape_filters():
    budget = 20.0
    rows = [
        discover.judge(cand("unsloth/moe-35B-A3B-GGUF", "unsloth", [("moe-35B-A3B-Q4_K_M.gguf", 18.0), ("moe-35B-A3B-Q4_K_S.gguf", 16.0)], last_modified=RECENT), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
        discover.judge(cand("unsloth/dense-8B-GGUF", "unsloth", [("dense-8B-Q4_K_M.gguf", 5.0)], last_modified=RECENT), budget_gb=budget, trusted=discover.TRUSTED_QUANTISERS),
    ]
    assert [r["id"] for r in discover.apply_filters(rows, moe="moe")] == ["unsloth/moe-35B-A3B-GGUF"]
    assert [r["id"] for r in discover.apply_filters(rows, moe="dense")] == ["unsloth/dense-8B-GGUF"]
    assert [r["id"] for r in discover.apply_filters(rows, max_gb=6.0)] == ["unsloth/dense-8B-GGUF"]
    assert [r["id"] for r in discover.apply_filters(rows, min_params=20)] == ["unsloth/moe-35B-A3B-GGUF"]
    kept = discover.apply_filters(rows, quants=["q4_k_s"])
    assert [r["id"] for r in kept] == ["unsloth/moe-35B-A3B-GGUF"] and [o["quant"] for o in kept[0]["options"]] == ["Q4_K_S"]
    assert discover.apply_filters(rows, quants=["q6_k"]) == []


def test_hub_url_keeps_the_whole_hub_one_click_away():
    assert discover.hub_url("qwen") == "https://huggingface.co/models?filter=gguf&search=qwen&sort=trending"
    assert "author=unsloth" in discover.hub_url("", owner="unsloth")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_search_endpoint_judges_against_this_server(client, monkeypatch):
    from types import SimpleNamespace
    from app import api

    async def search_models(query, limit=30, sort="downloads"):
        return [SimpleNamespace(id="unsloth/fits-8B-GGUF", downloads=1000, likes=10, last_modified=RECENT, tags=["license:apache-2.0"], pipeline_tag="text-generation", gguf_count=1),
                SimpleNamespace(id="stranger/huge-70B-GGUF", downloads=5_000_000, likes=900, last_modified=RECENT, tags=[], pipeline_tag="text-generation", gguf_count=1)]

    async def repo_files(repo):
        return [{"path": "fits-8B-Q4_K_M.gguf", "size": 5_000_000_000}] if repo.startswith("unsloth") else [{"path": "huge-70B-Q4_K_M.gguf", "size": 40_000_000_000}]

    async def avatars(owners):
        return {}
    monkeypatch.setattr(api.hf, "search_models", search_models)
    monkeypatch.setattr(api, "_repo_files_cached", repo_files)
    monkeypatch.setattr(api.hf, "owner_avatars", avatars)
    from app import main as mm
    monkeypatch.setattr(mm, "_backend_list", lambda: [{"name": "engine", "vram_gb": 14.0}])

    body = client.get("/api/v1/search?q=fits").json()
    assert [r["id"] for r in body["results"]] == ["unsloth/fits-8B-GGUF"]
    assert body["counts"]["found"] == 2 and body["counts"]["hiddenUntrusted"] == 1
    assert body["results"][0]["best"]["quant"] == "Q4_K_M" and body["results"][0]["license"] == "apache-2.0"
    assert body["budgetGb"] > 10 and "huggingface.co/models" in body["hubUrl"]
    wide = client.get("/api/v1/search?q=fits&trustedOnly=false&showUnsuitable=true").json()
    assert [r["id"] for r in wide["results"]] == ["unsloth/fits-8B-GGUF", "stranger/huge-70B-GGUF"]
    assert wide["results"][1]["suitable"] is False and "does not fit" in wide["results"][1]["reasons"][0]


def test_stray_files_without_a_quantisation_are_not_offered_as_models():
    judged = discover.judge(cand("unsloth/M-27B-GGUF", "unsloth", [
        ("M-27B-UD-IQ3_XXS.gguf", 10.9), ("M-27B-UD-Q3_K_XL.gguf", 13.1),
        ("M-27B-index.gguf", 0.01), ("extras/notes.gguf", 0.2)]), budget_gb=12.5, trusted=discover.TRUSTED_QUANTISERS)
    assert [o["path"] for o in judged["options"]] == ["M-27B-UD-Q3_K_XL.gguf", "M-27B-UD-IQ3_XXS.gguf"]
    assert not judged["suitable"], "its only fitting file is below Q4"
    assert judged["best"] is None
    assert "below Q4" in judged["reasons"][0] or "does not fit" in judged["reasons"][0]


def test_imatrix_files_are_never_offered_as_models():
    """Issue #342: an imatrix.gguf is llama.cpp quantisation calibration data, not weights.
    It must never appear as a downloadable option, even alongside a real, fitting quant."""
    judged = discover.judge(cand("bartowski/Qwen_Qwen3.5-4B-GGUF", "bartowski", [
        ("Qwen_Qwen3.5-4B-Q4_K_M.gguf", 2.6), ("Qwen_Qwen3.5-4B-imatrix.gguf", 0.0035)]),
        budget_gb=13.5, trusted=discover.TRUSTED_QUANTISERS)
    assert [o["path"] for o in judged["options"]] == ["Qwen_Qwen3.5-4B-Q4_K_M.gguf"]


def test_is_model_weight_file_rejects_imatrix_regardless_of_size_or_naming():
    # The normal case: tiny, no quant token.
    assert discover.is_model_weight_file("x/Qwen_Qwen3.5-4B-imatrix.gguf", 3_500_000) is False
    # Case-insensitive, and even a suspiciously large or quant-looking name must not slip through
    # — "imatrix" in the basename is disqualifying on its own.
    assert discover.is_model_weight_file("x/model-IMatrix.gguf", 1_000_000) is False
    assert discover.is_model_weight_file("x/model-Q4_K_M-imatrix.gguf", 5_000_000_000) is False
    # A real quantised model file of the same rough size is accepted.
    assert discover.is_model_weight_file("x/Qwen_Qwen3.5-4B-Q4_K_M.gguf", 2_600_000_000) is True
    # Stray files without a quant token, or too small, are rejected the same way build_options
    # already rejected them — the helper must not loosen that.
    assert discover.is_model_weight_file("x/index.gguf", 10_000) is False
    assert discover.is_model_weight_file("x/no-quant-token.gguf", 5_000_000_000) is False


def test_is_stray_gguf_does_not_require_a_quant_token_unlike_is_model_weight_file():
    """The repo-detail view (search_repo) uses this looser check: a real, big file with no
    recognised quant token (a plain "model.gguf", or a scheme our regex doesn't know) is a
    model, not a stray — only imatrix and genuinely tiny files are excluded."""
    assert discover.is_stray_gguf("x/model.gguf", 2_000_000_000) is False
    assert discover.is_stray_gguf("x/model-TQ1_0.gguf", 2_000_000_000) is False
    assert discover.is_stray_gguf("x/model-imatrix.gguf", 3_500_000) is True
    assert discover.is_stray_gguf("x/model-IMatrix.GGUF", 5_000_000_000) is True, \
        "imatrix disqualifies a file regardless of size"
    assert discover.is_stray_gguf("x/extras/stray.gguf", 5_000_000) is True
    # is_model_weight_file is strictly stronger: it additionally requires a quant token.
    assert discover.is_model_weight_file("x/model.gguf", 2_000_000_000) is False
