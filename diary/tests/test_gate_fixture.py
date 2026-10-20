"""Diary gate tooling (diary/gate): deterministic synthetic fixture, real daily format, signer and scoring."""
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / "gate"
sys.path.insert(0, str(GATE))

import gen_fixture  # noqa: E402
import gatelib  # noqa: E402
import score  # noqa: E402
from agent import corpus as fmt  # noqa: E402
from agent import tenant_assertion  # noqa: E402

ARGS = dict(seed=20261020, anchor=date(2030, 1, 6), days=60, tz="UTC", commit="0" * 40)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("gate") / "fix"
    gen_fixture.write(out, **ARGS)
    return out


def tree_bytes(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_two_runs_are_byte_identical(fixture_dir, tmp_path):
    again = tmp_path / "again"
    manifest = gen_fixture.write(again, **ARGS)
    assert tree_bytes(fixture_dir) == tree_bytes(again)
    files = {rel: hashlib.sha256((again / rel).read_bytes()).hexdigest() for rel in manifest["files"]}
    assert files == manifest["files"]
    assert gen_fixture.tree_hash(files) == manifest["tree_sha256"]


def test_cli_matches_library_and_refuses_non_empty_output(fixture_dir, tmp_path, capsys):
    out = tmp_path / "cli"
    assert gen_fixture.main(["--out", str(out), "--generator-commit", "0" * 40]) == 0
    assert tree_bytes(out) == tree_bytes(fixture_dir)
    with pytest.raises(SystemExit):
        gen_fixture.main(["--out", str(out), "--generator-commit", "0" * 40])
    with pytest.raises(SystemExit):
        gen_fixture.main(["--out", str(tmp_path / "tz"), "--tz", "Europe/London"])


def test_a_different_seed_changes_the_fixture(fixture_dir, tmp_path):
    other = gen_fixture.write(tmp_path / "other", **{**ARGS, "seed": 7})
    assert other["tree_sha256"] != json.loads((fixture_dir / "manifest.json").read_text())["tree_sha256"]


def test_manifest_records_inputs_and_query_clock(fixture_dir):
    m = json.loads((fixture_dir / "manifest.json").read_text())
    assert (m["seed"], m["anchor"], m["days"], m["tz"]) == (20261020, "2030-01-06", 60, "UTC")
    assert m["query_clock"] == "2030-03-06T12:00:00Z"
    assert m["generator_commit"] == "0" * 40
    c = m["counts"]
    assert c["entries"] == 180 and c["queries"] == 60 and c["needles"] == 40 and c["distractors"] == 40
    assert (c["multi_entry"], c["unanswerable"], c["ambiguous"]) == (8, 6, 4)
    assert all(v >= 13 for v in c["positions"].values())
    assert c["mutations"] == {"paraphrase": 4, "correction": 4, "deletion": 3}


def test_day_files_are_what_the_diary_writer_produces(fixture_dir):
    entries = gatelib.read_jsonl(fixture_dir / "truth" / "entries.jsonl")
    files = gatelib.fixture_files(fixture_dir)
    assert len(files) == 60
    by_day = {}
    for e in entries:
        if e["kind"] != "correction":
            by_day.setdefault(e["day"], []).append(e)
    for rel in files:
        text = (fixture_dir / "fixture" / rel).read_text(encoding="utf-8")
        days = fmt.parse_diary(text)
        assert len(days) == 1 and days[0].date is not None
        day = days[0].date
        rows = by_day[day.isoformat()]
        assert rel == f"Entries/{day.year}/{day.strftime('%B')}/{day.strftime('%B')} {day.day}, {day.year}.md"
        assert len(rows) == 3
        # Rebuild the file with the diary's own writer, one exchange at a time.
        rebuilt = None
        for e in rows:
            sub = next(s for s in days[0].subsections if s.header == e["header"])
            ex = sub.exchanges[0]
            assert ex.xid == e["xid"]
            rebuilt = fmt.append_to_month_text(rebuilt, day, e["header"], fmt.render_exchange(ex.me, ex.claude, e["xid"]))
        assert rebuilt == text


def test_queries_reference_real_entries_and_mutations(fixture_dir):
    entries = {e["id"]: e for e in gatelib.read_jsonl(fixture_dir / "truth" / "entries.jsonl")}
    mutations = {m["id"]: m for m in gatelib.read_jsonl(fixture_dir / "truth" / "mutations.jsonl")}
    queries = gatelib.read_jsonl(fixture_dir / "truth" / "queries.jsonl")
    assert len({q["id"] for q in queries}) == len(queries) == 60
    for q in queries:
        assert set(q) >= {"id", "text", "valid_as_of", "relevant_entry_ids", "expected_facts", "forbidden_facts", "expect_abstain"}
        for alternative in q["relevant_entry_ids"]:
            assert alternative and all(i in entries for i in alternative)
        assert all(m in mutations for m in q["mutations"])
        assert not set(q["expected_facts"]) & set(q["forbidden_facts"])
    cats = [q["category"] for q in queries]
    assert cats.count("unanswerable") == 6 and cats.count("ambiguous") == 4 and cats.count("multi-entry") == 8
    assert sum(1 for q in queries if q["expect_abstain"] and q["category"] == "unanswerable") == 6
    assert any(len(q["relevant_entry_ids"]) > 1 for q in queries)
    for m in mutations.values():
        if m["op"] == "add_entry":
            assert m["entry_id"] in entries and ("xid:" + m["xid"]) in m["block"]
            assert m["path"] in gatelib.fixture_files(fixture_dir)  # a correction lands on an existing fixture day


def test_fixture_is_synthetic_and_has_no_credentials(fixture_dir):
    text = "".join((fixture_dir / "fixture" / rel).read_text(encoding="utf-8") for rel in gatelib.fixture_files(fixture_dir))
    lower = text.lower()
    for word in ("password", "passcode", "token", "secret", "api key", "api_key", "pin code", "http"):
        assert word not in lower
    for _, topic in gen_fixture.UNANSWERABLE:
        assert topic.lower() not in lower


def test_gate_signer_matches_the_diary_reference_signer():
    key, user = "k" * 64, "0c7b4f8e-2a55-4d3f-a1e6-9d0b6c2f4e83"
    body = b'{"path": "Notes/a.md"}'
    ours = gatelib.sign(key, user, "POST", "/api/file", query=b"x=1", body=body, ts=1900000000, nonce="ab" * 16)
    ref = tenant_assertion.sign(key, user, "POST", "/api/file", {}, 1900000000, "ab" * 16, query=b"x=1",
                                body_hash=hashlib.sha256(body).hexdigest())
    assert ours == ref
    stored = gatelib.sign(key, user, "GET", "/api/health", storage=gatelib.LOCAL_STORAGE, ts=1900000000, nonce="cd" * 16)
    assert stored == tenant_assertion.sign(key, user, "GET", "/api/health", {"X-Cowork-Storage": gatelib.LOCAL_STORAGE},
                                           1900000000, "cd" * 16)


def test_retrieval_metrics():
    sets = [["E1"], ["E2", "E3"]]
    assert score.best_recall(sets, ["E9", "E2"]) == 0.5
    assert score.best_recall(sets, ["E1"]) == 1.0
    assert score.precision(sets, ["E1", "E9", "E3", "E8"]) == 0.5
    assert score.mrr(sets, ["E9", "E3"]) == 0.5
    assert score.ndcg([["E1"]], ["E1", "E2"], 8) == 1.0
    assert 0 < score.ndcg([["E1"]], ["E2", "E1"], 8) < 1


def test_answer_signals_and_declared_thresholds():
    rubric = json.loads((GATE / "rubric.json").read_text())
    q = {"expected_facts": ["Plot 27"], "forbidden_facts": ["plot 33"], "citation_days": ["2030-01-21"]}
    s = score.answer_signals(q, "On January 21 you noted plot 27.", rubric)
    assert s == {"facts_present": True, "forbidden_present": [], "cited": True, "abstained": False}
    s = score.answer_signals({"expected_facts": [], "forbidden_facts": [], "citation_days": []},
                             "I can't find that in your diary.", rubric)
    assert s["abstained"] is True
    th = json.loads((GATE / "thresholds.json").read_text())
    assert th["candidate_retrieval"]["recall_at_k_min"] == 0.90 and th["candidate_retrieval"]["precision_at_k_min"] == 0.40
    assert th["reranking"]["ndcg_at_k_min"] == 0.80 and th["context_load"]["recall_min"] == 0.90
    assert (th["answer"]["abstentions_min"], th["answer"]["abstentions_of"], th["answer"]["false_abstentions_max"]) == (5, 6, 1)


def test_score_without_traces_reports_untested_and_no_blended_score(fixture_dir, tmp_path):
    run = gatelib.evidence_dir(tmp_path / "ev", "c" * 40, "pre")
    gatelib.write_json(run / "stages" / "load.json", {"ok": True})
    result = score.score(run, fixture_dir)
    stages = {s["stage"]: s["status"] for s in result["stages"]}
    assert stages["load"] == "pass" and stages["candidate_retrieval"].startswith("untested")
    assert result["blended_score"] is None and result["verdict"] != "pass"
    assert (run / "matrix.csv").read_text().startswith("kind,name,status")


def test_evidence_dir_is_refused_inside_git(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(SystemExit):
        gatelib.evidence_dir(tmp_path / "repo" / "ev", "a" * 40, "pre")
    run = gatelib.evidence_dir(tmp_path / "ev", "b" * 40, "post")
    assert run.name == "b" * 12 + "-post" and run.parent.name == "diary-gate"
    assert all((run / d).is_dir() for d in ("stages", "workflows", "logs")) and (run / "blockers.md").exists()
