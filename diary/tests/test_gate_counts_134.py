"""#134: the gate records truncation and error codes per answer row and score.py totals them."""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / "gate"
sys.path.insert(0, str(GATE))

import gatelib as g  # noqa: E402
import gen_fixture  # noqa: E402
import harness  # noqa: E402
import score  # noqa: E402


class FakeClient:
    def __init__(self, status, body):
        self.reply = g.Response(status, body if isinstance(body, bytes) else json.dumps(body).encode(), {})

    def request(self, method, path, **kw):
        return self.reply


def ask(status, body):
    return harness.ask(FakeClient(status, body), "u", "hello", "2030-03-07T10:00:00Z", "s")


def ok_body(finish="stop", truncated=None):
    diary = {"decision": "ok", "xid": "x1"}
    if truncated is not None:
        diary["truncated"] = truncated
    return {"choices": [{"message": {"role": "assistant", "content": "an answer"}, "finish_reason": finish}], "diary": diary}


# ---------------- response shapes -> ask() result ----------------

def test_truncated_answer_is_recorded():
    res = ask(200, ok_body("length", True))
    assert res["status"] == 200 and res["truncated"] is True and res["finish_reason"] == "length"
    assert "error_code" not in res


def test_length_finish_reason_alone_counts_as_truncated():
    res = ask(200, ok_body("length"))
    assert res["truncated"] is True and res["finish_reason"] == "length"


def test_complete_answer_is_recorded_as_not_truncated():
    res = ask(200, ok_body("stop"))
    assert res["truncated"] is False and res["finish_reason"] == "stop" and "error_code" not in res


def test_chat_timeout_502_records_its_code():
    body = {"error": {"message": "the model did not answer before the diary deadline", "type": "api_error", "code": "chat_timeout"}}
    res = ask(502, body)
    assert res["status"] == 502 and res["error_code"] == "chat_timeout" and "truncated" not in res


def test_other_502_records_a_different_code():
    res = ask(502, {"error": {"message": "x", "type": "api_error", "code": "model_unavailable"}})
    assert res["error_code"] == "model_unavailable"


@pytest.mark.parametrize("body", [{"error": {"message": "x"}}, {"error": "boom"}, b"<html>bad gateway</html>", b"", {}])
def test_502_without_a_code_records_none(body):
    res = ask(502, body)
    assert res["status"] == 502 and "error_code" in res and res["error_code"] is None


def test_stream_error_reason_chat_timeout_wins_over_model_failed():
    res = ask(504, {"type": "error", "code": "modelFailed", "reason": "chat_timeout", "saved": False, "text": "t"})
    assert res["error_code"] == "chat_timeout"


def test_stream_error_without_the_timeout_reason_uses_its_code():
    assert ask(504, {"type": "error", "code": "modelFailed", "saved": False})["error_code"] == "modelFailed"


def test_v1_502_length_cap_code_is_recorded():
    body = {"error": {"message": "the answer hit the length cap", "type": "api_error", "code": "length_cap"}}
    assert ask(502, body)["error_code"] == "length_cap"


@pytest.mark.parametrize("reason", ["length_cap", "chat_timeout", "anything_else"])
def test_any_non_empty_stream_reason_wins_over_the_code(reason):
    res = ask(504, {"type": "error", "code": "modelFailed", "reason": reason, "saved": False})
    assert res["error_code"] == reason


@pytest.mark.parametrize("reason", ["", None, 5, ["length_cap"], {"a": 1}, True])
def test_an_empty_or_non_string_stream_reason_falls_back_to_the_code(reason):
    assert ask(504, {"type": "error", "code": "modelFailed", "reason": reason})["error_code"] == "modelFailed"


def test_error_code_of_reads_the_reason_from_the_nested_error_too():
    assert harness.error_code_of({"error": {"code": "modelFailed", "reason": "length_cap"}}) == "length_cap"
    assert harness.error_code_of({"error": {"code": "length_cap"}}) == "length_cap"
    assert harness.error_code_of({"error": {"code": "modelFailed", "reason": ""}}) == "modelFailed"


def test_a_nested_error_code_beats_a_top_level_reason_but_reason_wins_inside_one_object():
    # #144: the nested object is read first; only inside one object does `reason` win over `code`.
    assert harness.error_code_of({"error": {"code": "length_cap"}, "reason": "x"}) == "length_cap"
    assert harness.error_code_of({"error": {"code": "length_cap", "reason": "y"}, "reason": "x"}) == "y"
    assert harness.error_code_of({"error": "text", "code": "modelFailed", "reason": "x"}) == "x"


@pytest.mark.parametrize("diary", ["oops", 7, ["x"], True])
def test_a_truthy_non_object_diary_field_does_not_raise(diary):
    body = ok_body()
    body["diary"] = diary
    res = ask(200, body)
    assert res["status"] == 200 and res["answer"] == "an answer" and res["decision"] is None and res["xid"] is None
    err = ask(502, {"error": {"code": "length_cap"}, "diary": diary})
    assert err["error_code"] == "length_cap" and err["decision"] is None


def test_a_transport_failure_has_status_0_and_no_code():
    res = ask(0, b"")
    assert res["status"] == 0 and res["error_code"] is None


# ---------------- rows written by both writers ----------------

def test_answer_extras_only_carry_what_the_response_provided():
    assert harness.answer_extras({"status": 200, "answer": "a"}) == {}
    res = {"status": 200, "truncated": True, "finish_reason": "length"}
    assert harness.answer_extras(res) == {"truncated": True, "finish_reason": "length"}
    assert harness.answer_extras({"status": 502, "error_code": None}) == {"error_code": None}


def test_stage_and_area_rows_carry_the_new_fields():
    q = {"id": "Q1", "category": "recall"}
    res = {"status": 200, "answer": "a", "ms": 1, "decision": None, "xid": None, "idle_wait_s": 0.0,
           "truncated": True, "finish_reason": "length"}
    stage = harness.stage_row(q, 0, res, "available")
    assert stage["truncated"] is True and stage["finish_reason"] == "length" and stage["status"] == 200
    spec = importlib.util.spec_from_file_location("model_area", GATE / "areas" / "_model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    row = model.area_row("semantic", q, 1, "text", {**res, "status": 502, "error_code": "chat_timeout"}, None, "trace unavailable", None)
    assert row["http"] == 502 and row["error_code"] == "chat_timeout" and row["id"] == "semantic:Q1#1"


def test_old_response_dicts_produce_old_rows():
    q = {"id": "Q1", "category": "recall"}
    stage = harness.stage_row(q, 0, {"status": 200, "answer": "a", "ms": 1, "decision": None, "xid": None, "idle_wait_s": 0.0}, "available")
    assert not {"truncated", "finish_reason", "error_code"} & set(stage)


# ---------------- score.py totals ----------------

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("counts") / "fix"
    gen_fixture.write(out, 20261020, date(2030, 1, 6), 60, "UTC", "0" * 40)
    return out


def _stage_row(q, status=200, **extra):
    return {"id": q["id"], "query": q["id"], "turn": 0, "category": q["category"], "status": status, "answer": "x",
            "verdict": "needs-judge", **extra}


def _run(tmp_path, fixture_dir, name, stage_extras, area_rows):
    run = tmp_path / name
    (run / "stages").mkdir(parents=True)
    queries = g.read_jsonl(fixture_dir / "truth" / "queries.jsonl")
    for i, q in enumerate(queries[:4]):
        g.append_jsonl(run / "stages" / "answers.jsonl", _stage_row(q, **stage_extras(i)))
    for area, rows in area_rows.items():
        (run / "workflows" / area).mkdir(parents=True)
        g.write_json(run / "workflows" / area / "result.json", {"area": area, "status": "untested", "parallel_safe": True, "checks": []})
        for r in rows:
            g.append_jsonl(run / "workflows" / area / "answers.jsonl", r)
    return run


def _area_row(i, http=200, **extra):
    return {"id": f"a:Q{i}#0", "query": f"Q{i}", "turn": 0, "http": http, "answer": "x", "verdict": "needs-judge", **extra}


def test_score_totals_truncation_and_each_error_code_per_area_and_overall(tmp_path, fixture_dir):
    def stage_extras(i):
        return [dict(truncated=True, finish_reason="length"), dict(truncated=False, finish_reason="stop"),
                dict(status=502, error_code="chat_timeout"), dict(status=502, error_code="model_unavailable")][i]
    areas = {"semantic": [_area_row(1, truncated=True, finish_reason="length"), _area_row(2, 502, error_code="chat_timeout"),
                          _area_row(3, 502, error_code="chat_timeout"), _area_row(4, 502, error_code=None)],
             "temporal": [_area_row(5), _area_row(6, truncated=True)]}
    counts = score.score(_run(tmp_path, fixture_dir, "r", stage_extras, areas), fixture_dir)["counts"]
    assert counts["by_area"]["stages"] == {"rows": 4, "truncated": 1, "error_codes": {"chat_timeout": 1, "model_unavailable": 1}}
    assert counts["by_area"]["semantic"] == {"rows": 4, "truncated": 1, "error_codes": {"chat_timeout": 2, "unknown": 1}}
    assert counts["by_area"]["temporal"] == {"rows": 2, "truncated": 1, "error_codes": {}}
    assert counts["overall"] == {"rows": 10, "truncated": 3, "error_codes": {"chat_timeout": 3, "model_unavailable": 1, "unknown": 1}}
    saved = json.loads((tmp_path / "r" / "stages" / "score.json").read_text())
    assert saved["counts"] == counts


def test_old_format_rows_score_exactly_as_before_and_the_new_fields_change_no_verdict(tmp_path, fixture_dir):
    old = _run(tmp_path, fixture_dir, "old", lambda i: {"status": 502} if i == 2 else {},
               {"semantic": [_area_row(1), _area_row(2, 502)]})
    new = _run(tmp_path, fixture_dir, "new",
               lambda i: {"status": 502, "error_code": "chat_timeout"} if i == 2 else {"truncated": True, "finish_reason": "length"},
               {"semantic": [_area_row(1, truncated=True, finish_reason="length"), _area_row(2, 502, error_code="chat_timeout")]})
    a, b = score.score(old, fixture_dir), score.score(new, fixture_dir)
    for key in ("stages", "areas", "verdict", "blended_score", "judgements"):
        assert a[key] == b[key], key
    assert a["counts"]["overall"] == {"rows": 6, "truncated": 0, "error_codes": {"unknown": 2}}  # no new fields: nothing truncated
    assert b["counts"]["overall"]["truncated"] == 4 and b["counts"]["overall"]["error_codes"] == {"chat_timeout": 2}


def test_counts_with_no_answers_are_zero(tmp_path, fixture_dir):
    run = tmp_path / "empty"
    (run / "stages").mkdir(parents=True)
    assert score.score(run, fixture_dir)["counts"] == {"overall": {"rows": 0, "truncated": 0, "error_codes": {}}, "by_area": {}}
