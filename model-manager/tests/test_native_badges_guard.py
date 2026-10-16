"""noevia#1318 and #1905: the badge writers.

Under NOEVIA_NATIVE_MODEL_MANAGER_BADGES_CONFIRMED=1 the private Python child refuses PUT and DELETE
/api/v1/badges (the native parent answers them) and keeps the badge reads. The Python routes carry
the #1905 fixes too, so python mode answers what the native routes answer.

Synthetic store rows and tokens only; FastAPI startup never runs.
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_BADGES_CONFIRMED"
TOKEN = "synthetic-badges-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PATH = "/api/v1/badges"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)
    db.init()
    return TestClient(app)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_both_writers_after_token_gate(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    writes = []
    monkeypatch.setattr(db, "badge_set", lambda *a, **k: writes.append(a) or (True, ""))
    monkeypatch.setattr(db, "badge_clear", lambda *a, **k: writes.append(a))
    assert client.put(PATH, json={}).status_code == 401
    for method in ("put", "delete"):
        kwargs = {"json": {"alias": "m", "category": "coding", "rating": 3}} if method == "put" else {"params": {"alias": "m", "category": "coding"}}
        refused = getattr(client, method)(PATH, headers=auth(), **kwargs)
        assert refused.status_code == 503
        assert refused.content == REFUSED
    assert writes == []
    assert client.get(PATH, headers=auth()).status_code == 405


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    answer = client.put(PATH, headers=auth(), json={"alias": "m", "category": "coding", "rating": 3})
    assert answer.status_code == 200
    assert [b["rating"] for b in answer.json()["badges"]] == [3]


@pytest.mark.parametrize("body,message", [
    ({"rating": "abc"}, "rating must be a number"),
    ({"rating": True}, "rating must be a number"),
    ({"rating": 2.9}, "rating must be a number"),
    ({"rating": [1]}, "rating must be a number"),
    ({"rating": 3, "runId": "x"}, "runId must be a number"),
    ({"rating": 3, "runId": 2 ** 70}, "runId must be a number"),
    ({"rating": 10 ** 30}, "rating must be 1-5"),
    ({"rating": 1e300}, "rating must be 1-5"),
])
def test_1905_non_integers_answer_400_not_500(client, body, message):
    answer = client.put(PATH, headers=auth(), json={"alias": "m", "category": "coding", **body})
    assert answer.status_code == 400
    assert answer.json() == {"detail": message}
    assert db.badges_for("m") == []


@pytest.mark.parametrize("rating,stored", [("3", 3), (" 4 ", 4), ("+5", 5), (3.0, 3), ("1_0", None)])
def test_integer_forms_python_accepted_still_work(client, rating, stored):
    answer = client.put(PATH, headers=auth(), json={"alias": "m", "category": "coding", "rating": rating})
    if stored is None:
        assert answer.json() == {"detail": "rating must be 1-5"}
    else:
        assert [b["rating"] for b in answer.json()["badges"]] == [stored]


def test_1905_padded_alias_is_answered_and_clearable(client):
    answer = client.put(PATH, headers=auth(), json={"alias": " p ", "category": "coding", "rating": 3})
    assert [b["alias"] for b in answer.json()["badges"]] == ["p"]
    cleared = client.delete(PATH, headers=auth(), params={"alias": " p ", "category": "coding"})
    assert cleared.json() == {"badges": []}
    assert db.badges_for("p") == []
