"""
Smoke tests for the HaemoRL backend.

Run from the v3/ directory:
    python -m pytest -q

The app runs against a temporary DATA_DIR (see conftest.py).

Verifies the app boots (seeds data) and every key endpoint responds 200 with
sane JSON — including /api/llm/decide, which previously crashed on None peld_score.
"""
import pytest


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "HaemoRL" in r.text


GET_ENDPOINTS = [
    "/api/stats",
    "/api/dashboard",
    "/api/patients",
    "/api/donors",
    "/api/blood-bank",
    "/api/hospitals",
    "/api/transport",
    "/api/allocations",
    "/api/analytics",
    "/api/hla/matrix",
    "/api/kpe/candidates",
    "/api/rl/matches",
    "/api/rl/fairness",
    "/api/forecast/demand",
    "/api/llm/log",
    "/state",
    "/tasks",
]


@pytest.mark.parametrize("path", GET_ENDPOINTS)
def test_get_endpoints_ok(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    assert r.headers["content-type"].startswith("application/json")
    r.json()  # must be valid JSON


def test_stats_shape(client):
    s = client.get("/api/stats").json()
    assert s["total_patients"] > 0
    for key in ("critical", "donors_total", "allocations_total", "blood_units"):
        assert key in s


def test_auto_match(client):
    r = client.post("/api/allocations/auto-match")
    assert r.status_code == 200
    assert "matched" in r.json()


def test_llm_decide_no_crash(client):
    # Regression: used to 500 on None peld_score comparison.
    r = client.post("/api/llm/decide")
    assert r.status_code == 200
    body = r.json()
    assert "decision" in body or "action" in body or "mode" in body


def test_grade(client):
    r = client.post("/grade")
    assert r.status_code == 200
    assert "grader_results" in r.json()


def test_create_patient(client):
    before = client.get("/api/stats").json()["total_patients"]
    r = client.post("/api/patients", json={"category": "cardiac", "urgency": "critical",
                                           "name": "Smoke Test", "age": 40})
    assert r.status_code in (200, 201)
    after = client.get("/api/stats").json()["total_patients"]
    assert after == before + 1
