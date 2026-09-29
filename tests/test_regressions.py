"""Regression tests for bugs fixed in the RL environment, allocations and patient API."""
from pathlib import Path

from core.constants import CITY_STATE
from core.database import DB, make_patient
from services.allocation import compute_reward, gini_coefficient


# ── Pure functions ───────────────────────────────────────────────────────────

def test_gini_is_zero_for_equal_values_and_positive_for_unequal():
    assert gini_coefficient([1, 1, 1, 1]) == 0.0
    assert gini_coefficient([0, 0, 0, 10]) == 0.75
    assert 0 < gini_coefficient([1, 2, 3, 4]) < 1


def test_expiry_penalty_is_capped_at_minus_030():
    p = {"blood_type": "O+", "hla": {}, "age": 40, "ischaemia_h": 10}
    d = {"blood_type": "O+", "hla": {}}
    assert compute_reward(p, d, "", expired=1)["breakdown"]["expiry"] == -0.05
    assert compute_reward(p, d, "", expired=20)["breakdown"]["expiry"] == -0.30


def test_new_patients_live_in_their_citys_state():
    for _ in range(50):
        p = make_patient()
        assert p["state"] == CITY_STATE[p["city"]]
    DB._pctr -= 50  # make_patient only bumps the counter; nothing was stored


# ── RL environment ───────────────────────────────────────────────────────────

def _match(client):
    d = client.get("/api/rl/smart-decide").json()["decision"]
    assert d, "seed data should always offer a match"
    return d


def test_each_expired_organ_is_counted_once_and_clocks_renew(client):
    client.post("/reset", json={"task": "crisis_routing"})
    missed = lambda: sum(p.get("missed_offers", 0) for p in DB.patients.values())
    before = missed()
    for _ in range(5):
        client.post("/step", json={"action_type": "skip"})
    # the episode's expired count equals exactly the organ offers that lapsed
    assert client.get("/state").json()["expired_organs"] == missed() - before
    # …and no waiting patient is left with a dead clock
    assert not any(p.get("urgency") == "critical" and not p.get("is_allocated") and p.get("ischaemia_h") == 0
                   for p in DB.patients.values())


def test_reset_clears_expired_count_and_frees_donors(client):
    client.post("/reset", json={"task": "batch_allocation"})
    m = _match(client)
    client.post("/step", json=m)
    assert DB.donors[m["donor_id"]]["available"] is False

    client.post("/reset", json={})
    assert client.get("/state").json()["expired_organs"] == 0
    assert DB.donors[m["donor_id"]]["available"] is True
    assert DB.patients[m["patient_id"]]["is_allocated"] is False


def test_step_refuses_to_allocate_a_patient_twice(client):
    client.post("/reset", json={"task": "batch_allocation"})
    m = _match(client)
    first = client.post("/step", json=m).json()
    assert first["info"]["last_action_error"] is None
    other = next(d for d in DB.donors.values() if d.get("available"))
    second = client.post("/step", json={**m, "donor_id": other["id"]}).json()
    assert second["info"]["last_action_error"] == "patient already allocated"
    assert sum(1 for a in DB.allocations.values() if a["patient_id"] == m["patient_id"]) == 1


# ── Allocations & patients ───────────────────────────────────────────────────

def test_cancelling_an_allocation_frees_the_donor_and_cannot_be_reopened(client):
    client.post("/reset", json={})
    m = _match(client)
    alloc = client.post("/api/allocations/commit", json=m).json()
    assert DB.donors[m["donor_id"]]["available"] is False

    r = client.patch(f"/api/allocations/{alloc['id']}", json={"status": "cancelled"})
    assert r.status_code == 200
    assert DB.donors[m["donor_id"]]["available"] is True
    assert DB.patients[m["patient_id"]]["is_allocated"] is False

    assert client.patch(f"/api/allocations/{alloc['id']}", json={"status": "active"}).status_code == 409


def test_deleting_an_allocated_patient_frees_the_donor(client):
    client.post("/reset", json={})
    m = _match(client)
    alloc = client.post("/api/allocations/commit", json=m).json()
    assert client.delete(f"/api/patients/{m['patient_id']}").status_code == 200
    assert DB.donors[m["donor_id"]]["available"] is True
    assert DB.allocations[alloc["id"]]["status"] == "cancelled"


def test_create_patient_validates_input(client):
    bad = [{"age": "abc"}, {"age": 500}, {"blood_type": "C+"}, {"urgency": "whenever"}]
    for body in bad:
        assert client.post("/api/patients", json=body).status_code == 422, body
    ok = client.post("/api/patients", json={"age": "42", "blood_type": "AB-", "urgency": "stable"})
    assert ok.status_code == 201 and ok.json()["age"] == 42


# ── Config & spec ────────────────────────────────────────────────────────────

def test_llm_is_off_without_config_and_chat_explains_how_to_enable(client):
    assert client.get("/health").json()["llm_enabled"] is False
    reply = client.post("/api/chat", json={"message": "hi"}).json()["response"]
    assert "LLM_API_KEY" in reply


def test_openenv_yaml_is_served_from_the_spec_file(client):
    served = client.get("/openenv.yaml").text
    on_disk = (Path(__file__).parent.parent / "openenv.yaml").read_text(encoding="utf-8")
    assert served == on_disk


def test_tests_do_not_write_to_the_real_database(client):
    import os
    assert "haemorl-test-" in os.environ["DATA_DIR"]
