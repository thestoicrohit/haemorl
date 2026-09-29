"""
HaemoRL — In-memory database + seed logic.
Uses a single DB singleton. Thread-safe reads; writes should go through service layer.
"""
from __future__ import annotations
import json
import math
import random
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import (
    BLOOD_TYPES, BLOOD_COMPAT, HLA_POOL, ORGAN_VIABILITY, ORGAN_LIST,
    HOSPITALS, HOSPITAL_COORDS, HOSPITAL_STATES, CITIES, CITY_STATE,
    FIRST_NAMES, LAST_NAMES, OCCUPATIONS, DISEASE_DB,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

def uid() -> str:
    return str(uuid.uuid4())[:8]

def ri(a: int, b: int) -> int:
    return random.randint(a, b)

def rf(a: float, b: float) -> float:
    return round(random.uniform(a, b), 2)


# ── HLA / blood helpers ────────────────────────────────────────────────────────

def blood_ok(donor_bt: str, patient_bt: str) -> bool:
    """Return True if donor blood type is compatible with patient."""
    return patient_bt in BLOOD_COMPAT.get(donor_bt, set())

def make_hla() -> dict[str, list[str]]:
    return {locus: [random.choice(HLA_POOL[locus]), random.choice(HLA_POOL[locus])]
            for locus in ("A", "B", "DR")}

def hla_score(patient_hla: dict, donor_hla: dict) -> float:
    """6-antigen HLA match score [0,1]."""
    if not patient_hla or not donor_hla:
        return 0.0
    matches = total = 0
    for locus in ("A", "B", "DR"):
        pa = set(patient_hla.get(locus, []))
        da = set(donor_hla.get(locus, []))
        total += 2
        matches += len(pa & da)
    return round(matches / max(total, 1), 3)


# ── PELD score ─────────────────────────────────────────────────────────────────

def peld_score(age: int, bilirubin: float, inr: float, albumin: float, growth_failure: bool) -> int:
    if age >= 18:
        return 0
    score = (
        4.80 * math.log(max(bilirubin, 0.01))
        + 18.57 * math.log(max(inr, 0.01))
        - 6.87 * math.log(max(albumin, 0.01))
        + (4.36 if age < 1 else 0)
        + (6.67 if growth_failure else 0)
    )
    return max(1, min(99, int(score)))


# ── factories ─────────────────────────────────────────────────────────────────

def make_patient(category: str | None = None, urgency: str | None = None) -> dict[str, Any]:
    DB._pctr += 1
    pid = f"P-{str(DB._pctr).zfill(4)}"
    cat = category or random.choice(list(DISEASE_DB.keys()))
    dinfo = DISEASE_DB[cat]
    disease = random.choice(dinfo["diseases"])
    urg = urgency or random.choices(
        ["critical", "urgent", "moderate", "stable"],
        weights=[15, 25, 35, 25],
    )[0]

    age = ri(0, 85)
    is_paed = age <= 17
    city = random.choice(CITIES)
    state = CITY_STATE[city]
    bt = random.choice(BLOOD_TYPES)
    fn = random.choice(FIRST_NAMES)
    ln = random.choice(LAST_NAMES)
    wait_m = ri(1, 60)

    viab = ORGAN_VIABILITY.get(dinfo["organ"], 24)
    factor = rf(0.1, 0.95) if urg in ("critical", "urgent") else rf(0.5, 1.0)
    isch_base = viab * factor

    # Paediatric liver — PELD
    bilirubin_v = rf(0.5, 35)
    inr_v = rf(1.0, 4.5)
    albumin_v = rf(1.5, 4.0)
    growth_fail = random.random() < 0.3
    peld = peld_score(age, bilirubin_v, inr_v, albumin_v, growth_fail) if (is_paed and cat == "hepatic") else 0

    paed_bonus = 0.20 if age <= 1 else (0.15 if is_paed and age <= 12 else (0.10 if is_paed else 0.0))

    return {
        "id": pid,
        "name": f"{fn} {ln}",
        "age": age,
        "gender": random.choice(["M", "F", "M", "M", "F"]),
        "city": city,
        "state": state,
        "blood_type": bt,
        "hla": make_hla(),
        "category": cat,
        "disease": disease,
        "urgency": urg,
        "need_type": dinfo["need_type"],
        "organ_needed": dinfo["organ"],
        "ischaemia_h": round(isch_base, 2),
        "ischaemia_total": viab,
        "wait_months": wait_m,
        "is_paediatric": is_paed,
        "is_allocated": False,
        "allocation_id": None,
        "hospital": random.choice(HOSPITALS),
        "meld_score": ri(20, 40) if cat == "hepatic" else None,
        "peld_score": peld if peld > 0 else None,
        "cd4_count": ri(10, 200) if cat == "hiv" else None,
        "ef_percent": ri(10, 20) if cat == "cardiac" else None,
        "fev1_percent": ri(15, 30) if cat == "pulmonary" else None,
        "bilirubin": round(bilirubin_v, 1) if cat == "hepatic" else None,
        "inr": round(inr_v, 1) if cat in ("hepatic", "trauma") else None,
        "albumin": round(albumin_v, 1) if cat in ("hepatic", "autoimmune") else None,
        "dialysis_status": random.choice(["HD", "PD", "None", "None"]) if cat == "renal" else None,
        "crossmatch_status": random.choices(["negative", "positive", "unknown"], weights=[70, 10, 20])[0],
        "kpe_eligible": cat == "renal" and random.random() < 0.25,
        "symptoms": random.sample(dinfo["symptoms"], min(3, len(dinfo["symptoms"]))),
        "medications": random.sample(dinfo["medications"], min(2, len(dinfo["medications"]))),
        "lab_markers": dinfo["lab_markers"],
        "backstory": (
            f"{fn} is a {age}-year-old {random.choice(OCCUPATIONS)} from {city}, {state}. "
            f"Has been on the waiting list for {wait_m} months. "
            + random.choice([
                "Family is hopeful.",
                "Urgently needs intervention.",
                "Has been deteriorating rapidly.",
                "Responds well to current therapy.",
            ])
        ),
        "admitted_at": ts(),
        "paed_bonus": paed_bonus,
        "version": "v5",
    }


def make_donor() -> dict[str, Any]:
    DB._dctr += 1
    did = f"D-{str(DB._dctr).zfill(4)}"
    fn = random.choice(FIRST_NAMES)
    ln = random.choice(LAST_NAMES)
    n_organs = ri(1, 4)
    orgs = random.sample(ORGAN_LIST, min(n_organs, len(ORGAN_LIST)))
    bt = random.choice(BLOOD_TYPES)
    age = ri(18, 60)
    city = random.choice(CITIES)
    state = CITY_STATE[city]
    return {
        "id": did,
        "name": f"{fn} {ln}",
        "age": age,
        "gender": random.choice(["M", "F"]),
        "blood_type": bt,
        "hla": make_hla(),
        "organs": [
            {
                "organ": o,
                "status": "available",
                "viability_h": ORGAN_VIABILITY.get(o, 24),
                "harvested_at": ts(),
            }
            for o in orgs
        ],
        "available": True,
        "donor_type": random.choices(["DBD", "DCD"], weights=[75, 25])[0],
        "hospital": random.choice(HOSPITALS),
        "city": city,
        "state": state,
        "registered_at": ts(),
    }


# ── Database singleton ─────────────────────────────────────────────────────────

class _Database:
    """
    In-memory store for all platform entities.
    Uses dicts keyed by ID for O(1) lookup.
    Logs use deque for bounded memory.
    """

    def __init__(self) -> None:
        self.patients:    dict[str, dict] = {}
        self.donors:      dict[str, dict] = {}
        self.allocations: dict[str, dict] = {}
        self.blood_bank:  dict[str, dict] = {}
        self.hospitals:   dict[str, dict] = {}
        self.routes:      list[dict] = []

        # Bounded logs — no manual trimming needed
        self.alerts:       deque[dict] = deque(maxlen=300)
        self.analytics:    deque[dict] = deque(maxlen=500)
        self.llm_log:      deque[dict] = deque(maxlen=100)
        self.chat_history: deque[dict] = deque(maxlen=300)
        self.entropy_log:  deque[float] = deque(maxlen=50)
        self.fairness_log: deque[dict] = deque(maxlen=50)

        # RL episode state
        self.ep_id:       str   = uid()
        self.ep_step:     int   = 0
        self.ep_task:     str   = "crisis_routing"
        self.ep_maxsteps: int   = 60
        self.ep_cum:      float = 0.0
        self.ep_done:     bool  = False
        self.ep_expired:  int   = 0

        # Counters
        self._pctr:  int  = 0
        self._dctr:  int  = 0
        self._actr:  int  = 0
        self._alctr: int  = 0
        self._seeded: bool = False

    def next_alloc_id(self) -> str:
        self._actr += 1
        return f"A-{str(self._actr).zfill(4)}"

    def next_alert_id(self) -> str:
        self._alctr += 1
        return f"AL-{str(self._alctr).zfill(4)}"


DB = _Database()


# ── Seed ──────────────────────────────────────────────────────────────────────

SEED_COUNT = 750

def seed(count: int = SEED_COUNT) -> None:
    if DB._seeded:
        return
    random.seed(42)
    cats = list(DISEASE_DB.keys())
    per_cat = count // len(cats)
    remainder = count % len(cats)
    for i, cat in enumerate(cats):
        n = per_cat + (1 if i < remainder else 0)
        for _ in range(n):
            p = make_patient(cat)
            DB.patients[p["id"]] = p

    for _ in range(60):
        d = make_donor()
        DB.donors[d["id"]] = d

    for bt in BLOOD_TYPES:
        DB.blood_bank[bt] = {
            "blood_type": bt,
            "units": ri(25, 160),
            "component": "Whole Blood",
            "expiry_days": ri(5, 42),
        }

    # Backfill timeline history so the analytics/fairness/entropy charts
    # render immediately on a fresh seed (instead of waiting ~60s for the
    # first background tick to append a data point).
    from datetime import timedelta
    _now = datetime.utcnow()
    _crit = sum(1 for p in DB.patients.values() if p.get("urgency") == "critical")
    for i in range(24, 0, -1):
        _t = (_now - timedelta(minutes=i)).isoformat() + "Z"
        _allocs = (24 - i) // 3
        DB.analytics.append({
            "ts": _t,
            "total": len(DB.patients),
            "critical": max(0, _crit + random.randint(-8, 8)),
            "allocations": _allocs,
            "expired": (24 - i) // 6,
            "kpe": random.randint(0, 12),
        })
        DB.entropy_log.append(round(random.uniform(0.40, 0.95), 3))
        DB.fairness_log.append({
            "ts": _t,
            "gini_age":  round(random.uniform(0.18, 0.32), 3),
            "gini_wait": round(random.uniform(0.20, 0.40), 3),
            "paed_pct":  round(random.uniform(10, 28), 1),
            "n_allocs":  _allocs,
        })

    DB._seeded = True


def init_hospitals() -> None:
    if DB.hospitals:
        return
    for i, name in enumerate(HOSPITALS):
        DB.hospitals[name] = {
            "id": f"H-{str(i + 1).zfill(3)}",
            "name": name,
            "load_pct": ri(45, 95),
            "beds_total": ri(300, 3500),
            "beds_available": ri(15, 450),
            "icu_beds": ri(40, 350),
            "icu_available": ri(3, 70),
            "has_transplant": True,
            "coords": list(HOSPITAL_COORDS[i]),
            "state": HOSPITAL_STATES[i],
        }


def init_routes() -> None:
    if DB.routes:
        return
    for i in range(len(HOSPITALS)):
        for j in range(i + 1, len(HOSPITALS)):
            if random.random() < 0.40:
                x1, y1 = HOSPITAL_COORDS[i]
                x2, y2 = HOSPITAL_COORDS[j]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                DB.routes.append({
                    "from": HOSPITALS[i],
                    "to": HOSPITALS[j],
                    "hours": round(dist / 40 + random.uniform(0.5, 2.5), 1),
                    "active": random.random() < 0.35,
                    "dist_km": round(dist * 12),
                })


# ── Persistence ───────────────────────────────────────────────────────────────

def _serialize_db() -> dict:
    return {
        "patients": DB.patients,
        "donors": DB.donors,
        "allocations": DB.allocations,
        "blood_bank": DB.blood_bank,
        "hospitals": DB.hospitals,
        "routes": DB.routes,
        "alerts": list(DB.alerts),
        "analytics": list(DB.analytics),
        "llm_log": list(DB.llm_log),
        "chat_history": list(DB.chat_history),
        "entropy_log": list(DB.entropy_log),
        "fairness_log": list(DB.fairness_log),
        "ep_id": DB.ep_id,
        "ep_step": DB.ep_step,
        "ep_task": DB.ep_task,
        "ep_maxsteps": DB.ep_maxsteps,
        "ep_cum": DB.ep_cum,
        "ep_done": DB.ep_done,
        "ep_expired": DB.ep_expired,
        "_pctr": DB._pctr,
        "_dctr": DB._dctr,
        "_actr": DB._actr,
        "_alctr": DB._alctr,
        "_seeded": DB._seeded,
        "version": "5.1.0",
    }


def save(path: Path) -> None:
    try:
        path.write_text(json.dumps(_serialize_db()), encoding="utf-8")
    except Exception as e:
        import logging
        logging.getLogger("haemorl").error(f"Save failed: {e}")


def load(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        DB.patients    = data.get("patients", {})
        DB.donors      = data.get("donors", {})
        DB.allocations = data.get("allocations", {})
        DB.blood_bank  = data.get("blood_bank", {})
        DB.hospitals   = data.get("hospitals", {})
        DB.routes      = data.get("routes", [])
        DB.alerts      = deque(data.get("alerts", []), maxlen=300)
        DB.analytics   = deque(data.get("analytics", []), maxlen=500)
        DB.llm_log     = deque(data.get("llm_log", []), maxlen=100)
        DB.chat_history= deque(data.get("chat_history", []), maxlen=300)
        DB.entropy_log = deque(data.get("entropy_log", []), maxlen=50)
        DB.fairness_log= deque(data.get("fairness_log", []), maxlen=50)
        DB.ep_id       = data.get("ep_id", uid())
        DB.ep_step     = data.get("ep_step", 0)
        DB.ep_task     = data.get("ep_task", "crisis_routing")
        DB.ep_maxsteps = data.get("ep_maxsteps", 60)
        DB.ep_cum      = data.get("ep_cum", 0.0)
        DB.ep_done     = data.get("ep_done", False)
        DB.ep_expired  = data.get("ep_expired", 0)
        DB._pctr       = data.get("_pctr", 0)
        DB._dctr       = data.get("_dctr", 0)
        DB._actr       = data.get("_actr", 0)
        DB._alctr      = data.get("_alctr", 0)
        DB._seeded     = data.get("_seeded", False)
        return len(DB.patients) > 0
    except Exception as e:
        import logging
        logging.getLogger("haemorl").error(f"Load failed: {e}")
        return False
