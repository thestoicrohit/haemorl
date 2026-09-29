"""
HaemoRL — Allocation & Reward Engine
Single source of truth for all reward computation and matching logic.
8-component reward: blood, HLA, ischaemia, paediatric, hospital-load,
                   survival, geographic, organ-expiry.
"""
from __future__ import annotations
import math
import random
from typing import Any

from core.database import DB, blood_ok, hla_score


# ── Fairness metrics ───────────────────────────────────────────────────────────

def gini_coefficient(values: list[float]) -> float:
    """Gini index in [0, 1): 0 = perfectly equal, → 1 = maximally unequal."""
    if not values:
        return 0.0
    n = len(values)
    s = sorted(values)
    total = sum(s)
    if total == 0:
        return 0.0
    cumulative = sum((i + 1) * s[i] for i in range(n))
    return round(2 * cumulative / (n * total) - (n + 1) / n, 3)


def policy_entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return 1.0
    probs = [(s - mn) / (mx - mn + 1e-9) for s in scores]
    total = sum(probs) + 1e-9
    probs = [p / total for p in probs]
    return round(
        -sum(p * math.log(p + 1e-9) for p in probs if p > 0) / math.log(len(probs) + 1),
        3,
    )


# ── Geographic distance ────────────────────────────────────────────────────────

def geo_dist(hospital_a: str, hospital_b: str) -> float:
    """Euclidean distance in SVG map space between two hospitals."""
    c1 = DB.hospitals.get(hospital_a, {}).get("coords", [250, 180])
    c2 = DB.hospitals.get(hospital_b, {}).get("coords", [250, 180])
    if len(c1) == 2 and len(c2) == 2:
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    return 50.0


# ── 8-component reward ────────────────────────────────────────────────────────

def compute_reward(
    patient: dict[str, Any] | None,
    donor:   dict[str, Any] | None,
    hospital_name: str,
    expired: int = 0,
    crit_remaining: int = 0,
    action_type: str = "match_organ",
) -> dict[str, Any]:
    """
    Returns:
        value:       float [-1, 1]
        hla_score:   float [0, 1]
        breakdown:   dict  {component: contribution}
        explanation: str   human-readable summary
        blood_ok:    bool
        survival_pct: int
    """
    if action_type == "skip":
        v = -0.30 if crit_remaining > 0 else 0.0
        return {
            "value": v, "hla_score": 0.0,
            "breakdown": {"skip": v},
            "explanation": f"Skip ({crit_remaining} critical waiting)",
            "blood_ok": False, "survival_pct": 0,
        }
    if not patient:
        return {"value": -0.50, "hla_score": 0.0, "breakdown": {"invalid": -0.50},
                "explanation": "Patient not found", "blood_ok": False, "survival_pct": 0}
    if not donor:
        return {"value": -0.40, "hla_score": 0.0, "breakdown": {"no_donor": -0.40},
                "explanation": "No donor available", "blood_ok": False, "survival_pct": 0}

    r = 0.0
    bd: dict[str, float] = {}

    # ── 1. BLOOD COMPATIBILITY ─────────────────────────────────────────────────
    bok = blood_ok(donor.get("blood_type", ""), patient.get("blood_type", ""))
    bd["blood"] = 0.25 if bok else -0.15
    r += bd["blood"]

    # ── 2. HLA TISSUE TYPING — 6-antigen ──────────────────────────────────────
    hsc = hla_score(patient.get("hla", {}), donor.get("hla", {}))
    bd["hla"] = round(hsc * 0.35, 4)
    r += bd["hla"]

    # ── 3. ISCHAEMIA URGENCY ───────────────────────────────────────────────────
    ih    = patient.get("ischaemia_h", 24)
    total = patient.get("ischaemia_total", 24)
    frac  = ih / max(total, 1)
    if ih <= 0:      isch = -0.30
    elif ih < 1:     isch = 0.22
    elif ih < 2:     isch = 0.20
    elif ih < 4:     isch = 0.15
    elif ih < 6:     isch = 0.10
    elif ih < 12:    isch = 0.05
    elif frac > 0.8: isch = -0.02
    else:            isch = 0.01
    bd["isch"] = round(isch, 4)
    r += bd["isch"]

    # ── 4. PAEDIATRIC PRIORITY (NOTTO guideline) ───────────────────────────────
    age = patient.get("age", 40)
    if age <= 1:    paed = 0.20
    elif age <= 12: paed = 0.15
    elif age <= 17: paed = 0.10
    else:           paed = 0.0
    peld = patient.get("peld_score") or 0
    if peld >= 30:   paed = min(paed + 0.10, 0.30)
    elif peld >= 20: paed = min(paed + 0.05, 0.25)
    bd["paed"] = paed
    r += bd["paed"]

    # ── 5. HOSPITAL LOAD (ICU-aware) ───────────────────────────────────────────
    load = DB.hospitals.get(hospital_name, {}).get("load_pct", 70)
    icu  = DB.hospitals.get(hospital_name, {}).get("icu_available", 10)
    if load > 95:   hosp = -0.25
    elif load > 90: hosp = -0.20
    elif load > 85: hosp = -0.12
    elif load > 75: hosp = -0.05
    elif load < 45: hosp = 0.08
    elif load < 55: hosp = 0.05
    else:           hosp = 0.0
    if icu <= 2:    hosp -= 0.05
    bd["hosp"] = round(hosp, 4)
    r += bd["hosp"]

    # ── 6. SURVIVAL PROBABILITY (MELD/PELD/CD4/EF/FEV1) ──────────────────────
    base_surv = max(0.0, 1.0 - age / 130)
    disease_penalty = 0.0
    meld = patient.get("meld_score")
    if meld:
        if meld >= 40:   disease_penalty += 0.25
        elif meld >= 35: disease_penalty += 0.15
        elif meld >= 30: disease_penalty += 0.10
        elif meld >= 25: disease_penalty += 0.05
    if peld >= 30:   disease_penalty += 0.12
    elif peld >= 20: disease_penalty += 0.06
    cd4 = patient.get("cd4_count")
    if cd4 and cd4 < 50:   disease_penalty += 0.10
    ef = patient.get("ef_percent")
    if ef and ef < 15:     disease_penalty += 0.08
    fev1 = patient.get("fev1_percent")
    if fev1 and fev1 < 20: disease_penalty += 0.06
    if patient.get("crossmatch_status") == "positive": disease_penalty += 0.05
    surv_score = max(0.0, base_surv - disease_penalty)
    bd["surv"] = round(surv_score * 0.15, 4)
    r += bd["surv"]

    # ── 7. GEOGRAPHIC PROXIMITY ────────────────────────────────────────────────
    pt_hosp = patient.get("hospital", "")
    if pt_hosp and hospital_name:
        dist = geo_dist(pt_hosp, hospital_name)
        if dist < 20:    geo = 0.06
        elif dist < 40:  geo = 0.03
        elif dist < 80:  geo = 0.01
        elif dist > 150: geo = -0.04
        else:            geo = 0.0
    else:
        geo = 0.0
    bd["geo"] = round(geo, 4)
    r += bd["geo"]

    # ── 8. ORGAN EXPIRY PENALTY ────────────────────────────────────────────────
    if expired > 0:
        # -0.05 per expired organ, capped at -0.30
        bd["expiry"] = round(max(-0.05 * expired, -0.30), 3)
        r += bd["expiry"]

    # ── Bonus: wait-time fairness & dialysis ───────────────────────────────────
    wait = patient.get("wait_months", 0)
    if wait > 24:   r += 0.03
    elif wait > 12: r += 0.015
    if patient.get("dialysis_status") in ("HD", "PD"):
        r += 0.02

    final = round(max(-1.0, min(1.0, r)), 3)

    # Human-readable summary
    xm = patient.get("crossmatch_status", "unknown")
    explanation = (
        f"Blood={'✓' if bok else '✗'} "
        f"HLA={int(hsc * 100)}% "
        f"Isch={ih:.1f}h({int(frac * 100)}%) "
        f"Paed={'YES(+' + str(round(paed, 2)) + ')' if paed > 0 else 'no'} "
        f"Load={load}% "
        f"Surv={int(surv_score * 100)}% "
        f"Geo={geo:+.2f} "
        f"Wait={wait}mo"
        + (f" PELD={peld}" if peld > 0 else "")
        + (f" XM={xm}" if xm != "unknown" else "")
    )

    return {
        "value": final,
        "hla_score": round(hsc, 3),
        "breakdown": bd,
        "explanation": explanation,
        "blood_ok": bok,
        "survival_pct": int(surv_score * 100),
    }


# ── Allocation bookkeeping ────────────────────────────────────────────────────

def release_allocation(alloc: dict[str, Any]) -> None:
    """Undo an allocation's hold: the donor becomes available and the patient re-enters the queue."""
    d = DB.donors.get(alloc.get("donor_id") or "")
    if d:
        d["available"] = True
    p = DB.patients.get(alloc.get("patient_id") or "")
    if p and p.get("allocation_id") == alloc.get("id"):
        p["is_allocated"] = False
        p["allocation_id"] = None


def _renew_window(p: dict[str, Any]) -> None:
    """The organ offer lapsed: log it and start a fresh window for the next offer."""
    p["missed_offers"] = p.get("missed_offers", 0) + 1
    p["ischaemia_h"] = round((p.get("ischaemia_total") or 24) * random.uniform(0.5, 1.0), 2)
    p.pop("_expired", None)


def tick_ischaemia(hours: float) -> int:
    """Advance every unallocated critical/urgent clock by `hours`; returns organs that expired.

    Each expiry is counted once (DB.ep_expired) and the patient stays on the waitlist with a
    new window, so the simulation never fills up with clocks stuck at zero.
    """
    newly = 0
    for p in DB.patients.values():
        if p.get("urgency") in ("critical", "urgent") and not p.get("is_allocated"):
            prev = p.get("ischaemia_h") or 0.0
            p["ischaemia_h"] = max(0.0, round(prev - hours, 4))
            if p["ischaemia_h"] == 0.0:
                newly += 1
                _renew_window(p)
    DB.ep_expired += newly
    return newly


def revive_stalled_clocks() -> int:
    """Give a fresh window to patients left at 0h by older versions (runs at startup)."""
    n = 0
    for p in DB.patients.values():
        if p.get("urgency") in ("critical", "urgent") and not p.get("is_allocated") \
                and (p.get("_expired") or not p.get("ischaemia_h")):
            _renew_window(p)
            n += 1
    return n


# ── Best-match finder ─────────────────────────────────────────────────────────

def find_best_match(
    patient: dict[str, Any],
    donors: list[dict[str, Any]],
    hospitals: list[dict[str, Any]],
    expired: int = 0,
    crit_count: int = 0,
) -> tuple[dict | None, dict | None, dict | None]:
    """
    Find the (donor, hospital, reward) triple that maximises reward for a patient.
    Returns (best_donor, best_hospital, best_reward) or (None, None, None).
    """
    best_donor = best_hospital = best_reward = None
    best_score = -999.0
    best_hosp = min(hospitals, key=lambda h: h.get("load_pct", 100)) if hospitals else None
    for d in donors:
        if not d.get("available"):
            continue
        h = best_hosp or {"name": "AIIMS New Delhi"}
        rew = compute_reward(patient, d, h["name"], expired, crit_count)
        if rew["value"] > best_score:
            best_score = rew["value"]
            best_donor = d
            best_hospital = h
            best_reward = rew
    return best_donor, best_hospital, best_reward


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_single() -> dict:
    done = [a for a in DB.allocations.values() if a["status"] in ("complete", "active", "pending")]
    if not done:
        return {"task_id": "single_match", "score": 0.001, "passed": False,
                "details": "No allocations", "metrics": {}}
    best = max(done, key=lambda a: a.get("hla_score", 0) * 0.5 + a.get("reward", {}).get("value", 0) * 0.5)
    s = round(min(1.0,
        0.40
        + (0.30 if best.get("reward", {}).get("breakdown", {}).get("blood", 0) > 0 else 0)
        + best.get("hla_score", 0) * 0.30
        + (0.05 if best.get("is_paediatric") else 0)
    ), 3)
    return {
        "task_id": "single_match",
        "score": round(max(0.001, min(0.999, s)), 3),
        "passed": s >= 0.25,
        "details": f"hla={int(best.get('hla_score', 0) * 100)}% score={s}",
        "metrics": {"n": len(done)},
    }


def grade_batch() -> dict:
    done = [a for a in DB.allocations.values() if a["status"] in ("complete", "active", "pending")]
    n = len(done)
    if n == 0:
        return {"task_id": "batch_allocation", "score": 0.001, "passed": False,
                "details": "No allocations", "metrics": {}}
    avg_hla = sum(a.get("hla_score", 0) for a in done) / n
    blood_ok_pct = sum(1 for a in done if a.get("reward", {}).get("breakdown", {}).get("blood", 0) > 0) / n
    paed = sum(1 for a in done if a.get("is_paediatric", False))
    s = round(min(1.0,
        min(1.0, n / 5) * 0.50
        + avg_hla * 0.20
        + blood_ok_pct * 0.10
        + min(0.10, paed * 0.05)
        - min(0.15, DB.ep_expired * 0.03)
    ), 3)
    return {
        "task_id": "batch_allocation",
        "score": round(max(0.001, min(0.999, s)), 3),
        "passed": s >= 0.30,
        "details": f"n={n} avg_hla={int(avg_hla * 100)}% score={s}",
        "metrics": {"matches": n, "avg_hla": round(avg_hla, 2)},
    }


def grade_crisis() -> dict:
    done = list(DB.allocations.values())
    n = len(done)
    step = DB.ep_step
    cum = DB.ep_cum
    exp = DB.ep_expired
    rs = min(0.35, max(0.0, (cum / max(step, 1)) * 0.35))
    crit_m = sum(1 for a in done if a.get("reward", {}).get("breakdown", {}).get("isch", 0) >= 0.10)
    paed_m = sum(1 for a in done if a.get("is_paediatric") and a.get("reward", {}).get("breakdown", {}).get("isch", 0) >= 0.10)
    s = round(min(1.0,
        rs
        + min(0.25, crit_m * 0.05)
        + max(0.0, 0.20 - exp * 0.04)
        + min(0.10, paed_m * 0.05)
    ), 3)
    return {
        "task_id": "crisis_routing",
        "score": round(max(0.001, min(0.999, s)), 3),
        "passed": s >= 0.20,
        "details": f"steps={step} cum={cum:.2f} crit={crit_m} exp={exp}",
        "metrics": {"steps": step, "cum": round(cum, 3), "crit_matched": crit_m},
    }


# ── Observation snapshot ──────────────────────────────────────────────────────

def build_observation() -> dict:
    pts = list(DB.patients.values())
    crit = sorted(
        [p for p in pts if p.get("urgency") == "critical"],
        key=lambda p: p.get("ischaemia_h", 999),
    )
    avd  = [d for d in DB.donors.values() if d.get("available")]
    orgs = sum(
        len([o for o in d.get("organs", []) if o.get("status") == "available"])
        for d in avd
    )
    bl = sum(b.get("units", 0) for b in DB.blood_bank.values())
    avg_h = round(
        sum(h.get("load_pct", 0) for h in DB.hospitals.values()) / max(len(DB.hospitals), 1),
        1,
    )
    avg_hla = round(
        sum(hla_score(crit[0].get("hla", {}), d.get("hla", {})) for d in avd[:5])
        / max(len(avd[:5]), 1),
        3,
    ) if crit and avd else 0.0

    def pv(lst, idx, key, default):
        p = lst[idx] if idx < len(lst) else {}
        return p.get(key, default) if p else default

    return {
        "episode_id": DB.ep_id,
        "step": DB.ep_step,
        "total_patients": len(pts),
        "critical_count": len(crit),
        "urgent_count": sum(1 for p in pts if p.get("urgency") == "urgent"),
        "organ_queue": sum(1 for p in pts if p.get("need_type") == "organ"),
        "blood_units_total": bl,
        "donors_available": len(avd),
        "organs_available": orgs,
        "avg_hla_score": avg_hla,
        "min_ischaemia_remaining": round(min((p.get("ischaemia_h", 999) for p in crit), default=0), 2),
        "paediatric_critical": sum(1 for p in crit if p.get("is_paediatric")),
        "expired_organs": DB.ep_expired,
        "hospital_avg_load": avg_h,
        "hospitals_overloaded": sum(1 for h in DB.hospitals.values() if h.get("load_pct", 0) > 85),
        "active_allocations": sum(1 for a in DB.allocations.values() if a.get("status") == "active"),
        "completed_allocations": sum(1 for a in DB.allocations.values() if a.get("status") == "complete"),
        "cumulative_reward": round(DB.ep_cum, 3),
        "kpe_candidates": sum(1 for p in pts if p.get("kpe_eligible") and not p.get("is_allocated")),
        # top-3 critical patients (abbreviated)
        "p1": {k: crit[0].get(k) for k in ("id","name","blood_type","urgency","ischaemia_h","is_paediatric","disease","peld_score","meld_score")} if len(crit) > 0 else {},
        "p2": {k: crit[1].get(k) for k in ("id","name","blood_type","urgency","ischaemia_h","is_paediatric","disease","peld_score","meld_score")} if len(crit) > 1 else {},
        "p3": {k: crit[2].get(k) for k in ("id","name","blood_type","urgency","ischaemia_h","is_paediatric","disease","peld_score","meld_score")} if len(crit) > 2 else {},
        # top-3 available donors (abbreviated)
        "d1": {k: avd[0].get(k) for k in ("id","name","blood_type","hospital")} if len(avd) > 0 else {},
        "d2": {k: avd[1].get(k) for k in ("id","name","blood_type","hospital")} if len(avd) > 1 else {},
        "d3": {k: avd[2].get(k) for k in ("id","name","blood_type","hospital")} if len(avd) > 2 else {},
    }
