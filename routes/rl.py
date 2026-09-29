"""
HaemoRL — RL / AI Allocation routes
GET  /api/rl/matches          — top N RL-scored match candidates
GET  /api/rl/smart-decide     — best single decision (deterministic)
POST /api/rl/run-episode      — simulate a full episode
GET  /api/rl/episode-summary  — current episode metrics
GET  /api/rl/fairness         — Gini + category equity metrics

POST /reset     — reset RL episode
POST /step      — advance one RL step
GET  /state     — full current state
POST /grade     — compute grader scores
GET  /validate  — environment spec validation
GET  /tasks     — task definitions
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.database import DB, make_patient, ts, uid, hla_score
from core.constants import TASK_INFO, DISEASE_DB
from services.allocation import (
    compute_reward, find_best_match, build_observation,
    grade_single, grade_batch, grade_crisis,
    gini_coefficient, policy_entropy,
    release_allocation, tick_ischaemia,
)

router = APIRouter(tags=["rl"])


def _save():
    from core.database import save
    d = os.getenv("DATA_DIR", "")
    p = (Path(d) / "haemorl_db.json") if d else (Path(__file__).parent.parent / "data" / "haemorl_db.json")
    save(p)


# ── RL environment endpoints (reset / step / state / grade) ──────────────────────────────────────

@router.post("/reset")
async def reset(body: dict = {}):
    task = (body or {}).get("task", "crisis_routing")
    if task not in TASK_INFO:
        task = "crisis_routing"
    info = TASK_INFO[task]
    DB.ep_id = uid()
    DB.ep_step = 0
    DB.ep_task = task
    DB.ep_maxsteps = info["max_steps"]
    DB.ep_cum = 0.0
    DB.ep_done = False
    DB.ep_expired = 0
    # Free every donor/patient held by the previous episode before dropping its allocations
    for a in DB.allocations.values():
        release_allocation(a)
    DB.allocations.clear()
    return {
        "observation": build_observation(),
        "done": False,
        "info": {
            "task": task, "episode_id": DB.ep_id,
            "max_steps": info["max_steps"],
            "patients": len(DB.patients),
            "critical": sum(1 for p in DB.patients.values() if p.get("urgency") == "critical"),
        },
    }


@router.post("/step")
async def step(body: dict):
    if not isinstance(body, dict):
        body = {}
    if DB.ep_done:
        raise HTTPException(400, "Episode done. Call /reset first.")
    if DB.ep_step >= DB.ep_maxsteps:
        DB.ep_done = True
        return {"observation": build_observation(), "reward": {"value": 0.0}, "done": True, "info": {"reason": "max_steps"}}

    DB.ep_step += 1
    delta = {"single_match": 2.0, "batch_allocation": 1.5, "crisis_routing": 1.0}.get(DB.ep_task, 1.0)

    tick_ischaemia(delta)

    # Inject trauma in crisis mode
    if DB.ep_task == "crisis_routing" and DB.ep_step % 5 == 0:
        tp = make_patient("trauma", "critical")
        DB.patients[tp["id"]] = tp

    patient = DB.patients.get(body.get("patient_id", ""))
    donor   = DB.donors.get(body.get("donor_id", "")) if body.get("donor_id") else None
    if donor and not donor.get("available"):
        donor = None

    err = None
    if patient and patient.get("is_allocated") and body.get("action_type", "match_organ") != "skip":
        patient, err = None, "patient already allocated"

    crit_count = sum(1 for p in DB.patients.values() if p.get("urgency") == "critical")
    rew = compute_reward(patient, donor, body.get("hospital", ""), DB.ep_expired, crit_count, body.get("action_type", "match_organ"))

    if body.get("action_type", "match_organ") != "skip" and patient and donor:
        aid = DB.next_alloc_id()
        DB.allocations[aid] = {
            "id": aid,
            "patient_id": patient["id"],    "patient_name": patient.get("name"),
            "donor_id":   donor["id"],      "donor_name":   donor.get("name"),
            "organ": donor.get("organs", [{"organ": "Unknown"}])[0].get("organ"),
            "hospital": body.get("hospital"), "action_type": body.get("action_type"),
            "status": "pending", "reward": rew, "hla_score": rew["hla_score"],
            "is_paediatric": patient.get("is_paediatric", False),
            "peld_score": patient.get("peld_score"),
            "step": DB.ep_step, "created_at": ts(), "disease": patient.get("disease"),
        }
        donor["available"] = False
        patient["is_allocated"] = True
        patient["allocation_id"] = aid
        DB.ep_cum += rew["value"]
    elif body.get("action_type", "match_organ") != "skip" and not err:
        err = "patient or donor not found"

    crit_left = sum(1 for p in DB.patients.values() if p.get("urgency") == "critical")
    done = (DB.ep_step >= DB.ep_maxsteps) or (crit_left == 0 and DB.ep_task == "single_match")
    DB.ep_done = done

    return {
        "observation": build_observation(), "reward": rew, "done": done,
        "info": {
            "step": DB.ep_step, "cumulative_reward": round(DB.ep_cum, 3),
            "critical_remaining": crit_left, "expired_organs": DB.ep_expired,
            "last_action_error": err,
        },
    }


@router.get("/state")
def state():
    return {
        "episode_id": DB.ep_id, "step": DB.ep_step, "task": DB.ep_task,
        "done": DB.ep_done, "cumulative_reward": round(DB.ep_cum, 3),
        "expired_organs": DB.ep_expired,
        "patients": list(DB.patients.values())[:20],
        "donors": list(DB.donors.values()),
        "blood_bank": list(DB.blood_bank.values()),
        "hospitals": list(DB.hospitals.values()),
        "allocations": list(DB.allocations.values()),
        "observation": build_observation(),
    }


@router.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"task_id": k, "difficulty": v["difficulty"], "max_steps": v["max_steps"],
             "description": v["desc"], "baseline_score": v["baseline"]}
            for k, v in TASK_INFO.items()
        ]
    }


@router.post("/grade")
def grade():
    r = {
        "single_match":     grade_single(),
        "batch_allocation": grade_batch(),
        "crisis_routing":   grade_crisis(),
    }
    for k in r:
        r[k]["score"] = round(max(0.001, min(0.999, r[k].get("score", 0.001))), 3)
    mean = round(max(0.001, min(0.999, sum(v["score"] for v in r.values()) / 3)), 3)
    return {
        "episode_id": DB.ep_id, "grader_results": r, "mean_score": mean,
        "all_passed": all(v["passed"] for v in r.values()), "graded_at": ts(),
    }


@router.get("/validate")
@router.post("/validate")
def validate():
    return {
        "valid": True, "name": "haemorl-organ-allocation", "version": "5.1.0",
        "spec_version": "openenv-1.0", "tasks": list(TASK_INFO.keys()),
        "difficulties":    {k: v["difficulty"] for k, v in TASK_INFO.items()},
        "max_steps":       {k: v["max_steps"]  for k, v in TASK_INFO.items()},
        "baseline_scores": {k: v["baseline"]   for k, v in TASK_INFO.items()},
        "action_space": {"type": "structured", "fields": ["patient_id","donor_id","hospital","action_type"]},
        "observation_space": {"type": "dict", "fields": len(build_observation())},
        "reward_range": [-1.0, 1.0], "shaped_reward": True, "partial_progress": True,
        "reward_components": 8,
    }


# ── RL/AI Analytics endpoints ─────────────────────────────────────────────────

@router.get("/api/rl/matches")
def rl_matches(n: int = 5):
    crit = [p for p in DB.patients.values() if p.get("urgency") == "critical" and not p.get("is_allocated")]
    avd  = [d for d in DB.donors.values() if d.get("available")]
    if not crit or not avd:
        return {"matches": []}
    hospitals = list(DB.hospitals.values())
    results = []
    for p in crit[:n]:
        for d in avd[:4]:
            bh = min(hospitals, key=lambda h: h.get("load_pct", 100)) if hospitals else {"name": "AIIMS New Delhi"}
            rew = compute_reward(p, d, bh.get("name", ""), DB.ep_expired, len(crit))
            results.append({
                "patient": {k: p.get(k) for k in ("id","name","blood_type","ischaemia_h","is_paediatric","disease","peld_score","meld_score","city","urgency")},
                "donor":   {k: d.get(k) for k in ("id","name","blood_type","hospital")},
                "donor_organs": [o["organ"] for o in d.get("organs", []) if o.get("status") == "available"],
                "hospital": bh.get("name"),
                "rl_score": rew["value"],
                "hla_score": rew["hla_score"],
                "blood_ok": rew["blood_ok"],
                "survival_pct": rew["survival_pct"],
                "breakdown": rew["breakdown"],
                "explanation": rew["explanation"],
            })
    results.sort(key=lambda x: x["rl_score"], reverse=True)
    return {"matches": results[:n]}


@router.get("/api/rl/smart-decide")
def smart_decide():
    crit = [p for p in DB.patients.values() if p.get("urgency") == "critical" and not p.get("is_allocated")]
    avd  = [d for d in DB.donors.values() if d.get("available")]
    if not crit or not avd:
        return {"decision": None, "reason": "No critical patients or donors"}

    hospitals = list(DB.hospitals.values())
    best_score = -999.0
    best_p = best_d = best_rew = best_bh = None
    for p in crit[:10]:
        for d in avd[:10]:
            bh = min(hospitals, key=lambda h: h.get("load_pct", 100)) if hospitals else {"name": "AIIMS New Delhi"}
            rew = compute_reward(p, d, bh.get("name", ""), DB.ep_expired, len(crit))
            if rew["value"] > best_score:
                best_score = rew["value"]
                best_p, best_d, best_rew, best_bh = p, d, rew, bh

    if not best_p:
        return {"decision": None, "reason": "No valid match found"}

    return {
        "decision": {
            "patient_id": best_p["id"], "donor_id": best_d["id"],
            "hospital": best_bh.get("name"), "action_type": "match_organ",
        },
        "predicted_reward": best_score,
        "blood_ok": best_rew["blood_ok"],
        "hla_score": best_rew["hla_score"],
        "survival_pct": best_rew["survival_pct"],
        "breakdown": best_rew["breakdown"],
        "explanation": best_rew["explanation"],
        "patient": {k: best_p.get(k) for k in ("name","disease","urgency","ischaemia_h","peld_score","city","blood_type","age")},
        "donor":   {k: best_d.get(k) for k in ("name","blood_type","hospital")},
        "donor_organs": [o["organ"] for o in best_d.get("organs", []) if o.get("status") == "available"],
    }


@router.post("/api/rl/run-episode")
async def run_episode(body: dict):
    task = body.get("task", "crisis_routing")
    if task not in TASK_INFO:
        task = "crisis_routing"
    info = TASK_INFO[task]
    steps = []
    cum = 0.0
    entropies = []

    pts  = [p for p in DB.patients.values() if p.get("urgency") == "critical" and not p.get("is_allocated")]
    dons = [d for d in DB.donors.values() if d.get("available")]
    hospitals = list(DB.hospitals.values())
    max_s = min(info["max_steps"], max(len(pts), 1), 20)

    for i in range(max_s):
        if not pts or not dons:
            break
        p = pts[i % len(pts)]
        d = dons[i % len(dons)]
        bh = min(hospitals, key=lambda h: h.get("load_pct", 100)) if hospitals else {"name": "AIIMS New Delhi"}
        rew = compute_reward(p, d, bh.get("name", ""), 0, len(pts))
        cum += rew["value"]
        scores = [compute_reward(p, dd, bh.get("name", ""), 0, len(pts))["value"] for dd in dons[:4]]
        ent = policy_entropy(scores)
        entropies.append(ent)
        steps.append({
            "step": i + 1,
            "action": f"{p.get('name','?')[:12]}→{d.get('name','?')[:12]}",
            "detail": (
                f"Blood:{p.get('blood_type')}/{d.get('blood_type')} "
                f"HLA:{int(rew['hla_score'] * 100)}% "
                f"Isch:{p.get('ischaemia_h', 0):.1f}h"
                + (" PAED" if p.get("is_paediatric") else "")
                + (f" PELD={p.get('peld_score')}" if p.get("peld_score") else "")
            ),
            "reward": rew["value"],
            "cum": round(cum, 3),
            "hla_score": rew["hla_score"],
            "blood_ok": rew["blood_ok"],
            "survival_pct": rew["survival_pct"],
            "is_paediatric": p.get("is_paediatric", False),
            "policy_entropy": ent,
            "breakdown": rew["breakdown"],
        })

    # Score from actual performance (no randomisation)
    avg_r = cum / max(len(steps), 1)
    base = info["baseline"]
    score = round(max(0.001, min(0.999, base * 0.5 + (avg_r + 1) / 2 * 0.5)), 3)

    return {
        "task": task, "difficulty": info["difficulty"], "steps": steps,
        "total_reward": round(cum, 3), "avg_reward": round(avg_r, 3),
        "score": score, "success": score >= 0.30,
        "rewards":     [s["reward"] for s in steps],
        "avg_entropy": round(sum(entropies) / max(len(entropies), 1), 3),
    }


@router.get("/api/rl/episode-summary")
def episode_summary():
    allocs = list(DB.allocations.values())
    rewards    = [a.get("reward", {}).get("value", 0) for a in allocs]
    hla_scores = [a.get("hla_score", 0) for a in allocs]
    paed       = [a for a in allocs if a.get("is_paediatric")]
    blood_compat = [a for a in allocs if a.get("reward", {}).get("blood_ok", False)]
    return {
        "episode_id": DB.ep_id, "task": DB.ep_task,
        "step": DB.ep_step, "max_steps": DB.ep_maxsteps,
        "cumulative_reward": round(DB.ep_cum, 3),
        "expired_organs": DB.ep_expired,
        "total_allocations": len(allocs),
        "avg_reward":  round(sum(rewards) / max(len(rewards), 1), 3),
        "min_reward":  round(min(rewards, default=0), 3),
        "max_reward":  round(max(rewards, default=0), 3),
        "avg_hla_score": round(sum(hla_scores) / max(len(hla_scores), 1), 3),
        "paediatric_allocations": len(paed),
        "blood_compatible_pct": round(len(blood_compat) / max(len(allocs), 1) * 100, 1),
        "policy_entropy": DB.entropy_log[-1] if DB.entropy_log else 0.0,
        "entropy_trend": list(DB.entropy_log)[-5:],
        "gini_index": DB.fairness_log[-1].get("gini_wait", 0) if DB.fairness_log else 0.0,
    }


@router.get("/api/rl/fairness")
def rl_fairness():
    pts = list(DB.patients.values())
    allocs = list(DB.allocations.values())
    allocated_pts = [DB.patients.get(a.get("patient_id"), {}) for a in allocs if a.get("status") in ("complete","active","pending")]
    cat_alloc = {cat: sum(1 for p in allocated_pts if p.get("category") == cat) for cat in DISEASE_DB.keys()}
    cat_total = {cat: sum(1 for p in pts          if p.get("category") == cat) for cat in DISEASE_DB.keys()}
    cat_rate  = {cat: round(cat_alloc.get(cat, 0) / max(cat_total.get(cat, 1), 1) * 100, 1) for cat in DISEASE_DB.keys()}
    ages  = [p.get("age", 40)       for p in allocated_pts]
    waits = [p.get("wait_months", 0) for p in allocated_pts]
    paed_total = sum(1 for p in pts            if p.get("is_paediatric"))
    paed_alloc = sum(1 for p in allocated_pts  if p.get("is_paediatric"))
    return {
        "gini_age":  gini_coefficient(ages),
        "gini_wait": gini_coefficient(waits),
        "paed_allocation_rate": round(paed_alloc / max(paed_total, 1) * 100, 1),
        "paed_total": paed_total, "paed_allocated": paed_alloc,
        "category_allocation_rates": cat_rate,
        "category_totals":   cat_total,
        "category_allocated": cat_alloc,
        "fairness_history": list(DB.fairness_log)[-10:],
        "equity_score": round(1 - gini_coefficient(list(cat_rate.values())), 3),
    }


@router.get("/api/forecast/demand")
def forecast_demand():
    import random
    pts  = list(DB.patients.values())
    crit = [p for p in pts if p.get("urgency") == "critical"]
    cat_demand = {cat: sum(1 for p in crit if p.get("category") == cat) for cat in DISEASE_DB.keys()}
    expiring_6h = sum(1 for p in crit if 0 < p.get("ischaemia_h", 999) <= 6)
    forecast = []
    for h in range(1, 25, 6):
        new_trauma = random.randint(0, 2)
        forecast.append({
            "hour": h,
            "estimated_critical": max(0, len(crit) - h // 2 + new_trauma),
            "new_trauma": new_trauma,
            "organs_expiring": max(0, expiring_6h - h // 6),
        })
    return {
        "current_critical": len(crit),
        "current_donors": sum(1 for d in DB.donors.values() if d.get("available")),
        "24h_forecast": forecast,
        "high_demand_categories": sorted(cat_demand.items(), key=lambda x: x[1], reverse=True)[:5],
        "supply_gap": max(0, len(crit) - sum(1 for d in DB.donors.values() if d.get("available"))),
        "recommendations": [
            "Activate regional donor networks" if len(crit) > 20 else "Monitor ischaemia clocks",
            "Prioritise paediatric cases" if sum(1 for p in crit if p.get("is_paediatric")) > 3 else "Standard allocation protocol",
        ],
    }
