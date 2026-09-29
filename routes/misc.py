"""
HaemoRL — Miscellaneous routes
GET  /ping
GET  /health
GET  /api/stats
GET  /api/dashboard
GET  /api/analytics
GET  /api/alerts
POST /api/alerts/{aid}/resolve
GET  /api/disease-catalog
GET  /openenv.yaml

POST /api/admin/inject-trauma
POST /api/admin/reseed
GET  /api/admin/stats
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import PlainTextResponse

from core.database import DB, make_patient, ts, seed as _seed, init_hospitals, init_routes
from core.constants import DISEASE_DB, TASK_INFO, BLOOD_TYPES
from services.llm import llm_enabled
from services.websocket import manager

router = APIRouter(tags=["misc"])


def _save():
    from core.database import save
    d = os.getenv("DATA_DIR", "")
    p = (Path(d) / "haemorl_db.json") if d else (Path(__file__).parent.parent / "data" / "haemorl_db.json")
    save(p)


# ── System ────────────────────────────────────────────────────────────────────

@router.get("/ping")
def ping():
    return {"pong": True, "ts": ts(), "version": "5.1.0"}


@router.get("/health")
def health():
    d = os.getenv("DATA_DIR", "")
    save_file = (Path(d) / "haemorl_db.json") if d else (Path(__file__).parent.parent / "data" / "haemorl_db.json")
    return {
        "status": "healthy", "ts": ts(), "version": "5.1.0",
        "patients":  len(DB.patients),
        "critical":  sum(1 for p in DB.patients.values() if p.get("urgency") == "critical"),
        "donors":    len(DB.donors),
        "allocations": len(DB.allocations),
        "llm_enabled": llm_enabled(),
        "model": os.getenv("LLM_MODEL") if llm_enabled() else "rule_based",
        "persistent_storage": save_file.exists(),
        "save_file_size_kb": round(save_file.stat().st_size / 1024, 1) if save_file.exists() else 0,
        "hospitals": len(DB.hospitals),
        "kpe_candidates": sum(1 for p in DB.patients.values() if p.get("kpe_eligible") and not p.get("is_allocated")),
        "reward_components": 8,
    }


@router.get("/api/stats")
def api_stats():
    pts  = list(DB.patients.values())
    crit = [p for p in pts if p.get("urgency") == "critical"]
    return {
        "total_patients": len(pts),
        "critical":  len(crit),
        "urgent":    sum(1 for p in pts if p.get("urgency") == "urgent"),
        "moderate":  sum(1 for p in pts if p.get("urgency") == "moderate"),
        "stable":    sum(1 for p in pts if p.get("urgency") == "stable"),
        "organ_queue":  sum(1 for p in pts if p.get("need_type") == "organ"),
        "blood_queue":  sum(1 for p in pts if p.get("need_type") == "blood"),
        "marrow_queue": sum(1 for p in pts if p.get("need_type") == "marrow"),
        "paediatric":      sum(1 for p in pts  if p.get("is_paediatric")),
        "paed_critical":   sum(1 for p in crit if p.get("is_paediatric")),
        "kpe_candidates":  sum(1 for p in pts  if p.get("kpe_eligible") and not p.get("is_allocated")),
        "blood_units":     sum(b.get("units", 0) for b in DB.blood_bank.values()),
        "donors_total":    len(DB.donors),
        "donors_available": sum(1 for d in DB.donors.values() if d.get("available")),
        "organs_available": sum(
            len([o for o in d.get("organs", []) if o.get("status") == "available"])
            for d in DB.donors.values() if d.get("available")
        ),
        "allocations_total":   len(DB.allocations),
        "allocations_pending": sum(1 for a in DB.allocations.values() if a.get("status") == "pending"),
        "allocations_active":  sum(1 for a in DB.allocations.values() if a.get("status") == "active"),
        "allocations_done":    sum(1 for a in DB.allocations.values() if a.get("status") == "complete"),
        "expired_organs":      DB.ep_expired,
        "hospitals_overloaded": sum(1 for h in DB.hospitals.values() if h.get("load_pct", 0) > 85),
        "ischaemia_danger":  sum(1 for p in crit if p.get("ischaemia_h", 999) <= 2),
        "ischaemia_warning": sum(1 for p in crit if 2 < p.get("ischaemia_h", 999) <= 6),
        "cumulative_reward": round(DB.ep_cum, 3),
        "llm_enabled": llm_enabled(),
        "disease_distribution": {
            cat: sum(1 for p in pts if p.get("category") == cat)
            for cat in DISEASE_DB.keys()
        },
    }


@router.get("/api/dashboard")
def api_dashboard():
    pts = list(DB.patients.values())
    cat_d: dict = {}
    bt_d:  dict = {}
    urg_d = {"critical": 0, "urgent": 0, "moderate": 0, "stable": 0}
    age_d = {"0-12": 0, "13-19": 0, "20-39": 0, "40-59": 0, "60-74": 0, "75+": 0}

    for p in pts:
        cat  = p.get("category", "other")
        bt   = p.get("blood_type", "?")
        urg  = p.get("urgency", "stable")
        age  = p.get("age", 40)
        cat_d[cat] = cat_d.get(cat, 0) + 1
        bt_d[bt]   = bt_d.get(bt, 0) + 1
        urg_d[urg] = urg_d.get(urg, 0) + 1
        if   age <= 12: age_d["0-12"]  += 1
        elif age <= 19: age_d["13-19"] += 1
        elif age <= 39: age_d["20-39"] += 1
        elif age <= 59: age_d["40-59"] += 1
        elif age <= 74: age_d["60-74"] += 1
        else:           age_d["75+"]   += 1

    return {
        "urgency_dist": urg_d,
        "disease_dist": dict(sorted(cat_d.items(), key=lambda x: x[1], reverse=True)),
        "blood_dist":   bt_d,
        "age_dist":     age_d,
        "hospital_loads": [
            {"name": h.get("name"), "load_pct": h.get("load_pct"),
             "beds_available": h.get("beds_available"), "icu_available": h.get("icu_available")}
            for h in sorted(DB.hospitals.values(), key=lambda h: h.get("load_pct", 0), reverse=True)
        ],
        "blood_bank":    list(DB.blood_bank.values()),
        "recent_alerts": [a for a in list(DB.alerts) if not a.get("resolved")][:10],
        "llm_log":       list(DB.llm_log)[-5:],
        "kpe_count":     sum(1 for p in pts if p.get("kpe_eligible") and not p.get("is_allocated")),
        "policy_entropy": list(DB.entropy_log)[-1] if DB.entropy_log else None,
        "gini_index":     DB.fairness_log[-1].get("gini_wait") if DB.fairness_log else None,
    }


@router.get("/api/analytics")
def api_analytics():
    return {
        "history": list(DB.analytics)[-100:],
        "current": {
            "ts": ts(),
            "total": len(DB.patients),
            "critical": sum(1 for p in DB.patients.values() if p.get("urgency") == "critical"),
            "allocations": len(DB.allocations),
            "expired": DB.ep_expired,
        },
        "entropy_log":  list(DB.entropy_log)[-20:],
        "fairness_log": list(DB.fairness_log)[-10:],
    }


@router.get("/api/disease-catalog")
def disease_catalog():
    return {
        "categories": list(DISEASE_DB.keys()),
        "total_diseases": sum(len(v["diseases"]) for v in DISEASE_DB.values()),
        "catalog": {
            k: {
                "diseases": v["diseases"], "treatment": v["treatment"], "organ": v["organ"],
                "symptoms": v["symptoms"], "medications": v["medications"],
            }
            for k, v in DISEASE_DB.items()
        },
    }


# ── Alerts ────────────────────────────────────────────────────────────────────

@router.get("/api/alerts")
def api_alerts(resolved: bool = Query(False), limit: int = Query(50, ge=1, le=200)):
    return {"alerts": [a for a in list(DB.alerts) if a.get("resolved", False) == resolved][:limit]}


@router.post("/api/alerts/{aid}/resolve")
async def resolve_alert(aid: str, bg: BackgroundTasks):
    for a in DB.alerts:
        if a.get("id") == aid:
            a["resolved"] = True
            a["resolved_at"] = ts()
            _save()
            return {"resolved": aid}
    raise HTTPException(404, f"Alert {aid} not found")


# ── Admin ─────────────────────────────────────────────────────────────────────

@router.post("/api/admin/inject-trauma")
async def inject_trauma(bg: BackgroundTasks):
    p = make_patient("trauma", "critical")
    DB.patients[p["id"]] = p
    DB.alerts.append({
        "id": DB.next_alert_id(), "severity": "critical",
        "title": f"TRAUMA ALERT: {p['name']}",
        "message": f"{p.get('disease')} — {p.get('ischaemia_h', 0):.1f}h ischaemia",
        "patient_id": p["id"], "created_at": ts(), "resolved": False,
    })
    _save()
    bg.add_task(manager.broadcast, {"event": "trauma_injected", "patient": p, "ts": ts()})
    return {"injected": True, "patient": p}


@router.post("/api/admin/reseed")
async def reseed(bg: BackgroundTasks):
    DB.patients.clear(); DB.donors.clear(); DB.allocations.clear(); DB.alerts.clear()
    DB._pctr = 0; DB._dctr = 0; DB._actr = 0; DB._alctr = 0
    DB._seeded = False; DB.ep_cum = 0.0; DB.ep_expired = 0
    DB.entropy_log.clear(); DB.fairness_log.clear()
    _seed(); init_hospitals(); init_routes()
    bg.add_task(manager.broadcast, {"event": "reseeded", "patients": len(DB.patients), "ts": ts()})
    return {"reseeded": True, "patients": len(DB.patients), "donors": len(DB.donors)}


@router.get("/api/admin/stats")
def admin_stats():
    d = os.getenv("DATA_DIR", "")
    save_file = (Path(d) / "haemorl_db.json") if d else (Path(__file__).parent.parent / "data" / "haemorl_db.json")
    return {
        "patients": len(DB.patients), "donors": len(DB.donors),
        "allocations": len(DB.allocations), "alerts": len(DB.alerts),
        "chat_messages": len(DB.chat_history), "llm_decisions": len(DB.llm_log),
        "save_file": str(save_file), "file_exists": save_file.exists(),
        "file_size_kb": round(save_file.stat().st_size / 1024, 1) if save_file.exists() else 0,
        "disease_categories": len(DISEASE_DB),
        "total_diseases": sum(len(v["diseases"]) for v in DISEASE_DB.values()),
        "version": "5.1.0", "reward_components": 8, "hospitals": len(DB.hospitals),
    }


# ── Environment spec ───────────────────────────────────────────────────────────

@router.get("/openenv.yaml")
def openenv_yaml():
    """Serve the environment spec file (single source of truth: openenv.yaml)."""
    f = Path(__file__).parent.parent / "openenv.yaml"
    if not f.exists():
        raise HTTPException(404, "openenv.yaml not found")
    return PlainTextResponse(f.read_text(encoding="utf-8"), media_type="text/yaml")
