"""
HaemoRL — Patient routes
GET  /api/patients          — list with filtering & pagination
GET  /api/patients/{pid}    — single patient
POST /api/patients          — create
DELETE /api/patients/{pid}  — delete
GET  /api/patients/{pid}/hla-matches
GET  /api/patients/{pid}/timeline
"""
from __future__ import annotations
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from core.database import DB, make_patient, save, ts
from core.constants import DISEASE_DB, BLOOD_TYPES
from services.allocation import release_allocation
from services.websocket import manager

router = APIRouter(prefix="/api/patients", tags=["patients"])

def _save_file():
    from pathlib import Path
    return Path(os.getenv("DATA_DIR", "")) / "haemorl_db.json" if os.getenv("DATA_DIR") else Path(__file__).parent.parent / "data" / "haemorl_db.json"


@router.get("")
def list_patients(
    page:      int  = Query(1, ge=1),
    per_page:  int  = Query(20, ge=1, le=100),
    urgency:   str  = Query(None),
    category:  str  = Query(None),
    need_type: str  = Query(None),
    q:         str  = Query(None),
    kpe_only:  bool = Query(False),
):
    pts = list(DB.patients.values())
    if urgency   and urgency   != "all": pts = [p for p in pts if p.get("urgency")   == urgency]
    if category  and category  != "all": pts = [p for p in pts if p.get("category")  == category]
    if need_type and need_type != "all": pts = [p for p in pts if p.get("need_type") == need_type]
    if kpe_only:  pts = [p for p in pts if p.get("kpe_eligible") and not p.get("is_allocated")]
    if q:
        ql = q.lower()
        pts = [p for p in pts if any(
            ql in str(p.get(f, "")).lower()
            for f in ("name", "id", "disease", "category", "city")
        )]
    pts = sorted(
        pts,
        key=lambda p: (
            {"critical": 0, "urgent": 1, "moderate": 2, "stable": 3}.get(p.get("urgency", "stable"), 4),
            p.get("ischaemia_h", 999),
        ),
    )
    total = len(pts)
    start = (page - 1) * per_page
    return {
        "total": total, "page": page, "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
        "patients": pts[start: start + per_page],
    }


@router.get("/{pid}")
def get_patient(pid: str):
    p = DB.patients.get(pid)
    if not p:
        raise HTTPException(404, f"Patient {pid} not found")
    return p


@router.post("", status_code=201)
async def create_patient(body: dict, bg: BackgroundTasks):
    cat = body.get("category", "oncology")
    if cat not in DISEASE_DB:
        cat = "oncology"
    if body.get("urgency") is not None and body["urgency"] not in ("critical", "urgent", "moderate", "stable"):
        raise HTTPException(422, "urgency must be one of: critical, urgent, moderate, stable")
    if body.get("blood_type") is not None and body["blood_type"] not in BLOOD_TYPES:
        raise HTTPException(422, f"blood_type must be one of: {', '.join(BLOOD_TYPES)}")
    if body.get("age") is not None:
        try:
            body["age"] = int(body["age"])
        except (TypeError, ValueError):
            raise HTTPException(422, "age must be a whole number")
        if not 0 <= body["age"] <= 120:
            raise HTTPException(422, "age must be between 0 and 120")
    p = make_patient(cat, body.get("urgency"))
    for key in ("name", "age", "gender", "blood_type", "disease", "urgency", "hospital"):
        if body.get(key) is not None:
            p[key] = body[key]
    if body.get("age") is not None:
        p["is_paediatric"] = int(body["age"]) <= 17
    p["added_by"] = body.get("added_by", "user")
    DB.patients[p["id"]] = p

    if p.get("urgency") == "critical":
        DB.alerts.append({
            "id": DB.next_alert_id(), "severity": "critical",
            "title": f"New Critical: {p['name']}",
            "message": f"{p.get('disease')} — {p.get('ischaemia_h', 0):.1f}h ischaemia",
            "patient_id": p["id"], "created_at": ts(), "resolved": False,
        })

    save(_save_file())
    bg.add_task(manager.broadcast, {
        "event": "patient_added", "patient_id": p["id"],
        "urgency": p.get("urgency"), "name": p.get("name"), "ts": ts(),
    })
    return p


@router.delete("/{pid}")
async def delete_patient(pid: str, bg: BackgroundTasks):
    if pid not in DB.patients:
        raise HTTPException(404, f"Patient {pid} not found")
    # Cancel any allocation holding this patient so its donor is freed
    for a in DB.allocations.values():
        if a.get("patient_id") == pid and a.get("status") not in ("failed", "cancelled"):
            release_allocation(a)
            a["status"] = "cancelled"
            a["updated_at"] = ts()
    del DB.patients[pid]
    save(_save_file())
    bg.add_task(manager.broadcast, {"event": "patient_removed", "patient_id": pid, "ts": ts()})
    return {"deleted": pid}


@router.get("/{pid}/hla-matches")
def hla_matches(pid: str, top: int = Query(5, ge=1, le=20)):
    from core.database import hla_score, blood_ok
    p = DB.patients.get(pid)
    if not p:
        raise HTTPException(404, f"Patient {pid} not found")
    avd = [d for d in DB.donors.values() if d.get("available")]
    sc = [
        {
            "donor_id": d["id"], "donor_name": d.get("name"),
            "hla_score": round(hla_score(p.get("hla", {}), d.get("hla", {})), 3),
            "blood_ok": blood_ok(d.get("blood_type", ""), p.get("blood_type", "")),
            "donor_blood": d.get("blood_type"),
            "organs": [o["organ"] for o in d.get("organs", []) if o.get("status") == "available"],
            "hospital": d.get("hospital"),
        }
        for d in avd
    ]
    sc.sort(key=lambda x: x["hla_score"] * 0.6 + (0.4 if x["blood_ok"] else 0), reverse=True)
    return {"patient_id": pid, "matches": sc[:top]}


@router.get("/{pid}/timeline")
def patient_timeline(pid: str):
    p = DB.patients.get(pid)
    if not p:
        raise HTTPException(404, f"Patient {pid} not found")
    alloc = DB.allocations.get(p.get("allocation_id", ""))
    events = [{"ts": p.get("admitted_at", ts()), "event": "Patient registered",
               "detail": f"Admitted with {p.get('disease')}", "type": "info"}]
    if p.get("urgency") == "critical":
        events.append({"ts": p.get("admitted_at", ts()), "event": "Critical status assigned",
                       "detail": f"Ischaemia window: {p.get('ischaemia_total', 24)}h", "type": "critical"})
    if alloc:
        events.append({"ts": alloc.get("created_at", ts()), "event": "Organ allocated",
                       "detail": f"{alloc.get('organ')} from {alloc.get('donor_name')} at {alloc.get('hospital')}",
                       "type": "success", "reward": alloc.get("reward", {}).get("value")})
    return {"patient": p, "timeline": events, "allocation": alloc}
