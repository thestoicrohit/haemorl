"""
HaemoRL — Donor & Blood Bank routes
GET  /api/donors             — list donors
GET  /api/donors/{did}       — single donor
POST /api/donors             — register donor
POST /api/donors/{did}/refresh — mark donor available again

GET  /api/blood-bank         — inventory summary
POST /api/blood-bank/add     — add units
POST /api/blood-bank/dispense — dispense units

GET  /api/hospitals          — list hospitals
GET  /api/transport          — organ transport routes
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from core.database import DB, make_donor, save, ts
from core.constants import ORGAN_VIABILITY, BLOOD_TYPES
from services.websocket import manager

router = APIRouter(tags=["donors"])


def _save_file() -> Path:
    d = os.getenv("DATA_DIR", "")
    return (Path(d) / "haemorl_db.json") if d else (Path(__file__).parent.parent / "data" / "haemorl_db.json")


# ── Donors ────────────────────────────────────────────────────────────────────

@router.get("/api/donors")
def list_donors(available_only: bool = Query(False)):
    donors = list(DB.donors.values())
    if available_only:
        donors = [d for d in donors if d.get("available")]
    return {"total": len(donors), "donors": donors}


@router.get("/api/donors/{did}")
def get_donor(did: str):
    d = DB.donors.get(did)
    if not d:
        raise HTTPException(404, f"Donor {did} not found")
    return d


@router.post("/api/donors", status_code=201)
async def create_donor(body: dict, bg: BackgroundTasks):
    d = make_donor()
    for key in ("name", "age", "blood_type", "donor_type", "hospital"):
        if body.get(key):
            d[key] = body[key]
    if body.get("organs"):
        d["organs"] = [
            {"organ": o, "status": "available", "viability_h": ORGAN_VIABILITY.get(o, 24), "harvested_at": ts()}
            for o in body["organs"]
        ]
    d["added_by"] = body.get("added_by", "user")
    DB.donors[d["id"]] = d
    save(_save_file())
    bg.add_task(manager.broadcast, {"event": "donor_added", "donor_id": d["id"], "ts": ts()})
    return d


@router.post("/api/donors/{did}/refresh")
async def refresh_donor(did: str, bg: BackgroundTasks):
    d = DB.donors.get(did)
    if not d:
        raise HTTPException(404, f"Donor {did} not found")
    d["available"] = True
    for o in d.get("organs", []):
        o["status"] = "available"
    save(_save_file())
    bg.add_task(manager.broadcast, {"event": "donor_refreshed", "donor_id": did, "ts": ts()})
    return d


# ── Blood Bank ────────────────────────────────────────────────────────────────

@router.get("/api/blood-bank")
def blood_bank():
    return {
        "summaries": list(DB.blood_bank.values()),
        "total_units": sum(b.get("units", 0) for b in DB.blood_bank.values()),
        "critical_low": [bt for bt, b in DB.blood_bank.items() if b.get("units", 0) < 30],
        "expiring_soon": [bt for bt, b in DB.blood_bank.items() if b.get("expiry_days", 99) < 7],
    }


@router.post("/api/blood-bank/add")
async def add_blood(body: dict, bg: BackgroundTasks):
    bt    = body.get("blood_type", "O+")
    units = int(body.get("units", 0))
    if not units:
        raise HTTPException(400, "units required")
    if bt not in BLOOD_TYPES:
        raise HTTPException(400, f"Invalid blood type: {bt}")
    if bt in DB.blood_bank:
        DB.blood_bank[bt]["units"] += units
    else:
        DB.blood_bank[bt] = {
            "blood_type": bt, "units": units,
            "component": body.get("component", "Whole Blood"),
            "expiry_days": int(body.get("expiry_days", 42)),
        }
    save(_save_file())
    bg.add_task(manager.broadcast, {"event": "blood_added", "blood_type": bt, "units": units, "ts": ts()})
    return DB.blood_bank[bt]


@router.post("/api/blood-bank/dispense")
async def dispense_blood(body: dict, bg: BackgroundTasks):
    bt    = body.get("blood_type")
    units = int(body.get("units", 2))
    if not bt or bt not in BLOOD_TYPES:
        raise HTTPException(400, "Valid blood_type required")
    available = DB.blood_bank.get(bt, {}).get("units", 0)
    if available < units:
        raise HTTPException(409, f"Insufficient {bt}: {available} units available, {units} requested")
    DB.blood_bank[bt]["units"] -= units
    save(_save_file())
    bg.add_task(manager.broadcast, {"event": "blood_dispensed", "blood_type": bt, "units": units, "ts": ts()})
    return {"dispensed": units, "blood_type": bt, "remaining": DB.blood_bank[bt]["units"]}


# ── Hospitals & Transport ─────────────────────────────────────────────────────

@router.get("/api/hospitals")
def hospitals():
    return {"hospitals": list(DB.hospitals.values()), "total": len(DB.hospitals)}


@router.get("/api/transport")
def transport(active_only: bool = Query(False)):
    routes = DB.routes if not active_only else [r for r in DB.routes if r.get("active")]
    return {
        "total": len(routes),
        "active": sum(1 for r in DB.routes if r.get("active")),
        "routes": sorted(routes, key=lambda r: r.get("hours", 0)),
    }


@router.get("/api/organs/viability")
def organ_viability():
    avd = [d for d in DB.donors.values() if d.get("available")]
    organs = []
    for d in avd:
        for o in d.get("organs", []):
            if o.get("status") == "available":
                organs.append({
                    "donor_id": d["id"], "donor_name": d.get("name"),
                    "organ": o["organ"], "viability_h": o.get("viability_h", 24),
                    "blood_type": d.get("blood_type"), "hospital": d.get("hospital"),
                    "harvested_at": o.get("harvested_at"), "status": "available",
                })
    organs.sort(key=lambda o: o["viability_h"])
    return {
        "total": len(organs),
        "critical_viability": [o for o in organs if o["viability_h"] <= 4],
        "warning_viability":  [o for o in organs if 4 < o["viability_h"] <= 12],
        "ok_viability":       [o for o in organs if o["viability_h"] > 12],
        "organs": organs,
    }


@router.get("/api/kpe/candidates")
def kpe_candidates():
    from core.database import hla_score, blood_ok
    kpe_pts = [p for p in DB.patients.values() if p.get("kpe_eligible") and not p.get("is_allocated")]
    chains = []
    for i in range(0, min(len(kpe_pts) - 1, 10), 2):
        p1 = kpe_pts[i]
        p2 = kpe_pts[i + 1] if i + 1 < len(kpe_pts) else None
        if not p2:
            break
        compat_12 = blood_ok(p2.get("blood_type", ""), p1.get("blood_type", ""))
        compat_21 = blood_ok(p1.get("blood_type", ""), p2.get("blood_type", ""))
        hla_cross  = round((hla_score(p1.get("hla", {}), p2.get("hla", {})) +
                            hla_score(p2.get("hla", {}), p1.get("hla", {}))) / 2, 3)
        chains.append({
            "chain_id": f"KPE-{i // 2 + 1:03d}",
            "pair": [
                {"patient_id": p1["id"], "name": p1.get("name"), "blood": p1.get("blood_type"), "city": p1.get("city")},
                {"patient_id": p2["id"], "name": p2.get("name"), "blood": p2.get("blood_type"), "city": p2.get("city")},
            ],
            "cross_compatible": compat_12 and compat_21,
            "hla_cross_score": hla_cross,
            "benefit_score": round(hla_cross * 0.6 + (0.4 if compat_12 and compat_21 else 0.2), 3),
        })
    chains.sort(key=lambda c: c["benefit_score"], reverse=True)
    return {
        "total_eligible": len(kpe_pts),
        "chains": chains,
        "description": "Kidney Paired Exchange — incompatible donor-recipient pairs that can swap for mutual benefit",
    }
