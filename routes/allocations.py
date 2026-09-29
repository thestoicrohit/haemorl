"""
HaemoRL — Allocation routes
GET  /api/allocations          — list allocations
POST /api/allocations/commit   — manual commit
POST /api/allocations/auto-match — automatic batch matching
PATCH /api/allocations/{aid}   — update status
GET  /api/hla/matrix           — HLA compatibility matrix
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from core.database import DB, ts, hla_score, blood_ok
from services.allocation import compute_reward, find_best_match, release_allocation
from services.websocket import manager

router = APIRouter(prefix="/api", tags=["allocations"])


def _save():
    from core.database import save
    d = os.getenv("DATA_DIR", "")
    p = (Path(d) / "haemorl_db.json") if d else (Path(__file__).parent.parent / "data" / "haemorl_db.json")
    save(p)


VALID_STATUSES = {"pending", "active", "complete", "failed", "cancelled"}


@router.get("/allocations")
def list_allocations(status: str = Query(None)):
    allocs = list(DB.allocations.values())
    if status and status != "all":
        allocs = [a for a in allocs if a.get("status") == status]
    allocs.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return {
        "total":    len(allocs),
        "pending":  sum(1 for a in allocs if a.get("status") == "pending"),
        "active":   sum(1 for a in allocs if a.get("status") == "active"),
        "complete": sum(1 for a in allocs if a.get("status") == "complete"),
        "failed":   sum(1 for a in allocs if a.get("status") == "failed"),
        "allocations": allocs,
    }


@router.post("/allocations/commit")
async def commit_allocation(body: dict, bg: BackgroundTasks):
    pid  = body.get("patient_id")
    did  = body.get("donor_id")
    hosp = body.get("hospital")

    p = DB.patients.get(pid)
    if not p:
        raise HTTPException(404, f"Patient {pid} not found")
    if p.get("is_allocated"):
        raise HTTPException(409, "Patient already has an active allocation")

    d = DB.donors.get(did) if did else None
    if d and not d.get("available"):
        raise HTTPException(409, "Donor is not currently available")

    crit_count = sum(1 for x in DB.patients.values() if x.get("urgency") == "critical")
    rew = compute_reward(p, d, hosp or "", DB.ep_expired, crit_count, body.get("action_type", "match_organ"))
    aid = DB.next_alloc_id()

    alloc = {
        "id": aid,
        "patient_id": p["id"],       "patient_name": p.get("name"),
        "donor_id":   d["id"] if d else None,
        "donor_name": d.get("name") if d else None,
        "organ":  d.get("organs", [{"organ": "Unknown"}])[0].get("organ") if d else p.get("organ_needed"),
        "hospital": hosp,
        "action_type": body.get("action_type", "match_organ"),
        "status": "pending",
        "reward": rew,
        "hla_score": rew["hla_score"],
        "is_paediatric": p.get("is_paediatric", False),
        "peld_score": p.get("peld_score"),
        "step": DB.ep_step,
        "created_at": ts(),
        "disease": p.get("disease"),
        "source": "manual",
    }

    DB.allocations[aid] = alloc
    DB.ep_cum += rew["value"]
    if d:
        d["available"] = False
    p["is_allocated"] = True
    p["allocation_id"] = aid

    _save()
    bg.add_task(manager.broadcast, {
        "event": "allocation_created", "allocation_id": aid,
        "score": rew["value"], "ts": ts(),
    })
    return alloc


@router.post("/allocations/auto-match")
async def auto_match(bg: BackgroundTasks):
    crit = [p for p in DB.patients.values() if p.get("urgency") == "critical" and not p.get("is_allocated")]
    avd  = [d for d in DB.donors.values() if d.get("available")]
    if not crit or not avd:
        return {"matched": 0, "allocations": [], "message": "No critical patients or donors available"}

    hospitals = list(DB.hospitals.values())
    made = []
    for p in crit[:6]:
        if not avd:
            break
        best_d, best_h, best_rew = find_best_match(p, avd, hospitals, DB.ep_expired, len(crit))
        if not best_d:
            continue
        avd = [d for d in avd if d["id"] != best_d["id"]]
        aid = DB.next_alloc_id()
        alloc = {
            "id": aid,
            "patient_id": p["id"],        "patient_name": p.get("name"),
            "donor_id":   best_d["id"],   "donor_name":   best_d.get("name"),
            "organ": best_d.get("organs", [{"organ": "Unknown"}])[0].get("organ"),
            "hospital": best_h.get("name") if best_h else None,
            "action_type": "match_organ", "status": "pending",
            "reward": best_rew, "hla_score": best_rew["hla_score"],
            "is_paediatric": p.get("is_paediatric", False),
            "peld_score": p.get("peld_score"),
            "step": DB.ep_step, "created_at": ts(),
            "disease": p.get("disease"), "source": "auto_match",
        }
        DB.allocations[aid] = alloc
        best_d["available"] = False
        p["is_allocated"] = True
        p["allocation_id"] = aid
        DB.ep_cum += best_rew["value"]
        made.append(alloc)

    _save()
    bg.add_task(manager.broadcast, {"event": "auto_match", "count": len(made), "ts": ts()})
    return {"matched": len(made), "allocations": made}


@router.patch("/allocations/{aid}")
async def update_allocation(aid: str, body: dict, bg: BackgroundTasks):
    a = DB.allocations.get(aid)
    if not a:
        raise HTTPException(404, f"Allocation {aid} not found")
    new_status = body.get("status")
    if new_status not in VALID_STATUSES:
        raise HTTPException(400, f"Invalid status. Must be one of: {', '.join(sorted(VALID_STATUSES))}")
    closed = ("failed", "cancelled")
    if a.get("status") in closed and new_status not in closed:
        raise HTTPException(409, "Allocation is closed; create a new allocation instead")
    if new_status in closed and a.get("status") not in closed:
        release_allocation(a)   # the donor and patient go back into the pool
    a["status"] = new_status
    a["updated_at"] = ts()
    if new_status == "complete":
        a["completed_at"] = ts()
    _save()
    bg.add_task(manager.broadcast, {"event": "allocation_updated", "id": aid, "status": new_status, "ts": ts()})
    return a


@router.get("/hla/matrix")
def hla_matrix(n_patients: int = Query(5, ge=1, le=20), n_donors: int = Query(5, ge=1, le=20)):
    pts  = [p for p in DB.patients.values() if p.get("urgency") == "critical"][:n_patients]
    dons = [d for d in DB.donors.values() if d.get("available")][:n_donors]
    return {
        "donors": [{"id": d["id"], "name": d.get("name"), "blood_type": d.get("blood_type")} for d in dons],
        "matrix": [
            {
                "patient_id": p["id"], "patient_name": p.get("name"),
                "blood_type": p.get("blood_type"),
                "is_paediatric": p.get("is_paediatric"),
                "disease": p.get("disease"),
                "peld_score": p.get("peld_score"),
                "scores": {
                    d["id"]: {
                        "hla_pct": int(hla_score(p.get("hla", {}), d.get("hla", {})) * 100),
                        "blood_ok": blood_ok(d.get("blood_type", ""), p.get("blood_type", "")),
                        "crossmatch": p.get("crossmatch_status", "unknown"),
                    }
                    for d in dons
                },
            }
            for p in pts
        ],
    }
