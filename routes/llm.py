"""
HaemoRL — LLM / Chat routes
POST /api/llm/decide       — AI allocation decision
GET  /api/llm/log          — LLM decision log
GET  /api/llm/models       — available models

POST /api/chat             — HaemoBot message
GET  /api/chat/history     — conversation history
DELETE /api/chat/history   — clear history
"""
from __future__ import annotations
import os
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel

from core.database import DB, ts
from services.allocation import build_observation, compute_reward
from services.websocket import manager

router = APIRouter(prefix="/api", tags=["llm"])


def _save():
    from core.database import save
    d = os.getenv("DATA_DIR", "")
    p = (Path(d) / "haemorl_db.json") if d else (Path(__file__).parent.parent / "data" / "haemorl_db.json")
    save(p)


def _get_config():
    from services.llm import llm_config
    return llm_config()


def _enabled(cfg: dict) -> bool:
    return all(cfg.values())


# Simple in-memory rate limiter
_rate_limits: dict[str, list] = {}

def _rate_ok(key: str, max_per_minute: int = 20) -> bool:
    now = time.time()
    if key not in _rate_limits:
        _rate_limits[key] = []
    _rate_limits[key] = [t for t in _rate_limits[key] if now - t < 60]
    if len(_rate_limits[key]) >= max_per_minute:
        return False
    _rate_limits[key].append(now)
    return True


# ── AI Decision ───────────────────────────────────────────────────────────────

@router.post("/llm/decide")
async def llm_decide(bg: BackgroundTasks):
    cfg = _get_config()
    from services.llm import ai_decide
    obs = build_observation()
    result = await ai_decide(obs, list(DB.patients.values()), list(DB.donors.values()), **cfg)

    DB.llm_log.append({
        "ts": ts(), "mode": result.get("mode"),
        "reasoning": result.get("reasoning", ""),
        "confidence": result.get("confidence", 0),
        "action": result.get("action", {}),
        "priority_factors": result.get("priority_factors", []),
    })

    action = result.get("action", {})
    committed = None
    if action.get("patient_id") and action.get("action_type") != "skip":
        p = DB.patients.get(action.get("patient_id", ""))
        d = DB.donors.get(action.get("donor_id", "")) if action.get("donor_id") else None
        if p and not p.get("is_allocated") and (not d or d.get("available")):
            crit_count = sum(1 for x in DB.patients.values() if x.get("urgency") == "critical")
            rew = compute_reward(p, d, action.get("hospital", ""), DB.ep_expired, crit_count)
            aid = DB.next_alloc_id()
            alloc = {
                "id": aid,
                "patient_id": p["id"],    "patient_name": p.get("name"),
                "donor_id": d["id"] if d else None,
                "donor_name": d.get("name") if d else None,
                "organ": d.get("organs", [{"organ": "Unknown"}])[0].get("organ") if d else p.get("organ_needed"),
                "hospital": action.get("hospital"),
                "action_type": "match_organ", "status": "active",
                "reward": rew, "hla_score": rew["hla_score"],
                "is_paediatric": p.get("is_paediatric", False),
                "peld_score": p.get("peld_score"),
                "step": DB.ep_step, "created_at": ts(),
                "source": "llm_decision",
                "llm_reasoning": result.get("reasoning", ""),
                "disease": p.get("disease"),
            }
            DB.allocations[aid] = alloc
            DB.ep_cum += rew["value"]
            if d:
                d["available"] = False
            p["is_allocated"] = True
            p["allocation_id"] = aid
            committed = alloc

    _save()
    await manager.broadcast({
        "event": "llm_decision", "mode": result.get("mode"),
        "reasoning": result.get("reasoning", ""),
        "committed": committed is not None, "ts": ts(),
    })
    return {"decision": result, "committed_allocation": committed}


@router.get("/llm/log")
def llm_log(limit: int = Query(20, ge=1, le=100)):
    cfg = _get_config()
    return {
        "total": len(DB.llm_log),
        "log": list(DB.llm_log)[-limit:],
        "has_llm": _enabled(cfg),
        "model": cfg["model"] if _enabled(cfg) else "rule_based",
    }


@router.get("/llm/models")
def llm_models():
    cfg = _get_config()
    on = _enabled(cfg)
    return {
        "models": [{"id": cfg["model"], "name": cfg["model"]}] if on else [],
        "current":  cfg["model"] if on else "rule_based",
        "has_key": on,
    }


# ── HaemoBot Chat ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    session_id: str = "global"


@router.post("/chat")
async def chat(body: ChatMessage, bg: BackgroundTasks, request: Request):
    client_ip = request.client.host if request.client else "anon"
    if not _rate_ok(f"chat_{client_ip}", 20):
        raise HTTPException(429, "Too many requests. Please wait a moment.")

    cfg = _get_config()
    pts  = list(DB.patients.values())
    crit = [p for p in pts if p.get("urgency") == "critical"]

    context = (
        f"Patients:{len(pts)} Critical:{len(crit)} "
        f"Paed-crit:{sum(1 for p in crit if p.get('is_paediatric'))} "
        f"Allocations:{len(DB.allocations)} "
        f"Donors-avail:{sum(1 for d in DB.donors.values() if d.get('available'))} "
        f"Expired:{DB.ep_expired} "
        f"KPE:{sum(1 for p in pts if p.get('kpe_eligible') and not p.get('is_allocated'))}"
    )

    session_msgs = [
        {"role": h["role"], "content": h["content"]}
        for h in list(DB.chat_history)[-10:]
        if h.get("session") in (body.session_id, "global")
    ]
    session_msgs.append({"role": "user", "content": body.message})

    from services.llm import chat as llm_chat
    response = await llm_chat(session_msgs, context, cfg["api_base"], cfg["model"], cfg["token"])

    DB.chat_history.append({"role": "user", "content": body.message, "ts": ts(), "session": body.session_id})
    DB.chat_history.append({"role": "assistant", "content": response,    "ts": ts(), "session": body.session_id})
    _save()
    bg.add_task(manager.broadcast, {"event": "chat", "session": body.session_id, "ts": ts()})

    return {
        "response": response,
        "model": cfg["model"] if _enabled(cfg) else "no_llm",
        "ts": ts(), "session_id": body.session_id,
    }


@router.get("/chat/history")
def chat_history(session_id: str = Query("global"), limit: int = Query(50, ge=1, le=200)):
    cfg = _get_config()
    msgs = [h for h in list(DB.chat_history) if h.get("session") in (session_id, "global")]
    return {
        "messages": msgs[-limit:],
        "total": len(msgs),
        "has_llm": _enabled(cfg),
    }


@router.delete("/chat/history")
async def clear_chat(bg: BackgroundTasks):
    DB.chat_history.clear()
    _save()
    bg.add_task(manager.broadcast, {"event": "chat_cleared", "ts": ts()})
    return {"cleared": True}
