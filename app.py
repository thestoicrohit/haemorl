"""
HaemoRL v5.1 — Production Backend
AI-powered organ allocation platform for India.

Architecture:
  core/constants.py     — static data (blood types, HLA pool, hospitals, diseases)
  core/database.py      — in-memory DB singleton + seed + persistence
  services/allocation.py — 8-component reward engine + graders
  services/llm.py       — LLM + deterministic fallback
  services/websocket.py — WS connection manager
  routes/patients.py
  routes/donors.py
  routes/allocations.py
  routes/rl.py
  routes/llm.py
  routes/misc.py
"""
from __future__ import annotations
import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("haemorl")

# ── Configuration ──────────────────────────────────────────────────────────────
PORT      = int(os.getenv("PORT", "7860"))
_data_dir = os.getenv("DATA_DIR", "")
DATA_DIR  = Path(_data_dir) if _data_dir else Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVE_FILE = DATA_DIR / "haemorl_db.json"

# Comma-separated allowed origins (default "*" for local dev). Set CORS_ORIGINS
# to a specific list in production, e.g. "https://app.example.com".
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
# When DEBUG is unset, the global error handler hides internal exception text.
DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes", "on")

# ── Lifespan (startup / shutdown) ───────────────────────────────────────────────
# Defined before the app; it references _run_startup() / _background_loop() which
# are declared lower in this module — fine, since the body only runs at startup.
_bg_task: "asyncio.Task | None" = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _bg_task
    await _run_startup()
    _bg_task = asyncio.create_task(_background_loop())
    try:
        yield
    finally:
        if _bg_task:
            _bg_task.cancel()


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HaemoRL",
    version="5.1.0",
    description="AI-powered organ allocation platform for India",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    content = {"error": "Internal server error"}
    if DEBUG:                       # only expose internals when explicitly debugging
        content["detail"] = str(exc)[:200]
    return JSONResponse(status_code=500, content=content)


# ── Register routers ───────────────────────────────────────────────────────────
from routes.patients    import router as patients_router
from routes.donors      import router as donors_router
from routes.allocations import router as allocations_router
from routes.rl          import router as rl_router
from routes.llm         import router as llm_router
from routes.misc        import router as misc_router

app.include_router(patients_router)
app.include_router(donors_router)
app.include_router(allocations_router)
app.include_router(rl_router)
app.include_router(llm_router)
app.include_router(misc_router)


# ── Frontend ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    f = Path(__file__).parent / "index.html"
    if f.exists():
        return HTMLResponse(f.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>HaemoRL v5.1 — index.html not found</h1>", status_code=503)


# ── WebSocket ─────────────────────────────────────────────────────────────────
from services.websocket import manager
from services.llm import llm_enabled
from core.database import DB, ts as _ts


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "event": "connected",
            "patients": len(DB.patients),
            "critical": sum(1 for p in DB.patients.values() if p.get("urgency") == "critical"),
            "llm_enabled": llm_enabled(),
            "version": "5.1.0",
            "ts": _ts(),
        })
        while True:
            await asyncio.sleep(10)
            await websocket.send_json({
                "event": "heartbeat", "ts": _ts(),
                "critical":       sum(1 for p in DB.patients.values() if p.get("urgency") == "critical"),
                "allocations":    len(DB.allocations),
                "expired_organs": DB.ep_expired,
                "kpe": sum(1 for p in DB.patients.values() if p.get("kpe_eligible") and not p.get("is_allocated")),
                "entropy": list(DB.entropy_log)[-1] if DB.entropy_log else 0.0,
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ── Startup ────────────────────────────────────────────────────────────────────
async def _run_startup():
    logger.info("HaemoRL v5.1 starting …")
    from core.database import load, seed, init_hospitals, init_routes
    try:
        loaded = load(SAVE_FILE)
        if not loaded:
            logger.info("No saved state — seeding fresh data …")
            seed()
        else:
            logger.info(f"Loaded {len(DB.patients)} patients from {SAVE_FILE}")
        init_hospitals()
        init_routes()
        from services.allocation import revive_stalled_clocks
        revived = revive_stalled_clocks()
        if revived:
            logger.info(f"Renewed {revived} ischaemia clocks that were stuck at 0h")
    except Exception as e:
        logger.critical(f"Startup error: {e}")
        traceback.print_exc()
        try:
            from core.database import seed
            seed()
        except Exception:
            pass

    llm_on = llm_enabled()
    logger.info(
        f"Ready — {len(DB.patients)} patients · {len(DB.donors)} donors · "
        f"{len(DB.hospitals)} hospitals · LLM={'on' if llm_on else 'off (rule-based)'}"
    )


# ── Background loop ────────────────────────────────────────────────────────────
async def _background_loop():
    consecutive_errors = 0
    while True:
        try:
            await asyncio.sleep(60)
            await _background_tick()
            consecutive_errors = 0
        except asyncio.CancelledError:
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Background tick #{consecutive_errors}: {e}")
            if consecutive_errors > 10:
                await asyncio.sleep(300)
                consecutive_errors = 0


async def _background_tick():
    from core.database import save
    from services.allocation import gini_coefficient, policy_entropy, tick_ischaemia

    tick_ischaemia(1.0)

    pts    = list(DB.patients.values())
    crit_c = sum(1 for p in pts if p.get("urgency") == "critical")

    DB.analytics.append({
        "ts": _ts(), "total": len(pts), "critical": crit_c,
        "allocations": len(DB.allocations), "expired": DB.ep_expired,
        "kpe": sum(1 for p in pts if p.get("kpe_eligible") and not p.get("is_allocated")),
    })

    alloc_pts = [
        DB.patients.get(a.get("patient_id"), {})
        for a in DB.allocations.values()
        if a.get("status") in ("complete", "active", "pending")
    ]
    ages  = [p.get("age", 40)         for p in alloc_pts]
    waits = [p.get("wait_months", 0)   for p in alloc_pts]
    paed_total = sum(1 for p in pts       if p.get("is_paediatric"))
    paed_alloc = sum(1 for p in alloc_pts if p.get("is_paediatric"))
    DB.fairness_log.append({
        "ts": _ts(),
        "gini_age":  gini_coefficient(ages),
        "gini_wait": gini_coefficient(waits),
        "paed_pct":  round(paed_alloc / max(paed_total, 1) * 100, 1),
        "n_allocs":  len(DB.allocations),
    })

    recent_rewards = [a.get("reward", {}).get("value", 0) for a in list(DB.allocations.values())[-20:]]
    if recent_rewards:
        DB.entropy_log.append(round(policy_entropy(recent_rewards), 3))

    save(SAVE_FILE)
    await manager.broadcast({
        "event": "tick", "ts": _ts(), "critical": crit_c,
        "expired": DB.ep_expired,
        "kpe_candidates": sum(1 for p in pts if p.get("kpe_eligible") and not p.get("is_allocated")),
    })


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False, log_level="info")


if __name__ == "__main__":
    main()
