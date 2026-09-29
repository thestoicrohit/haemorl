# HaemoRL — Smart Organ Allocation for India

> India loses viable organs to delays. HaemoRL tracks every ischaemia clock and uses RL to route
> each organ to the right patient and hospital.

HaemoRL is a hospital-style dashboard and reinforcement-learning environment for allocating organs
and blood across India's transplant network. Transplant coordinators see critical patients, live
ischaemia clocks, donor organs, blood stock and hospital capacity in one place, while an RL
environment and a rule-based (optionally LLM-assisted) agent recommend the best
patient → donor → hospital match.

## Highlights

- **Clinical interface** — clean light "hospital" theme by default with a calm night mode
  (remembered per browser); compact charts; 8 menu items grouped into Overview, Resources and
  RL + AI, with tabs inside each.
- **Landing page** — live numbers from the backend (critical patients, donors available,
  partner hospitals) and one-click entry into any section.
- **Official India map** — the Hospital Network page opens on a map of India drawn to the official
  Government of India boundary (all 36 states/UTs, including Ladakh, full J&K and Arunachal
  Pradesh). Tap it to explore: zoom, pan, and select any of the 20 transplant centres to see live
  load, free beds and ICU capacity. Works offline — the map is embedded in the page.
- **Live ischaemia clocks** — every critical and urgent patient has a countdown; when an organ
  offer lapses it's logged as a missed offer and the patient gets a fresh window, so the
  simulation never stalls.
- **RL environment** — three graded tasks and an 8-component shaped reward, with an example agent
  (`inference.py`) whose composite policy clearly beats the baselines
  (avg reward per step: composite **+0.56**, urgency +0.29, random +0.02).
- **Optional AI assistant** — HaemoBot chat and AI allocation decisions through any standard
  `/chat/completions` LLM endpoint; without one, a deterministic rule-based engine takes over.

## Quick start (Windows)

Double-click **`start.bat`** — it installs dependencies on first run, starts the server and opens
the app in your browser.

Or manually (Python 3.11+):

```bat
python -m pip install -r requirements.txt
python app.py
```

Open **http://localhost:7860** — interactive API docs are at `/docs`.

## What's in the app

| Section   | Menu item        | Tabs                                             |
|-----------|------------------|--------------------------------------------------|
| Overview  | Dashboard        | Command Center · Analytics                       |
|           | Patients         | Registry · Ischaemia Clocks                      |
|           | Allocations      | Allocations · Kidney Exchange                    |
| Resources | Donors & Blood   | Donors & Organs · Blood Bank · HLA Matrix        |
|           | Hospital Network | India Map · Transport Routes                     |
| RL + AI   | RL Lab           | Environment · Training · Leaderboard · Inference |
|           | AI Assistant     | HaemoBot · LLM Decisions                         |
|           | Simulations      | Crisis Mode · What-If                            |

On narrow screens the sidebar collapses to an icon rail.

## Project layout

```
app.py              FastAPI app: serves index.html at /, REST API under /api, WebSocket at /ws
core/               constants (blood types, HLA pool, hospitals, cities→states, diseases),
                    in-memory DB, seeding and JSON persistence
services/           allocation.py — reward engine, graders, ischaemia clock, allocation bookkeeping
                    llm.py        — LLM client (httpx) with rule-based fallback
                    websocket.py  — live-update broadcaster
routes/             API routers: patients, donors, allocations, rl, llm, misc
index.html          single-page frontend (inline CSS/JS; Chart.js + Google Fonts from CDN)
inference.py        example agent that plays the RL environment over HTTP
openenv.yaml        environment spec (observation, action, reward, tasks) — served at /openenv.yaml
tests/              pytest suite — runs against a temporary data folder
data/               persisted database (haemorl_db.json), created on first run
```

## RL environment

1. `POST /reset {"task": "crisis_routing"}` — start an episode
2. `POST /step {"patient_id", "donor_id", "hospital", "action_type": "match_organ" | "skip"}` — repeat
3. `POST /grade` — score the episode

| Task               | Difficulty | Max steps | What's tested                                           |
|--------------------|------------|-----------|---------------------------------------------------------|
| `single_match`     | easy       | 10        | one critical patient → best compatible donor            |
| `batch_allocation` | medium     | 30        | five patients, five donors, full medical constraints    |
| `crisis_routing`   | hard       | 60        | live clocks, trauma arrivals, overloaded hospitals      |

Rewards are shaped in [-1, 1] from 8 components: blood compatibility, HLA match, ischaemia
urgency, paediatric priority (NOTTO), hospital load, survival estimate, geographic distance, and
an organ-expiry penalty (−0.05 per expired organ, capped at −0.30). An invalid action (unknown
patient or donor, or a patient who's already allocated) returns `info.last_action_error` and a
negative reward. Full contract: `openenv.yaml`.

Run the example agent while the server is up:

```bat
python inference.py                                 :: composite policy, crisis_routing
python inference.py --policy random --episodes 5    :: baseline for comparison
python inference.py --task batch_allocation --steps 30 --out results.json
```

## Configuration

All environment variables are optional:

| Variable       | Default  | Purpose                                                          |
|----------------|----------|------------------------------------------------------------------|
| `PORT`         | `7860`   | Server port (`python app.py`)                                    |
| `DATA_DIR`     | `./data` | Where the database file is saved                                 |
| `CORS_ORIGINS` | `*`      | Comma-separated allowed origins                                  |
| `DEBUG`        | off      | Include error details in 500 responses                           |
| `LLM_BASE_URL` | —        | Base URL of a `/chat/completions` LLM API (e.g. `https://…/v1`)   |
| `LLM_MODEL`    | —        | Model name to request                                            |
| `LLM_API_KEY`  | —        | API key for that endpoint                                        |

The AI features switch on only when all three `LLM_*` variables are set; otherwise they run in
deterministic rule-based mode.

**Fresh demo data:** stop the server, delete `data/haemorl_db.json` and restart — or call
`POST /api/admin/reseed`. (This replaces all current patients, donors and allocations.)

## Tests

```bat
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

35 tests cover the key API routes plus regressions for the reward engine, fairness (Gini) metric,
episode reset, double allocation, cancelled allocations, patient validation, and LLM
configuration. Tests use a temporary data folder and never touch `data/haemorl_db.json`.

## Docker

```bat
docker build -t haemorl .
docker run -p 7860:7860 haemorl
```

The container runs a single worker because the database lives in process memory.

## Recent changes

**Interface**
- Redesigned as a clinical hospital-style UI (light default, softer night mode, Inter font).
- Navigation reduced from 25 items to 8 grouped menu items with tabs; gimmick and duplicate pages
  removed (Clock Wall, Hospital Battle, Daily Report, Multiplayer, Patient Stories, Episode
  Replay, Comparative).
- New landing page with live stats; dashboard charts made compact (one row of four).
- Hospital Network rebuilt: official-boundary India map → interactive zoom/pan explorer using the
  backend's real hospital data (the old map showed random loads).

**Backend fixes**
- Expired organs counted once (they were re-counted every step); expiry penalty capped correctly.
- `/reset` now clears the expiry count and frees donors/patients held by the previous episode.
- A patient can't be allocated twice; cancelling an allocation or deleting a patient frees the donor.
- Fairness (Gini) metric corrected — it previously returned negative values.
- Ischaemia clocks renew after a missed offer instead of sitting at 0h forever.
- Patient creation validates age, blood type and urgency.
- Cities and hospitals are placed in their correct states.
- LLM mode works via `httpx` with neutral `LLM_*` settings (it previously needed an uninstalled package).
- `inference.py` rewritten to match the real API; `openenv.yaml` now describes the actual
  observation/action/reward and is the single source served at `/openenv.yaml`.
- Docker runs one worker so all requests see the same data.

## Known limitations

- Several pages (Dashboard, Patients, Clocks, Donors, Allocations) still render a browser-side
  demo dataset; wiring them to the backend API is the next step.
- No login or roles yet — don't expose the server publicly (`/api/admin/*` is open).
- Storage is a single JSON file; a real database with an allocation audit trail is planned.
- The Training tab visualises a simulated learning curve rather than a trained policy.

## Map data

India state boundaries follow the official Government of India depiction, from
[Vardhan Maps](https://github.com/Vardhan-Systems/vardhan-maps) — © OpenStreetMap contributors, ODbL.
