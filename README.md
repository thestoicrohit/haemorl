---
title: HaemoRL Smart Organ Allocation
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: RL organ allocation, 1500 patients, live LLM
tags:
  - openenv
  - medical
  - rl
  - organ-allocation
---

# HaemoRL — RL-Based Smart Organ Allocation
### *"As efficient as a factory, as precise as a surgeon — every second counts."*

**1,500 patients · Persistent data · Live LLM decisions · OpenEnv compliant**  
**India performs ~15,000 transplants/year but needs 500,000+. Every bad allocation is a death.**

---

## 🚀 Deploy (5 files, Docker Space)

```
app.py           ← FastAPI backend (serves UI + API + OpenEnv spec)
index.html       ← Full dark UI dashboard (13 pages)
Dockerfile       ← HF Spaces Docker config (port 7860)
requirements.txt ← Dependencies
README.md        ← This file
```

## ⚙️ Environment Variables

| Variable | Value | Required |
|---|---|---|
| `HF_TOKEN` | Your HF token from hf.co/settings/tokens | For LLM |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Optional |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Optional |
| `SEED_PATIENTS` | `1500` | Optional |

**Without HF_TOKEN** → rule-based RL agent (still passes all hackathon checks)  
**With HF_TOKEN** → Qwen 2.5 72B makes real decisions with clinical reasoning

## 💾 Persistent Storage

Enable **Persistent Storage** in Space Settings for data to survive restarts.  
Without it, data resets on restart (patients still re-seed from scratch, allocations are lost).

All data persists to `/data/haemorl.json`:
- All 1,500 seeded patients (saved to `/data/haemorl.json` on first run)
- Every patient added through the UI or API
- All allocations, donors, blood bank, LLM decision log

## 🎯 OpenEnv Endpoints

| Endpoint | Description |
|---|---|
| `POST /reset` | Start new episode |
| `POST /step` | Take action, get shaped reward |
| `GET /state` | Full 42-dim observation |
| `GET /tasks` | 3 tasks with metadata |
| `POST /grade` | Run all 3 deterministic graders |
| `GET /validate` | OpenEnv spec compliance check |
| `GET /openenv.yaml` | Machine-readable spec |
| `GET /health` | Health check (for automated ping) |
| `GET /docs` | Swagger UI — all 45+ endpoints |

## 📊 Baseline Scores

| Task | Difficulty | Max Steps | Score |
|---|---|---|---|
| `single_match` | Easy | 10 | **0.847** |
| `batch_allocation` | Medium | 30 | **0.693** |
| `crisis_routing` | Hard | 60 | **0.521** |
| **Mean** | — | — | **0.687** |

## 🧠 What Makes This Novel

- **First organ allocation environment in OpenEnv**
- **HLA-A/B/DR tissue typing** — compatibility beyond blood type
- **Live ischaemia countdowns** — organs expire in real time (critical: 6–24h)
- **Paediatric priority weighting** — +15% RL score for patients ≤17 (medical ethics encoded)
- **7-component shaped reward** — not binary, signals throughout episode
- **Real Indian hospital network** — 15 hospitals, real city coordinates, GPS routing
- **LLM integration** — model explains every allocation decision in clinical language
- **Collaborative** — multiple users, all see live updates via WebSocket

## 🏥 Disease Categories (13)

Oncology · Haematology · HIV/AIDS · Cardiac · Renal · Hepatic · Pulmonary ·  
Neurological · Diabetes · Autoimmune · Trauma · Infectious · Genetic

## 📡 Full API

`GET /api/stats` · `GET /api/patients` · `POST /api/patients` · `GET /api/donors`  
`POST /api/allocations/auto-match` · `POST /api/llm/decide` · `POST /api/llm/chat`  
`GET /api/hla/matrix` · `GET /api/analytics` · `GET /api/alerts` · `GET /api/transport`

