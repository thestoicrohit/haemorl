"""
HaemoRL — LLM Service (HaemoBot + AI Decision Engine)

Talks to any standard `/chat/completions` LLM endpoint over plain HTTP.
Configure with LLM_BASE_URL, LLM_MODEL and LLM_API_KEY; when any is missing
the platform runs in deterministic rule-based mode.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os

import httpx

from core.database import DB, hla_score, blood_ok
from services.allocation import compute_reward

logger = logging.getLogger("haemorl")


def llm_config() -> dict:
    """LLM settings from the environment (read on every call so changes apply live)."""
    return {
        "api_base": os.getenv("LLM_BASE_URL", "").rstrip("/"),
        "model":    os.getenv("LLM_MODEL", ""),
        "token":    os.getenv("LLM_API_KEY", ""),
    }


def llm_enabled() -> bool:
    return all(llm_config().values())


async def _complete(api_base: str, model: str, token: str, messages: list[dict],
                    temperature: float, max_tokens: int, timeout: float) -> str:
    """One chat-completions call; returns the assistant message text."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

CHAT_SYSTEM = """You are HaemoBot v2 — the AI medical assistant for HaemoRL India, a smart organ allocation platform.

Your expertise covers:
- Organ allocation protocols and NOTTO (India) guidelines
- HLA tissue typing and 6-antigen matching
- Cold ischaemia windows by organ type
- MELD and PELD scoring systems
- Kidney Paired Exchange (KPE) methodology
- Paediatric transplant prioritisation
- Geographic organ routing optimisation
- India's organ donation and transplant crisis

Guidelines:
- Be clinical, accurate, and empathetic
- Use correct medical terminology
- Keep responses concise and actionable
- When live platform data is provided, reference it specifically

Current platform context will be appended below."""


async def chat(messages: list[dict], context: str, api_base: str, model: str, token: str) -> str:
    if not (api_base and model and token):
        return (
            "HaemoBot's AI assistant isn't configured. Set LLM_BASE_URL, LLM_MODEL and "
            "LLM_API_KEY on the server to enable it — the rule-based engine keeps working meanwhile."
        )
    system_msg = CHAT_SYSTEM + (f"\n\nLive platform data:\n{context}" if context else "")
    for attempt in range(2):
        try:
            return await _complete(
                api_base, model, token,
                [{"role": "system", "content": system_msg}] + messages[-10:],
                temperature=0.35, max_tokens=700,
                timeout=28.0 if attempt == 0 else 45.0,
            )
        except (asyncio.TimeoutError, httpx.TimeoutException):
            if attempt == 0:
                continue
            return "AI is taking too long. Please try again in a moment."
        except Exception as e:
            logger.error(f"LLM chat error (attempt {attempt}): {e}")
            if attempt == 0:
                continue
            return "AI is temporarily unavailable. The rule-based system remains fully operational."
    return "Could not get AI response."


def _rule_based_decision(
    critical_patients: list[dict],
    available_donors: list[dict],
) -> dict:
    """Deterministic fallback when LLM is unavailable."""
    if not critical_patients or not available_donors:
        return {
            "action": {"patient_id": "", "donor_id": None, "hospital": None, "action_type": "skip"},
            "reasoning": "No critical patients or donors available.",
            "confidence": 1.0,
            "priority_factors": [],
            "mode": "rule_based",
        }

    # Prioritise paediatric patients with imminent ischaemia
    paed = [p for p in critical_patients if p.get("is_paediatric") and (p.get("ischaemia_h") if p.get("ischaemia_h") is not None else 999) < 6]
    target = paed[0] if paed else critical_patients[0]

    scored = [
        (d, hla_score(target.get("hla", {}), d.get("hla", {})),
         blood_ok(d.get("blood_type", ""), target.get("blood_type", "")))
        for d in available_donors
    ]
    scored.sort(key=lambda x: x[1] * 0.6 + (0.4 if x[2] else 0), reverse=True)
    best_donor = scored[0][0]

    hospitals = list(DB.hospitals.values())
    best_hosp = min(hospitals, key=lambda h: h.get("load_pct", 100)) if hospitals else {"name": "AIIMS New Delhi"}

    priority_factors = ["ischaemia", "geographic"]
    if target.get("is_paediatric"):       priority_factors.append("paediatric")
    if scored[0][2]:                       priority_factors.append("blood_compatible")
    if (target.get("peld_score") or 0) > 20:  priority_factors.append("high_peld")

    return {
        "action": {
            "patient_id": target["id"],
            "donor_id": best_donor["id"],
            "hospital": best_hosp["name"],
            "action_type": "match_organ",
        },
        "reasoning": (
            f"{target.get('name')} ({target.get('disease')}, "
            f"{(target.get('ischaemia_h') or 0):.1f}h ischaemia) matched to "
            f"{best_donor.get('name')} — HLA {int(scored[0][1] * 100)}%"
        ),
        "confidence": round(0.60 + scored[0][1] * 0.30, 2),
        "priority_factors": priority_factors,
        "mode": "rule_based",
    }


async def ai_decide(
    obs: dict,
    patients: list[dict],
    donors: list[dict],
    api_base: str,
    model: str,
    token: str,
) -> dict:
    """
    LLM-powered allocation decision with rule-based fallback.
    """
    critical = sorted(
        [p for p in patients if p.get("urgency") == "critical" and not p.get("is_allocated")],
        key=lambda p: p.get("ischaemia_h") if p.get("ischaemia_h") is not None else 999,
    )
    available = [d for d in donors if d.get("available")]
    fallback = _rule_based_decision(critical, available)

    if not (api_base and model and token):
        return fallback

    try:
        pt_summary = [
            {k: v for k, v in p.items()
             if k in ("id","name","age","blood_type","urgency","ischaemia_h",
                      "is_paediatric","disease","peld_score","meld_score","kpe_eligible")}
            for p in critical[:4]
        ]
        dn_summary = [
            {k: v for k, v in d.items()
             if k in ("id","name","blood_type","organs","hospital")}
            for d in available[:4]
        ]

        prompt = (
            f"Critical={obs.get('critical_count',0)} "
            f"MinIsch={(obs.get('min_ischaemia_remaining') or 0):.1f}h "
            f"KPE={obs.get('kpe_candidates',0)}\n"
            f"Patients: {json.dumps(pt_summary)}\n"
            f"Donors: {json.dumps(dn_summary)}\n"
            "Return JSON only: "
            "{\"action\":{\"patient_id\":\"...\",\"donor_id\":\"...\",\"hospital\":\"...\",\"action_type\":\"match_organ\"},"
            "\"reasoning\":\"...\",\"confidence\":0.0,\"priority_factors\":[],\"mode\":\"llm\"}"
        )

        text = (await _complete(
            api_base, model, token,
            [
                {"role": "system", "content": (
                    "You are an RL organ allocation agent for India. "
                    "Consider MELD/PELD, ischaemia, HLA, blood type, geography. "
                    "Return JSON only."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.10, max_tokens=350, timeout=18.0,
        )).strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        result = json.loads(text)
        result["mode"] = "llm"
        return result

    except Exception as e:
        logger.warning(f"AI decide fallback: {e}")
        return fallback
