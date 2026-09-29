#!/usr/bin/env python3
"""
HaemoRL — example agent for the RL environment API.

Runs one or more episodes against a running server:
    python inference.py                        # composite policy, 20 steps
    python inference.py --policy random        # baseline for comparison
    python inference.py --steps 40 --task batch_allocation
    python inference.py --host http://my-server:7860

Each step sends a structured action {patient_id, donor_id, hospital, action_type}
to POST /step and reads back {observation, reward{value, breakdown, …}, done, info}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

# donor blood type -> recipient blood types it can safely give to
BLOOD_COMPAT = {
    "O-":  {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"},
    "O+":  {"A+", "B+", "AB+", "O+"},
    "A-":  {"A+", "A-", "AB+", "AB-"},
    "A+":  {"A+", "AB+"},
    "B-":  {"B+", "B-", "AB+", "AB-"},
    "B+":  {"B+", "AB+"},
    "AB-": {"AB+", "AB-"},
    "AB+": {"AB+"},
}


def hla_match(p: dict, d: dict) -> float:
    ph, dh = p.get("hla") or {}, d.get("hla") or {}
    return sum(len(set(ph.get(l, [])) & set(dh.get(l, []))) for l in ("A", "B", "DR")) / 6


# ── Policies: (patients, donors, hospitals) -> action or None ───────────────

def policy_random(patients, donors, hospitals):
    if not patients or not donors:
        return None
    return {"patient_id": random.choice(patients)["id"], "donor_id": random.choice(donors)["id"],
            "hospital": random.choice(hospitals)["name"] if hospitals else "", "action_type": "match_organ"}


def policy_urgency(patients, donors, hospitals):
    """Most urgent patient (shortest ischaemia clock) + first blood-compatible donor."""
    for p in sorted(patients, key=lambda p: p.get("ischaemia_h", 999)):
        ok = [d for d in donors if p.get("blood_type") in BLOOD_COMPAT.get(d.get("blood_type"), ())]
        if ok:
            h = min(hospitals, key=lambda h: h.get("load_pct", 100)) if hospitals else {"name": ""}
            return {"patient_id": p["id"], "donor_id": ok[0]["id"], "hospital": h["name"], "action_type": "match_organ"}
    return None


def policy_composite(patients, donors, hospitals):
    """Score every pairing on urgency, blood compatibility, HLA match and paediatric priority."""
    best, best_s = None, -1.0
    for p in sorted(patients, key=lambda p: p.get("ischaemia_h", 999))[:15]:
        urgency = 1.0 - min(p.get("ischaemia_h", 48), 48) / 48
        for d in donors:
            if p.get("blood_type") not in BLOOD_COMPAT.get(d.get("blood_type"), ()):
                continue
            s = 0.45 * urgency + 0.40 * hla_match(p, d) + (0.15 if p.get("is_paediatric") else 0.0)
            if s > best_s:
                best, best_s = (p, d), s
    if not best:
        return None
    h = min(hospitals, key=lambda h: h.get("load_pct", 100)) if hospitals else {"name": ""}
    return {"patient_id": best[0]["id"], "donor_id": best[1]["id"], "hospital": h["name"], "action_type": "match_organ"}


POLICIES = {"composite": policy_composite, "urgency": policy_urgency, "random": policy_random}


# ── Episode loop ─────────────────────────────────────────────────────────────

def run_episode(base_url: str, n_steps: int, policy: str, task: str, verbose: bool = True) -> dict[str, Any]:
    client = httpx.Client(base_url=base_url, timeout=30.0)
    choose = POLICIES[policy]

    reset = client.post("/reset", json={"task": task}).json()
    info = reset.get("info", {})
    if verbose:
        print(f"\n{'=' * 60}\n  HaemoRL episode {info.get('episode_id')} · task={info.get('task')} · policy={policy}\n{'=' * 60}")

    rewards: list[float] = []
    for step in range(1, n_steps + 1):
        patients = client.get("/api/patients", params={"urgency": "critical", "per_page": 100}).json()["patients"]
        patients = [p for p in patients if not p.get("is_allocated")]
        state = client.get("/state").json()
        donors = [d for d in state.get("donors", []) if d.get("available")]
        action = choose(patients, donors, state.get("hospitals", []))
        if action is None:
            action = {"action_type": "skip"}

        result = client.post("/step", json=action).json()
        if "reward" not in result:          # e.g. "Episode done. Call /reset first."
            if verbose:
                print(f"  stopped: {result.get('detail', result)}")
            break
        r = result["reward"].get("value", 0.0)
        rewards.append(r)
        if verbose and (step <= 5 or step % 10 == 0 or result.get("done")):
            print(f"  step {step:3d} | {action.get('patient_id', 'skip'):>7} → {action.get('donor_id', '-'):>7} "
                  f"| reward {r:+.3f} | cum {sum(rewards):+.3f}"
                  + (f" | {result['info'].get('last_action_error')}" if result.get("info", {}).get("last_action_error") else ""))
        if result.get("done"):
            break

    grade = client.post("/grade").json()
    client.close()
    summary = {
        "policy": policy, "task": task, "steps": len(rewards),
        "total_reward": round(sum(rewards), 4),
        "avg_reward": round(sum(rewards) / max(len(rewards), 1), 4),
        "mean_grade": grade.get("mean_score"),
        "graders": {k: v.get("score") for k, v in grade.get("grader_results", {}).items()},
    }
    if verbose:
        print(f"\n  steps {summary['steps']} · total {summary['total_reward']:+.3f} · "
              f"avg {summary['avg_reward']:+.3f} · mean grade {summary['mean_grade']}")
        for k, v in summary["graders"].items():
            print(f"    {k:<18} {v}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="HaemoRL example agent",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default="http://localhost:7860", help="API base URL")
    ap.add_argument("--steps", type=int, default=20, help="max steps per episode")
    ap.add_argument("--policy", choices=list(POLICIES), default="composite")
    ap.add_argument("--task", choices=["single_match", "batch_allocation", "crisis_routing"], default="crisis_routing")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--quiet", action="store_true", help="only print the final summary")
    ap.add_argument("--out", default="", help="write summaries to this JSON file")
    args = ap.parse_args()

    runs = [run_episode(args.host, args.steps, args.policy, args.task, not args.quiet) for _ in range(args.episodes)]
    if args.episodes > 1 or args.quiet:
        print(f"\n{args.policy}: mean avg-reward over {len(runs)} episode(s) = "
              f"{sum(r['avg_reward'] for r in runs) / len(runs):+.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
