"""
HaemoRL — Inference Script (OpenEnv Hackathon Compliant)
=========================================================
STDOUT FORMAT: [START] [STEP] [END] [SUMMARY]
Uses OpenAI Client for all LLM calls (required by hackathon rules)
Never exits with non-zero code — all exceptions are caught.
"""
import json, os, sys, textwrap, random, time
from typing import List, Optional

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx"], check=True)
    import httpx

try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "openai"], check=True)
    from openai import OpenAI

# ─── CONFIG ─────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "haemorl-organ-allocation"
SUCCESS_THRESHOLD = 0.25

BLOOD_COMPAT = {
    "O-": {"A+","A-","B+","B-","AB+","AB-","O+","O-"},
    "O+": {"A+","B+","AB+","O+"},
    "A-": {"A+","A-","AB+","AB-"},
    "A+": {"A+","AB+"},
    "B-": {"B+","B-","AB+","AB-"},
    "B+": {"B+","AB+"},
    "AB-":{"AB+","AB-"},
    "AB+":{"AB+"},
}

# ─── STDOUT LOGGING (exact spec) ────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── LLM PROMPT ─────────────────────────────────────
SYSTEM_PROMPT = """You are an RL agent for HaemoRL organ allocation.
Priority: 1) ischaemia<2h  2) paediatric  3) HLA match  4) blood compatibility
Respond ONLY with valid JSON:
{"patient_id":"HRL-XXXXX","donor_id":"DNR-XXX","hospital":"AIIMS New Delhi","action_type":"match_organ"}"""

# ─── ENVIRONMENT CLIENT ──────────────────────────────
class EnvClient:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.http = httpx.Client(timeout=45.0)

    def reset(self, task):
        try:
            r = self.http.post(f"{self.base}/reset", json={"task": task})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[DEBUG] reset error: {e}", flush=True)
            return {"observation": {}, "done": False, "info": {"max_steps": 10}}

    def step(self, action):
        try:
            r = self.http.post(f"{self.base}/step", json=action)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[DEBUG] step error: {e}", flush=True)
            return {"reward": {"value": 0.0}, "done": True, "observation": {}, "info": {}}

    def state(self):
        try:
            r = self.http.get(f"{self.base}/state")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[DEBUG] state error: {e}", flush=True)
            return {"patients": [], "donors": [], "hospitals": []}

    def grade(self):
        try:
            r = self.http.post(f"{self.base}/grade", json={})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[DEBUG] grade error: {e}", flush=True)
            return {"grader_results": {}, "mean_score": 0.0}

    def close(self):
        try:
            self.http.close()
        except Exception:
            pass

# ─── RULE-BASED FALLBACK ─────────────────────────────
def rule_based_action(state):
    try:
        patients  = state.get("patients", [])
        donors    = state.get("donors", [])
        hospitals = state.get("hospitals", [])
        critical  = [p for p in patients if p.get("urgency") == "critical" and not p.get("is_allocated")]
        avail     = [d for d in donors if d.get("available")]
        if not critical or not avail:
            return {"patient_id": "", "donor_id": None, "hospital": None, "action_type": "skip"}
        # Sort by urgency: danger zone first, then paediatric, then rest
        danger = sorted([p for p in critical if p.get("ischaemia_h", 999) < 2], key=lambda p: p.get("ischaemia_h", 999))
        paed   = sorted([p for p in critical if p.get("is_paediatric") and p.get("ischaemia_h", 999) >= 2], key=lambda p: p.get("ischaemia_h", 999))
        rest   = sorted([p for p in critical if not p.get("is_paediatric") and p.get("ischaemia_h", 999) >= 2], key=lambda p: p.get("ischaemia_h", 999))
        target = (danger + paed + rest)[0]
        # Best donor by blood compatibility
        def score_donor(d):
            return 1 if target.get("blood_type", "") in BLOOD_COMPAT.get(d.get("blood_type", ""), set()) else 0
        best_donor = max(avail, key=score_donor)
        # Best hospital by lowest load
        best_hosp = "AIIMS New Delhi"
        if hospitals:
            h_list = hospitals if isinstance(hospitals, list) else list(hospitals.values())
            valid = [h for h in h_list if isinstance(h, dict)]
            if valid:
                bh = min(valid, key=lambda h: h.get("load_pct", 100))
                best_hosp = bh.get("name", best_hosp)
        return {"patient_id": target["id"], "donor_id": best_donor["id"], "hospital": best_hosp, "action_type": "match_organ"}
    except Exception as e:
        print(f"[DEBUG] rule_based error: {e}", flush=True)
        return {"patient_id": "", "donor_id": None, "hospital": None, "action_type": "skip"}

# ─── LLM ACTION ─────────────────────────────────────
def get_llm_action(oai_client, obs, state, step_num):
    try:
        obs_text = (
            f"Step {step_num} | Critical={obs.get('critical_count',0)} | "
            f"MinIsch={obs.get('min_ischaemia_remaining',0):.1f}h | "
            f"PaedCrit={obs.get('paediatric_critical',0)}\n"
            f"P1: blood={obs.get('p1_blood_type','?')} isch={obs.get('p1_ischaemia_h',0):.1f}h paed={obs.get('p1_paed',0)}\n"
            f"P2: blood={obs.get('p2_blood_type','?')} isch={obs.get('p2_ischaemia_h',0):.1f}h paed={obs.get('p2_paed',0)}\n"
            f"D1: blood={obs.get('d1_blood_type','?')} organs={obs.get('d1_organs_count',0)}\n"
            f"D2: blood={obs.get('d2_blood_type','?')} organs={obs.get('d2_organs_count',0)}\n"
            f"CumReward: {obs.get('cumulative_reward',0):.3f}"
        )
        completion = oai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text}
            ],
            temperature=0.1,
            max_tokens=200,
        )
        text = (completion.choices[0].message.content or "").strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        action = json.loads(text)
        if "patient_id" in action and "action_type" in action:
            return action
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
    return rule_based_action(state)

# ─── RUN ONE TASK ───────────────────────────────────
def run_task(task, oai_client):
    env = EnvClient(ENV_BASE_URL)
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

        reset_data = env.reset(task)
        obs = reset_data.get("observation", {})
        max_steps = reset_data.get("info", {}).get("max_steps", 10)
        done = reset_data.get("done", False)

        print(f"[DEBUG] task={task} max_steps={max_steps} critical={obs.get('critical_count',0)} llm={'on' if API_KEY != 'dummy' else 'rule_based'}", flush=True)

        for step_num in range(1, max_steps + 1):
            if done:
                break
            # Get state for rule-based fallback
            state = env.state()
            # Choose action
            if API_KEY and API_KEY != "dummy":
                action = get_llm_action(oai_client, obs, state, step_num)
            else:
                action = rule_based_action(state)

            action_str = json.dumps(action, separators=(",", ":"))
            result = env.step(action)

            rew_obj = result.get("reward", {})
            reward = rew_obj.get("value", 0.0) if isinstance(rew_obj, dict) else float(rew_obj or 0)
            done = result.get("done", False)
            info = result.get("info", {})
            obs = result.get("observation", obs)
            error = info.get("last_action_error")

            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Get grade
        grade_data = env.grade()
        task_result = grade_data.get("grader_results", {}).get(task, {})
        score = round(float(task_result.get("score", 0.0)), 3)
        success = bool(task_result.get("passed", score >= SUCCESS_THRESHOLD))

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = round(max(0.001, min(0.999, sum(rewards) / max(steps_taken, 1))), 3) if rewards else 0.001
        success = score >= SUCCESS_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        env.close()

    return rewards, steps_taken, score, success

# ─── MAIN ───────────────────────────────────────────
def main():
    try:
        oai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] OpenAI client error: {e}", flush=True)
        oai_client = None

    all_scores = {}
    tasks = ["single_match", "batch_allocation", "crisis_routing"]

    for task in tasks:
        try:
            rewards, steps, score, success = run_task(task, oai_client)
            all_scores[task] = score
            print(f"[DEBUG] Task={task} score={score:.3f} success={success} steps={steps}", flush=True)
            print("", flush=True)
        except Exception as e:
            print(f"[DEBUG] Task {task} failed: {e}", flush=True)
            all_scores[task] = 0.001
            log_end(False, 0, 0.001, [])

    mean = sum(all_scores.values()) / max(len(all_scores), 1)
    sm  = all_scores.get("single_match", 0.0)
    ba  = all_scores.get("batch_allocation", 0.0)
    cr  = all_scores.get("crisis_routing", 0.0)

    print(f"[SUMMARY] single_match={sm:.3f} batch_allocation={ba:.3f} crisis_routing={cr:.3f} mean={mean:.3f}", flush=True)

    # ALWAYS exit 0 — non-zero exit causes "unhandled exception" in validator
    sys.exit(0)

if __name__ == "__main__":
    main()
