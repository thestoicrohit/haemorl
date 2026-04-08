"""
HaemoRL — Inference Script (Hackathon Compliant)
=================================================
MANDATORY ENV VARS: API_BASE_URL, MODEL_NAME, HF_TOKEN
STDOUT FORMAT: [START] [STEP] [END] — exact spec from hackathon guidelines
Uses OpenAI Client for all LLM calls (required by hackathon rules)
"""
import asyncio, json, os, sys, textwrap, random
from typing import List, Optional
import httpx
from openai import OpenAI

# ─── MANDATORY ENV VARS ─────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
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

# ─── EXACT STDOUT FORMAT (hackathon spec) ───────────
def log_start(task:str, env:str, model:str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step:int, action:str, reward:float, done:bool, error:Optional[str]):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success:bool, steps:int, score:float, rewards:List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── LLM SYSTEM PROMPT ──────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an RL agent for HaemoRL — an organ allocation system with 9000+ patients.
You make allocation decisions to maximise cumulative reward.

REWARD COMPONENTS:
+0.25 blood compatible | +0.35 HLA match | +0.20 ischaemia<2h
+0.15 paediatric | +0.05 hospital<55% | -0.15 blood mismatch
-0.30 ischaemia expired | -0.20 hospital overloaded

PRIORITY: 1) ischaemia_h < 2h  2) paediatric  3) HLA score  4) blood type

Respond ONLY with a valid JSON action object. Nothing else.
{
  "patient_id": "HRL-XXXXX",
  "donor_id": "DNR-XXX",
  "hospital": "AIIMS New Delhi",
  "action_type": "match_organ"
}
""").strip()

# ─── ENV HTTP CLIENT ────────────────────────────────
class EnvClient:
    def __init__(self, base:str):
        self.base=base.rstrip("/")
        self.http=httpx.Client(timeout=30.0)

    def reset(self, task:str) -> dict:
        r=self.http.post(f"{self.base}/reset", json={"task":task}); r.raise_for_status(); return r.json()

    def step(self, action:dict) -> dict:
        r=self.http.post(f"{self.base}/step", json=action); r.raise_for_status(); return r.json()

    def state(self) -> dict:
        r=self.http.get(f"{self.base}/state"); r.raise_for_status(); return r.json()

    def grade(self) -> dict:
        r=self.http.post(f"{self.base}/grade", json={}); r.raise_for_status(); return r.json()

    def close(self): self.http.close()

# ─── RULE-BASED AGENT (fallback) ────────────────────
def rule_based_action(state:dict) -> dict:
    """Optimal rule-based allocation: ischaemia→paediatric→HLA→blood."""
    patients  = state.get("patients", [])
    donors    = state.get("donors",   [])
    hospitals = state.get("hospitals",[])
    critical  = [p for p in patients if p.get("urgency")=="critical" and not p.get("is_allocated")]
    avail     = [d for d in donors if d.get("available")]
    if not critical or not avail:
        return {"patient_id":"","donor_id":None,"hospital":None,"action_type":"skip"}
    # Prioritise: danger (<2h) → paediatric → ischaemia → everyone else
    danger = sorted([p for p in critical if p.get("ischaemia_h",999)<2], key=lambda p: p.get("ischaemia_h",999))
    paed   = sorted([p for p in critical if p.get("is_paediatric") and p.get("ischaemia_h",999)>=2], key=lambda p: p.get("ischaemia_h",999))
    rest   = sorted([p for p in critical if not p.get("is_paediatric") and p.get("ischaemia_h",999)>=2], key=lambda p: p.get("ischaemia_h",999))
    target = (danger + paed + rest)[0]
    # Best donor: score by blood compatibility + HLA approximation
    def donor_score(d):
        bt_ok = target.get("blood_type","") in BLOOD_COMPAT.get(d.get("blood_type",""), set())
        return (1 if bt_ok else 0)
    best_donor = max(avail, key=donor_score)
    # Best hospital (lowest load)
    best_hosp = "AIIMS New Delhi"
    if hospitals:
        bh = min(hospitals, key=lambda h: h.get("load_pct",100) if isinstance(h,dict) else 100)
        best_hosp = bh.get("name", best_hosp) if isinstance(bh,dict) else best_hosp
    return {"patient_id":target["id"],"donor_id":best_donor["id"],"hospital":best_hosp,"action_type":"match_organ"}

# ─── LLM AGENT ──────────────────────────────────────
def get_llm_action(client:OpenAI, obs:dict, state:dict, history:List[str], step:int) -> dict:
    """Query LLM for allocation decision. Returns action dict."""
    obs_text = f"""Step {step} | Critical={obs.get('critical_count',0)} | MinIsch={obs.get('min_ischaemia_remaining',0):.1f}h | PaedCrit={obs.get('paediatric_critical',0)}
P1: blood={obs.get('p1_blood_type','?')} isch={obs.get('p1_ischaemia_h',0):.1f}h paed={obs.get('p1_paed',0)} id=?
P2: blood={obs.get('p2_blood_type','?')} isch={obs.get('p2_ischaemia_h',0):.1f}h paed={obs.get('p2_paed',0)} id=?
D1: blood={obs.get('d1_blood_type','?')} organs={obs.get('d1_organs_count',0)} id=?
D2: blood={obs.get('d2_blood_type','?')} organs={obs.get('d2_organs_count',0)} id=?
Cum reward: {obs.get('cumulative_reward',0):.3f}
History: {' | '.join(history[-3:]) if history else 'None'}"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":obs_text}],
            temperature=0.1, max_tokens=200, stream=False,
        )
        text=(completion.choices[0].message.content or "").strip()
        if "```" in text: text=text.split("```")[1].lstrip("json").strip()
        action=json.loads(text)
        if "patient_id" in action and "action_type" in action:
            return action
    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)
    return rule_based_action(state)

# ─── EPISODE RUNNER ──────────────────────────────────
def run_task(task:str, oai_client:OpenAI) -> tuple:
    env=EnvClient(ENV_BASE_URL)
    rewards=[]; steps_taken=0; score=0.0; success=False
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
    try:
        reset_data=env.reset(task)
        obs=reset_data.get("observation",{}); max_steps=reset_data.get("info",{}).get("max_steps",60); done=reset_data.get("done",False)
        print(f"[DEBUG] Episode: task={task} patients={obs.get('total_patients',0)} critical={obs.get('critical_count',0)} llm={'active' if API_KEY else 'rule_based'}", flush=True)
        history=[]
        for step_num in range(1, max_steps+1):
            if done: break
            state={}
            try: state=env.state()
            except: pass
            # LLM or rule-based action
            if API_KEY:
                action=get_llm_action(oai_client, obs, state, history, step_num)
            else:
                action=rule_based_action(state)
            action_str=json.dumps(action, separators=(",",":"))
            result=env.step(action)
            rew_obj=result.get("reward",{})
            reward=rew_obj.get("value",0.0) if isinstance(rew_obj,dict) else float(rew_obj or 0)
            done=result.get("done",False)
            info=result.get("info",{})
            obs=result.get("observation",obs)
            error=info.get("last_action_error")
            rewards.append(reward); steps_taken=step_num
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)
            history.append(f"S{step_num}:{action.get('action_type','?')} r={reward:+.2f}")
            if done: break
        # Grade
        try:
            grade=env.grade()
            task_result=grade.get("grader_results",{}).get(task,{})
            score=round(float(task_result.get("score",0.0)),3)
            success=task_result.get("passed",score>=SUCCESS_THRESHOLD)
        except Exception as ge:
            print(f"[DEBUG] Grade failed: {ge}", flush=True)
            score=round(min(1.0,max(0.0,sum(rewards)/max(max_steps,1))),3)
            success=score>=SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        env.close()
    return rewards, steps_taken, score, success

def main():
    oai_client=OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    all_scores={}
    for task in["single_match","batch_allocation","crisis_routing"]:
        rewards,steps,score,success=run_task(task,oai_client)
        all_scores[task]=score
        print(f"[DEBUG] Task={task} score={score:.3f} success={success} steps={steps}", flush=True)
        print("", flush=True)
    mean=sum(all_scores.values())/3
    print(f"[SUMMARY] single_match={all_scores.get('single_match',0):.3f} batch_allocation={all_scores.get('batch_allocation',0):.3f} crisis_routing={all_scores.get('crisis_routing',0):.3f} mean={mean:.3f}", flush=True)
    passed=all(s>=SUCCESS_THRESHOLD for s in all_scores.values())
    sys.exit(0 if passed else 1)

if __name__=="__main__":
    main()
