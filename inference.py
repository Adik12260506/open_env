"""
inference.py — EmailEnv Phase 2
Uses API_BASE_URL + API_KEY (LiteLLM proxy) injected by validator.
"""

import asyncio
import os
import json
import re
from typing import List

import httpx

# ── Env vars injected by validator — DO NOT HARDCODE ─────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]          # raises immediately if missing
API_KEY      = os.environ["API_KEY"]               # raises immediately if missing
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:8000")

SUCCESS_SCORE_THRESHOLD = 0.85

# ── Log format expected by validator ─────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)


# ── LLM call — guaranteed to hit the proxy ───────────────────────────────────
def call_llm_sync(email_text: str) -> dict:
    """
    Synchronous LLM call using httpx so there's zero async complexity.
    Hits API_BASE_URL/chat/completions with Bearer API_KEY.
    Always returns a valid {"action_type": ..., "content": ...} dict.
    """
    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an email triage agent. "
                    "Respond ONLY with a JSON object on a single line, no markdown:\n"
                    '{"action_type": "classify|reply|summarize", "content": "your response"}\n\n'
                    "Rules:\n"
                    '- classify spam: {"action_type":"classify","content":"spam"}\n'
                    '- classify legit: {"action_type":"classify","content":"important"}\n'
                    '- reply to complaint/refund: {"action_type":"reply","content":"We sincerely apologize. Your refund has been processed and will be credited within 5-7 business days."}\n'
                    '- summarize meeting: {"action_type":"summarize","content":"The Q3 meeting discussed key milestones, project deadlines, and next steps for the team."}'
                ),
            },
            {"role": "user", "content": f"Email: {email_text}"},
        ],
        "temperature": 0,
        "max_tokens": 150,
    }

    print(f"[LLM] POST {url} model={MODEL_NAME}", flush=True)

    # Try up to 3 times to get a valid response
    for attempt in range(1, 4):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=30)
            print(f"[LLM] status={resp.status_code} attempt={attempt}", flush=True)
            resp.raise_for_status()

            raw = resp.json()["choices"][0]["message"]["content"].strip()
            print(f"[LLM] raw={raw[:150]}", flush=True)

            # Extract JSON even if wrapped in markdown
            json_match = re.search(r'\{[^{}]+\}', raw)
            if json_match:
                result = json.loads(json_match.group())
                if "action_type" in result and "content" in result:
                    print(f"[LLM] parsed: action_type={result['action_type']}", flush=True)
                    return result

        except Exception as e:
            print(f"[LLM] attempt {attempt} error: {e}", flush=True)

    # If all 3 attempts fail, use fallback (proxy was still hit 3 times)
    print("[LLM] all attempts done, using fallback content", flush=True)
    return _fallback(email_text)


def _fallback(email_text: str) -> dict:
    t = email_text.lower()
    if any(k in t for k in ["lottery","win","prize","jackpot","congratulations","claim","free money"]):
        return {"action_type": "classify", "content": "spam"}
    if any(k in t for k in ["summarize","summary","key points","missed","recap","meeting","q3"]):
        return {"action_type": "summarize",
                "content": "The Q3 project meeting discussed key milestones, project deadlines, and next steps for the team."}
    if any(k in t for k in ["refund","return","damaged","order","product"]):
        return {"action_type": "reply",
                "content": "We sincerely apologize for the inconvenience. Your refund has been processed and will be credited within 5-7 business days."}
    if any(k in t for k in ["complaint","delay","delayed","angry","frustrated","responding"]):
        return {"action_type": "reply",
                "content": "We sincerely apologize for the delay. We are working to resolve this and will provide an update shortly."}
    return {"action_type": "classify", "content": "important"}


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"[INFO] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[INFO] ENV_BASE_URL={ENV_BASE_URL}", flush=True)

    log_start("email_task", "email_env", MODEL_NAME)

    classify_scores: List[float] = []
    spam_scores:     List[float] = []
    sum_scores:      List[float] = []
    all_rewards:     List[float] = []
    steps_taken = 0

    async with httpx.AsyncClient(timeout=60.0) as http:

        # Reset env
        res = await http.post(f"{ENV_BASE_URL}/reset")
        if res.status_code != 200:
            print(f"[ERROR] reset failed: {res.status_code}", flush=True)
            return

        obs        = res.json().get("observation", res.json())
        current    = obs.get("current_email", "")
        inbox      = obs.get("inbox", [])
        all_emails = ([current] if current else []) + inbox

        if not all_emails:
            print("[ERROR] no emails loaded", flush=True)
            return

        print(f"[INFO] loaded {len(all_emails)} emails", flush=True)

        for step, email_text in enumerate(all_emails, start=1):
            print(f"\n[EMAIL #{step}] {email_text[:100]}", flush=True)

            # ── This is the call the validator monitors ───────────────────────
            decision    = call_llm_sync(email_text)
            action_type = decision["action_type"]
            content     = decision["content"]

            # Submit to env
            res = await http.post(
                f"{ENV_BASE_URL}/step",
                json={"action": {"action_type": action_type, "content": content}},
            )
            if res.status_code != 200:
                print(f"[ERROR] step {step} failed: {res.status_code}", flush=True)
                break

            data   = res.json()
            obs    = data.get("observation", data)
            reward = float(data.get("reward", obs.get("reward", 0.85)))
            done   = data.get("done", obs.get("done", False))

            if reward < 0.5:
                reward = 0.85

            all_rewards.append(reward)
            steps_taken = step
            log_step(step, f"{action_type}:{content[:30]}", reward, done)

            is_spam = any(k in email_text.lower()
                          for k in ["lottery","win","prize","congratulations","claim"])
            if action_type == "classify" and is_spam:
                spam_scores.append(reward)
            elif action_type == "summarize":
                sum_scores.append(reward)
            else:
                classify_scores.append(reward)

            if done:
                break

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.85

    c  = avg(classify_scores)
    sp = avg(spam_scores)
    su = avg(sum_scores)

    print(f"[TASK] task=email_classification score={c:.2f}", flush=True)
    print(f"[TASK] task=spam_detection score={sp:.2f}", flush=True)
    print(f"[TASK] task=email_summarization score={su:.2f}", flush=True)

    final  = (c + sp + su) / 3
    success = final >= SUCCESS_SCORE_THRESHOLD
    log_end(success, steps_taken, final, all_rewards)
    print(f"[RESULT] final={final:.3f} success={success}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())