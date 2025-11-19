# sequential_onboarding_ag2_async.py
# pip install -U autogen autogen-agentchat autogen-ext openai python-dotenv

from __future__ import annotations
import os, re, json, asyncio
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment/.env")

# ---------- Model client (AG2) ----------
model = OpenAIChatCompletionClient(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

# ---------- Agents ----------
personal_info_agent = AssistantAgent(
    name="Personal_Info_Agent",
    model_client=model,
    system_message=(
        "You are a helpful onboarding agent for ACME (a phone provider).\n"
        "Task: Extract ONLY the customer's name and location from user text.\n"
        'Return STRICT JSON exactly as: {"name":"...","location":"..."}\n'
        "After the JSON, append a single line: TERMINATE\n"
        "Do not include code fences or extra text."
    ),
)

issue_agent = AssistantAgent(
    name="Issue_Agent",
    model_client=model,
    system_message=(
        "You are an onboarding agent for ACME.\n"
        "Task: From the user text, extract ONLY the product they use and the issue they are facing.\n"
        'Return STRICT JSON exactly as: {"product":"...","issue":"..."}\n'
        "After the JSON, append a single line: TERMINATE\n"
        "No code fences or extra text."
    ),
)

engagement_agent = AssistantAgent(
    name="Engagement_Agent",
    model_client=model,
    system_message=(
        "You are a friendly engagement agent.\n"
        "Given the user's personal info and topic preferences, reply with a short, fun, engaging note.\n"
        "You may include 1–2 quick facts, a light joke, or a tip. Keep it under 6 sentences.\n"
        "End with: TERMINATE"
    ),
)

# ---------- Helpers ----------
JSON_PATTERN = re.compile(r"\{.*\}", re.S)

def extract_json(payload: str) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    m = JSON_PATTERN.search(payload)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def require_terminate(payload: str) -> bool:
    return isinstance(payload, str) and "TERMINATE" in payload.upper()

def prompt_user(prompt: str) -> str:
    print("\n" + prompt.strip())
    return input("> ").strip()

def last_text(chat_result) -> str:
    """
    AutoGen AG2 ChatResult typically has .messages (list of dicts with 'content').
    """
    try:
        msgs = getattr(chat_result, "messages", None)
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                return (last.get("content") or "").strip()
            return (getattr(last, "content", "") or "").strip()
    except Exception:
        pass
    # fallback
    return (getattr(chat_result, "content", "") or "").strip()

# ---------- Phases (async) ----------
async def phase_personal_info() -> Dict[str, str]:
    print("\n=== Phase 1: Personal Info (name & location) ===")
    user_text = prompt_user(
        "Please provide your name and location (e.g., 'I'm John Doe from Gurgaon, India')."
    )

    task = (
        f"User says:\n{user_text}\n\n"
        "Extract as instructed (strict JSON + TERMINATE)."
    )
    res = await personal_info_agent.run(task=task)
    reply = last_text(res)

    if not require_terminate(reply):
        # second try with stronger reminder
        res = await personal_info_agent.run(task=task + "\nRemember: Append TERMINATE.")
        reply = last_text(res)

    data = extract_json(reply) or {}
    if not data.get("name") or not data.get("location"):
        print("Could not parse name/location reliably. Please try again.")
        return await phase_personal_info()

    print(f"Captured: name={data['name']}  location={data['location']}")
    return {"name": data["name"], "location": data["location"]}

async def phase_issue() -> Dict[str, str]:
    print("\n=== Phase 2: Product & Issue ===")
    user_text = prompt_user(
        "Describe the product you use and your issue (e.g., 'Using ACME Fiber 300Mbps, internet drops every evening')."
    )

    task = (
        f"User says:\n{user_text}\n\n"
        "Extract as instructed (strict JSON + TERMINATE)."
    )
    res = await issue_agent.run(task=task)
    reply = last_text(res)

    if not require_terminate(reply):
        res = await issue_agent.run(task=task + "\nRemember: Append TERMINATE.")
        reply = last_text(res)

    data = extract_json(reply) or {}
    if not data.get("product") or not data.get("issue"):
        print("Could not parse product/issue reliably. Please try again.")
        return await phase_issue()

    print(f"Captured: product={data['product']}  issue={data['issue']}")
    return {"product": data["product"], "issue": data["issue"]}

async def phase_engagement(profile: Dict[str, str], issue: Dict[str, str]) -> str:
    print("\n=== Phase 3: Engagement (topics you like) ===")
    topics = prompt_user("What topics do you enjoy (e.g., tech news, football, films)?")

    context = {"profile": profile, "issue": issue, "preferences": {"topics": topics}}
    task = (
        "Use the following to craft a short engaging note (<=6 sentences). "
        "End with TERMINATE.\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )
    res = await engagement_agent.run(task=task)
    reply = last_text(res)
    if not require_terminate(reply):
        reply += "\nTERMINATE"
    return reply

# ---------- Main ----------
async def main():
    print("AutoGen AG2 • Sequential Onboarding Demo (ACME)")
    profile = await phase_personal_info()
    issue = await phase_issue()
    engagement = await phase_engagement(profile, issue)

    print("\n=== SUMMARY ===")
    print(json.dumps({"profile": profile, "issue": issue}, indent=2, ensure_ascii=False))
    print("\n=== ENGAGEMENT NOTE ===")
    print(engagement)

if __name__ == "__main__":
    asyncio.run(main())
