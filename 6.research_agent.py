# research_autogen_ag2_tavily_structured.py
# pip install -U autogen-agentchat autogen-ext openai python-dotenv requests pydantic

from __future__ import annotations
import os, json, asyncio, time
from typing import List
import requests
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment/.env")
if not TAVILY_API_KEY:
    raise RuntimeError("Set TAVILY_API_KEY in your environment/.env")

# ---------- Tool: Tavily Search ----------
def tavily_search(query: str) -> List[dict]:
    """
    Web search via Tavily. Returns a list of {title, url, content}.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": int(6),
        "search_depth": "advanced",
        "include_answer": False,
        "include_images": False,
        "include_domains": None,
        "exclude_domains": None,
    }
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=25)
            r.raise_for_status()
            data = r.json()
            out = []
            for item in (data.get("results") or [])[:6]:
                out.append({
                    "title": item.get("title", "") or "",
                    "url": item.get("url", "") or "",
                    "content": (item.get("content") or "")[:1500],
                })
            return out
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(0.8 * (attempt + 1))
    return []  # fallback (shouldn't reach here)

# ---------- Structured Output Schema ----------
class Source(BaseModel):
    title: str
    url: str

class ResearchReport(BaseModel):
    question: str
    findings: List[str] = Field(description="Bullet points with concrete, sourced findings.")
    sources: List[Source] = Field(description="Unique, high-quality sources used.")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in the findings.")
    next_steps: List[str] = Field(default_factory=list, description="Optional follow-ups.")

# ---------- Model Client ----------
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",            # use gpt-4o-mini if you want to save cost
    api_key=OPENAI_API_KEY,
)

tool = FunctionTool(tavily_search, description="Web Search", strict=True)

# ---------- Agent (uses structured output) ----------
researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[tool],
    system_message=(
        "You are a careful research assistant.\n"
        "1) Use tavily_search(query) to find authoritative sources (recent docs, official pages, reputable blogs/papers).\n"
        "2) Synthesize concrete findings (avoid fluff). Prefer items that are verifiable from returned results.\n"
        "3) Limit sources to 3–6 unique, high-quality links and ensure they come from tavily_search output.\n"
        "4) Return a structured ResearchReport (no code fences, no extra keys).\n"
        "5) If the query is ambiguous, note assumptions in findings and suggest 'next_steps'."
    ),
    output_content_type=ResearchReport,  # ⬅️ validated structured output
    # reflect_on_tool_use=True (default when output_content_type is set)
)

# ---------- Run ----------
async def main():
    question = "What are current best practices to evaluate production RAG systems, and which open tools support them?"
    result = await researcher.run(task=question)

    # The last message is a StructuredMessage with a Pydantic object in .content
    last = result.messages[-1]
    assert isinstance(last, StructuredMessage), f"Expected StructuredMessage, got {type(last)}"
    report: ResearchReport = last.content

    print("\n=== QUESTION ===")
    print(report.question)

    print("\n=== FINDINGS ===")
    for i, f in enumerate(report.findings, 1):
        print(f"{i}. {f}")

    print("\n=== SOURCES ===")
    for s in report.sources:
        print(f"- {s.title} — {s.url}")

    print("\nConfidence:", report.confidence)
    if report.next_steps:
        print("\nNext steps:")
        for step in report.next_steps:
            print("-", step)

    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
