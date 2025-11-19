# streamlit_autogen_ag2_financial.py
# pip install -U autogen autogen-agentchat autogen-ext openai streamlit yfinance python-dotenv

from __future__ import annotations
import os, json, re, asyncio, inspect
from datetime import datetime

import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# --- AutoGen AG2 imports ---
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Financial Insights (AG2)", layout="wide")
st.title("📈 Financial Insights — AutoGen AG2 (gpt-4o-mini)")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in your environment/.env"); st.stop()

# ---------- Tool (async wrapper) ----------
async def fetch_stock_data(ticker: str) -> dict:
    """
    Return key ratios + 1-month daily close prices for a ticker.
    Runs yfinance in a thread to avoid blocking the event loop.
    """
    def _work():
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            hist = t.history(period="1mo")
            prices = {}
            if not hist.empty and "Close" in hist.columns:
                prices = {str(idx.date()): float(val) for idx, val in hist["Close"].dropna().items()}
            return {
                "name": info.get("longName") or info.get("shortName") or "",
                "symbol": ticker.upper(),
                "pe_ratio": float(info["trailingPE"]) if info.get("trailingPE") is not None else None,
                "forward_pe": float(info["forwardPE"]) if info.get("forwardPE") is not None else None,
                "dividend_rate": float(info["dividendRate"]) if info.get("dividendRate") is not None else None,
                "price_to_book": float(info["priceToBook"]) if info.get("priceToBook") is not None else None,
                "debt_to_equity": float(info["debtToEquity"]) if info.get("debtToEquity") is not None else None,
                "roe": float(info["returnOnEquity"]) if info.get("returnOnEquity") is not None else None,
                "prices": prices,
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    return await asyncio.to_thread(_work)

# ---------- Model clients ----------
model_client = OpenAIChatCompletionClient(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
)

writer_model_client = OpenAIChatCompletionClient(
    api_key=OPENAI_API_KEY,
    model="gpt-4o",   # stronger for writing
)

# ---------- Agents (stream OFF for simple return handling) ----------
financial_agent = AssistantAgent(
    name="FinancialAgent",
    model_client=model_client,
    tools=[fetch_stock_data],           # pass async tool directly
    system_message=(
        "You are a careful financial analyst.\n"
        "1) Call fetch_stock_data(ticker) exactly once using the provided ticker.\n"
        "2) RETURN ONLY strict JSON with keys:\n"
        '   {"name":..., "symbol":..., "pe_ratio":..., "forward_pe":..., "dividend_rate":..., '
        '"price_to_book":..., "debt_to_equity":..., "roe":..., "prices":{date->close}}\n'
        "No explanations. No code fences. No extra keys."
    ),
    model_client_stream=False,
)

writer_agent = AssistantAgent(
    name="Writer",
    model_client=writer_model_client,
    system_message=(
        "You are a professional financial report writer. Given valid JSON of metrics & prices, "
        "write a polished **Markdown** report containing:\n"
        "- Overview (company, symbol)\n"
        "- Key metrics table\n"
        "- 1-month price summary (aggregate stats, brief trend)\n"
        "- 2–3 scenario outlook bullets\n"
        "Return ONLY Markdown. No code fences."
    ),
    model_client_stream=False,
)

# ---------- Helpers ----------
def _strip_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)  # drop ```json or ```
        s = s.rstrip("`").strip()
    return s

def _content_from_message(msg) -> str:
    """
    Robustly extract a string from a single message (object or dict).
    """
    # object-style message
    if hasattr(msg, "content"):
        cont = msg.content
        if isinstance(cont, str):
            return cont
        # some SDKs return list-of-parts like [{"type":"text","text":"..."}]
        if isinstance(cont, list):
            parts = []
            for p in cont:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    parts.append(p.get("text") or p.get("content") or "")
            return "\n".join([x for x in parts if x])
    # dict-style message
    if isinstance(msg, dict):
        cont = msg.get("content", "")
        if isinstance(cont, str):
            return cont
        if isinstance(cont, list):
            parts = []
            for p in cont:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    parts.append(p.get("text") or p.get("content") or "")
            return "\n".join([x for x in parts if x])
    return ""

def _extract_last_text(result) -> str:
    """
    Get the last assistant text from an AG2 run result.
    """
    # 1) messages attribute (common)
    msgs = getattr(result, "messages", None)
    if isinstance(msgs, list) and msgs:
        # take the last non-empty content, scanning backwards
        for m in reversed(msgs):
            text = _content_from_message(m)
            if text:
                return text

    # 2) reply attribute
    reply = getattr(result, "reply", None)
    if reply:
        text = _content_from_message(reply)
        if text:
            return text

    # 3) summary (sometimes present)
    summary = getattr(result, "summary", None)
    if isinstance(summary, str) and summary.strip():
        return summary.strip()

    # 4) content attribute directly
    if hasattr(result, "content") and isinstance(result.content, str):
        return result.content

    # 5) Nothing found
    return ""

async def run_workflow(ticker: str, debug: bool = False):
    date_str = datetime.now().strftime("%Y-%m-%d")
    task_analyst = (
        f"Today is {date_str}. For ticker '{ticker}', call fetch_stock_data('{ticker}') once and return ONLY the strict JSON."
    )

    # Step 1: Ask analyst to fetch & return strict JSON
    res1 = await financial_agent.run(task=task_analyst)
    text1 = _extract_last_text(res1)
    text1 = _strip_fences(text1)

    # Try to parse JSON
    data_json = None
    if text1:
        try:
            data_json = json.loads(text1)
        except Exception:
            m = re.search(r"\{.*\}", text1, re.S)
            if m:
                try:
                    data_json = json.loads(m.group(0))
                except Exception:
                    pass

    # Fallback: if model didn’t return parsable JSON, call the tool directly
    if not isinstance(data_json, dict) or not data_json:
        data_json = await fetch_stock_data(ticker)
        # If even tool fails, bubble up minimal error
        if not isinstance(data_json, dict) or not data_json:
            data_json = {"error": "Could not obtain data", "raw_model_output": text1}

    # Step 2: Ask writer to produce Markdown from JSON
    task_writer = (
        "Write a markdown report from this JSON (do not add code fences):\n"
        f"{json.dumps(data_json, ensure_ascii=False)}"
    )
    res2 = await writer_agent.run(task=task_writer)
    report_md = _extract_last_text(res2)

    return data_json, (report_md or "_No report generated._"), (res1 if debug else None), (res2 if debug else None)

# ---------- UI ----------
ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, TCS.NS):")
show_debug = st.checkbox("Show debug info")
btn = st.button("Run Analysis")

if btn and ticker:
    with st.spinner("Running AG2 workflow..."):
        data, report, dbg1, dbg2 = asyncio.run(run_workflow(ticker, debug=show_debug))

    st.subheader("Structured Data (JSON)")
    st.json(data)
    st.subheader("Report (Markdown)")
    st.markdown(report, unsafe_allow_html=True)

    if show_debug:
        with st.expander("🔧 Debug: financial_agent result"):
            st.write(repr(dbg1))
            st.write("Has .messages:", hasattr(dbg1, "messages"))
            if hasattr(dbg1, "messages"):
                st.write([type(m).__name__ for m in dbg1.messages])
                # Try to pretty-print messages content
                st.write([getattr(m, "content", getattr(m, "get", lambda k, d=None: None)("content", "")) for m in dbg1.messages])
        with st.expander("🔧 Debug: writer_agent result"):
            st.write(repr(dbg2))

st.caption(
    "AG2 demo • Packages: autogen, autogen-ext, openai, streamlit, yfinance • "
    "Models: gpt-4o-mini (analysis) + gpt-4o (writing)"
)
