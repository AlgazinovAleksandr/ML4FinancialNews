import json
import re

from openai import OpenAI

from .config import API_KEY, BASE_URL, MODEL_NAME
from .schemas import AgentOutput, TickerContext

_SYSTEM_PROMPT = """\
You are a quantitative equity analyst.  Your job is to assess how a news article \
will affect a stock's price on the next trading day and give a clear investment recommendation.

You will receive:
  - The stock ticker
  - The news article text
  - Company context extracted from a financial dataset (sector, recent returns, volatility)
  - An ML model forecast for the next-day price change (may be null if unavailable)

Guidelines:
  - Be conservative.  News impact is usually smaller than it first appears.
  - If the ML forecast is null, rely on the news and your general knowledge of the company.
  - confidence (1–5):  1 = very uncertain,  5 = high conviction.
    Base it on the clarity of the news signal, not on the size of the move.
  - key_factors:   2–4 concrete reasons that support your view.
  - risk_factors:  2–3 specific things that could prove the recommendation wrong.

Return ONLY a valid JSON object — no markdown, no extra text — matching this schema exactly:
{
  "price_impact":       "positive" | "neutral" | "negative",
  "recommendation":     "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
  "model_forecast_pct": <float or null>,
  "confidence":         <integer 1–5>,
  "reasoning":          "<one paragraph>",
  "key_factors":        ["<factor>", ...],
  "risk_factors":       ["<risk>", ...]
}
"""


def _build_user_message(ticker: str, news_text: str, ctx: TickerContext) -> str:
    lines = [
        f"Ticker: {ticker}",
        "",
        "News article:",
        news_text.strip(),
        "",
        "Company context (last available observation in the training dataset):",
    ]

    lines.append(f"  Sector:    {ctx.sector or 'Unknown'}")
    lines.append(f"  Industry:  {ctx.industry or 'Unknown'}")

    if ctx.marketcap is not None:
        lines.append(f"  Market cap: ${ctx.marketcap:,.0f}")

    if ctx.current_day_return is not None:
        lines.append(f"  Same-day return (article date):  {ctx.current_day_return * 100:.2f}%")
    if ctx.recent_return_2d is not None:
        lines.append(f"  2-day trailing return:           {ctx.recent_return_2d * 100:.2f}%")
    if ctx.recent_return_7d is not None:
        lines.append(f"  7-day trailing return:           {ctx.recent_return_7d * 100:.2f}%")
    if ctx.volatility_7d is not None:
        lines.append(f"  7-day historical volatility:     {ctx.volatility_7d * 100:.2f}%")

    if ctx.last_date:
        lines.append(f"  (Data as of: {ctx.last_date} — training set, not live)")

    if not ctx.found:
        lines.append("  NOTE: ticker not found in training dataset; context unavailable.")

    lines += [
        "",
        "ML model forecast (next-day % price change): "
        + (f"{ctx.forecast_pct * 100:.3f}%" if ctx.forecast_pct is not None
           else "null — model not yet integrated"),
    ]

    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    """
    Extract the first JSON object from the LLM response.
    Handles markdown code fences (```json ... ```) and stray leading/trailing text.
    """
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"Could not extract JSON from LLM response:\n{raw}")


def run_llm(ticker: str, news_text: str, ctx: TickerContext) -> AgentOutput:
    """Call the LLM and return a validated AgentOutput."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    user_message = _build_user_message(ticker, news_text, ctx)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    data = _extract_json(raw)
    # Always echo the actual model value (not whatever the LLM wrote)
    data["model_forecast_pct"] = ctx.forecast_pct
    return AgentOutput(ticker=ticker, **data)
