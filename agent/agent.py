"""
Pipeline
1. model_tool — load last training observation for the ticker, run CatBoost
                 inference (text embedding + numeric features) → TickerContext
2. llm_chain — build prompt, call LLM, parse structured JSON → AgentOutput
"""

from .llm_chain import run_llm
from .model_tool import get_ticker_context
from .schemas import AgentOutput

def run_agent(ticker: str, news_text: str) -> AgentOutput:
    context = get_ticker_context(ticker, news_text)
    return run_llm(ticker, news_text, context)
