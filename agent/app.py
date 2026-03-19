"""
app.py — FastAPI demo server
----------------------------
Run locally:
    uvicorn agent.app:app --reload --host 0.0.0.0 --port 8000

Run via Docker (build from project root):
    docker build -f agent/Dockerfile -t ml4fin-agent .
    docker run -p 8000:8000 \
        -v $(pwd)/modeling:/app/modeling \
        -v $(pwd)/data:/app/data \
        --env-file agent/.env \
        ml4fin-agent
"""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent import run_agent
from .schemas import AgentInput

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="ML4FinancialNews Agent")

_STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/tickers")
def tickers():
    """Return the list of tickers the model was trained on."""
    meta_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "modeling", "best_model_meta.json")
    )
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        known = sorted(
            n.replace("ticker_", "")
            for n in meta.get("feature_names", [])
            if n.startswith("ticker_") and n != "ticker_OTHER"
        )
        if known:
            return JSONResponse(known)

    # Fallback — S&P 100 subset used in training
    fallback = [
        "AAPL", "ADBE", "AMZN", "AVGO", "BA", "BAC", "C", "CMCSA", "COST",
        "CRM", "CSCO", "CVX", "DIS", "GE", "GOOGL", "GS", "HD", "INTC",
        "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MRNA", "MSFT",
        "MU", "NFLX", "NVDA", "ORCL", "PFE", "PG", "PYPL", "QCOM", "T",
        "TSLA", "UNH", "V", "VZ", "WFC", "WMT", "XOM",
    ]
    return JSONResponse(fallback)


@app.post("/predict")
def predict(body: AgentInput):
    """Run the full agent pipeline and return a structured recommendation."""
    try:
        result = run_agent(ticker=body.ticker, news_text=body.news_text)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
