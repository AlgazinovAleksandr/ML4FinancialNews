from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PriceImpact(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Recommendation(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class AgentInput(BaseModel):
    ticker: str
    news_text: str


class TickerContext(BaseModel):
    """Company context retrieved from the last training-set observation for a ticker."""
    found: bool
    ticker: str
    last_date: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    marketcap: Optional[float] = None
    # Recent price momentum
    recent_return_2d: Optional[float] = None
    recent_return_7d: Optional[float] = None
    current_day_return: Optional[float] = None
    # Risk
    volatility_7d: Optional[float] = None
    # Placeholder — will hold the CatBoost prediction once the model is saved
    forecast_pct: Optional[float] = None


class AgentOutput(BaseModel):
    ticker: str
    price_impact: PriceImpact
    recommendation: Recommendation
    model_forecast_pct: Optional[float] = Field(
        default=None,
        description="Predicted next-day % price change from the ML model. "
                    "Null until the trained CatBoost model is serialized to disk.",
    )
    confidence: int = Field(ge=1, le=5, description="1 = very uncertain, 5 = high conviction")
    reasoning: str = Field(description="One-paragraph explanation of the recommendation")
    key_factors: list[str] = Field(description="2-4 specific drivers supporting the view")
    risk_factors: list[str] = Field(description="2-3 things that could invalidate this call")
