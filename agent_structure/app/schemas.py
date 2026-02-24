from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class RecommendationCategory(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"



class AgentInput(BaseModel):
    news: str
    ticker: str
    current_price: float
    model_forecast_pct: float
    metadata: Optional[Dict[str, Any]] = None


class AgentOutput(BaseModel):
    recommendation_category: RecommendationCategory
    confidence: int = Field(ge=1, le=5)
    reasoning: str