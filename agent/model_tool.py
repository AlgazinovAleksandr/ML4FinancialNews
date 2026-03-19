"""

Loads the saved CatBoost model, retrieves the last training-set observation for
a given ticker, reconstructs the feature vector, runs inference, and returns a
TickerContext with a populated forecast_pct.

"""

import json as _json
import os
import sys

import numpy as np
import pandas as pd

from .config import DATA_PATH
from .schemas import TickerContext

_MODELING_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "modeling"))
_MODEL_CBM      = os.path.join(_MODELING_DIR, "best_model.cbm")
_MODEL_PKL      = os.path.join(_MODELING_DIR, "best_model.pkl")
_MODEL_META     = os.path.join(_MODELING_DIR, "best_model_meta.json")
_SCALER_PKL     = os.path.join(_MODELING_DIR, "best_model_scaler.pkl")
_TEXT_PKL       = os.path.join(_MODELING_DIR, "best_text_model.pkl")   # TF-IDF SVD pipeline
_TEXT_BIN       = os.path.join(_MODELING_DIR, "best_text_model.bin")   # Word2Vec KeyedVectors

_df_cache: pd.DataFrame | None = None
_model_cache = None
_scaler_cache = None
_meta_cache: dict | None = None

def _load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_parquet(DATA_PATH)
        _df_cache["date"] = pd.to_datetime(_df_cache["date"])
    return _df_cache


def _load_model():
    """Load the model, scaler, and metadata. Returns (None, None, None) if files are missing."""
    global _model_cache, _scaler_cache, _meta_cache

    if _meta_cache is None:
        if not os.path.exists(_MODEL_META):
            return None, None, None
        with open(_MODEL_META) as f:
            _meta_cache = _json.load(f)

    if _model_cache is None:
        model_type = _meta_cache.get("model_type", "catboost")
        if model_type == "catboost":
            if not os.path.exists(_MODEL_CBM):
                return None, None, None
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(_MODEL_CBM)
            _model_cache = m
        else:
            if not os.path.exists(_MODEL_PKL):
                return None, None, None
            import pickle
            with open(_MODEL_PKL, "rb") as f:
                _model_cache = pickle.load(f)

    if _scaler_cache is None and os.path.exists(_SCALER_PKL):
        import pickle
        with open(_SCALER_PKL, "rb") as f:
            _scaler_cache = pickle.load(f)

    return _model_cache, _scaler_cache, _meta_cache

def _count_json_list(s) -> int:
    try:
        lst = _json.loads(str(s))
        return len(lst) if isinstance(lst, list) else 0
    except Exception:
        return 0


def _build_numeric_vector(row: pd.Series, feature_names: list[str]) -> np.ndarray:
    """
    Reconstruct the numeric + one-hot portion of the feature vector from a
    single dataset row, matching the exact feature names used at training time.

    One-hot groups handled:
      dow_*     ← date_day_of_week column
      ticker_*  ← ticker column  (unknown tickers map to ticker_OTHER)
      sector_*  ← sector column
    """
    extras = {
        "n_mentioned": float(_count_json_list(row.get("mentioned_companies", "[]"))),
        "n_related":   float(_count_json_list(row.get("related_companies",   "[]"))),
    }

    result: list[float] = []
    for fname in feature_names:
        if fname.startswith("dow_"):
            val = float(f"dow_{row.get('date_day_of_week', '')}" == fname)

        elif fname.startswith("ticker_"):
            canonical = f"ticker_{row.get('ticker', '')}"
            if canonical not in feature_names:
                canonical = "ticker_OTHER"
            val = float(canonical == fname)

        elif fname.startswith("sector_"):
            val = float(f"sector_{row.get('sector', '') or ''}" == fname)

        elif fname in extras:
            val = extras[fname]

        else:
            raw = row.get(fname)
            try:
                val = float(raw) if raw is not None else 0.0
                if np.isnan(val):
                    val = 0.0
            except (TypeError, ValueError):
                val = 0.0

        result.append(val)

    return np.array(result, dtype=np.float32).reshape(1, -1)


def _get_text_embedding(news_text: str, text_method: str, n_text_dims: int) -> np.ndarray:
    """
    Vectorize news_text using the same method as training.

    For tf-idf-svd / word2vec the saved model file is loaded; for bert / finbert
    the pre-trained transformer is loaded directly from HuggingFace (no file needed).
    The modeling/ directory is added to sys.path to reuse vectorizer.py.
    """
    if _MODELING_DIR not in sys.path:
        sys.path.insert(0, _MODELING_DIR)

    from vectorizer import preprocess_tokenize, vectorize  # type: ignore

    preprocessed, tokenized = preprocess_tokenize([news_text])

    if text_method == "tf-idf-svd" and os.path.exists(_TEXT_PKL):
        import pickle
        with open(_TEXT_PKL, "rb") as f:
            saved_pipeline = pickle.load(f)
        # Apply the saved fitted pipeline directly (no re-fitting)
        X_text = saved_pipeline.transform(preprocessed)
        if hasattr(X_text, "toarray"):
            X_text = X_text.toarray()
        return X_text.reshape(1, -1).astype(np.float32)

    if text_method == "word2vec" and os.path.exists(_TEXT_BIN):
        from gensim.models import KeyedVectors  # type: ignore
        kv = KeyedVectors.load(_TEXT_BIN)
        # Mean-pool word vectors
        vectors = [kv[w] for tokens in tokenized for w in tokens if w in kv]
        mean_vec = np.mean(vectors, axis=0) if vectors else np.zeros(kv.vector_size)
        return mean_vec.reshape(1, -1).astype(np.float32)

    # bert / finbert — or fallback when saved files are missing
    X_text, _ = vectorize(text_method, preprocessed, tokenized)
    return X_text.reshape(1, -1).astype(np.float32)

def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if np.isnan(v) else round(v, 6)
    except Exception:
        return None


def get_ticker_context(ticker: str, news_text: str = "") -> TickerContext:
    """
    Return a TickerContext for the given ticker.

    If the saved CatBoost model is available, runs a forecast and populates
    forecast_pct.  Otherwise forecast_pct stays None and the agent relies on
    LLM-only analysis.

    Parameters
    ----------
    ticker    : Stock ticker symbol, e.g. "AAPL".
    news_text : The news article text supplied by the user.  Used for the text
                embedding when the best model includes a text method.
    """
    df = _load_data()
    ticker_upper = ticker.strip().upper()

    subset = df[df["ticker"] == ticker_upper].sort_values("date")
    if subset.empty:
        return TickerContext(found=False, ticker=ticker_upper)

    row = subset.iloc[-1]

    ctx = TickerContext(
        found=True,
        ticker=ticker_upper,
        last_date=str(row["date"].date()),
        sector=str(row.get("sector", "") or "") or None,
        industry=str(row.get("industry", "") or "") or None,
        marketcap=_safe_float(row.get("marketcap")),
        recent_return_2d=_safe_float(row.get("ret_lag2")),
        recent_return_7d=_safe_float(row.get("ret_lag7")),
        current_day_return=_safe_float(row.get("curr_prev_per_change")),
        volatility_7d=_safe_float(row.get("hist_volatility_7d")),
        forecast_pct=None,
    )

    # ИНФЕРЕНС!
    model, scaler, meta = _load_model()
    if model is None or meta is None:
        return ctx   # model not yet saved — agent will rely on LLM only

    feature_names = meta["feature_names"]
    text_method   = meta.get("text_method")   # None for baseline
    model_type    = meta.get("model_type", "catboost")

    # Identify how many leading entries in feature_names are text dimensions
    _TEXT_PREFIXES = ("tfidf_", "w2v_", "bert_", "finbert_")
    n_text        = sum(1 for f in feature_names if f.startswith(_TEXT_PREFIXES))
    numeric_names = feature_names[n_text:]    # the non-text portion

    # Build numeric feature vector from the last training observation
    X_num = _build_numeric_vector(row, numeric_names)

    if text_method and n_text > 0 and news_text.strip():
        X_text = _get_text_embedding(news_text, text_method, n_text)
        X = np.hstack([X_text, X_num])
    else:
        X = X_num

    # Non-CatBoost models were trained on scaled features
    if model_type != "catboost" and scaler is not None:
        X = scaler.transform(X)

    ctx.forecast_pct = round(float(model.predict(X)[0]), 6)
    return ctx
