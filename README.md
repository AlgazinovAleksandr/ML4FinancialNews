# ML4FinancialNews

**This project was developed in collaboration with [Oleg Kharitonov](https://github.com/gibon228)**

Predicting next-day stock price movements from financial news text. The repository covers the full
pipeline: news and price acquisition, preprocessing, feature engineering, text vectorization,
regression/classification modeling, and an LLM agent that turns a model forecast into a structured
investment recommendation.

---

## Contents

| Folder | What's inside |
|---|---|
| `data/` | Preprocessing script, postprocessing / feature-engineering notebooks, EDA, processed datasets |
| `NewsScraper/` | Finnhub news scraper for S&P 100 tickers (with checkpointing) |
| `StockPriceScraper/` | S&P 500 daily OHLCV downloader (stooq.com, threaded) |
| `modeling/` | `vectorizer.py` + regression and classification notebooks, saved models and plots |
| `agent/` | LLM agent: CatBoost inference tool + LLM chain, FastAPI server, CLI, web UI, Dockerfile |
| `agent_structure/` | Earlier LangGraph-based agent prototype (kept for reference) |
| `literature/` | Literature review and paper draft (LaTeX, ACM/VLDB template) |
| `presentation/` | Beamer presentation of the project (metropolis theme) |

---

## Data

Raw data comes from [felixdrinkall/financial-news-dataset](https://github.com/felixdrinkall/financial-news-dataset).

**1. Decompress the raw archives**

```bash
xz -d data/*.xz
```

**2. Merge all `.json` articles into one `.csv`** — run from the folder containing the decompressed files:

```bash
python data/data_preprocessing.py
```

By default the output is saved to `news_prices.csv`. Each row is one (article, ticker) pair with
`prev_day_price`, `curr_day_price`, `next_day_price` and pre-computed sentiment/emotion scores.

**3. Postprocessing** — `data/data_postprocessing.ipynb` fills missing prices via stooq.com and
writes `data/data_sample/news_prices_full_processed.parquet` (100,428 rows × 27 columns).

**4. Target + feature generation** — `data/target_creation_feature_generation.ipynb` builds the
targets and the full feature set, producing
`data/data_sample/news_prices_full_processed_with_target_v2.parquet`
(84,658 rows × 156 columns, 43 tickers, 2017-01-04 → 2023-12-27).

**5. EDA** — `data/complex_eda.ipynb` (missing-value audit, target deep-dive, temporal/ticker/sector
patterns, sentiment and text-length signals).

> Note: `data_sample/` is git-ignored, so the parquet/CSV files are local artifacts, not part of the repo.

### Targets

- `next_curr_per_change` — **primary target**, continuous % change from current to next day close.
- `next_curr_pos_change` / `direction` — binary variant (1 = up).

### Feature groups (156 columns)

Text (`title`, `description`, `maintext`, and their concatenation), pre-computed sentiment
(neg/neu/pos) and emotion (anger, fear, joy, sadness, disgust, surprise, neutral) scores, text stats,
current-day OHLCV, lagged close/volume/S&P500 at lags {2, 3, 4, 5, 7, 14, 21, 28}, lag returns and
directions, alpha vs. the market, momentum spreads, relative volume, price position within recent
highs/lows, historical volatility, calendar features, and company metadata (sector, industry,
market cap, EBITDA, revenue growth, employees, index weight).

Leakage control: `next_day_price`, `next_curr_pos_change` and `future_date` are dropped before any
feature is constructed; every lag feature looks strictly backward from `date`.

---

## Modeling

### `modeling/vectorizer.py`

Shared vectorization module — every notebook and the agent import from it.

```python
from vectorizer import get_available_methods, preprocess, preprocess_tokenize, vectorize
```

| Method | Description | Dims |
|---|---|---|
| `tf-idf` | Sparse TF-IDF, bigrams, 10k features | 10,000 |
| `tf-idf-svd` | TF-IDF (50k vocab) → TruncatedSVD | 300 |
| `word2vec` | Skip-gram Word2Vec, mean-pooled | 300 |
| `word2vec-tfidf` | TF-IDF-weighted Word2Vec | 300 |
| `bert` | Sentence-BERT `all-MiniLM-L6-v2` | 384 |
| `finbert` | FinBERT `yiyanghkust/finbert-pretrain`, CLS token | 768 |

Torch-based methods auto-detect the device: MPS (Apple Silicon) → CUDA → CPU.

### Regression — `modeling/modeling_regression.ipynb`

CatBoost on `next_curr_per_change`, **time-based 60/20/20 split** (earliest 60% of unique dates →
train, next 20% → val, final 20% → test). Every text method is compared against a no-text baseline
on identical split indices. Test set, sorted by R²:

| Method | RMSE | MAE | R² | Dir. Accuracy | IC (Spearman) |
|---|---|---|---|---|---|
| **BERT + features** | 0.01534 | 0.01135 | 0.00006 | **0.5855** | **0.1669** |
| TF-IDF SVD + features | 0.01540 | 0.01141 | −0.0080 | 0.5560 | −0.0189 |
| Baseline (no text) | 0.01542 | 0.01147 | −0.0102 | 0.5446 | 0.0085 |
| FinBERT + features | 0.01543 | 0.01144 | −0.0116 | 0.5340 | 0.1134 |
| Word2Vec + features | 0.01558 | 0.01159 | −0.0321 | 0.4628 | −0.0977 |

**Interpretation.** Under an honest temporal split, next-day returns are essentially unpredictable in
the R² sense — every model sits at or below zero. The usable signal is *directional*: Sentence-BERT
embeddings lift directional accuracy from 54.5% to 58.6% and Spearman IC from 0.008 to 0.167 over the
no-text baseline. FinBERT adds IC but not accuracy; Word2Vec hurts. This model is the one saved to
`best_model.cbm` / `best_model_meta.json` and served by the agent.

### Regression, alternative split — `modeling/modeling_regression_stratified_split.ipynb`

Same pipeline with a **ticker-stratified random 90/5/5 split** and four model families (CatBoost,
KNN, DecisionTree, …). Best result there is Baseline (no text) + CatBoost with R² = 0.849,
directional accuracy 0.913, IC 0.924 (saved as `best_model_regression.cbm`).

⚠️ **These numbers are optimistic and should not be read as forecasting performance.** A random split
puts articles from adjacent dates for the same ticker on both sides of the split, so the model
interpolates between neighbouring price observations rather than forecasting forward. The time-based
results above are the honest ones; this notebook is kept as an ablation showing exactly how much a
leaky split inflates the metrics.

### Classification — `modeling/modeling_classification.ipynb`

Text-only direction classification (`title + description + maintext` → TF-IDF + TruncatedSVD),
stratified random 90/5/5, sweeping SVD dimensionality × {CatBoost, Logistic Regression, KNN,
Decision Tree}, selecting on validation F1.

Best: **DecisionTree, 150 components** — test Accuracy 0.531, Precision 0.524, Recall 0.977,
F1 0.683, ROC-AUC 0.547, PR-AUC 0.563.

ROC-AUC of 0.547 means the text carries only a weak directional signal, and the 0.977 recall shows
the classifier is close to a majority-class predictor. Threshold analysis
(`modeling/pictures/threshold_analysis.png`) shows precision can be traded against coverage, which is
the only regime in which this would be tradeable. Plots: `class_balance.png`,
`metrics_vs_embedding_size.png`, `roc_pr_curves.png`.

### Saved artifacts (`modeling/`)

| File | Contents |
|---|---|
| `best_model.cbm` + `best_model_meta.json` | BERT + features CatBoost (time split) — used by the agent |
| `best_model_regression.cbm` + `_meta.json` + `_scaler.pkl` | Best model from the stratified-split notebook |
| `best_clf_model.pkl`, `best_clf_svd.pkl`, `best_clf_meta.json` | Best direction classifier + its SVD pipeline |

Earlier iterations live in `modeling/modeling_iterations/` (`baseline.ipynb`,
`advanced_modeling_v_1.ipynb`, `advanced_modeling_v_2.ipynb`) and `data/iterations/`.

---

## Agent

`agent/` turns a news article into a structured recommendation in two steps:

1. **`model_tool.py`** — loads the last observation for the ticker, embeds the supplied news text
   with the same vectorizer used at training time, runs CatBoost inference → `TickerContext`
   (sector, industry, market cap, recent returns, 7-day volatility, `forecast_pct`).
2. **`llm_chain.py`** — builds a prompt from the article plus that context, calls an LLM through an
   OpenAI-compatible endpoint (OpenRouter by default), and parses the reply into a validated
   `AgentOutput`: `price_impact`, `recommendation` (strong_buy … strong_sell), `model_forecast_pct`,
   `confidence` (1–5), `reasoning`, `key_factors`, `risk_factors`.

If the saved model is missing, `forecast_pct` stays `None` and the agent falls back to LLM-only
analysis. The `model_forecast_pct` returned to the caller is always the real model output, never
whatever the LLM wrote.

**Configure**

```bash
cp agent/.env.example agent/.env
# API_KEY, MODEL_NAME, BASE_URL (defaults to https://openrouter.ai/api/v1)
pip install -r agent/requirements.txt
```

**CLI**

```bash
python -m agent.main --ticker AAPL --news "Apple reported record quarterly revenue..."
```

**API + web UI** (`agent/static/index.html`, "Stock News Analyzer")

```bash
uvicorn agent.app:app --reload --host 0.0.0.0 --port 8000
```

Endpoints: `GET /` (UI), `GET /tickers` (tickers the model was trained on), `POST /predict`.

**Docker** (build from the project root)

```bash
docker build -f agent/Dockerfile -t ml4fin-agent .
docker run -p 8000:8000 \
    -v $(pwd)/modeling:/app/modeling \
    -v $(pwd)/data:/app/data \
    --env-file agent/.env \
    ml4fin-agent
```

---

## Scrapers

- **`NewsScraper/scraper.py`** — Finnhub news for S&P 100 tickers, ~1 call/sec, resumes from the last
  saved ticker. Needs `NewsScraper/.env`:
  ```bash
  cp NewsScraper/.env.example NewsScraper/.env   # FINNHUB_API_KEY = 'your_key_here'
  ```
- **`StockPriceScraper/get_stock_prices.ipynb`** — S&P 500 daily OHLCV from stooq.com via
  `ThreadPoolExecutor`, written to `StockPriceScraper/stock_data/`.

---

## Environment

```
pandas  numpy  scikit-learn  gensim  torch  transformers  sentence-transformers
catboost  pyarrow  finnhub-python  yfinance  pandas-datareader  python-dotenv
fastapi  uvicorn  openai  pydantic
```

---

## Writeups

- `literature/literature_review.tex` + `references.bib` (35 entries) — literature review;
  `literature/main.tex` / `main.pdf` — paper draft (ACM/VLDB template).
- `presentation/main.tex` — Beamer deck covering the pipeline, methods and results.

---

## TODO

- [ ] SHAP explainability analysis on the best regression model
- [ ] Reconcile the two split strategies in the writeup; report time-based results as the headline
- [ ] Evaluate finance-specific LLMs (e.g. [fin-llama](https://github.com/Bavest/fin-llama)) against
      general-purpose ones for the recommendation step
- [ ] Finish `NewsScraper/scraper.py` (Finnhub path is still incomplete)
- [ ] Fix the Docker build for the agent
- [ ] Finalize the paper title and the results section in `literature/main.tex`

---

## References

- [felixdrinkall/financial-news-dataset](https://github.com/felixdrinkall/financial-news-dataset) — source news data
- [S&P 500 stocks dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks) — index and constituent metadata
- [stooq.com](https://stooq.com) — historical daily prices
- [Finnhub](https://finnhub.io) — news API

---

## License

MIT — see [LICENSE](LICENSE).
