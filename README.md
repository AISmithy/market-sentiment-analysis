# StockSense

**Market Sentiment & Risk Analytics**

A compact Python project for fetching financial data and running AI-powered sentiment analysis on recent news for a given stock ticker. The app combines Yahoo Finance data with a FinBERT-based sentiment model from Hugging Face and exposes results through a modern, interactive web UI (Django-based).

## Main objective

The primary goal of this project is to provide a lightweight, reproducible pipeline that combines financial data (prices, company profile, news) with an AI-based sentiment analysis model (FinBERT) and expose the results through an intelligent web UI. Key objectives include:

- Fetch historical price data and current price for a given stock ticker.
- Collect recent news items about the company and score each item for sentiment (Positive/Negative/Neutral) using a FinBERT model from Hugging Face.
- **Compute sentiment-driven risk scores (0-100)** based on recent news sentiment distribution.
- **Generate daily sentiment heatmap** with clickable date filtering for temporal analysis.
- **Recommend similar tickers** from the same sector with comparative sentiment analysis.
- Present the results in an interactive, modern web-friendly view with:
  - Candlestick chart for historical price trends
  - Circular SVG risk gauge with color-coded severity levels (low/medium/high)
  - Daily sentiment heatmap with sentiment-based color coding
  - Tabbed news interface (Recent/Filtered by Date)
  - Sector peer recommendations with sentiment indicators
  - Company profile, sentiment badges, confidence percentages, and raw JSON for debugging
- Support both a full analysis endpoint (news + model inference + charts) and a lightweight price-only endpoint for efficient real-time polling (30s intervals).
- Keep the core ingestion and analysis logic reusable so the same functions can power different UIs.

This repository is intended for experimentation and prototyping; it's not hardened for production use (no auth, limited rate-limiting, model loading occurs on first request). See "Development notes" and "Troubleshooting" for operational guidance.

## Quick features

- ✅ Fetch historical price data (via `yfinance`).
- ✅ Retrieve recent news items for a ticker and analyze sentiment using FinBERT (`ProsusAI/finbert`) via `transformers`.
- ✅ **Sentiment-driven risk scoring** — generates 0-100 risk score based on negative/positive/neutral news ratios.
- ✅ **Daily sentiment aggregation** — groups news by date and computes daily sentiment statistics.
- ✅ **Sector peer discovery** — recommends 3-5 related tickers from 8 industry categories (mega tech, AI/ML, cloud, semiconductors, finance, consumer, energy, pharma).
- ✅ **Interactive heatmap** — clickable daily sentiment timeline with color-coded sentiment levels.
- ✅ **Live polling** — automatic price/risk updates every 30 seconds.
- ✅ **Modern responsive UI** — CSS Grid layout, card-based design, hover effects, sentiment badges, confidence percentages.
- ✅ **News filtering** — toggle between all news and date-filtered views via tabbed interface.

## Repository layout

- `src/` — core ingestion and analysis logic (re-usable between UIs):
	- `data_ingestion.py` — helpers for fetching stock data, company info and news.
	- `sentiment_analyzer.py` — loads the Hugging Face FinBERT pipeline and maps model outputs.
	- `utils.py` — small utilities (logging, helpers).
- `script/` — standalone utility scripts for model evaluation and dataset creation:
	- `baseline_finbert_eval.py` — evaluate FinBERT on labeled headline datasets; outputs metrics, confusion matrix, and optional APA-formatted .docx report.
	- `build_silver_dataset.py` — fetch news for a ticker and label each headline with FinBERT sentiment; saves CSV for training/evaluation.
- `market_site/` — Django project scaffolding (development server and settings).
- `sentiment/` — Django app that exposes the web UI and JSON endpoints (`/analyze/`, `/price/`).
- `templates/sentiment/` — Django HTML templates for the web UI.
- `config/settings.py` — basic configuration (model name, etc.).
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes cache, reports, build artifacts, and virtual environments from git tracking.

## Requirements

- Python 3.8+ (3.10/3.11 recommended)
- Recommended system: Windows, macOS or Linux with access to internet for model download and Yahoo Finance APIs.

Python dependencies are listed in `requirements.txt` and include:

- **Data & Finance**: pandas, yfinance, yahoo_fin
- **Web/UI**: django, plotly
- **ML/NLP**: transformers, torch, accelerate
- **Evaluation**: matplotlib, scikit-learn
- **Reporting**: python-docx

Install them with pip:

```powershell
python -m pip install -r requirements.txt
```

If you use a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Configuration

The simple configuration for the sentiment model is in `config/settings.py`:

```python
# config/settings.py
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
```

You can change the model name to another Hugging Face-compatible model if needed. Keep in mind some models require GPU or specific tokenizer handling.

## API Endpoints

The Django app exposes two main HTTP endpoints for querying sentiment and risk data:

### `/analyze/?ticker=XXX`

Runs full analysis for the given ticker and returns comprehensive JSON data. Used on initial page load.

**Response includes:**
- `ticker` — the ticker symbol
- `company` — company profile (name, sector, market_cap, business_summary, current_price, change, change_pct)
- `news` — array of analyzed news items with `title`, `link`, `published`, `sentiment_label`, `sentiment_score`
- `risk_score` — 0-100 risk score computed from sentiment distribution
- `risk_level` — 'low' (0-32), 'medium' (33-66), or 'high' (67-100)
- `risk_sentiment_counts` — breakdown of positive/negative/neutral articles
- `daily_stats` — array of daily sentiment statistics (by date: count, positive, negative, neutral, avg_score, dominant_sentiment, net_sentiment)
- `ticker_suggestions` — array of related tickers with their sentiment profiles (for peer comparison)
- `history` — historical candlestick data (OHLCV format)

### `/price/?ticker=XXX`

Lightweight endpoint for real-time price and risk updates. Polled every 30 seconds during the session.

**Response includes:**
- `current_price`, `previous_close`, `change`, `change_pct` — price data
- `risk_score`, `risk_level`, `risk_sentiment_counts` — live risk metrics

Quick start (Windows PowerShell):

```powershell
# create and activate a virtual environment (if not already created)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install -r requirements.txt

# apply Django migrations (SQLite DB used by default)
# using the convenience launcher `app.py`
.\.venv\Scripts\python.exe app.py migrate

# start the development server (binds to 127.0.0.1:8000 by default)
.\.venv\Scripts\python.exe app.py runserver
```

## UI Features

The modern dashboard (`http://127.0.0.1:8000/`) provides:

### Layout (Responsive CSS Grid)

- **Header** — StockSense branding with tagline, ticker search box, Show JSON toggle
- **Company Card** — gradient header with ticker, company name, current price (color-coded delta), sector, market cap, business summary
- **KPI Row** — side-by-side risk gauge (SVG circular meter) and sentiment distribution pie chart
- **Candlestick Chart** — Plotly interactive historical price chart (OHLCV)
- **Heatmap + Peers** — two-column layout with daily sentiment timeline (clickable cells) and sector peer cards
- **News Section** — tabbed interface (Recent/Filtered by Date) with scrollable list of news cards
- **Ticker Suggestions** — grid of related tickers with sentiment-based border colors and stat breakdown
- **Raw JSON** — toggle-able debug view of API response

### Interactive Elements

- **Risk Gauge** — circular SVG arc that visualizes risk (green=low, orange=medium, red=high), updates on polling
- **Heatmap Cells** — click to filter news by date; hover for tooltip with sentiment breakdown
- **News Cards** — hover effects, sentiment badge (green/yellow/red), confidence percentage, publication date
- **Ticker Cards** — click to navigate to related ticker; color-coded border by dominant sentiment
- **Tab Switching** — toggle between all news and date-filtered news without page reload
- **Live Polling** — price and risk gauge update automatically every 30 seconds

Notes:
- The first request that triggers model loading may be slow while the FinBERT weights download. Consider pre-warming the model in a background task if you plan to serve many users.
- The Django app is intended for local development and prototyping; production deployment needs additional work (WSGI/ASGI configuration, reverse proxy, caching, and security).

## Common tasks

- Update dependencies:

```powershell
python -m pip install --upgrade -r requirements.txt
```

- Run a quick smoke test (start the app and try a ticker like `AAPL`).

- Evaluate the FinBERT baseline on a labeled CSV:

```powershell
python script/baseline_finbert_eval.py --data data/headlines.csv --out runs/baseline_eval
```

- Build a labeled dataset from news headlines:

```powershell
python script/build_silver_dataset.py --ticker AAPL --out data/aapl_headlines.csv --limit 300
```

## Development notes

## Code Architecture

### Services Layer (`sentiment/services.py`)

Core business logic functions that compute analytics from raw news data:

- `get_company_data(ticker)` — fetches company profile, historical prices, and current price info
- `analyze_news_for_ticker(ticker)` — fetches recent news and analyzes sentiment
- `compute_risk_score(news_items)` — **NEW:** calculates 0-100 risk score from sentiment distribution
  - Base risk: 50 (neutral)
  - Negative sentiment: increases risk (weighted +50 per article)
  - Positive sentiment: decreases risk (weighted -25 per article)
  - Returns: `{risk_score, risk_level, sentiment_counts}`
- `compute_daily_sentiment_stats(news_items)` — **NEW:** groups news by date, computes daily metrics
  - Returns: dict keyed by date (YYYY-MM-DD) with `{count, positive, negative, neutral, avg_score, dominant_sentiment, net_sentiment}`
- `get_related_tickers(ticker)` — **NEW:** maps ticker to industry group, returns 3-5 related tickers
  - Supports 8 categories: mega_tech, ai_ml, cloud, semiconductors, finance, consumer, energy, pharma
- `compute_sentiment_summary(news_items)` — **NEW:** fast aggregation of sentiment stats
- `get_ticker_suggestions_with_sentiment(ticker)` — **NEW:** fetches related tickers with sentiment profiles

### Views (`sentiment/views.py`)

Django view handlers that orchestrate services and format responses:

- `index()` — renders the main UI template
- `analyze(request)` — `/analyze/` endpoint that calls all service functions and returns full JSON response
- `price(request)` — `/price/` endpoint that returns lightweight price + risk data for polling

### Frontend (`templates/sentiment/index.html`)

Interactive dashboard built with vanilla JavaScript, CSS Grid, SVG, and Plotly:

- `updateRiskGauge(riskScore, riskLevel, sentimentCounts)` — updates SVG circular gauge with color-coded arc
- `renderHeatmap(dailyStats)` — creates clickable date cells with sentiment coloring
- `filterNewsByDate(date, cellEl)` — toggles date filter and updates news display
- `displayNewsList(newsArray)` — renders news items as styled cards with sentiment badges
- `renderTickerSuggestions(suggestions)` — creates grid of peer ticker cards with sentiment indicators
- `pollPrice()` — 30-second polling for live price/risk updates

- Git and artifacts:
	- The `.gitignore` file excludes cache/, reports/, runs/, __pycache__/, .venv/, and other temporary artifacts.
	- All evaluation outputs (confusion matrices, metrics, reports) are saved to `runs/` or `reports/` and are not tracked in git.

## Troubleshooting

- Transformer/torch errors on model load:
	- Ensure `torch` is installed and compatible with your system (CPU-only builds are available). On Windows, prefer installing a matching `torch` wheel as recommended by PyTorch's website.
	- If memory is limited, consider using a smaller model or running on CPU.

- No news / empty responses:
	- Yahoo RSS feeds and `yahoo_fin` rely on the external service; if a ticker returns no news the app will show an empty list.

## Tests

This repository does not include automated tests yet. For small additions, add unit tests under a `tests/` folder and run them with `pytest`.

Smoke test:
```powershell
# Import all modules and validate they load without error
python -c "from src.data_ingestion import get_stock_news; from src.sentiment_analyzer import load_sentiment_model; print('All modules OK')"
```

## Contributing

Contributions and issues are welcome. A typical workflow:

1. Fork the repo and create a feature branch.
2. Add tests for new behavior where appropriate.
3. Open a pull request describing the change.


