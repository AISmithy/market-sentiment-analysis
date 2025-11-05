# Market Sentiment Analysis

A compact Python project for fetching financial data and running AI-powered sentiment analysis on recent news for a given stock ticker. The app combines Yahoo Finance data with a FinBERT-based sentiment model from Hugging Face and exposes results through a web UI (Django-based by default).

## Main objective

The primary goal of this project is to provide a lightweight, reproducible pipeline that combines financial data (prices, company profile, news) with an AI-based sentiment analysis model (FinBERT) and expose the results through a simple web UI. Key objectives include:

- Fetch historical price data and current price for a given stock ticker.
- Collect recent news items about the company and score each item for sentiment (Positive/Negative/Neutral) using a FinBERT model from Hugging Face.
- Present the results in an interactive, web-friendly view: candlestick chart, sentiment distribution chart, readable news list with sentiment badges, company profile and raw JSON for debugging.
- Support both a full analysis endpoint (news + model inference + charts) and a lightweight price-only endpoint for efficient real-time polling.
- Keep the core ingestion and analysis logic reusable across UIs (Streamlit and Django) so the same functions can power different frontends.

This repository is intended for experimentation and prototyping; it's not hardened for production use (no auth, limited rate-limiting, model loading occurs on first request). See "Development notes" and "Troubleshooting" for operational guidance.

## Quick features

- Fetch historical price data (via `yfinance`).
- Retrieve recent news items for a ticker and analyze sentiment using a FinBERT model (`ProsusAI/finbert`) via `transformers`.
- Interactive web dashboard with candlestick chart, company profile, and news sentiment summaries (Django templates and Plotly on the client).

## Repository layout

- `src/` — core ingestion and analysis logic (re-usable between UIs):
	- `data_ingestion.py` — helpers for fetching stock data, company info and news.
	- `sentiment_analyzer.py` — loads the Hugging Face FinBERT pipeline and maps model outputs.
	- `utils.py` — small utilities (logging, helpers).
- `market_site/` — Django project scaffolding (development server and settings).
- `sentiment/` — Django app that exposes the web UI and JSON endpoints (`/analyze/`, `/price/`).
- `config/settings.py` — basic configuration (model name, etc.).
- `requirements.txt` — Python dependencies.

## Requirements

- Python 3.8+ (3.10/3.11 recommended)
- Recommended system: Windows, macOS or Linux with access to internet for model download and Yahoo Finance APIs.

Python dependencies are listed in `requirements.txt` and include:

- pandas, yfinance, yahoo_fin, streamlit, plotly, transformers, torch, accelerate

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

## Streamlit (removed / optional)

This project now uses Django as the primary web UI. The Streamlit prototype that used to live in `src/dashboard.py` has been marked optional. If you do not need Streamlit, you can remove `src/dashboard.py` and delete `streamlit` from `requirements.txt` to keep the repository lean.

Notes about model download and caching:
- The FinBERT model is downloaded the first time the `transformers` pipeline is created. This requires internet connectivity and may take a minute.
- The core `src/` modules were made import-safe so they can be used under the Django app without Streamlit runtime present.

## Running the app (Django)

This repository also includes a minimal Django app that wraps the same ingestion and analysis logic and provides two HTTP endpoints useful for web UIs:

- `/analyze/?ticker=XXX` — runs the full analysis (news fetch + sentiment scoring + history) and returns JSON used to render the full UI.
- `/price/?ticker=XXX` — lightweight endpoint that returns only the current price and change (useful for frequent polling).

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

Open `http://127.0.0.1:8000/` in your browser. The default UI accepts a `ticker` and will call `/analyze/` to render the full page; the client also polls `/price/` for lightweight updates.

### History endpoint

You can now fetch historical OHLCV data via the lightweight `/history/` endpoint. Example:

GET (curl):

```powershell
curl "http://127.0.0.1:8000/history/?ticker=AAPL&period=6mo&interval=1d"
```

Python (requests):

```python
import requests
r = requests.get('http://127.0.0.1:8000/history/', params={'ticker':'AAPL','period':'6mo','interval':'1d'})
data = r.json()
history = data.get('history', [])
print(len(history), history[:2])
```

Query params supported: `ticker` (required), `period` (default `1y`), `interval` (default `1d`), `start` and `end` (ISO YYYY-MM-DD to override period).

Note: the `/history/` endpoint uses a short server-side cache to reduce repeated calls to external services — results are cached in-memory for 300 seconds by default. This cache is lightweight and intended for development; for production consider using Django's cache framework or an external cache (Redis/Memcached) for persistence and multi-process sharing.

Bypass cache:

You can force the `/history/` endpoint to bypass the server cache by adding `?nocache=1` to the request. Example:

```powershell
curl "http://127.0.0.1:8000/history/?ticker=AAPL&period=6mo&interval=1d&nocache=1"
```

Running with Redis (recommended for multi-process deployments):

1. Install `django-redis` into your virtualenv:

```powershell
python -m pip install django-redis
```

2. Start Redis and the web service using the provided `docker-compose.yml` example (it sets REDIS_URL for the web container):

```powershell
docker-compose up --build
```

3. When using Redis, the Django cache backend will be used and the `/history/` results will be stored in Redis and available to all web workers. The local in-memory index/enforcement is only used for the default locmem backend.

Management command: clear_history_cache

There is a management command to clear history-related cache keys. It works with Redis (via `django-redis`) or the default locmem index:

```powershell
# clear keys matching the default pattern (history:*)
python app.py clear_history_cache

# specify a different pattern
python app.py clear_history_cache --pattern "history:*"
```

Docker Compose

An example `docker-compose.yml` and `Dockerfile` are provided to run Redis and a web image with the application's Python requirements preinstalled. To run the stack:

```powershell
docker-compose up --build
```

This builds the `web` image (installs `requirements.txt` into a virtualenv inside the image) and starts Redis and the Django dev server bound to 0.0.0.0:8000. The `REDIS_URL` environment variable is passed into the container so Django will pick the Redis cache backend automatically.

Notes:
- The first request that triggers model loading may be slow while the FinBERT weights download. Consider pre-warming the model in a background task if you plan to serve many users.
- The Django app is intended for local development and prototyping; production deployment needs additional work (WSGI/ASGI configuration, reverse proxy, caching, and security).

## Common tasks

- Update dependencies:

```powershell
python -m pip install --upgrade -r requirements.txt
```

- Run a quick smoke test (start the app and try a ticker like `AAPL`).

## Development notes

- Code entry points:
	- `app.py` — convenience launcher for the Django dev server (aliases manage commands; defaults to `runserver`).
	- `sentiment/services.py` — Django-side wrappers that call into `src/` and prepare JSON for templates.
	- `src/data_ingestion.py` — contains `get_stock_data`, `get_company_info`, `get_stock_news`.
	- `src/sentiment_analyzer.py` — contains `load_sentiment_model` and `analyze_sentiment`.
	- `src/dashboard.py` — optional Streamlit prototype (kept for quick visualization experiments).

- Caching and development notes:
	- Streamlit caching was used in the prototype to reduce repeated downloads and model initialization. The `src/` helpers were made import-safe so they can be used by Django without Streamlit runtime.
	- If you are iterating on model or data functions, restarting the Django devserver or clearing any Streamlit cache (if you run the prototype) ensures fresh behavior.

- If you want to replace the sentiment pipeline with a different approach (local model, remote API), update `src/sentiment_analyzer.py` and keep the `analyze_sentiment(text, classifier)` contract:

	- inputs: `text: str`, `classifier: HF pipeline or similar`
	- outputs: dict with keys `label` (one of `Positive`/`Negative`/`Neutral`) and `score` (float)

## Troubleshooting

- Transformer/torch errors on model load:
	- Ensure `torch` is installed and compatible with your system (CPU-only builds are available). On Windows, prefer installing a matching `torch` wheel as recommended by PyTorch's website.
	- If memory is limited, consider using a smaller model or running on CPU.

- No news / empty responses:
	- Yahoo RSS feeds and `yahoo_fin` rely on the external service; if a ticker returns no news the app will show an empty list.

- Streamlit shows cached / stale data:
	- Use the Streamlit menu to "Clear cache" or restart the app.

## Tests

This repository does not include automated tests yet. For small additions, add unit tests under a `tests/` folder and run them with `pytest`.

## Contributing

Contributions and issues are welcome. A typical workflow:

1. Fork the repo and create a feature branch.
2. Add tests for new behavior where appropriate.
3. Open a pull request describing the change.


