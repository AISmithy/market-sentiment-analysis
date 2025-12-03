# Market Sentiment Analysis

A compact Python project for fetching financial data and running AI-powered sentiment analysis on recent news for a given stock ticker. The app combines Yahoo Finance data with a FinBERT-based sentiment model from Hugging Face and exposes results through a web UI (Django-based by default).

## Main objective

The primary goal of this project is to provide a lightweight, reproducible pipeline that combines financial data (prices, company profile, news) with an AI-based sentiment analysis model (FinBERT) and expose the results through a simple web UI. Key objectives include:

- Fetch historical price data and current price for a given stock ticker.
- Collect recent news items about the company and score each item for sentiment (Positive/Negative/Neutral) using a FinBERT model from Hugging Face.
- Present the results in an interactive, web-friendly view: candlestick chart, sentiment distribution chart, readable news list with sentiment badges, company profile and raw JSON for debugging.
- Support both a full analysis endpoint (news + model inference + charts) and a lightweight price-only endpoint for efficient real-time polling.
	- Keep the core ingestion and analysis logic reusable so the same functions can power different UIs.

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

- Code entry points:
	- `app.py` — convenience launcher for the Django dev server (aliases manage commands; defaults to `runserver`).
	- `manage.py` — standard Django management script.
	- `sentiment/services.py` — Django-side wrappers that call into `src/` and prepare JSON for templates.
	- `sentiment/views.py` — Django views that handle `/analyze/` and `/price/` endpoints.
	- `src/data_ingestion.py` — contains `get_stock_data`, `get_company_info`, `get_stock_news`.
	- `src/sentiment_analyzer.py` — contains `load_sentiment_model` and `analyze_sentiment`.
	- `script/baseline_finbert_eval.py` — standalone evaluation script (requires sklearn, matplotlib, python-docx).
	- `script/build_silver_dataset.py` — standalone script to build labeled datasets from news.
- Caching and development notes:
	- The `src/` helpers are import-safe and can be used independently by Django or other UIs.
	- If you are iterating on model or data functions, restarting the Django devserver ensures fresh behavior.

- If you want to replace the sentiment pipeline with a different approach (local model, remote API), update `src/sentiment_analyzer.py` and keep the `analyze_sentiment(text, classifier)` contract:

	- inputs: `text: str`, `classifier: HF pipeline or similar`
	- outputs: dict with keys `label` (one of `Positive`/`Negative`/`Neutral`) and `score` (float)

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


