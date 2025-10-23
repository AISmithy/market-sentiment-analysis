# Market Sentiment Analysis

A compact Python project for fetching financial data and running AI-powered sentiment analysis on recent news for a given stock ticker. The app combines Yahoo Finance data, a FinBERT-based sentiment model from Hugging Face, and a Streamlit dashboard to visualize price history and recent news sentiment.

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
- Interactive Streamlit dashboard with candlestick chart, company profile, and news sentiment summaries.

## Repository layout

- `src/`
	- `dashboard.py` — Streamlit app (entry point).
	- `data_ingestion.py` — helpers for fetching stock data, company info and news.
	- `sentiment_analyzer.py` — loads the Hugging Face FinBERT pipeline and maps model outputs.
	- `utils.py` — small utilities (logging, helpers).
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

## Running the app (Streamlit)

Start the dashboard locally with Streamlit. From the repository root run:

```powershell
streamlit run src/dashboard.py
```

Default behavior:
- Enter a stock ticker in the sidebar (e.g., AAPL, TSLA).
- The app shows a candlestick chart, company profile, and recent news with sentiment labels and a sentiment distribution chart.

Notes about model download and caching:
- The FinBERT model is downloaded the first time the `transformers` pipeline is created. This requires internet connectivity and may take a minute.
- The code uses Streamlit cache decorators to avoid reloading data/models repeatedly.

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
.\.venv\Scripts\python.exe manage.py migrate

# start the development server (binds to 127.0.0.1:8000 by default)
.\.venv\Scripts\python.exe manage.py runserver
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

## Development notes

- Code entry points:
	- `src/dashboard.py` — UI and orchestration.
	- `src/data_ingestion.py` — contains `get_stock_data`, `get_company_info`, `get_stock_news`.
	- `src/sentiment_analyzer.py` — contains `load_sentiment_model` and `analyze_sentiment`.

- Caching in Streamlit is used to reduce calls to remote services. If you are iterating on the model or data functions, either restart Streamlit or clear the cache from the UI.

- If you want to replace the sentiment pipeline with a different approach (local model, remote API), update `sentiment_analyzer.py` and keep the `analyze_sentiment(text, classifier)` contract:

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

## License

This project does not include a license file. Add a `LICENSE` (for example MIT) to make usage terms explicit.

---

If you want, I can also:

- add a `requirements-dev.txt` and a small `Makefile`/`tasks.json` for common dev tasks,
- add a minimal `tests/` folder with a basic smoke test for `get_stock_data` and `analyze_sentiment`, or
- add a `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

Tell me which of those you'd like next and I'll implement it.
