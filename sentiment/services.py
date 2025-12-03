import logging
from pathlib import Path
import importlib
import sys

logger = logging.getLogger(__name__)


def _import_src_module(mod_name: str):
    """Import a module from the top-level `src` package.

    Tries an absolute import first (recommended). If that fails, adds the
    project root to sys.path and retries once. Returns the module or None.
    """
    try:
        return importlib.import_module(f"src.{mod_name}")
    except Exception:
        # Attempt to add project root to sys.path and retry
        try:
            root = Path(__file__).resolve().parents[1]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return importlib.import_module(f"src.{mod_name}")
        except Exception:
            return None


data_ingestion = _import_src_module("data_ingestion")
sentiment_analyzer = _import_src_module("sentiment_analyzer")


def get_company_data(ticker):
    """Return company profile and historical data summary if available."""
    if data_ingestion is None:
        logger.warning('data_ingestion module not available')
        return {'name': None}

    try:
        profile = data_ingestion.get_company_info(ticker)
        hist = data_ingestion.get_stock_data(ticker)

        history_serialized = None
        if hist is not None and not hist.empty:
            # Limit to the last 180 days (or available rows)
            hist2 = hist.tail(180).copy()
            # Ensure index is timezone-naive ISO format
            hist2.index = hist2.index.tz_convert(None)
            history_serialized = []
            for idx, row in hist2.iterrows():
                history_serialized.append({
                    'date': idx.isoformat(),
                    'open': float(row.get('Open', None) or 0.0),
                    'high': float(row.get('High', None) or 0.0),
                    'low': float(row.get('Low', None) or 0.0),
                    'close': float(row.get('Close', None) or 0.0),
                    'volume': int(row.get('Volume', 0) or 0),
                })

        # price + change (best-effort)
        price_info = None
        try:
            price_info = data_ingestion.get_price_and_change(ticker)
        except Exception:
            price_info = None

        return {
            'profile': profile,
            'history_head': hist.head(5).to_dict() if hist is not None else None,
            'history': history_serialized,
            'price': price_info,
        }
    except Exception as e:
        logger.exception('Error getting company data')
        return {'error': str(e)}


def analyze_news_for_ticker(ticker):
    """Fetch latest news and analyze sentiment. Returns list of simplified dicts."""
    if data_ingestion is None or sentiment_analyzer is None:
        logger.warning('data_ingestion or sentiment_analyzer unavailable')
        return []

    try:
        news_items = data_ingestion.get_stock_news(ticker)
        # news_items are already analyzed in the current implementation
        simplified = []
        for n in news_items:
            simplified.append({
                'title': n.get('title'),
                'link': n.get('link'),
                'published': n.get('published'),
                'sentiment_label': n.get('sentiment_label'),
                'sentiment_score': n.get('sentiment_score'),
            })
        return simplified
    except Exception as e:
        logger.exception('Error analyzing news')
        return []
