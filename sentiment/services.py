import logging
from pathlib import Path
import importlib

logger = logging.getLogger(__name__)

# Try to import the existing src modules (they are not a package, so adapt import path)
try:
    import src.data_ingestion as data_ingestion
    import src.sentiment_analyzer as sentiment_analyzer
except Exception:
    # If direct import fails, try relative location
    try:
        from ..src import data_ingestion, sentiment_analyzer
    except Exception:
        data_ingestion = None
        sentiment_analyzer = None


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

        return {'profile': profile, 'history_head': hist.head(5).to_dict() if hist is not None else None, 'history': history_serialized}
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
