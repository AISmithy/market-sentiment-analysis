import yfinance as yf
from yahoo_fin import news as yf_news

# Try to import Streamlit caching wrappers when available. If not, provide no-op decorators.
try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:
    def cache_data(ttl=None):
        def _decorator(fn):
            return fn
        return _decorator

from .sentiment_analyzer import load_sentiment_model, analyze_sentiment


@cache_data(ttl=3600)
def get_stock_data(ticker, period="5y"):
    """ Fetches historical stock data. """
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period)
        return hist_data if not hist_data.empty else None
    except:
        return None


@cache_data(ttl=86400)
def get_company_info(ticker):
    """ Fetches company profile information. """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        profile_data = {
            "Company Name": info.get('longName', 'N/A'),
            "Sector": info.get('sector', 'N/A'),
            "Market Cap": f"${info.get('marketCap', 0):,}",
            "Business Summary": info.get('longBusinessSummary', 'N/A')
        }
        return profile_data
    except:
        return None


@cache_data(ttl=60)
def get_current_price(ticker):
    """Return the current market price (float) for the ticker or None."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        # Try common fields
        price = info.get('regularMarketPrice') or info.get('currentPrice')
        if price is None:
            # try fast_info if available
            fast = getattr(stock, 'fast_info', None)
            if fast:
                price = fast.get('lastPrice') if isinstance(fast, dict) else getattr(fast, 'lastPrice', None)
        if price is not None:
            return float(price)
        # Fallback: recent close
        hist = stock.history(period='2d')
        if hist is not None and not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None


@cache_data(ttl=60)
def get_price_and_change(ticker):
    """Return a dict with current_price, previous_close, change, and change_pct."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        current = info.get('regularMarketPrice') or info.get('currentPrice')
        prev = info.get('regularMarketPreviousClose') or info.get('previousClose')

        # Fallback to history when fields are missing
        if (current is None or prev is None):
            hist = stock.history(period='3d')
            if hist is not None and len(hist) >= 2:
                # last row = most recent close
                prev = float(hist['Close'].iloc[-2]) if prev is None else prev
                current = float(hist['Close'].iloc[-1]) if current is None else current
            elif hist is not None and len(hist) == 1:
                prev = float(hist['Close'].iloc[0]) if prev is None else prev
                current = float(hist['Close'].iloc[0]) if current is None else current

        if current is None:
            return {'current_price': None, 'previous_close': None, 'change': None, 'change_pct': None}

        current = float(current)
        prev = float(prev) if prev is not None else None
        if prev is None:
            change = None
            pct = None
        else:
            change = current - prev
            try:
                pct = (change / prev) * 100 if prev != 0 else None
            except Exception:
                pct = None

        return {
            'current_price': current,
            'previous_close': prev,
            'change': float(change) if change is not None else None,
            'change_pct': float(pct) if pct is not None else None,
        }
    except Exception:
        return {'current_price': None, 'previous_close': None, 'change': None, 'change_pct': None}


@cache_data(ttl=1800)
def get_stock_news(ticker):
    """ Fetches and analyzes the latest news articles. """
    classifier = load_sentiment_model()
    if classifier is None:
        # If running without Streamlit, just return an empty list
        try:
            import streamlit as st
            st.error("Sentiment model failed to load. News analysis disabled.")
        except Exception:
            pass
        return []

    try:
        news_list = yf_news.get_yf_rss(ticker)
        analyzed_news = []
        for item in news_list:
            title = item.get('title', 'No Title')
            sentiment = analyze_sentiment(title, classifier)

            analyzed_news.append({
                'title': title,
                'link': item.get('link', '#'),
                'published': item.get('published', 'N/A'),
                'sentiment_label': sentiment['label'],
                'sentiment_score': sentiment['score']
            })
        return analyzed_news
    except Exception as e:
        print(f"Error fetching or analyzing news: {e}")
        return []