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