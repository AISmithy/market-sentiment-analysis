from django.shortcuts import render
from django.http import JsonResponse
from .services import get_company_data, analyze_news_for_ticker, compute_risk_score, compute_daily_sentiment_stats, get_ticker_suggestions_with_sentiment, compute_regime


def index(request):
    return render(request, 'sentiment/index.html')


def analyze(request):
    ticker = request.GET.get('ticker', 'AAPL').upper()
    company_data = get_company_data(ticker)

    # Normalize company output so callers can use `company.name`
    profile = company_data.get('profile') if isinstance(company_data, dict) else None
    company = {
        'name': None,
        'sector': None,
        'market_cap': None,
        'business_summary': None,
    }
    if profile:
        company['name'] = profile.get('Company Name')
        company['sector'] = profile.get('Sector')
        company['market_cap'] = profile.get('Market Cap')
        company['business_summary'] = profile.get('Business Summary')
    # price and change info if available
    if isinstance(company_data, dict):
        price_info = company_data.get('price') or {}
        company['current_price'] = float(price_info.get('current_price')) if price_info.get('current_price') is not None else None
        company['previous_close'] = float(price_info.get('previous_close')) if price_info.get('previous_close') is not None else None
        company['change'] = float(price_info.get('change')) if price_info.get('change') is not None else None
        company['change_pct'] = float(price_info.get('change_pct')) if price_info.get('change_pct') is not None else None
    else:
        company['current_price'] = None
        company['previous_close'] = None
        company['change'] = None
        company['change_pct'] = None

    news = analyze_news_for_ticker(ticker)

    # Compute risk score based on sentiment
    risk_data = compute_risk_score(news)

    # Compute daily sentiment statistics
    daily_stats = compute_daily_sentiment_stats(news)
    # Convert to sorted list by date (newest first)
    daily_stats_list = sorted(daily_stats.values(), key=lambda x: x['date'], reverse=True)

    # Get ticker suggestions with sentiment
    try:
        ticker_suggestions = get_ticker_suggestions_with_sentiment(ticker)
    except Exception as e:
        logger.warning(f"Failed to get ticker suggestions: {e}")
        ticker_suggestions = []

    # Compute market regime
    try:
        regime_data = compute_regime(company_data, news)
    except Exception as e:
        logger.debug(f"Failed to compute regime: {e}")
        regime_data = {
            'regime': 'Unknown',
            'trend': 'unknown',
            'sentiment_bias': 'unknown',
            'color': '#999',
            'price_change_pct': 0.0
        }

    # Include serialized history if available from the service
    history = None
    if isinstance(company_data, dict):
        history = company_data.get('history')

    data = {
        'ticker': ticker,
        'company': company,
        'news': news,
        'news_available': bool(news),
        'history': history,
        'history_available': bool(history),
        'risk_score': risk_data['risk_score'],
        'risk_level': risk_data['risk_level'],
        'risk_sentiment_counts': risk_data['sentiment_counts'],
        'risk_explanation': risk_data.get('risk_explanation'),
        'daily_stats': daily_stats_list,
        'ticker_suggestions': ticker_suggestions,
        'regime': regime_data,
    }

    return JsonResponse(data, safe=False)


def price(request):
    """Lightweight endpoint that returns only the price/change info and risk score for a ticker."""
    ticker = request.GET.get('ticker', 'AAPL').upper()
    company_data = get_company_data(ticker)
    price_info = None
    if isinstance(company_data, dict):
        price_info = company_data.get('price') or {}
    else:
        price_info = {}

    # Include risk score (requires fetching news)
    news = analyze_news_for_ticker(ticker)
    risk_data = compute_risk_score(news)

    data = {
        'ticker': ticker,
        'current_price': price_info.get('current_price'),
        'previous_close': price_info.get('previous_close'),
        'change': price_info.get('change'),
        'change_pct': price_info.get('change_pct'),
        'risk_score': risk_data['risk_score'],
        'risk_level': risk_data['risk_level'],
        'risk_sentiment_counts': risk_data['sentiment_counts'],
        'risk_explanation': risk_data.get('risk_explanation'),
    }
    return JsonResponse(data)

