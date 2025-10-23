from django.shortcuts import render
from django.http import JsonResponse
from .services import get_company_data, analyze_news_for_ticker


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

    news = analyze_news_for_ticker(ticker)

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
    }

    return JsonResponse(data, safe=False)
