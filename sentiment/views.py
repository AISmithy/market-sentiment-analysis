"""\
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

from django.shortcuts import render
from django.http import JsonResponse
from .services import get_company_data, analyze_news_for_ticker
import yfinance as yf
from datetime import datetime
import time
from django.core.cache import cache
from django.conf import settings

# Use Django's cache framework. TTL and max items are configured in Django settings
# (HISTORY_CACHE_TTL, HISTORY_CACHE_MAX_ITEMS). For locmem backend we maintain a small
# index to enforce max-items; for production Redis/Memcached backends rely on the backend.
_HISTORY_TTL = getattr(settings, 'HISTORY_CACHE_TTL', 300)
_HISTORY_MAX_ITEMS = getattr(settings, 'HISTORY_CACHE_MAX_ITEMS', 200)
_HISTORY_INDEX_KEY = 'history_cache_index'


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


def price(request):
    """Lightweight endpoint that returns only the price/change info for a ticker."""
    ticker = request.GET.get('ticker', 'AAPL').upper()
    company_data = get_company_data(ticker)
    price_info = None
    if isinstance(company_data, dict):
        price_info = company_data.get('price') or {}
    else:
        price_info = {}

    data = {
        'ticker': ticker,
        'current_price': price_info.get('current_price'),
        'previous_close': price_info.get('previous_close'),
        'change': price_info.get('change'),
        'change_pct': price_info.get('change_pct'),
    }
    return JsonResponse(data)


def history(request):
    """Return historical OHLCV for a ticker.

    Query params:
      - ticker (required)
      - period (optional, default '1y') e.g. '1d','5d','1mo','3mo','6mo','1y','5y','max'
      - interval (optional, default '1d') e.g. '1m','2m','5m','15m','1h','1d','1wk','1mo'
      - start (optional, YYYY-MM-DD) and end (optional, YYYY-MM-DD) override period if provided
    """
    ticker = request.GET.get('ticker', 'AAPL').upper()
    period = request.GET.get('period') or '1y'
    interval = request.GET.get('interval') or '1d'
    start = request.GET.get('start')
    end = request.GET.get('end')

    cache_key = f"history:{ticker}|{period}|{interval}|{start or ''}|{end or ''}"

    # Allow callers to bypass cache
    nocache = request.GET.get('nocache')
    if nocache and str(nocache) in ('1', 'true', 'True'):
        cached = None
    else:
        cached = cache.get(cache_key)
    if cached is not None:
        return JsonResponse({'ticker': ticker, 'history': cached})

    try:
        tk = yf.Ticker(ticker)
        # Prefer start/end if provided
        if start:
            # validate date format
            try:
                _ = datetime.fromisoformat(start)
            except Exception:
                return JsonResponse({'error': 'start must be ISO YYYY-MM-DD'}, status=400)
            kwargs = {'start': start}
            if end:
                try:
                    _ = datetime.fromisoformat(end)
                except Exception:
                    return JsonResponse({'error': 'end must be ISO YYYY-MM-DD'}, status=400)
                kwargs['end'] = end
            hist = tk.history(interval=interval, **kwargs)
        else:
            hist = tk.history(period=period, interval=interval)

        if hist is None or hist.empty:
            out = []
            # store empty result in cache too
            cache.set(cache_key, out, timeout=_HISTORY_TTL)
            # maintain index for locmem backend
            try:
                backend = settings.CACHES.get('default', {}).get('BACKEND', '')
            except Exception:
                backend = ''
            if 'locmem' in backend:
                idx = cache.get(_HISTORY_INDEX_KEY) or {}
                idx[cache_key] = time.time()
                # evict oldest if necessary
                if len(idx) > _HISTORY_MAX_ITEMS:
                    items = sorted(idx.items(), key=lambda kv: kv[1])
                    remove_count = max(1, int(_HISTORY_MAX_ITEMS * 0.1))
                    for k, _v in items[:remove_count]:
                        try:
                            cache.delete(k)
                        except Exception:
                            pass
                        idx.pop(k, None)
                cache.set(_HISTORY_INDEX_KEY, idx, None)
            return JsonResponse({'ticker': ticker, 'history': out})

        # Serialize
        out = []
        # ensure timezone naive and ISO formatting
        try:
            idx = hist.index.tz_convert(None)
        except Exception:
            idx = hist.index
        for i, row in hist.iterrows():
            dt = i.isoformat() if hasattr(i, 'isoformat') else str(i)
            out.append({
                'date': dt,
                'open': float(row.get('Open') or 0.0),
                'high': float(row.get('High') or 0.0),
                'low': float(row.get('Low') or 0.0),
                'close': float(row.get('Close') or 0.0),
                'volume': int(row.get('Volume') or 0),
            })

        # Write to cache (Django cache backend handles TTL). For locmem backend maintain index
        cache.set(cache_key, out, timeout=_HISTORY_TTL)
        try:
            backend = settings.CACHES.get('default', {}).get('BACKEND', '')
        except Exception:
            backend = ''
        if 'locmem' in backend:
            idx = cache.get(_HISTORY_INDEX_KEY) or {}
            idx[cache_key] = time.time()
            if len(idx) > _HISTORY_MAX_ITEMS:
                items = sorted(idx.items(), key=lambda kv: kv[1])
                remove_count = max(1, int(_HISTORY_MAX_ITEMS * 0.1))
                for k, _v in items[:remove_count]:
                    try:
                        cache.delete(k)
                    except Exception:
                        pass
                    idx.pop(k, None)
            cache.set(_HISTORY_INDEX_KEY, idx, None)

        return JsonResponse({'ticker': ticker, 'history': out})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
