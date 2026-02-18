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


def compute_risk_score(news_items):
    """
    Compute a short-term risk score (0-100) based on recent news sentiment.
    
    Risk scoring logic:
    - Negative sentiment: increases risk
    - Neutral sentiment: neutral risk
    - Positive sentiment: decreases risk
    
    Returns a dict with:
    - risk_score: 0-100 (0=low risk, 100=high risk)
    - risk_level: 'low', 'medium', 'high'
    - sentiment_counts: dict with counts by sentiment
    """
    if not news_items:
        # No news = moderate risk (unknown)
        risk_score = 50.0
        risk_level = 'medium'
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'sentiment_counts': sentiment_counts,
            'risk_explanation': explain_risk_meter(
                risk_score=risk_score,
                risk_level=risk_level,
                sentiment_counts=sentiment_counts,
                total_items=0,
                base_risk=50.0,
                ratio_adjustment=0.0,
                confidence_adjustment=0.0,
            ),
        }
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    weighted_score = 0.0
    
    for item in news_items:
        label = (item.get('sentiment_label') or 'Neutral').lower()
        score = item.get('sentiment_score') or 0.0
        
        if label.startswith('posit'):
            positive_count += 1
            # Positive sentiment reduces risk: lower contribution
            weighted_score -= score * 25  # Positive contributions reduce the base
        elif label.startswith('neg'):
            negative_count += 1
            # Negative sentiment increases risk: higher contribution
            weighted_score += score * 50  # Negative contributions increase risk more
        else:
            neutral_count += 1
            # Neutral sentiment has minimal impact
            weighted_score += score * 5
    
    # Normalize: base risk is 50, then adjust based on sentiment distribution
    base_risk = 50
    
    # Calculate average sentiment impact
    total_items = len(news_items)
    positive_ratio = positive_count / total_items if total_items > 0 else 0
    negative_ratio = negative_count / total_items if total_items > 0 else 0
    
    # Adjust risk based on sentiment ratios
    risk_adjustment = (negative_ratio * 40) - (positive_ratio * 25)
    risk_score = base_risk + risk_adjustment + (weighted_score / max(total_items, 1))
    
    # Clamp to 0-100
    risk_score = max(0, min(100, risk_score))
    
    # Determine risk level
    if risk_score < 33:
        risk_level = 'low'
    elif risk_score < 67:
        risk_level = 'medium'
    else:
        risk_level = 'high'
    
    sentiment_counts = {
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count
    }
    rounded_score = round(risk_score, 1)
    confidence_adjustment = weighted_score / max(total_items, 1)

    return {
        'risk_score': rounded_score,
        'risk_level': risk_level,
        'sentiment_counts': sentiment_counts,
        'risk_explanation': explain_risk_meter(
            risk_score=rounded_score,
            risk_level=risk_level,
            sentiment_counts=sentiment_counts,
            total_items=total_items,
            base_risk=base_risk,
            ratio_adjustment=risk_adjustment,
            confidence_adjustment=confidence_adjustment,
        ),
    }


def explain_risk_meter(
    risk_score,
    risk_level,
    sentiment_counts,
    total_items,
    base_risk,
    ratio_adjustment,
    confidence_adjustment,
):
    """
    Build a human-readable explanation of how the risk score was determined.
    """
    positive = int(sentiment_counts.get('positive', 0) or 0)
    negative = int(sentiment_counts.get('negative', 0) or 0)
    neutral = int(sentiment_counts.get('neutral', 0) or 0)

    if total_items <= 0:
        return {
            'summary': 'No recent articles were found, so the meter defaults to Medium risk (50/100).',
            'components': [
                'Base risk: 50.0',
                'Ratio adjustment: +0.0 (no news mix available)',
                'Confidence adjustment: +0.0',
            ],
        }

    ratio_sign = '+' if ratio_adjustment >= 0 else ''
    conf_sign = '+' if confidence_adjustment >= 0 else ''
    final_sign = '+' if (ratio_adjustment + confidence_adjustment) >= 0 else ''
    shift = ratio_adjustment + confidence_adjustment

    summary = (
        f"{risk_level.title()} risk ({risk_score}/100) from {total_items} recent articles: "
        f"{negative} negative, {neutral} neutral, {positive} positive."
    )
    components = [
        f"Base risk: {base_risk:.1f}",
        f"Ratio adjustment: {ratio_sign}{ratio_adjustment:.1f} "
        "(negative coverage increases risk, positive coverage reduces risk)",
        f"Confidence adjustment: {conf_sign}{confidence_adjustment:.1f} "
        "(strong negative scores add more than strong positive scores subtract)",
        f"Net shift from base: {final_sign}{shift:.1f}",
    ]

    return {
        'summary': summary,
        'components': components,
    }


def compute_daily_sentiment_stats(news_items):
    """
    Group news items by day and compute daily sentiment statistics.
    
    Returns a dict mapping date (YYYY-MM-DD) to:
    - date: the date string
    - count: number of articles
    - positive: count of positive articles
    - negative: count of negative articles
    - neutral: count of neutral articles
    - avg_score: average sentiment score
    - dominant_sentiment: 'Positive', 'Negative', or 'Neutral'
    - net_sentiment: calculated sentiment bias (-1 to 1)
    """
    from datetime import datetime
    
    daily_stats = {}
    
    for item in news_items:
        # Parse the published date (format: "Day, dd Mon YYYY hh:mm:ss GMT" or similar)
        published_str = item.get('published') or ''
        try:
            # Try multiple date formats
            date_obj = None
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y %H:%M:%S', '%Y-%m-%d']:
                try:
                    date_obj = datetime.strptime(published_str.replace('GMT', '').strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if date_obj is None:
                # If parsing fails, try ISO format or skip
                try:
                    date_obj = datetime.fromisoformat(published_str.split('T')[0])
                except:
                    continue
            
            date_key = date_obj.strftime('%Y-%m-%d')
        except:
            continue
        
        if date_key not in daily_stats:
            daily_stats[date_key] = {
                'date': date_key,
                'count': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'total_score': 0.0,
            }
        
        stats = daily_stats[date_key]
        stats['count'] += 1
        
        label = (item.get('sentiment_label') or 'Neutral').lower()
        score = item.get('sentiment_score') or 0.0
        stats['total_score'] += score
        
        if label.startswith('posit'):
            stats['positive'] += 1
        elif label.startswith('neg'):
            stats['negative'] += 1
        else:
            stats['neutral'] += 1
    
    # Compute derived metrics
    for date_key, stats in daily_stats.items():
        count = stats['count']
        stats['avg_score'] = round(stats['total_score'] / count, 3) if count > 0 else 0.0
        
        # Determine dominant sentiment
        pos, neg, neu = stats['positive'], stats['negative'], stats['neutral']
        if pos > neg and pos > neu:
            stats['dominant_sentiment'] = 'Positive'
        elif neg > pos and neg > neu:
            stats['dominant_sentiment'] = 'Negative'
        else:
            stats['dominant_sentiment'] = 'Neutral'
        
        # Net sentiment: -1 (all negative) to 1 (all positive)
        net = (pos - neg) / max(count, 1)
        stats['net_sentiment'] = round(net, 3)
        
        # Remove the temporary total_score field
        del stats['total_score']
    
    return daily_stats

def get_related_tickers(ticker):
    """
    Get a list of related tickers based on industry/sector grouping.
    Returns 3-5 related tickers excluding the input ticker.
    """
    # Define ticker groups by category (tech, AI, finance, consumer, etc.)
    ticker_groups = {
        'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        'ai_ml': ['NVDA', 'MSFT', 'GOOGL', 'AMD', 'LRCX', 'ASML', 'BROADCOM'],
        'cloud': ['MSFT', 'GOOGL', 'AMZN', 'CRM', 'ADBE', 'OKTA', 'MDB'],
        'semiconductors': ['NVDA', 'AMD', 'QUALCOMM', 'INTC', 'LRCX', 'ASML', 'BROADCOM'],
        'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'AMEX', 'SCHW'],
        'consumer': ['AMZN', 'TSLA', 'AAPL', 'META', 'WMT', 'TGT', 'COST'],
        'energy': ['XOM', 'CVX', 'MPC', 'COP', 'EOG', 'FANG', 'OXY'],
        'pharma': ['JNJ', 'PFE', 'MRNA', 'ABBV', 'VRTX', 'BNTX', 'AMGN'],
    }
    
    ticker_upper = ticker.upper()
    related = []
    
    # Find which groups contain this ticker
    for group_name, tickers in ticker_groups.items():
        if ticker_upper in tickers:
            # Get up to 4 other tickers from this group
            other_tickers = [t for t in tickers if t != ticker_upper]
            related.extend(other_tickers[:4])
            break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_related = []
    for t in related:
        if t not in seen:
            seen.add(t)
            unique_related.append(t)
    
    # Return up to 5 suggestions
    return unique_related[:5]


def compute_sentiment_summary(news_items):
    """
    Compute a summary of sentiment for a list of news items.
    Returns: {positive: count, negative: count, neutral: count, avg_score: float}
    """
    if not news_items:
        return {'positive': 0, 'negative': 0, 'neutral': 0, 'avg_score': 0.0}
    
    positive = negative = neutral = 0
    total_score = 0.0
    
    for item in news_items:
        label = (item.get('sentiment_label') or 'Neutral').lower()
        score = item.get('sentiment_score') or 0.0
        total_score += score
        
        if label.startswith('posit'):
            positive += 1
        elif label.startswith('neg'):
            negative += 1
        else:
            neutral += 1
    
    return {
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'avg_score': round(total_score / len(news_items), 3) if news_items else 0.0
    }


def get_ticker_suggestions_with_sentiment(ticker):
    """
    Get related tickers with their current sentiment profiles.
    This is cached at the HTTP level to avoid expensive repeated analysis.
    
    Returns a list of dicts with:
    - ticker: the ticker symbol
    - sentiment: sentiment summary {positive, negative, neutral, avg_score}
    - count: number of articles analyzed
    """
    related_tickers = get_related_tickers(ticker)
    suggestions = []
    
    for suggested_ticker in related_tickers:
        try:
            # Get news and compute sentiment for this ticker
            news = data_ingestion.get_stock_news(suggested_ticker)
            if news:
                sentiment = compute_sentiment_summary(news)
                sentiment['count'] = len(news)
                suggestions.append({
                    'ticker': suggested_ticker,
                    'sentiment': sentiment,
                })
        except Exception as e:
            # Skip tickers that fail to fetch
            logger.debug(f"Failed to get news for {suggested_ticker}: {e}")
            continue
    
    return suggestions


def compute_regime(company_data, news_items):
    """
    Compute market regime label based on price trend and sentiment.
    
    Analyzes price history to determine trend (bullish/bearish/sideways)
    and compares with sentiment distribution to identify convergence or divergence.
    
    Returns dict with:
    - regime: string label (e.g., "Bullish + Positive", "Bearish / Negative Divergence")
    - trend: 'bullish', 'bearish', or 'sideways'
    - sentiment_bias: 'positive', 'negative', or 'mixed'
    - color: CSS color for chip (green, red, orange, yellow)
    """
    # Determine price trend from historical data
    trend = 'sideways'
    price_change_pct = 0.0
    
    try:
        if isinstance(company_data, dict):
            history = company_data.get('history')
            if history and len(history) > 1:
                # Get first and last prices from history
                try:
                    first_price = float(history[0].get('close', history[0].get('o')))
                    last_price = float(history[-1].get('close', history[-1].get('c')))
                    if first_price > 0:
                        price_change_pct = ((last_price - first_price) / first_price) * 100
                        
                        if price_change_pct > 2:
                            trend = 'bullish'
                        elif price_change_pct < -2:
                            trend = 'bearish'
                        else:
                            trend = 'sideways'
                except (ValueError, KeyError, TypeError):
                    trend = 'sideways'
    except Exception as e:
        logger.debug(f"Error computing price trend: {e}")
        trend = 'sideways'
    
    # Determine sentiment bias
    sentiment_summary = compute_sentiment_summary(news_items)
    pos = sentiment_summary.get('positive', 0)
    neg = sentiment_summary.get('negative', 0)
    neu = sentiment_summary.get('neutral', 0)
    total = pos + neg + neu
    
    sentiment_bias = 'mixed'
    if total > 0:
        pos_ratio = pos / total
        neg_ratio = neg / total
        
        if pos_ratio > 0.5:
            sentiment_bias = 'positive'
        elif neg_ratio > 0.5:
            sentiment_bias = 'negative'
        else:
            sentiment_bias = 'mixed'
    
    # Build regime label and determine color
    regime = ''
    color = '#9ca3af'  # default gray
    
    if trend == 'bullish' and sentiment_bias == 'positive':
        regime = 'Bullish + Positive'
        color = '#10b981'  # vibrant green - strong bullish
    elif trend == 'bullish' and sentiment_bias == 'negative':
        regime = 'Bullish / Negative Divergence'
        color = '#f97316'  # vibrant orange - warning
    elif trend == 'bullish' and sentiment_bias == 'mixed':
        regime = 'Bullish + Mixed Sentiment'
        color = '#f59e0b'  # vibrant amber - caution
    
    elif trend == 'bearish' and sentiment_bias == 'negative':
        regime = 'Bearish + Negative'
        color = '#ef4444'  # vibrant red - strong bearish
    elif trend == 'bearish' and sentiment_bias == 'positive':
        regime = 'Bearish / Positive Divergence'
        color = '#f97316'  # vibrant orange - opportunity
    elif trend == 'bearish' and sentiment_bias == 'mixed':
        regime = 'Bearish + Mixed Sentiment'
        color = '#f59e0b'  # vibrant amber - caution
    
    elif trend == 'sideways' and sentiment_bias == 'positive':
        regime = 'Consolidating, Positive Sentiment'
        color = '#06b6d4'  # vibrant cyan - sideways bullish
    elif trend == 'sideways' and sentiment_bias == 'negative':
        regime = 'Consolidating, Negative Sentiment'
        color = '#f97316'  # vibrant orange - sideways bearish
    else:
        regime = 'Sideways, Mixed Sentiment'
        color = '#6b7280'  # medium gray - neutral
    
    return {
        'regime': regime,
        'trend': trend,
        'sentiment_bias': sentiment_bias,
        'color': color,
        'price_change_pct': round(price_change_pct, 2)
    }
