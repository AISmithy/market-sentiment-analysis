#!/usr/bin/env python3
"""

Fetch recent news for a ticker using the project's existing ingestion helpers,
label each headline with the project's FinBERT pipeline, and save a lightweight
CSV for baseline training/evaluation.

Uses:
  - src.data_ingestion.get_stock_news(ticker) -> List[dict] with keys like 'title','link','published','source'
  - src.sentiment_analyzer.load_sentiment_model()
  - src.sentiment_analyzer.analyze_sentiment(text, classifier) -> {'label': 'Positive|Neutral|Negative', 'score': float}

Example:
  python scripts/build_silver_dataset.py --ticker AAPL --out data/headlines.csv --limit 300 --unique

Output CSV columns (superset; training needs only headline,label):
  headline,label,score,raw_label,source,published,link
"""
import argparse
import csv
import sys
from pathlib import Path

# Ensure project root on path so we can import src.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_ingestion import get_stock_news
from src.sentiment_analyzer import load_sentiment_model, analyze_sentiment


def normalize_label(raw: str) -> str:
    """Map raw FinBERT labels to {positive, neutral, negative} (lowercase)."""
    if not raw:
        return "neutral"
    r = raw.strip().lower()
    if "pos" in r:
        return "positive"
    if "neg" in r:
        return "negative"
    if "neu" in r:
        return "neutral"
    return "neutral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g., AAPL")
    ap.add_argument("--out", default="data/headlines.csv", help="Output CSV path")
    ap.add_argument("--limit", type=int, default=300, help="Max number of news items to process")
    ap.add_argument("--min-len", type=int, default=5, help="Minimum headline length to keep")
    ap.add_argument("--unique", action="store_true", help="Deduplicate identical headlines (case-insensitive)")
    args = ap.parse_args()

    # Fetch news via project helper
    items = get_stock_news(args.ticker) or []
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    # Prepare classifier
    classifier = load_sentiment_model()

    rows = []
    seen = set()
    for it in items:
        title = (it.get("title") or "").strip()
        if not title or len(title) < args.min_len:
            continue
        if args.unique:
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)

        result = analyze_sentiment(title, classifier) or {}
        raw_label = (result.get("label") or "").strip()
        score = result.get("score", None)
        norm = normalize_label(raw_label)

        rows.append(
            {
                "headline": title,
                "label": norm,
                "score": f"{score:.6f}" if isinstance(score, float) else "",
                "raw_label": raw_label,
                "source": (it.get("source") or ""),
                "published": (it.get("published") or ""),
                "link": (it.get("link") or ""),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["headline", "label", "score", "raw_label", "source", "published", "link"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Ticker: {args.ticker}")
    print(f"Wrote {len(rows)} rows -> {out_path.resolve()}")
    if rows:
        from collections import Counter

        dist = Counter(r["label"] for r in rows)
        print("Label distribution:", dict(dist))


if __name__ == "__main__":
    main()
