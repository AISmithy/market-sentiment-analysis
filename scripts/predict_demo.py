"""Demo script: fetch history for ticker and run LSTM predictor.

Usage: run from repo root with the virtualenv activated:
    python scripts/predict_demo.py --ticker AAPL --window 60 --epochs 5

This script is intentionally minimal — it demonstrates the end-to-end flow.
"""

import argparse
import logging
from pathlib import Path

try:
    import src.data_ingestion as data_ingestion
    import src.prediction_module4 as predictor
except Exception as e:
    raise RuntimeError("Required src modules not importable: " + str(e))

logging.basicConfig(level=logging.INFO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', default='AAPL')
    p.add_argument('--window', type=int, default=60)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--model-path', default=None)
    args = p.parse_args()

    df = data_ingestion.get_stock_data(args.ticker, period='3y')
    if df is None or df.empty:
        print('No data for', args.ticker)
        return

    model_path = args.model_path
    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    predicted, meta = predictor.predict_next_day_from_df(
        df=df,
        feature='Close',
        window=args.window,
        model_path=model_path,
        retrain=True,
        epochs=args.epochs,
    )

    print(f"Predicted next close for {args.ticker}: {predicted}")
    print('meta:', meta)


if __name__ == '__main__':
    main()
