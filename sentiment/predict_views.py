"""Lightweight prediction view kept separate from the main views to reduce risk.
"""
from django.http import JsonResponse

try:
    import src.data_ingestion as data_ingestion
    import src.prediction_module4 as predictor
except Exception:
    data_ingestion = None
    predictor = None


def predict(request):
    ticker = request.GET.get('ticker', 'AAPL').upper()
    try:
        window = int(request.GET.get('window', 60))
    except Exception:
        window = 60
    try:
        epochs = int(request.GET.get('epochs', 20))
    except Exception:
        epochs = 20
    retrain = str(request.GET.get('retrain', '0')).lower() in ('1', 'true', 'yes')
    model_path = request.GET.get('model_path')

    if data_ingestion is None or predictor is None:
        return JsonResponse({'error': 'prediction dependencies not available on server'}, status=500)

    try:
        df = data_ingestion.get_stock_data(ticker, period='3y')
        if df is None or df.empty:
            return JsonResponse({'error': 'no historical data available for ticker'}, status=404)

        predicted, meta = predictor.predict_next_day_from_df(
            df=df,
            feature='Close',
            window=window,
            model_path=model_path,
            retrain=retrain,
            epochs=epochs,
            batch_size=32,
        )

        return JsonResponse({'ticker': ticker, 'predicted_next_close': float(predicted), 'meta': meta})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
