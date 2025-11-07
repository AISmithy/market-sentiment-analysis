"""
Basic unit test for the LSTM prediction module using a tiny synthetic dataset.
This test trains for a very small number of epochs to keep runtime short.
"""

import unittest

import numpy as np
import pandas as pd


class PredictionModule4Test(unittest.TestCase):
    def test_predict_small_synthetic(self):
        try:
            import src.prediction_module4 as predictor
        except Exception:
            self.skipTest('prediction_module4 or dependencies not available')

        # synthetic linear-ish series with small noise
        n = 200
        dates = pd.date_range(end=pd.Timestamp('today'), periods=n)
        base = np.linspace(100.0, 120.0, n)
        noise = np.random.normal(0, 0.2, n)
        close = base + noise
        df = pd.DataFrame({'Close': close}, index=dates)

        # small window and few epochs to keep test fast
        pred, meta = predictor.predict_next_day_from_df(
            df=df,
            feature='Close',
            window=10,
            model_path=None,
            retrain=True,
            epochs=2,
            batch_size=16,
            val_split=0.1,
        )

        self.assertIsInstance(pred, float)
        self.assertIn('window', meta)
        self.assertEqual(meta['window'], 10)


if __name__ == '__main__':
    unittest.main()
