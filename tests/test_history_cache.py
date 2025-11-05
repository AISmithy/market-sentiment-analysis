"""\
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

from django.test import SimpleTestCase, Client
from unittest.mock import patch
import pandas as pd
from pandas import Timestamp
from django.core.cache import cache


class HistoryCacheTest(SimpleTestCase):
    def setUp(self):
        cache.clear()

    @patch('yfinance.Ticker.history')
    def test_history_cached_and_nocache(self, mock_history):
        # prepare a small dataframe
        idx = [Timestamp('2025-11-01'), Timestamp('2025-11-02')]
        df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [101.0, 102.0],
            'Low': [99.0, 100.0],
            'Close': [100.5, 101.5],
            'Volume': [1000, 1100]
        }, index=idx)
        mock_history.return_value = df

        client = Client()
        params = {'ticker': 'AAPL', 'period': '2d', 'interval': '1d'}
        # First call should invoke yfinance
        r1 = client.get('/history/', params)
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(mock_history.call_count, 1)
        data1 = r1.json()
        self.assertIn('history', data1)
        self.assertEqual(len(data1['history']), 2)

        # Second call should be served from cache (no additional yfinance calls)
        r2 = client.get('/history/', params)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(mock_history.call_count, 1)

        # Force bypass cache
        params['nocache'] = '1'
        r3 = client.get('/history/', params)
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(mock_history.call_count, 2)
