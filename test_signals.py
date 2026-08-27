import unittest

import numpy as np
import pandas as pd

from src import signals


class TestDualMaSignal(unittest.TestCase):
    def test_uptrend_gives_long_signal(self):
        prices = pd.Series(np.linspace(100, 200, 300))
        sig = signals.dual_ma_signal(prices, fast=10, slow=50)
        self.assertTrue((sig.dropna() == 1.0).all())

    def test_downtrend_gives_short_signal(self):
        prices = pd.Series(np.linspace(200, 100, 300))
        sig = signals.dual_ma_signal(prices, fast=10, slow=50)
        self.assertTrue((sig.dropna() == -1.0).all())


class TestEwmacSignal(unittest.TestCase):
    def test_no_lookahead_first_values_are_nan(self):
        prices = pd.Series(np.linspace(100, 110, 200))
        sig = signals.ewmac_signal(prices, fast_span=8, slow_span=32, vol_lookback=16)
        # forecast should be undefined before enough history has accumulated
        self.assertTrue(sig.iloc[:8].isna().all())

    def test_forecast_is_capped(self):
        # a sharp, sustained trend should saturate the forecast at the cap
        prices = pd.Series(np.concatenate([np.full(50, 100.0), np.linspace(100, 400, 250)]))
        sig = signals.ewmac_signal(prices, fast_span=8, slow_span=32, vol_lookback=16, cap=20.0)
        self.assertLessEqual(sig.dropna().abs().max(), 20.0 + 1e-9)

    def test_uptrend_forecast_is_positive(self):
        prices = pd.Series(np.linspace(100, 300, 300))
        sig = signals.ewmac_signal(prices, fast_span=8, slow_span=32, vol_lookback=16)
        self.assertTrue((sig.dropna().iloc[-50:] > 0).all())


if __name__ == "__main__":
    unittest.main()
