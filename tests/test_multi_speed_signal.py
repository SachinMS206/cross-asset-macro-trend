import unittest

import numpy as np
import pandas as pd

from src import signals


class TestMultiSpeedEwmac(unittest.TestCase):
    def test_forecast_is_capped(self):
        prices = pd.Series(np.concatenate([np.full(50, 100.0), np.linspace(100, 400, 250)]))
        f = signals.multi_speed_ewmac_signal(prices)
        self.assertLessEqual(f.dropna().abs().max(), 20.0 + 1e-9)

    def test_uptrend_gives_positive_forecast(self):
        prices = pd.Series(np.linspace(100, 300, 300))
        f = signals.multi_speed_ewmac_signal(prices)
        self.assertTrue((f.dropna().iloc[-50:] > 0).all())

    def test_downtrend_gives_negative_forecast(self):
        prices = pd.Series(np.linspace(300, 100, 300))
        f = signals.multi_speed_ewmac_signal(prices)
        self.assertTrue((f.dropna().iloc[-50:] < 0).all())

    def test_custom_speeds_accepted(self):
        prices = pd.Series(np.linspace(100, 200, 300))
        f = signals.multi_speed_ewmac_signal(prices, speeds=[(4, 16), (8, 32)])
        self.assertTrue(f.dropna().shape[0] > 0)

    def test_default_speeds_constant_used_when_none_passed(self):
        prices = pd.Series(np.linspace(100, 200, 300))
        f_default = signals.multi_speed_ewmac_signal(prices)
        f_explicit = signals.multi_speed_ewmac_signal(prices, speeds=signals.DEFAULT_SPEEDS)
        pd.testing.assert_series_equal(f_default, f_explicit)


if __name__ == "__main__":
    unittest.main()
