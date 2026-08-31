import unittest

import numpy as np
import pandas as pd

from src import risk


class TestVolTargeting(unittest.TestCase):
    def test_higher_vol_instrument_gets_smaller_position_for_same_forecast(self):
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        rng = np.random.default_rng(0)

        low_vol_returns = pd.Series(rng.normal(0, 0.5, 300), index=idx)
        high_vol_returns = pd.Series(rng.normal(0, 5.0, 300), index=idx)
        constant_forecast = pd.Series(10.0, index=idx)
        dummy_price = pd.Series(100.0, index=idx)

        pos_low_vol = risk.vol_target_position_size(
            constant_forecast, dummy_price, low_vol_returns, target_ann_vol=0.10
        )
        pos_high_vol = risk.vol_target_position_size(
            constant_forecast, dummy_price, high_vol_returns, target_ann_vol=0.10
        )

        # same forecast strength, but the noisier instrument should be sized
        # smaller so both contribute similar risk
        self.assertGreater(pos_low_vol.dropna().abs().mean(), pos_high_vol.dropna().abs().mean())

    def test_zero_forecast_gives_zero_position(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(np.random.default_rng(1).normal(0, 1, 100), index=idx)
        zero_forecast = pd.Series(0.0, index=idx)
        price = pd.Series(100.0, index=idx)

        pos = risk.vol_target_position_size(zero_forecast, price, returns)
        self.assertTrue(np.allclose(pos.dropna(), 0.0))


class TestRiskParityWeights(unittest.TestCase):
    def test_equal_variance_uncorrelated_assets_get_equal_weight(self):
        cov = np.diag([1.0, 1.0, 1.0])
        w = risk.risk_parity_weights(cov)
        np.testing.assert_allclose(w, np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-3)

    def test_weights_sum_to_one(self):
        cov = np.array([[1.0, 0.2, 0.1], [0.2, 4.0, 0.3], [0.1, 0.3, 0.5]])
        w = risk.risk_parity_weights(cov)
        self.assertAlmostEqual(w.sum(), 1.0, places=4)

    def test_lower_variance_asset_gets_higher_weight(self):
        # asset 0 has much lower variance than asset 1 -> risk parity should
        # allocate it MORE capital weight to equalize risk CONTRIBUTION
        cov = np.array([[0.5, 0.0], [0.0, 5.0]])
        w = risk.risk_parity_weights(cov)
        self.assertGreater(w[0], w[1])


if __name__ == "__main__":
    unittest.main()
