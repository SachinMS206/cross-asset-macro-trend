import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import combine_portfolios as cp  # noqa: E402


class TestRealizedAnnVol(unittest.TestCase):
    def test_nan_before_window_fills(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        s = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=idx)
        vol = cp.realized_ann_vol(s, window=60)
        self.assertTrue(vol.iloc[:59].isna().all())
        self.assertFalse(pd.isna(vol.iloc[-1]))


class TestEqualRiskBlend(unittest.TestCase):
    def test_blend_is_no_lookahead(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        rng = np.random.default_rng(1)
        trend = pd.Series(rng.normal(0, 0.01, 200), index=idx)
        rv = pd.Series(rng.normal(0, 0.005, 200), index=idx)
        blend = cp.equal_risk_blend(trend, rv, window=60)
        # 60 obs to build the trailing vol estimate, then one more day of
        # lag before the blend can use it -- first valid at index 60
        self.assertTrue(blend.iloc[:60].isna().all())
        self.assertFalse(pd.isna(blend.iloc[60]))
        self.assertFalse(pd.isna(blend.iloc[-1]))

    def test_scaled_components_have_similar_realized_vol(self):
        """
        A naive 50/50 average of raw returns would let whichever stream
        happens to run hotter dominate the combined risk. Directly check
        the defining property of the risk-scaling step: after rescaling
        each stream to the same target vol using its own trailing
        realized vol, the two streams' own realized vols post-scaling
        should land close together -- even though the raw inputs here
        differ by 10x in volatility.
        """
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        rng = np.random.default_rng(2)
        target = 0.10
        low_vol = pd.Series(rng.normal(0, 0.002, 500), index=idx)
        high_vol = pd.Series(rng.normal(0, 0.02, 500), index=idx)

        low_scaled = (low_vol * (target / cp.realized_ann_vol(low_vol, window=60).shift(1))).dropna()
        high_scaled = (high_vol * (target / cp.realized_ann_vol(high_vol, window=60).shift(1))).dropna()

        low_realized = low_scaled.std() * np.sqrt(cp.TRADING_DAYS)
        high_realized = high_scaled.std() * np.sqrt(cp.TRADING_DAYS)

        ratio = max(low_realized, high_realized) / min(low_realized, high_realized)
        self.assertLess(ratio, 1.5)


if __name__ == "__main__":
    unittest.main()
