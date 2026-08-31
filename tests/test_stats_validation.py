import unittest

import numpy as np
import pandas as pd

from src import stats_validation as sv


class TestNormalHelpers(unittest.TestCase):
    def test_norm_cdf_known_values(self):
        self.assertAlmostEqual(sv._norm_cdf(0.0), 0.5, places=6)
        self.assertAlmostEqual(sv._norm_cdf(1.959964), 0.975, places=4)

    def test_norm_ppf_known_values(self):
        self.assertAlmostEqual(sv._norm_ppf(0.5), 0.0, places=4)
        self.assertAlmostEqual(sv._norm_ppf(0.975), 1.959964, places=3)

    def test_norm_ppf_cdf_roundtrip(self):
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            x = sv._norm_ppf(p)
            self.assertAlmostEqual(sv._norm_cdf(x), p, places=4)


class TestExpectedMaxSharpeIID(unittest.TestCase):
    def test_raises_below_two_trials(self):
        with self.assertRaises(ValueError):
            sv.expected_max_sharpe_iid(1.0, 1)

    def test_increases_with_more_trials(self):
        low = sv.expected_max_sharpe_iid(1.0, 5)
        high = sv.expected_max_sharpe_iid(1.0, 50)
        self.assertGreater(high, low)

    def test_increases_with_variance(self):
        low = sv.expected_max_sharpe_iid(0.5, 10)
        high = sv.expected_max_sharpe_iid(2.0, 10)
        self.assertGreater(high, low)


class TestProbabilisticSharpeRatio(unittest.TestCase):
    def test_equals_half_when_observed_equals_benchmark(self):
        psr = sv.probabilistic_sharpe_ratio(1.0, 1.0, n_obs=252, skew=0.0, kurtosis=3.0)
        self.assertAlmostEqual(psr, 0.5, places=6)

    def test_increases_with_observed_sharpe(self):
        low = sv.probabilistic_sharpe_ratio(0.2, 0.0, n_obs=252)
        high = sv.probabilistic_sharpe_ratio(1.5, 0.0, n_obs=252)
        self.assertGreater(high, low)

    def test_more_observations_increases_confidence_when_positive(self):
        short = sv.probabilistic_sharpe_ratio(0.5, 0.0, n_obs=60)
        long = sv.probabilistic_sharpe_ratio(0.5, 0.0, n_obs=2520)
        self.assertGreater(long, short)


class TestDeflatedSharpeRatio(unittest.TestCase):
    def test_single_trial_matches_psr_vs_zero(self):
        result = sv.deflated_sharpe_ratio(observed_sharpe=0.8, trial_sharpes=[0.8], n_obs=500)
        self.assertAlmostEqual(result["benchmark_sharpe"], 0.0, places=6)
        self.assertAlmostEqual(result["deflated_sharpe_ratio"], result["psr_vs_zero"], places=6)

    def test_more_trials_never_increases_deflated_sharpe(self):
        few_trials = sv.deflated_sharpe_ratio(0.8, [0.1, 0.2, 0.8], n_obs=500)
        many_trials = sv.deflated_sharpe_ratio(0.8, [0.1, 0.2, 0.3, -0.1, 0.4, 0.8, -0.2, 0.5], n_obs=500)
        self.assertLessEqual(many_trials["deflated_sharpe_ratio"], few_trials["deflated_sharpe_ratio"])

    def test_deflated_sharpe_never_exceeds_psr_vs_zero(self):
        result = sv.deflated_sharpe_ratio(0.8, [0.1, 0.3, 0.8, -0.2, 0.5], n_obs=500)
        self.assertLessEqual(result["deflated_sharpe_ratio"], result["psr_vs_zero"] + 1e-9)


class TestPBOCSCV(unittest.TestCase):
    def test_genuinely_good_strategy_gets_low_pbo(self):
        rng = np.random.default_rng(0)
        n = 1000
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        returns = pd.DataFrame({
            "noise_1": rng.normal(0, 0.01, n),
            "noise_2": rng.normal(0, 0.01, n),
            "noise_3": rng.normal(0, 0.01, n),
            "genuinely_good": rng.normal(0.002, 0.01, n),
        }, index=idx)
        result = sv.pbo_cscv(returns, n_splits=10)
        self.assertLess(result["pbo"], 0.3)

    def test_pure_noise_gives_pbo_near_half(self):
        rng = np.random.default_rng(1)
        n = 1000
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        returns = pd.DataFrame({
            f"noise_{i}": rng.normal(0, 0.01, n) for i in range(6)
        }, index=idx)
        result = sv.pbo_cscv(returns, n_splits=10)
        self.assertGreater(result["pbo"], 0.3)
        self.assertLess(result["pbo"], 0.7)

    def test_rejects_nan_input(self):
        idx = pd.date_range("2015-01-01", periods=10, freq="B")
        returns = pd.DataFrame({"a": [0.01] * 9 + [np.nan], "b": [0.02] * 10}, index=idx)
        with self.assertRaises(ValueError):
            sv.pbo_cscv(returns, n_splits=2)

    def test_rejects_odd_n_splits(self):
        idx = pd.date_range("2015-01-01", periods=10, freq="B")
        returns = pd.DataFrame({"a": [0.01] * 10, "b": [0.02] * 10}, index=idx)
        with self.assertRaises(ValueError):
            sv.pbo_cscv(returns, n_splits=5)


if __name__ == "__main__":
    unittest.main()
