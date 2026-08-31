import unittest

import numpy as np
import pandas as pd

from src import backtest, data_loaders


class TestSingleInstrumentBacktest(unittest.TestCase):
    def test_position_is_lagged_no_lookahead(self):
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        prices = pd.Series(100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 300)), index=idx)
        result = backtest.run_single_instrument_backtest(prices)

        # position at t must equal (pre-lag) position computed using data
        # available at t-1 -- i.e. position.shift back should match a
        # forecast/vol snapshot strictly before t. We check the simple
        # invariant: position on day 0 (or first valid day) is NaN/0, since
        # there is no information yet to trade on.
        first_valid = result["position"].first_valid_index()
        self.assertIsNotNone(first_valid)
        self.assertGreater(idx.get_loc(first_valid), 0)

    def test_costs_are_nonnegative_and_reduce_returns(self):
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        prices = pd.Series(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 300)), index=idx)
        result = backtest.run_single_instrument_backtest(prices, cost_bps=5.0)

        self.assertTrue((result["cost"].dropna() >= 0).all())
        valid = result[["gross_return", "net_return", "cost"]].dropna()
        diff = valid["gross_return"] - valid["net_return"]
        np.testing.assert_allclose(diff.values, valid["cost"].values, atol=1e-9)

    def test_zero_cost_means_gross_equals_net(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        prices = pd.Series(100 + np.cumsum(np.random.default_rng(4).normal(0, 1, 200)), index=idx)
        result = backtest.run_single_instrument_backtest(prices, cost_bps=0.0)
        pd.testing.assert_series_equal(
            result["gross_return"].dropna(), result["net_return"].dropna(), check_names=False
        )


class TestPortfolioBacktest(unittest.TestCase):
    def test_runs_end_to_end_on_synthetic_universe(self):
        universe = data_loaders.generate_synthetic_macro_universe(n_days=400, seed=7)
        result = backtest.run_portfolio_backtest(universe)
        self.assertIn("portfolio_returns", result)
        self.assertGreater(result["portfolio_returns"].dropna().shape[0], 0)
        # portfolio returns should be finite (catches unit-mismatch bugs,
        # e.g. mixing percentage and price-difference returns)
        self.assertTrue(np.isfinite(result["portfolio_returns"].dropna()).all())

    def test_weights_sum_to_approximately_one_each_day(self):
        universe = data_loaders.generate_synthetic_macro_universe(n_days=400, seed=8)
        result = backtest.run_portfolio_backtest(universe, weighting="equal")
        row_sums = result["weights"].dropna().sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


class TestWalkForwardSplits(unittest.TestCase):
    def test_splits_are_contiguous_and_non_overlapping(self):
        idx = pd.bdate_range("2015-01-01", periods=2000)
        splits = list(backtest.walk_forward_splits(idx, n_splits=4, min_train_years=2.0))
        self.assertEqual(len(splits), 4)
        for train_idx, test_idx in splits:
            overlap = set(train_idx).intersection(set(test_idx))
            self.assertEqual(len(overlap), 0)

    def test_expanding_train_window(self):
        idx = pd.bdate_range("2015-01-01", periods=2000)
        splits = list(backtest.walk_forward_splits(idx, n_splits=4, min_train_years=2.0))
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        self.assertEqual(train_sizes, sorted(train_sizes))  # non-decreasing


if __name__ == "__main__":
    unittest.main()
