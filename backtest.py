"""
backtest.py

Vectorized, dependency-free backtest engine (pure pandas/numpy) plus a
walk-forward validation harness.

Why not just hand this to vectorbt? vectorbt is recommended in requirements.txt
and is a fine drop-in for the single-instrument case (and is faster for large
parameter sweeps), but a hand-rolled engine here makes the exact mechanics
(costs, lag, position limits) auditable in one place -- which matters more
for a CV project than raw speed, and it means this repo has zero heavy
dependencies to actually run and verify.

Core mechanics:
    - Signals are generated using data available up to t-1 only, and applied
      to returns realized from t to t+1 (no look-ahead).
    - Transaction costs are charged in bps of notional traded whenever the
      position changes.
    - Portfolio return = sum_i weight_i(t) * position_i(t) * instrument_return_i(t+1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import risk as risk_mod
from src import signals as signals_mod


def _instrument_returns(prices: pd.Series) -> pd.Series:
    """
    Return series used for BOTH signal-vol-normalization and P&L, defined
    uniformly as the raw price/level DIFFERENCE (not pct_change).

    This is a deliberate choice, not an oversight: several instruments in
    the universe are spreads (e.g. the 2s10s curve, quoted in bps) that
    cross zero, so pct_change() is undefined/explosive for them. Since
    every position is subsequently vol-targeted (scaled by
    target_vol / realized_vol of this same difference series), using a
    consistent price-difference return convention across instruments still
    produces comparable, correctly-scaled risk contributions -- this is the
    standard convention in futures/spread trend-following systems (e.g.
    pysystemtrade), where P&L is naturally computed in price-difference
    terms per contract rather than as a percentage.
    """
    return prices.diff()


def run_single_instrument_backtest(
    prices: pd.Series,
    fast_span: int = 16,
    slow_span: int = 64,
    target_ann_vol: float = 0.10,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Backtest one instrument's vol-targeted EWMAC trend strategy."""
    returns = _instrument_returns(prices)
    forecast = signals_mod.ewmac_signal(prices, fast_span, slow_span)
    position = risk_mod.vol_target_position_size(
        forecast, prices, returns, target_ann_vol=target_ann_vol
    )

    # lag position by 1 day: trade on today's close using yesterday's signal
    position_lagged = position.shift(1)
    gross_return = position_lagged * returns

    turnover = position_lagged.diff().abs().fillna(0)
    costs = turnover * (cost_bps / 10_000)
    net_return = gross_return - costs

    out = pd.DataFrame({
        "price": prices,
        "forecast": forecast,
        "position": position_lagged,
        "gross_return": gross_return,
        "cost": costs,
        "net_return": net_return,
    })
    return out


def run_portfolio_backtest(
    universe: pd.DataFrame,
    fast_span: int = 16,
    slow_span: int = 64,
    target_ann_vol: float = 0.10,
    cost_bps: float = 1.0,
    weighting: str = "inverse_vol",
    rebalance_window: int = 60,
) -> dict:
    """
    Run the full cross-instrument pipeline:
      1. per-instrument EWMAC forecast -> vol-targeted position
      2. per-instrument net returns (after costs)
      3. portfolio weights (inverse-vol or risk-parity) applied across
         instruments, rebalanced on a rolling basis
      4. combined portfolio return series

    Returns a dict with the per-instrument breakdown and the combined
    portfolio return series, so results can be inspected at either level.
    """
    instrument_returns = {}
    instrument_positions = {}

    for col in universe.columns:
        result = run_single_instrument_backtest(
            universe[col], fast_span, slow_span, target_ann_vol, cost_bps
        )
        instrument_returns[col] = result["net_return"]
        instrument_positions[col] = result["position"]

    net_returns_df = pd.DataFrame(instrument_returns).dropna(how="all")

    if weighting == "inverse_vol":
        weights = risk_mod.inverse_vol_weights(net_returns_df, window=rebalance_window)
    elif weighting == "equal":
        weights = pd.DataFrame(
            1.0 / net_returns_df.shape[1], index=net_returns_df.index, columns=net_returns_df.columns
        )
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting}")

    weights = weights.shift(1).fillna(1.0 / net_returns_df.shape[1])  # avoid look-ahead on weights too
    portfolio_returns = (net_returns_df * weights).sum(axis=1, skipna=True)
    portfolio_returns.name = "portfolio_net_return"

    return {
        "instrument_returns": net_returns_df,
        "weights": weights,
        "portfolio_returns": portfolio_returns,
    }


def run_vol_targeted_buy_and_hold(
    prices: pd.Series,
    target_ann_vol: float = 0.10,
    cost_bps: float = 1.0,
) -> pd.Series:
    """
    Benchmark: a constant max-long forecast (+10, i.e. no trend timing at
    all) put through the SAME vol-targeting machinery as the trend strategy.
    This is a fairer comparison than a naive equal-weight raw-return
    benchmark, because it isolates the value added by the *trend signal*
    specifically, rather than mixing in the effect of vol-targeting itself
    (which the strategy also benefits from).
    """
    returns = _instrument_returns(prices)
    constant_forecast = pd.Series(10.0, index=prices.index)
    position = risk_mod.vol_target_position_size(constant_forecast, prices, returns, target_ann_vol)
    position_lagged = position.shift(1)
    gross_return = position_lagged * returns
    turnover = position_lagged.diff().abs().fillna(0)
    costs = turnover * (cost_bps / 10_000)
    return (gross_return - costs).rename("net_return")


def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int = 4, min_train_years: float = 2.0):
    """
    Yield (train_idx, test_idx) index pairs for expanding-window walk-forward
    validation: each fold trains on everything up to a point and tests on the
    following out-of-sample block, then the window expands. This is used
    instead of a single static in/out-of-sample split to check whether
    performance is consistent across regimes rather than an artifact of one
    lucky test window.
    """
    n = len(index)
    min_train = int(min_train_years * 252)
    if min_train >= n:
        raise ValueError("Not enough data for the requested min_train_years.")

    test_block = (n - min_train) // n_splits
    for i in range(n_splits):
        train_end = min_train + i * test_block
        test_end = min_train + (i + 1) * test_block if i < n_splits - 1 else n
        yield index[:train_end], index[train_end:test_end]
