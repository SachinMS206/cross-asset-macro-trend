"""
backtest.py

Vectorized, dependency-free backtest engine plus a walk-forward validation
harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import risk as risk_mod
from src import signals as signals_mod


def _instrument_returns(prices: pd.Series) -> pd.Series:
    """
    Return series used for BOTH signal-vol-normalization and P&L, defined
    uniformly as the raw price/level DIFFERENCE (not pct_change), since
    several instruments (spreads) cross zero.
    """
    return prices.diff()


def run_single_instrument_backtest(
    prices: pd.Series,
    fast_span: int = 16,
    slow_span: int = 64,
    target_ann_vol: float = 0.10,
    cost_bps: float = 1.0,
    signal_fn=None,
) -> pd.DataFrame:
    """
    Backtest one instrument's vol-targeted trend strategy.

    signal_fn: optional callable(prices) -> forecast Series. Defaults to the
    single-speed ewmac_signal(prices, fast_span, slow_span) for backward
    compatibility. Pass signals_mod.multi_speed_ewmac_signal to use the
    blended-speed signal instead.
    """
    returns = _instrument_returns(prices)

    if signal_fn is None:
        forecast = signals_mod.ewmac_signal(prices, fast_span, slow_span)
    else:
        forecast = signal_fn(prices)

    position = risk_mod.vol_target_position_size(
        forecast, prices, returns, target_ann_vol=target_ann_vol
    )

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


def run_vol_targeted_buy_and_hold(
    prices: pd.Series,
    target_ann_vol: float = 0.10,
    cost_bps: float = 1.0,
) -> pd.Series:
    """Benchmark: constant max-long forecast through the same vol-targeting machinery."""
    returns = _instrument_returns(prices)
    constant_forecast = pd.Series(10.0, index=prices.index)
    position = risk_mod.vol_target_position_size(constant_forecast, prices, returns, target_ann_vol)
    position_lagged = position.shift(1)
    gross_return = position_lagged * returns
    turnover = position_lagged.diff().abs().fillna(0)
    costs = turnover * (cost_bps / 10_000)
    return (gross_return - costs).rename("net_return")


def run_portfolio_backtest(
    universe: pd.DataFrame,
    fast_span: int = 16,
    slow_span: int = 64,
    target_ann_vol: float = 0.10,
    cost_bps: float = 1.0,
    weighting: str = "inverse_vol",
    rebalance_window: int = 60,
    signal_fn=None,
) -> dict:
    instrument_returns = {}
    instrument_positions = {}

    for col in universe.columns:
        result = run_single_instrument_backtest(
            universe[col], fast_span, slow_span, target_ann_vol, cost_bps, signal_fn=signal_fn
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

    weights = weights.shift(1).fillna(1.0 / net_returns_df.shape[1])
    portfolio_returns = (net_returns_df * weights).sum(axis=1, skipna=True)
    portfolio_returns.name = "portfolio_net_return"

    return {
        "instrument_returns": net_returns_df,
        "weights": weights,
        "portfolio_returns": portfolio_returns,
    }


def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int = 4, min_train_years: float = 2.0):
    n = len(index)
    min_train = int(min_train_years * 252)
    if min_train >= n:
        raise ValueError("Not enough data for the requested min_train_years.")

    test_block = (n - min_train) // n_splits
    for i in range(n_splits):
        train_end = min_train + i * test_block
        test_end = min_train + (i + 1) * test_block if i < n_splits - 1 else n
        yield index[:train_end], index[train_end:test_end]
