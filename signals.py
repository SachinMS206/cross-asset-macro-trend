"""
signals.py

Trend-following signal construction.

Two signal types are provided:

1. `dual_ma_signal` — the simple fast/slow moving-average crossover used in
   the original 2s10s single-pair project (kept for continuity/comparison).

2. `ewmac_signal` — an exponentially-weighted moving-average crossover,
   normalized by the instrument's own volatility, in the style described in
   Rob Carver's "Systematic Trading". This is the signal used by the
   multi-instrument portfolio because it (a) reacts faster with less lag
   than SMA crossovers, and (b) produces a *scaled* forecast (roughly in
   [-20, +20]) that's comparable across instruments with very different
   volatilities -- which is what makes cross-asset combination sensible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dual_ma_signal(prices: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """+1 when fast SMA > slow SMA, -1 otherwise. Simple, easy to explain."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = np.where(fast_ma > slow_ma, 1.0, -1.0)
    return pd.Series(signal, index=prices.index, name="signal").where(slow_ma.notna())


def _ewmac(prices: pd.Series, fast_span: int, slow_span: int) -> pd.Series:
    fast_ema = prices.ewm(span=fast_span, min_periods=fast_span).mean()
    slow_ema = prices.ewm(span=slow_span, min_periods=slow_span).mean()
    return fast_ema - slow_ema


def ewmac_signal(
    prices: pd.Series,
    fast_span: int = 16,
    slow_span: int = 64,
    vol_lookback: int = 36,
    cap: float = 20.0,
) -> pd.Series:
    """
    Volatility-normalized EWMAC forecast.

    raw = EMA_fast(price) - EMA_slow(price)
    vol-adjusted forecast = raw / rolling_price_vol
    scaled forecast = forecast * scalar, clipped to [-cap, +cap]

    The scaling constant is chosen so the average absolute forecast is
    roughly 10 (Carver's convention), making forecasts from different
    instruments/speeds comparable before they're combined and vol-targeted.
    """
    raw = _ewmac(prices, fast_span, slow_span)
    daily_returns = prices.diff()
    price_vol = daily_returns.ewm(span=vol_lookback, min_periods=vol_lookback).std()

    vol_adj = raw / price_vol.replace(0, np.nan)

    # rescale so the historical average absolute forecast ~= 10
    avg_abs = vol_adj.abs().expanding(min_periods=slow_span).mean()
    scalar = 10.0 / avg_abs.replace(0, np.nan)
    forecast = (vol_adj * scalar).clip(-cap, cap)
    forecast.name = "signal"
    return forecast
