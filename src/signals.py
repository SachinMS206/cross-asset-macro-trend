"""
signals.py

Trend-following signal construction.

Three signal types are provided:

1. `dual_ma_signal` -- the simple fast/slow moving-average crossover used in
   the original 2s10s single-pair project (kept for continuity/comparison).

2. `ewmac_signal` -- a SINGLE-SPEED exponentially-weighted moving-average
   crossover, normalized by the instrument's own volatility, in the style
   described in Rob Carver's "Systematic Trading". This was the original
   signal used by the multi-instrument portfolio.

3. `multi_speed_ewmac_signal` -- NEW. Averages several EWMAC speeds together
   (a "trend blend") rather than committing to one fast/slow pair. This is
   the standard fix, per Carver, for the exact fragility flagged in
   CRITIQUE.md: a single-speed EWMAC's performance swings substantially with
   nearby, equally-defensible speed choices, which is a sign the single-speed
   result may be closer to noise than edge. Blending speeds is not a
   guarantee of better performance -- it's a bet that averaging over
   several noisy single-speed forecasts reduces the variance of the combined
   forecast without giving up much genuine trend-following signal, since
   different speeds tend to be partially uncorrelated in their errors even
   when they're all reacting to the same underlying trend.
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
    Single-speed volatility-normalized EWMAC forecast.

    raw = EMA_fast(price) - EMA_slow(price)
    vol-adjusted forecast = raw / rolling_price_vol
    scaled forecast = forecast * scalar, clipped to [-cap, +cap]
    """
    raw = _ewmac(prices, fast_span, slow_span)
    daily_returns = prices.diff()
    price_vol = daily_returns.ewm(span=vol_lookback, min_periods=vol_lookback).std()

    vol_adj = raw / price_vol.replace(0, np.nan)

    avg_abs = vol_adj.abs().expanding(min_periods=slow_span).mean()
    scalar = 10.0 / avg_abs.replace(0, np.nan)
    forecast = (vol_adj * scalar).clip(-cap, cap)
    forecast.name = "signal"
    return forecast


# Default speed set, following Carver's convention of doubling each pair:
# (8,32), (16,64), (32,128) -- fast/slow spans roughly 4x apart, each pair
# 2x the previous. This spans "fast trend" to "slow trend" reactions.
DEFAULT_SPEEDS = [(8, 32), (16, 64), (32, 128)]


def multi_speed_ewmac_signal(
    prices: pd.Series,
    speeds: list[tuple[int, int]] = None,
    vol_lookback: int = 36,
    cap: float = 20.0,
) -> pd.Series:
    """
    Blended EWMAC forecast: computes a single-speed forecast (as in
    `ewmac_signal`) for each (fast_span, slow_span) pair in `speeds`, then
    averages them together, and re-clips to the cap.

    This directly addresses CRITIQUE.md's finding that the single 16/64
    speed choice was arbitrary and neighbouring speeds swung the reported
    Sharpe substantially (0.117 to 0.445 across a handful of nearby
    choices). Averaging speeds doesn't eliminate that sensitivity, but it
    means the final signal isn't a bet on any one arbitrary speed choice
    being the "right" one.
    """
    if speeds is None:
        speeds = DEFAULT_SPEEDS

    individual_forecasts = [
        ewmac_signal(prices, fast_span=f, slow_span=s, vol_lookback=vol_lookback, cap=cap)
        for f, s in speeds
    ]
    blended = pd.concat(individual_forecasts, axis=1).mean(axis=1)

    # re-scale so the blended forecast's average absolute value is still
    # ~10, since averaging several already-scaled forecasts pulls the
    # average magnitude down (individual signals rarely agree in strength
    # and sign at the same time), then re-clip to the cap
    avg_abs = blended.abs().expanding(min_periods=max(s for _, s in speeds)).mean()
    scalar = 10.0 / avg_abs.replace(0, np.nan)
    forecast = (blended * scalar).clip(-cap, cap)
    forecast.name = "signal"
    return forecast
