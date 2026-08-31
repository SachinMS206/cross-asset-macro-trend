"""
risk.py

Position sizing and portfolio-level risk allocation.

Two layers, mirroring how systematic macro books (e.g. the pysystemtrade
framework) actually size risk:

1. INSTRUMENT-LEVEL vol targeting (`vol_target_position_size`): scale each
   instrument's raw forecast so that, in isolation, it targets a constant
   annualized volatility contribution. This stops the book being dominated
   by whichever instrument happens to be most volatile.

2. PORTFOLIO-LEVEL weighting (`inverse_vol_weights`, `risk_parity_weights`):
   allocate capital/risk across instruments. `inverse_vol_weights` is a
   simple, dependency-free approximation; `risk_parity_weights` solves the
   full risk-parity problem (equal risk contribution) via `scipy.optimize`,
   and there's a thin wrapper to swap in Riskfolio-Lib's HRP/HERC
   implementations if that package is installed (see `hrp_weights`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realized_vol(returns: pd.Series, window: int = 36, annualize: bool = True) -> pd.Series:
    """EW realized volatility of a returns series."""
    vol = returns.ewm(span=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def vol_target_position_size(
    forecast: pd.Series,
    price: pd.Series,
    instrument_returns: pd.Series,
    target_ann_vol: float = 0.10,
    vol_window: int = 36,
    forecast_cap: float = 20.0,
) -> pd.Series:
    """
    Convert a scaled forecast (roughly in [-cap, +cap], avg abs ~10) into a
    position size (in "vol-target units", i.e. fraction of capital notional)
    such that a forecast of +/-10 corresponds to holding a position sized to
    target `target_ann_vol` annualized volatility.

    position = (forecast / 10) * (target_ann_vol / realized_ann_vol)

    This is the core Carver-style vol-targeting formula: it means a fast,
    choppy instrument (e.g. FX) and a slow, low-vol instrument (e.g. a rate
    spread) contribute similar risk to the book for the same forecast
    strength, instead of the book being accidentally dominated by whichever
    instrument happens to have the biggest raw price moves.
    """
    ann_vol = realized_vol(instrument_returns, vol_window, annualize=True)
    ann_vol = ann_vol.replace(0, np.nan)

    normalized_forecast = forecast.clip(-forecast_cap, forecast_cap) / 10.0
    position = normalized_forecast * (target_ann_vol / ann_vol)
    position.name = "position"
    return position


def inverse_vol_weights(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Simple, dependency-free risk-based portfolio weights: weight each
    instrument inversely proportional to its rolling volatility, then
    normalize to sum to 1. Ignores cross-instrument correlation -- a
    reasonable, robust default when correlations are noisy/unstable, which
    is common across macro instruments.
    """
    vol = returns.rolling(window).std()
    inv_vol = 1.0 / vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights


def risk_parity_weights(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """
    Solve for the equal-risk-contribution (risk parity) portfolio given a
    covariance matrix, via scipy.optimize.minimize on the sum of squared
    deviations between each asset's risk contribution and the target
    (equal) share. Dependency-free alternative to Riskfolio-Lib for
    environments where that package isn't installed.
    """
    from scipy.optimize import minimize

    n = cov.shape[0]
    target = np.ones(n) / n

    def risk_contributions(w):
        port_var = w @ cov @ w
        marginal = cov @ w
        return (w * marginal) / port_var

    def objective(w):
        rc = risk_contributions(w)
        return np.sum((rc - target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(1e-6, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": tol},
    )
    return result.x if result.success else w0


def hrp_weights(returns: pd.DataFrame):
    """
    Optional wrapper around Riskfolio-Lib's Hierarchical Risk Parity, used
    if the package is installed. Falls back to `risk_parity_weights` above
    if Riskfolio-Lib isn't available, so the pipeline never hard-fails.

        pip install Riskfolio-Lib

    """
    try:
        import riskfolio as rp
    except ImportError:
        cov = returns.cov().values
        w = risk_parity_weights(cov)
        return pd.Series(w, index=returns.columns)

    port = rp.HCPortfolio(returns=returns.dropna())
    w = port.optimization(model="HRP", codependence="pearson", rm="MV", rf=0, linkage="single")
    return w["weights"]
