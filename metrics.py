"""
metrics.py

Standard risk-adjusted performance metrics, implemented directly so the repo
has zero hard dependency on quantstats to compute the numbers that go in the
README (quantstats is still used in report.py for the nicer HTML tearsheet,
if installed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    growth = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return growth ** (1 / years) - 1 if years > 0 else np.nan


def annualized_vol(returns: pd.Series) -> float:
    return returns.dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns.dropna() - rf / TRADING_DAYS
    vol = excess.std()
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS) if vol > 0 else np.nan


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns.dropna() - rf / TRADING_DAYS
    downside = excess[excess < 0]
    downside_vol = downside.std()
    return (excess.mean() / downside_vol) * np.sqrt(TRADING_DAYS) if downside_vol > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns.dropna()).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    return drawdown.min()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return annualized_return(returns) / abs(mdd) if mdd != 0 else np.nan


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    return (r > 0).sum() / len(r) if len(r) > 0 else np.nan


def tail_ratio(returns: pd.Series, pct: float = 0.05) -> float:
    r = returns.dropna()
    right = r.quantile(1 - pct)
    left = r.quantile(pct)
    return abs(right / left) if left != 0 else np.nan


def summary(returns: pd.Series, name: str = "Strategy") -> pd.Series:
    return pd.Series({
        "Ann. Return": annualized_return(returns),
        "Ann. Vol": annualized_vol(returns),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns),
        "Hit Rate": hit_rate(returns),
        "Tail Ratio": tail_ratio(returns),
    }, name=name)
