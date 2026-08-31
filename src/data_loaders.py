"""
data_loaders.py

Two kinds of data sources are provided:

1. REAL data loaders (`load_fred_series`, `load_yfinance_series`) — these hit
   FRED and Yahoo Finance and require an internet connection. Use these when
   running the strategy for real.

2. `generate_synthetic_macro_universe` — a seeded synthetic-data generator
   used ONLY for offline testing / demoing the pipeline (e.g. in a sandboxed
   CI environment with no network access). It is clearly separated from the
   real loaders so nobody mistakes synthetic output for live market data.

Instruments modeled (rates spreads are quoted in basis points, everything
else in price levels):
    - US2S10S : 10Y-2Y Treasury yield spread (bps)      -> live via FRED (DGS10 - DGS2)
    - US5S30S : 30Y-5Y Treasury yield spread (bps)       -> live via FRED (DGS30 - DGS5)
    - EURUSD  : EUR/USD spot                             -> live via yfinance ("EURUSD=X")
    - GOLD    : Gold spot (USD/oz)                        -> live via yfinance ("GC=F")
    - SPX     : S&P 500 index level                       -> live via yfinance ("^GSPC")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE = ["US2S10S", "US5S30S", "EURUSD", "GOLD", "SPX"]


# ---------------------------------------------------------------------------
# REAL DATA LOADERS (require internet — not used in the offline demo/tests)
# ---------------------------------------------------------------------------

def load_fred_series(series_id: str, start: str = "2015-01-01") -> pd.Series:
    """Pull a single FRED series (e.g. 'DGS10', 'DGS2') via pandas-datareader."""
    import pandas_datareader.data as web

    s = web.DataReader(series_id, "fred", start)[series_id]
    s.name = series_id
    return s.ffill().dropna()


def load_yfinance_series(ticker: str, start: str = "2015-01-01") -> pd.Series:
    """Pull a single close-price series via yfinance."""
    import yfinance as yf

    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    close = df["Close"]
    # newer yfinance versions return MultiIndex columns (one column per
    # ticker) even for a single-ticker download, so "Close" comes back as a
    # one-column DataFrame rather than a Series -- squeeze it before renaming.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.rename(ticker)
    return s.ffill().dropna()


def load_live_universe(start: str = "2015-01-01") -> pd.DataFrame:
    """
    Build the full instrument universe from live FRED + Yahoo Finance data.
    Requires internet access. Returns a DataFrame of price/level series
    aligned on a common business-day index, column names matching UNIVERSE.
    """
    dgs10 = load_fred_series("DGS10", start)
    dgs2 = load_fred_series("DGS2", start)
    dgs30 = load_fred_series("DGS30", start)
    dgs5 = load_fred_series("DGS5", start)

    us2s10s = (dgs10 - dgs2) * 100  # bps
    us5s30s = (dgs30 - dgs5) * 100  # bps

    eurusd = load_yfinance_series("EURUSD=X", start)
    gold = load_yfinance_series("GC=F", start)
    spx = load_yfinance_series("^GSPC", start)

    df = pd.concat(
        [us2s10s.rename("US2S10S"), us5s30s.rename("US5S30S"),
         eurusd.rename("EURUSD"), gold.rename("GOLD"), spx.rename("SPX")],
        axis=1,
    )
    return df.ffill().dropna()


# ---------------------------------------------------------------------------
# SYNTHETIC DATA (offline demo / unit tests only — clearly labeled as such)
# ---------------------------------------------------------------------------

def generate_synthetic_macro_universe(
    n_days: int = 1500,
    seed: int = 42,
    start_date: str = "2019-01-01",
) -> pd.DataFrame:
    """
    Generate a SYNTHETIC multi-instrument macro dataset for offline testing
    and demoing the pipeline without network access. This is NOT real market
    data — it is a regime-switching random walk designed to contain the kind
    of persistent trends/reversals that a trend-following system is built to
    capture, so the pipeline can be exercised end-to-end.

    Do not use synthetic output to justify any performance claim on a CV or
    in an interview — always re-run against `load_live_universe()` first.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    def regime_switching_walk(start_level, n, vol, drift_scale, n_regimes=6):
        # split the series into randomly-sized regimes, each with its own
        # drift, to create persistent trends of the kind trend-following aims
        # to exploit -- rather than a pure random walk with no exploitable signal
        breaks = np.sort(rng.choice(np.arange(50, n - 50), size=n_regimes - 1, replace=False))
        breaks = np.concatenate([[0], breaks, [n]])
        drifts = rng.normal(0, drift_scale, size=n_regimes)
        increments = np.zeros(n)
        for i in range(n_regimes):
            lo, hi = breaks[i], breaks[i + 1]
            increments[lo:hi] = rng.normal(drifts[i], vol, size=hi - lo)
        return start_level + np.cumsum(increments)

    us2s10s = regime_switching_walk(30, n_days, vol=3.0, drift_scale=0.15)
    us5s30s = regime_switching_walk(60, n_days, vol=2.5, drift_scale=0.12)
    eurusd = regime_switching_walk(1.10, n_days, vol=0.004, drift_scale=0.0002)
    gold = regime_switching_walk(1800, n_days, vol=12.0, drift_scale=0.6)
    spx = np.exp(regime_switching_walk(np.log(4000), n_days, vol=0.009, drift_scale=0.0004))

    df = pd.DataFrame(
        {
            "US2S10S": us2s10s,
            "US5S30S": us5s30s,
            "EURUSD": eurusd,
            "GOLD": gold,
            "SPX": spx,
        },
        index=dates,
    )
    return df
