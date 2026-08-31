"""
combine_portfolios.py

Tests a specific, well-founded portfolio-construction question: does
blending this repo's trend-following book with the mean-reversion book
from github.com/SachinMS206/cross-asset-rv-monitor -- two genuinely
different, independently-built strategies -- improve risk-adjusted
returns through diversification?

This does NOT re-derive either strategy's logic. It reads the real daily
portfolio_returns CSVs that scripts/run_demo.py and (in the other repo)
scripts/run_monitor.py already export via --export-returns, so the
combined result is built on the actual, independently-tested pipelines,
not a reimplementation.

Usage (see .github/workflows/combined-portfolio.yml for how this is
actually run, since it needs both repos checked out):

    python scripts/combine_portfolios.py \
        --trend-csv reports/trend_returns.csv \
        --rv-csv reports/rv_returns.csv \
        --out-doc PORTFOLIO_CONSTRUCTION.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import metrics

TRADING_DAYS = 252
START_MARKER = "<!-- LIVE_RESULTS_START -->"
END_MARKER = "<!-- LIVE_RESULTS_END -->"


def realized_ann_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def equal_risk_blend(trend: pd.Series, rv: pd.Series, target_ann_vol: float = 0.10, window: int = 60) -> pd.Series:
    """
    Rescale each return stream to the same target vol using its own
    trailing (lagged, no look-ahead) realized vol, then average 50/50 --
    a true equal-risk-contribution blend rather than a raw 50/50 of the
    unscaled returns, which would let whichever stream happens to run
    hotter dominate the combined risk.
    """
    trend_vol = realized_ann_vol(trend, window).shift(1)
    rv_vol = realized_ann_vol(rv, window).shift(1)

    trend_scaled = trend * (target_ann_vol / trend_vol)
    rv_scaled = rv * (target_ann_vol / rv_vol)

    blend = 0.5 * trend_scaled + 0.5 * rv_scaled
    blend.name = "blended_return"
    return blend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trend-csv", required=True)
    parser.add_argument("--rv-csv", required=True)
    parser.add_argument("--out-doc", default="PORTFOLIO_CONSTRUCTION.md")
    args = parser.parse_args()

    trend_df = pd.read_csv(args.trend_csv, index_col=0, parse_dates=True)
    rv_df = pd.read_csv(args.rv_csv, index_col=0, parse_dates=True)

    trend_returns = trend_df["trend_portfolio_return"]
    benchmark_returns = trend_df["benchmark_return"] if "benchmark_return" in trend_df.columns else None
    rv_returns = rv_df["rv_portfolio_return"]

    combined = pd.concat([trend_returns, rv_returns], axis=1).dropna()
    aligned_trend = combined["trend_portfolio_return"]
    aligned_rv = combined["rv_portfolio_return"]

    correlation = aligned_trend.corr(aligned_rv)
    blend = equal_risk_blend(aligned_trend, aligned_rv)

    rows = [metrics.summary(aligned_trend, "Trend book"), metrics.summary(aligned_rv, "RV book")]
    if benchmark_returns is not None:
        aligned_bh = benchmark_returns.reindex(combined.index)
        rows.append(metrics.summary(aligned_bh, "Benchmark (vol-targeted buy & hold)"))
    rows.append(metrics.summary(blend, "Trend + RV blend (equal risk)"))
    table = pd.concat(rows, axis=1)

    pct_rows = {"Ann. Return", "Ann. Vol", "Max Drawdown", "Hit Rate"}

    def fmt(row_name, value):
        if pd.isna(value):
            return "n/a"
        return f"{value:.2%}" if row_name in pct_rows else f"{value:.2f}"

    formatted = pd.DataFrame(
        {c: [fmt(idx, table.loc[idx, c]) for idx in table.index] for c in table.columns},
        index=table.index,
    )
    md_table = formatted.to_markdown()

    print(f"Overlapping sample: {combined.index[0].date()} to {combined.index[-1].date()} ({len(combined)} days)")
    print(f"Correlation between trend and RV book returns: {correlation:+.3f}")
    print()
    print(table)

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    diversification_note = (
        "genuinely close to zero, consistent with the two books trading different "
        "logic on different (mostly non-overlapping) instruments"
        if abs(correlation) < 0.2
        else "not close to zero -- the diversification argument below is weaker than "
        "hoped, and should be read with that in mind"
    )

    block = (
        f"{START_MARKER}\n"
        f"*Last refreshed automatically from live data: {timestamp}. "
        f"See `.github/workflows/combined-portfolio.yml`.*\n\n"
        f"Overlapping live sample: {combined.index[0].date()} to {combined.index[-1].date()} "
        f"({len(combined)} trading days).\n\n"
        f"**Correlation between the trend book's and RV book's daily returns: {correlation:+.3f}** "
        f"-- {diversification_note}.\n\n"
        f"{md_table}\n\n"
        f"{END_MARKER}"
    )

    out_path = Path(args.out_doc)
    doc = out_path.read_text()
    if START_MARKER not in doc or END_MARKER not in doc:
        raise SystemExit(f"{args.out_doc} is missing the {START_MARKER}/{END_MARKER} markers.")
    before = doc.split(START_MARKER)[0]
    after = doc.split(END_MARKER)[1]
    out_path.write_text(before + block + after)
    print(f"\nUpdated {args.out_doc}")


if __name__ == "__main__":
    main()
