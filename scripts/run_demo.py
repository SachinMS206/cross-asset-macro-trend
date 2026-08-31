"""
run_demo.py

End-to-end pipeline runner.

    python scripts/run_demo.py                       # synthetic data, single-speed signal
    python scripts/run_demo.py --live                 # real data, single-speed signal
    python scripts/run_demo.py --live --multi-speed    # real data, blended-speed signal (see CRITIQUE.md)

The --multi-speed flag switches from the original single 16/64-day EWMAC
speed to a blend of three speeds (8/32, 16/64, 32/128), addressing the
speed-sensitivity weakness flagged in CRITIQUE.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import backtest, data_loaders, metrics, report, signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use real FRED/Yahoo data instead of synthetic")
    parser.add_argument("--multi-speed", action="store_true", help="Use the blended multi-speed EWMAC signal instead of single-speed")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--weighting", default="inverse_vol", choices=["inverse_vol", "equal"])
    args = parser.parse_args()

    if args.live:
        print("Pulling live data from FRED / Yahoo Finance ...")
        universe = data_loaders.load_live_universe(start=args.start)
    else:
        print("Using SYNTHETIC data (offline demo mode) -- see src/data_loaders.py docstring.")
        universe = data_loaders.generate_synthetic_macro_universe()

    print(f"Universe: {list(universe.columns)}  |  {universe.index[0].date()} to {universe.index[-1].date()}")

    signal_fn = signals.multi_speed_ewmac_signal if args.multi_speed else None
    if args.multi_speed:
        print("Signal: multi-speed EWMAC blend (8/32, 16/64, 32/128)")
    else:
        print("Signal: single-speed EWMAC (16/64)")

    result = backtest.run_portfolio_backtest(
        universe,
        target_ann_vol=args.target_vol,
        cost_bps=args.cost_bps,
        weighting=args.weighting,
        signal_fn=signal_fn,
    )
    portfolio_returns = result["portfolio_returns"]

    bh_returns = {
        col: backtest.run_vol_targeted_buy_and_hold(universe[col], args.target_vol, args.cost_bps)
        for col in universe.columns
    }
    benchmark_returns = pd.DataFrame(bh_returns).mean(axis=1)

    print("\n=== Portfolio performance summary ===")
    print(metrics.summary(portfolio_returns, "Strategy"))
    print("\n=== Benchmark (vol-targeted buy & hold) ===")
    print(metrics.summary(benchmark_returns, "Benchmark"))

    png_path = report.plot_performance(portfolio_returns, benchmark_returns)
    print(f"\nSaved performance chart -> {png_path}")

    md_table = report.metrics_markdown_table(portfolio_returns, benchmark_returns)
    Path("reports/metrics_table.md").write_text(md_table)
    print(f"Saved metrics table -> reports/metrics_table.md")

    html_path = report.try_quantstats_html(portfolio_returns, benchmark_returns)
    if html_path:
        print(f"Saved quantstats tearsheet -> {html_path}")
    else:
        print("quantstats not installed -- skipping HTML tearsheet.")

    print("\n=== Walk-forward out-of-sample Sharpe by fold ===")
    for i, (train_idx, test_idx) in enumerate(backtest.walk_forward_splits(universe.index, n_splits=4)):
        fold_returns = portfolio_returns.reindex(test_idx)
        sharpe = metrics.sharpe_ratio(fold_returns)
        print(f"  Fold {i+1}: {test_idx[0].date()} to {test_idx[-1].date()}  ->  Sharpe {sharpe:.2f}")


if __name__ == "__main__":
    main()
