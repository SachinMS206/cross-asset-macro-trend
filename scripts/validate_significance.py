"""
validate_significance.py

Applies the Deflated Sharpe Ratio and Probability of Backtest Overfitting
to the actual set of EWMAC speed variants documented in CRITIQUE.md's own
sensitivity table (4/16, 8/32, 16/64, 32/128, 64/256) plus the multi-speed
blend that CRITIQUE.md recommends as the fix -- six real trials, not an
invented number, since this is genuinely the set of configurations that
was tried on this data.

Usage:
    python scripts/validate_significance.py --live
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import backtest, data_loaders, metrics, signals, stats_validation as sv

SPEED_TRIALS = [(4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]

START_MARKER = "<!-- LIVE_RESULTS_START -->"
END_MARKER = "<!-- LIVE_RESULTS_END -->"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--pbo-splits", type=int, default=10)
    parser.add_argument("--out-doc", default="STATISTICAL_VALIDATION.md")
    args = parser.parse_args()

    if args.live:
        print("Pulling live data from FRED / Yahoo Finance ...")
        universe = data_loaders.load_live_universe(start=args.start)
    else:
        print("Using SYNTHETIC data (offline demo mode).")
        universe = data_loaders.generate_synthetic_macro_universe()

    print(f"Universe: {list(universe.columns)}  |  {universe.index[0].date()} to {universe.index[-1].date()}")

    trial_returns = {}
    trial_sharpes = {}

    for fast, slow in SPEED_TRIALS:
        label = f"{fast}/{slow}"
        result = backtest.run_portfolio_backtest(universe, fast_span=fast, slow_span=slow)
        r = result["portfolio_returns"]
        trial_returns[label] = r
        trial_sharpes[label] = metrics.sharpe_ratio(r)
        print(f"  EWMAC {label:8s} Sharpe = {trial_sharpes[label]:.3f}")

    multi_speed_fn = signals.multi_speed_ewmac_signal
    result = backtest.run_portfolio_backtest(universe, signal_fn=multi_speed_fn)
    r = result["portfolio_returns"]
    trial_returns["multi-speed"] = r
    trial_sharpes["multi-speed"] = metrics.sharpe_ratio(r)
    print(f"  EWMAC {'multi-speed':8s} Sharpe = {trial_sharpes['multi-speed']:.3f}")

    chosen_label = "multi-speed"
    chosen_sharpe = trial_sharpes[chosen_label]
    chosen_returns = trial_returns[chosen_label]
    n_obs = chosen_returns.dropna().shape[0]
    skew = float(chosen_returns.dropna().skew())
    kurt = float(chosen_returns.dropna().kurtosis()) + 3.0  # pandas kurtosis is excess kurtosis

    dsr_result = sv.deflated_sharpe_ratio(
        observed_sharpe=chosen_sharpe,
        trial_sharpes=list(trial_sharpes.values()),
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurt,
    )

    print("\n=== Deflated Sharpe Ratio ===")
    print(f"Chosen strategy: {chosen_label}, observed Sharpe = {chosen_sharpe:.3f}, n_obs = {n_obs}")
    print(f"Trials considered: {dsr_result['n_trials']} ({list(trial_sharpes.keys())})")
    print(f"Std dev of Sharpe across trials: {dsr_result['trial_sharpe_std']:.3f}")
    print(f"Expected max Sharpe by chance alone given {dsr_result['n_trials']} trials: {dsr_result['benchmark_sharpe']:.3f}")
    print(f"PSR vs. zero (ignores selection bias): {dsr_result['psr_vs_zero']:.1%}")
    print(f"Deflated Sharpe Ratio (accounts for {dsr_result['n_trials']} trials): {dsr_result['deflated_sharpe_ratio']:.1%}")

    aligned = pd.DataFrame(trial_returns).dropna()
    pbo_result = sv.pbo_cscv(aligned, n_splits=args.pbo_splits)

    print("\n=== Probability of Backtest Overfitting (CSCV) ===")
    print(f"Aligned sample: {aligned.index[0].date()} to {aligned.index[-1].date()} ({len(aligned)} days)")
    print(f"Combinations tested: {pbo_result['n_combinations']}")
    print(f"Most frequent in-sample winner: {pbo_result['most_frequent_in_sample_winner']}")
    print(f"PBO: {pbo_result['pbo']:.1%}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    dsr_pct = dsr_result["deflated_sharpe_ratio"]
    if dsr_pct >= 0.95:
        dsr_read = f"**{chosen_label}** clears a high bar even after accounting for how many speeds were tried"
    elif dsr_pct >= 0.5:
        dsr_read = f"**{chosen_label}** still looks better than chance after deflation, but not decisively"
    else:
        dsr_read = f"once deflated for the number of trials, **{chosen_label}**'s Sharpe is NOT distinguishable from what chance alone would produce -- the honest reading is that choosing it over the alternatives is not statistically validated"

    best_live_label = max(trial_sharpes, key=trial_sharpes.get)
    chosen_is_best = best_live_label == chosen_label
    if chosen_is_best:
        best_note = f"**{chosen_label}** is also the best performer among all {len(trial_sharpes)} trials on this live run (Sharpe {trial_sharpes[best_live_label]:.3f})."
    else:
        best_note = (
            f"**On this live run, {chosen_label} is NOT the best performer** -- "
            f"**{best_live_label}** (Sharpe {trial_sharpes[best_live_label]:.3f}) outperforms it "
            f"(Sharpe {chosen_sharpe:.3f}). The multi-speed blend was adopted to reduce fragility "
            f"across nearby speed choices (CRITIQUE.md), not because it was expected to be the "
            f"single best performer -- but on real data, a single speed currently beats it outright, "
            f"which is worth stating plainly rather than only reporting the blend's own numbers."
        )

    in_sample_winner = pbo_result["most_frequent_in_sample_winner"]
    pbo_pct = pbo_result["pbo"]
    pbo_note = (
        f"**PBO evaluates whichever variant wins each in-sample half** (most often **{in_sample_winner}** "
        f"here, not necessarily {chosen_label}) and asks whether that in-sample winner still ranks "
        f"above-median out-of-sample. "
    )
    if pbo_pct >= 0.4:
        pbo_read = pbo_note + "A PBO this close to 50% is a real warning sign that the in-sample-best variant does not reliably generalize."
    elif pbo_pct >= 0.2:
        pbo_read = pbo_note + "A PBO in this range is meaningfully below 50% but not low enough to call the in-sample-best variant's edge robustly validated."
    else:
        pbo_read = pbo_note + "A low PBO here means the in-sample-best variant does tend to generalize out-of-sample -- a separate question from whether the *chosen* multi-speed blend specifically is statistically justified (that's what DSR above addresses)."

    trial_table = pd.Series(trial_sharpes).to_frame("Sharpe")
    trial_table.index.name = "Variant"

    block = (
        f"{START_MARKER}\n"
        f"*Last refreshed automatically from live data: {timestamp}. "
        f"See `.github/workflows/refresh-significance.yml`.*\n\n"
        f"**Trials considered** (the five speeds in CRITIQUE.md's own sensitivity table, plus the multi-speed blend):\n\n"
        f"{trial_table.map(lambda v: f'{v:.3f}').to_markdown()}\n\n"
        f"{best_note}\n\n"
        f"**Chosen strategy: {chosen_label}** (observed Sharpe {chosen_sharpe:.3f}, n={n_obs} days)\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Expected max Sharpe by chance alone ({dsr_result['n_trials']} trials) | {dsr_result['benchmark_sharpe']:.3f} |\n"
        f"| PSR vs. zero (no selection-bias correction) | {dsr_result['psr_vs_zero']:.1%} |\n"
        f"| **Deflated Sharpe Ratio (for {chosen_label})** | **{dsr_pct:.1%}** |\n"
        f"| **Probability of Backtest Overfitting** | **{pbo_pct:.1%}** |\n\n"
        f"**Deflated Sharpe Ratio, read honestly:** {dsr_read}.\n\n"
        f"**PBO, read honestly:** {pbo_read}\n"
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
